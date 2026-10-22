"""Language understanding for Tripwire.

Two jobs, both done through Amazon Bedrock's Converse API with *forced tool use* (the model must
answer by calling a function whose arguments match a JSON schema):

1. ``extract_decision``: turn a spoken, messy decision ("we're going with the 200 meg plan because
   it's cheapest, we'd switch if the price goes over fifty or outages hit more than twice a month")
   into a structured record with explicit, machine-checkable tripwires.
2. ``assess_fact``: given a new fact ("the provider raised the price to $55") and every armed
   tripwire, say which ones it relates to and what number was observed.

The model only *proposes*. ``service.py`` re-validates everything and performs the numeric
comparison itself, so a hallucinated verdict can never silently flip a decision.

If Bedrock is unavailable, a small rule-based fallback keeps the server usable (and the tests
hermetic). Results always say which engine produced them.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from typing import Any

OPS = (">", ">=", "<", "<=")
STOP = {"the", "a", "an", "our", "my", "we", "i", "it", "is", "are", "was", "to", "of", "and", "or", "in",
        "on", "for", "that", "this", "than", "per", "now", "has", "have", "had", "with", "at", "be", "will"}

NUMWORDS = {"once": 1, "twice": 2, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "twenty": 20, "thirty": 30, "forty": 40,
            "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "a": 1, "an": 1}
MAX_PROMPT_ITEMS = 60          # cap tripwires sent to the model (cost, latency, context limits)
MAX_TEXT = 1500                # hard cap on any user text we process


def norm(word: str) -> str:
    w = re.sub(r"[^a-z]", "", word.lower())
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def tokens(text: str) -> set[str]:
    return {norm(w) for w in re.findall(r"[A-Za-z']+", text) if norm(w) and norm(w) not in STOP}


# ------------------------------------------------------------------------------ schemas

EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Short topic name, 2-5 words, e.g. 'internet plan'."},
        "choice": {"type": "string", "description": "What was decided, e.g. 'the 200 Mbps plan'."},
        "options": {"type": "array", "items": {"type": "string"}, "description": "Alternatives that were considered."},
        "reason": {"type": "string", "description": "Why, in one short sentence."},
        "tripwires": {
            "type": "array",
            "description": "Conditions that would make the person reconsider. Prefer numeric ones.",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Plain-English, speakable, e.g. 'price goes above $50'."},
                    "metric": {"type": "string", "description": "What is measured, e.g. 'price' or 'outages per month'. Omit if not measurable."},
                    "op": {"type": "string", "enum": list(OPS)},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                },
                "required": ["description"],
            },
        },
        "revisit_in_days": {"type": "integer", "description": "Only if the person asked to be reminded or to check back after a period."},
    },
    "required": ["title", "choice", "reason", "tripwires"],
}

ASSESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "description": "One entry per tripwire the fact is relevant to. Omit unrelated tripwires.",
            "items": {
                "type": "object",
                "properties": {
                    "tripwire_id": {"type": "string"},
                    "observed_value": {"type": "number", "description": "The number the fact states for this tripwire's metric, if any."},
                    "verdict": {"type": "string", "enum": ["tripped", "not_tripped", "unsure"]},
                    "note": {"type": "string"},
                },
                "required": ["tripwire_id", "verdict"],
            },
        }
    },
    "required": ["observations"],
}

EXTRACT_SYSTEM = (
    "You are Tripwire, a voice assistant that remembers household decisions together with the conditions "
    "that would change them. The user's words come from speech recognition, so expect run-on sentences and "
    "numbers spelled out. Extract the decision and its tripwires by calling submit_decision exactly once. "
    "Convert spoken numbers to digits. Only include tripwires the person actually stated; never invent any. "
    "If a tripwire is a measurable threshold, fill metric/op/value/unit so it can be checked automatically."
)
ASSESS_SYSTEM = (
    "You check whether a newly reported fact trips any of a household's armed tripwires. Call submit_assessment "
    "exactly once. Only mention tripwires the fact is genuinely about (same subject). Report observed_value "
    "whenever the fact states the relevant quantity. Use verdict 'tripped' only when the fact clearly satisfies "
    "the condition, 'not_tripped' when it clearly does not, and 'unsure' when it is ambiguous."
)


# ------------------------------------------------------------------------------ Bedrock


def _bedrock_client():
    import boto3
    from botocore.config import Config

    # Bounded latency: a voice turn must not hang. Worst case ~ total_attempts x read_timeout (default 2 x 6s).
    # NB: botocore's `max_attempts` counts *retries* (so 2 would mean 3 tries); `total_max_attempts` is what we mean.
    cfg = Config(
        connect_timeout=float(os.environ.get("TRIPWIRE_BEDROCK_CONNECT_TIMEOUT", "3")),
        read_timeout=float(os.environ.get("TRIPWIRE_BEDROCK_READ_TIMEOUT", "6")),
        retries={"total_max_attempts": int(os.environ.get("TRIPWIRE_BEDROCK_ATTEMPTS", "2")), "mode": "standard"},
    )
    return boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION"), config=cfg)


def _request(system: str, user: str, tool_name: str, schema: dict[str, Any], model_id: str, with_temperature: bool) -> dict[str, Any]:
    inference: dict[str, Any] = {"maxTokens": 1500}
    if with_temperature:
        inference["temperature"] = 0
    return dict(
        modelId=model_id,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        toolConfig={
            "tools": [{"toolSpec": {"name": tool_name, "description": f"Submit the {tool_name[7:]}.", "inputSchema": {"json": schema}}}],
            "toolChoice": {"tool": {"name": tool_name}},
        },
        inferenceConfig=inference,
    )


def converse_tool(system: str, user: str, tool_name: str, schema: dict[str, Any], client: Any = None) -> dict[str, Any]:
    model_id = os.environ.get("BEDROCK_MODEL_ID")
    if not model_id:
        raise RuntimeError("BEDROCK_MODEL_ID is not set")
    client = client or _bedrock_client()
    try:
        resp = client.converse(**_request(system, user, tool_name, schema, model_id, True))
    except Exception as e:  # noqa: BLE001
        # Some models reject an explicit temperature; retry once without it. Anything else propagates.
        if "temperature" in str(e).lower() and "validation" in str(e).lower():
            resp = client.converse(**_request(system, user, tool_name, schema, model_id, False))
        else:
            raise
    # A truncated tool call can still parse into a *partial* dict (e.g. tripwires silently missing). Refuse it.
    if resp.get("stopReason") != "tool_use":
        raise RuntimeError(f"Bedrock stopped with {resp.get('stopReason')!r}; refusing a possibly truncated answer")
    for block in resp["output"]["message"]["content"]:
        if "toolUse" in block:
            return block["toolUse"]["input"]
    raise RuntimeError("Bedrock response contained no tool use")


# ------------------------------------------------------------------------------ circuit breaker
# After N consecutive Bedrock failures, stop trying for a cool-down so every voice turn doesn't pay the
# full timeout while Bedrock is down. In 'auto' mode we use the offline parser meanwhile; in 'bedrock' mode we fail fast.

class _Breaker:
    def __init__(self):
        self._lock = threading.Lock()
        self.fails, self.open_until = 0, 0.0

    @staticmethod
    def _limits() -> tuple[int, float]:
        return int(os.environ.get("TRIPWIRE_BREAKER_FAILS", "3")), float(os.environ.get("TRIPWIRE_BREAKER_COOLDOWN", "60"))

    def is_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self.open_until

    def success(self) -> None:
        with self._lock:
            self.fails, self.open_until = 0, 0.0

    def failure(self) -> None:
        threshold, cooldown = self._limits()
        with self._lock:
            self.fails += 1
            if self.fails >= threshold:
                self.open_until = time.monotonic() + cooldown

    def reset(self) -> None:
        self.success()


BREAKER = _Breaker()


def _finite(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clean_tripwire(t: dict[str, Any]) -> dict[str, Any] | None:
    desc = str(t.get("description", "")).strip()
    if not desc:
        return None
    out: dict[str, Any] = {"description": desc[:160], "metric": None, "op": None, "value": None, "unit": ""}
    value = _finite(t.get("value"))
    if t.get("op") in OPS and value is not None and str(t.get("metric", "")).strip():
        out.update(metric=str(t["metric"]).strip().lower()[:60], op=t["op"], value=value, unit=str(t.get("unit", "")).strip()[:20])
    return out


def validate_extraction(raw: dict[str, Any]) -> dict[str, Any]:
    choice = str(raw.get("choice", "")).strip()
    if not choice:
        raise ValueError("no decision found")
    tws = [c for c in (_clean_tripwire(t) for t in raw.get("tripwires", []) if isinstance(t, dict)) if c]
    days = _finite(raw.get("revisit_in_days"))
    return {
        "title": (str(raw.get("title", "")).strip() or choice)[:60].lower(),
        "choice": choice[:120],
        "options": [str(o).strip()[:80] for o in raw.get("options", []) if str(o).strip()][:6],
        "reason": str(raw.get("reason", "")).strip()[:240],
        "tripwires": tws[:8],
        "revisit_in_days": max(1, min(int(days), 3650)) if days else None,
    }


def validate_assessment(raw: dict[str, Any], known_ids: set[str]) -> list[dict[str, Any]]:
    out = []
    for o in raw.get("observations", []):
        if not isinstance(o, dict) or o.get("tripwire_id") not in known_ids:
            continue  # ignore hallucinated ids
        out.append({
            "tripwire_id": o["tripwire_id"],
            "observed_value": _finite(o.get("observed_value")) if not isinstance(o.get("observed_value"), bool) else None,
            "verdict": o.get("verdict") if o.get("verdict") in ("tripped", "not_tripped", "unsure") else "unsure",
            "note": str(o.get("note", ""))[:200],
        })
    return out


def assess_prompt(fact: str, items: list[dict[str, Any]]) -> str:
    lines = []
    for it in items:
        spec = f" [numeric: {it['metric']} {it['op']} {it['value']:g} {it.get('unit', '')}]" if it.get("metric") else ""
        lines.append(f"- id={it['id']} | decision: {it['decision']} | tripwire: {it['description']}{spec}")
    return "New fact: " + fact + "\n\nArmed tripwires:\n" + "\n".join(lines)


# ------------------------------------------------------------------------------ offline fallback

_DIR = r"(above|over|exceeds?|more than|higher than|at least|below|under|less than|fewer than|lower than)"
_NUM = r"(\$?\d[\d,]*(?:\.\d+)?|once|twice|one|two|three|four|five|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
_FILLER = {"the", "our", "my", "a", "an", "it", "goes", "gets", "go", "get", "rises", "climbs", "drops", "falls",
          "hits", "hit", "is", "are", "reaches", "reach", "ever", "ends", "up", "we", "i", "you", "they", "he", "she", "us"}
_UP = {"above", "over", "exceed", "exceeds", "more than", "higher than"}


def _num(s: str) -> float:
    s = s.replace("$", "").replace(",", "")
    return float(NUMWORDS[s]) if s in NUMWORDS else float(s)


def _clean_text(text: str) -> str:
    """Normalise what speech-to-text / phone keyboards produce: curly quotes, odd whitespace, absurd length."""
    t = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(t.split())[:MAX_TEXT]


def extract_offline(text: str) -> dict[str, Any]:
    low = _clean_text(text).lower()
    cm = re.search(
        r"(?:going with|go with|picking|pick|choosing|chose|decided on|decided to (?:get|buy|take|use)|"
        r"getting|buying|taking|we(?:'ll| will) (?:get|take|use))\s+(.+?)(?=\s+(?:because|since|but|we'd|we would|i'd|i would|if|unless)\b|[.,;]|$)", low)
    choice = cm.group(1).strip() if cm else ""
    rm = re.search(r"(?:because|since)\s+(.+?)(?=[.;,]|\s+(?:we'd|we would|i'd|i would)\b|$)", low)
    tws = []
    for m in re.finditer(_DIR + r"\s+" + _NUM + r"(?: times)?(?:\s+(?:a|per|each)\s+(month|week|year|day))?", low):
        seg = re.split(r"[,;.]|\b(?:if|or|and|unless|when|that|switch|change|than)\b", low[:m.start()])[-1]
        metric = " ".join(w for w in seg.split() if w not in _FILLER).strip()
        if not metric:
            continue
        op = ">" if m.group(1) in _UP else "<"
        if m.group(1) == "at least":
            op = ">="
        value = _num(m.group(2))
        unit = f"per {m.group(3)}" if m.group(3) else ("usd" if "$" in m.group(2) else "")
        shown = f"${value:g}" if "$" in m.group(2) else f"{value:g}"
        tws.append({"description": f"{metric} {m.group(1)} {shown}" + (f" {unit}" if unit.startswith("per") else ""),
                    "metric": metric + (f" {unit}" if unit.startswith("per") else ""), "op": op, "value": value, "unit": unit})
    rv = re.search(r"(?:revisit|check back|check in|review|remind (?:us|me))\b[^.]*?\b(?:in|after)\s+(\d+|a|one|two|three|four|six|twelve)\s+(day|week|month|year)s?", low)
    days = None
    if rv:
        n = float(NUMWORDS.get(rv.group(1), rv.group(1)) if not rv.group(1).isdigit() else rv.group(1))
        days = int(n * {"day": 1, "week": 7, "month": 30, "year": 365}[rv.group(2)])
    return validate_extraction({"title": " ".join([w for w in choice.split() if w not in {"the", "a", "an", "our", "my", "some"}][-2:]) if choice else "", "choice": choice,
                                "reason": rm.group(1) if rm else "", "tripwires": tws, "revisit_in_days": days})


def _last_number(text: str) -> float | None:
    found = re.findall(r"\$?\d[\d,]*(?:\.\d+)?|\b(?:once|twice|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b", _clean_text(text).lower())
    return _num(found[-1]) if found else None


def assess_offline(fact: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ftok, value, out = tokens(_clean_text(fact)), _last_number(fact), []
    numeric = [it for it in items if it.get("metric") and tokens(it["metric"]) and tokens(it["metric"]) <= ftok]
    for it in items:
        about_decision = bool(tokens(it["decision"]) & ftok)
        if it in numeric and value is not None and (about_decision or len(numeric) == 1):
            out.append({"tripwire_id": it["id"], "observed_value": value, "verdict": "unsure", "note": ""})
        elif not it.get("metric"):
            dt = tokens(it["description"])
            if dt and len(dt & ftok) / len(dt) >= 0.6:
                out.append({"tripwire_id": it["id"], "observed_value": None, "verdict": "unsure", "note": "wording overlaps"})
    return out


# ------------------------------------------------------------------------------ public API


def _mode() -> str:
    return os.environ.get("TRIPWIRE_AI", "auto").lower()


def _cap_items(fact: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bound prompt size: keep the tripwires most related to the fact when there are too many."""
    if len(items) <= MAX_PROMPT_ITEMS:
        return items
    ftok = tokens(fact)
    def rel(it):
        return len(ftok & (tokens(it["decision"]) | tokens(it["description"]) | tokens(it.get("metric") or "")))
    return sorted(items, key=rel, reverse=True)[:MAX_PROMPT_ITEMS]


def _via_bedrock(call):
    """Run a Bedrock call under the circuit breaker."""
    if BREAKER.is_open():
        raise RuntimeError("Bedrock temporarily unavailable (circuit open)")
    try:
        result = call()
    except Exception:
        BREAKER.failure()
        raise
    BREAKER.success()
    return result


def extract_decision(text: str) -> tuple[dict[str, Any], str]:
    text = _clean_text(text)
    if _mode() == "offline":
        return extract_offline(text), "offline"
    try:
        return validate_extraction(_via_bedrock(lambda: converse_tool(EXTRACT_SYSTEM, text, "submit_decision", EXTRACT_SCHEMA))), "bedrock"
    except Exception:
        if _mode() == "bedrock":
            raise
        return extract_offline(text), "offline-fallback"


def assess_fact(fact: str, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    fact = _clean_text(fact)
    if _mode() == "offline":
        return assess_offline(fact, items), "offline"
    sent = _cap_items(fact, items)
    known = {it["id"] for it in sent}
    try:
        raw = _via_bedrock(lambda: converse_tool(ASSESS_SYSTEM, assess_prompt(fact, sent), "submit_assessment", ASSESS_SCHEMA))
        return validate_assessment(raw, known), "bedrock"
    except Exception:
        if _mode() == "bedrock":
            raise
        return assess_offline(fact, items), "offline-fallback"
