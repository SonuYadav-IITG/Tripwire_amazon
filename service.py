"""Domain logic for Tripwire (transport-agnostic, so it is trivially testable).

Lifecycle of a decision:  active --(tripwire trips / revisit date passes)--> revisit
                          revisit --resolve(keep)--> active | resolve(change) --> changed | resolve(close) --> closed

Safety principles
* The language model may *extract* and *relate*; only this module changes state, and numeric tripwires are
  always compared here, never by the model.
* Tripwires have stable ids, so a verdict computed while someone else edits the decision can never be applied
  to the wrong tripwire.
* Every mutation is an atomic compare-and-swap read-modify-write (``Store.update``): concurrent requests are
  merged, not overwritten, and re-running an already-applied observation is a no-op (idempotent).
"""

from __future__ import annotations

import operator
import os
import re
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Callable

from . import ai
from .store import ConflictError, Store

KIND = "decision"
OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
NEAR = 0.10  # within 10% of a threshold counts as "getting close"
LIVE = ("active", "revisit")
MAX_DECISIONS, MAX_TRIPWIRES, MAX_HISTORY = 300, 8, 50


def _fmt(v: float) -> str:
    return f"{v:g}"


def _speak_list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _day(d: str) -> str:
    return date.fromisoformat(d).strftime("%B %-d")


def evaluate(tw: dict[str, Any], observed: float) -> bool:
    return OPS[tw["op"]](observed, tw["value"])


def _is_numeric(tw: dict[str, Any]) -> bool:
    return bool(tw.get("metric")) and tw.get("op") in OPS and tw.get("value") is not None


def default_today() -> date:
    """Today in TRIPWIRE_TZ (e.g. 'America/Los_Angeles'); falls back to the server's local date."""
    tz = os.environ.get("TRIPWIRE_TZ")
    if tz:
        try:
            from zoneinfo import ZoneInfo

            return datetime.now(ZoneInfo(tz)).date()
        except Exception:  # noqa: BLE001 - bad tz name must not break every request
            pass
    return date.today()


def _new_tripwire(t: dict[str, Any]) -> dict[str, Any]:
    return {**t, "id": uuid.uuid4().hex[:8], "state": "armed"}


class Tripwire:
    def __init__(self, store: Store, household: str = "home", today: Callable[[], date] = default_today):
        self.s, self.h, self._today = store, household, today

    # ------------------------------------------------------------------ helpers
    def _all(self) -> list[dict[str, Any]]:
        return [{**d, "key": d["_key"]} for d in self.s.list(self.h, KIND)]

    def _event(self, d: dict[str, Any], text: str) -> None:
        hist = d.setdefault("history", [])
        hist.append({"on": self._today().isoformat(), "text": text[:300]})
        if len(hist) > MAX_HISTORY:  # keep the original decision + the most recent events; documents stay small
            d["history"] = [hist[0]] + hist[-(MAX_HISTORY - 1):]

    def _find(self, topic: str, statuses: tuple[str, ...] | None = None):
        """Resolve a spoken topic ('the internet thing') to one decision, or return a speakable error."""
        topic = ai._clean_text(topic or "")
        cands = [d for d in self._all() if statuses is None or d["status"] in statuses]
        want = ai.tokens(topic)
        scored = []
        for d in cands:
            have = ai.tokens(" ".join([d["title"], d["choice"], d.get("reason", ""), " ".join(d.get("options", []))]))
            scored.append((len(want & have), d))
        scored.sort(key=lambda x: -x[0])
        if not scored or scored[0][0] == 0:
            return None, {"ok": False, "spoken": f"I don't have a decision about {topic or 'that'}."}
        if len(scored) > 1 and scored[1][0] == scored[0][0]:
            names = _speak_list([s[1]["title"] for s in scored if s[0] == scored[0][0]])
            return None, {"ok": False, "ambiguous": True, "spoken": f"Which one do you mean: {names}?"}
        return scored[0][1], None

    def _edit(self, key: str, statuses: tuple[str, ...] | None, fn: Callable[[dict[str, Any]], tuple[dict[str, Any] | None, dict[str, Any]]]) -> dict[str, Any]:
        """Atomically edit one decision. ``fn(doc) -> (new_doc | None, response)`` runs against the *freshest* copy
        (and again on conflict), so preconditions are re-checked on current data."""
        box: dict[str, Any] = {}

        def wrapper(doc):
            if doc is None or (statuses and doc["status"] not in statuses):
                box["resp"] = {"ok": False, "spoken": "That decision just changed. Please ask again."}
                return None
            new, box["resp"] = fn(doc)
            return new

        self.s.update(self.h, KIND, key, wrapper)
        return box["resp"]

    def _unique_key(self, title: str, taken: set[str]) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "decision"
        key, n = base, 2
        while key in taken:
            key, n = f"{base}-{n}", n + 1
        return key

    def _due_flag(self, d: dict[str, Any]) -> bool:
        return bool(d.get("revisit_on")) and date.fromisoformat(d["revisit_on"]) <= self._today()

    def _due(self, d: dict[str, Any]) -> bool:
        return self._due_flag(d) and d["status"] == "active"

    @staticmethod
    def _public(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if k not in ("_key",)}

    @staticmethod
    def _bad_number(n: Any, count: int) -> bool:
        return not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= count

    # ------------------------------------------------------------------ record
    def record_decision(self, what_was_said: str) -> dict[str, Any]:
        text = ai._clean_text(what_was_said or "")
        if len(text) < 8:
            return {"ok": False, "spoken": "I didn't catch that. Try: we're going with X because Y, and we'd change our mind if Z."}
        try:
            ext, engine = ai.extract_decision(text)
        except Exception:  # noqa: BLE001 - speakable, not a stack trace
            return {"ok": False, "spoken": "I couldn't make sense of that decision. Try: we're going with X because Y, and we'd change our mind if Z."}
        existing = self._all()
        # Idempotency: a client that times out and retries must not create a duplicate (which would make every
        # later "the internet one" ambiguous). Same live decision recorded today with the same content = same request.
        sig = (ai.tokens(ext["choice"]), tuple(sorted(t["description"].lower() for t in ext["tripwires"])))
        for d in existing:
            if (d["status"] in LIVE and d["created_on"] == self._today().isoformat()
                    and (ai.tokens(d["choice"]), tuple(sorted(t["description"].lower() for t in d["tripwires"]))) == sig):
                return {"ok": True, "engine": engine, "duplicate": True, "decision": d,
                        "spoken": f"I already have that one: {d['choice']}. Nothing changed."}
        if len(existing) >= MAX_DECISIONS:
            return {"ok": False, "spoken": "I'm tracking a lot of decisions already. Close a few and I'll make room."}
        today = self._today()
        doc = {
            "title": ext["title"], "choice": ext["choice"], "options": ext["options"], "reason": ext["reason"],
            "status": "active", "created_on": today.isoformat(),
            "tripwires": [_new_tripwire(t) for t in ext["tripwires"][:MAX_TRIPWIRES]],
            "revisit_on": (today + timedelta(days=ext["revisit_in_days"])).isoformat() if ext["revisit_in_days"] else None,
            "history": [],
        }
        self._event(doc, f"Decided: {doc['choice']}" + (f" because {doc['reason']}" if doc["reason"] else ""))
        taken = {d["key"] for d in existing}
        for _ in range(25):  # create-only write (expected_ver=0); on a name race, take the next free key
            key = self._unique_key(ext["title"], taken)
            try:
                self.s.put(self.h, KIND, key, doc, expected_ver=0)
                break
            except ConflictError:
                taken = {d["key"] for d in self._all()}
        else:
            raise ConflictError("could not allocate a decision key")
        spoken = f"Got it. You chose {doc['choice']}" + (f", because {doc['reason']}." if doc["reason"] else ".")
        if doc["tripwires"]:
            numbered = [f"{i}, {t['description']}" for i, t in enumerate(doc["tripwires"], 1)]
            spoken += f" I'm watching for: {'; '.join(numbered)}."
        else:
            spoken += " I didn't hear what would change your mind. Tell me and I'll add a tripwire."
        if doc["revisit_on"]:
            spoken += f" I'll flag it on {_day(doc['revisit_on'])}."
        return {"ok": True, "engine": engine, "decision": {**doc, "key": key}, "spoken": spoken}

    def add_tripwire(self, topic: str, condition: str) -> dict[str, Any]:
        condition = ai._clean_text(condition or "")
        if len(condition) < 3:
            return {"ok": False, "spoken": "What condition should I watch for?"}
        d, err = self._find(topic, LIVE)
        if err:
            return err
        engine, found = "none", []
        try:  # reuse the decision extractor, framing the condition so it parses like a spoken decision
            ext, engine = ai.extract_decision(f"We're going with {d['choice']}. We'd change our mind if {condition}")
            found = ext["tripwires"]
        except Exception:  # noqa: BLE001 - fall back to a plain, human-confirmed tripwire
            pass
        blank = {"metric": None, "op": None, "value": None, "unit": ""}
        new = [_new_tripwire(t) for t in found] or [_new_tripwire({"description": condition[:160], **blank})]

        def fn(doc):
            have = {t["description"].lower() for t in doc["tripwires"]}
            new_ = [t for t in new if t["description"].lower() not in have]
            if not new_:  # retry-safe: already watching exactly this
                return None, {"ok": True, "engine": engine, "spoken": f"I'm already watching for {new[0]['description']}."}
            room = MAX_TRIPWIRES - len(doc["tripwires"])
            if room <= 0:
                return None, {"ok": False, "spoken": f"{doc['title'].capitalize()} already has {MAX_TRIPWIRES} tripwires. Remove one first."}
            doc["tripwires"].extend(new_[:room])
            self._event(doc, f"Added tripwire: {new_[0]['description']}")
            return doc, {"ok": True, "engine": engine, "spoken": f"Added. For {doc['title']} I'm now watching {len(doc['tripwires'])} things, including {new_[0]['description']}."}

        return self._edit(d["key"], LIVE, fn)

    def remove_tripwire(self, topic: str, number: int) -> dict[str, Any]:
        d, err = self._find(topic, LIVE)
        if err:
            return err

        def fn(doc):
            if self._bad_number(number, len(doc["tripwires"])):
                return None, {"ok": False, "spoken": f"{doc['title'].capitalize()} has {len(doc['tripwires'])} tripwires. Which number?"}
            gone = doc["tripwires"].pop(number - 1)
            self._event(doc, f"Removed tripwire: {gone['description']}")
            if doc["status"] == "revisit" and not any(t["state"] in ("tripped", "pending") for t in doc["tripwires"]) and not self._due_flag(doc):
                doc["status"] = "active"
            return doc, {"ok": True, "spoken": f"Removed {gone['description']}."}

        return self._edit(d["key"], LIVE, fn)

    # ------------------------------------------------------------------ the core: reporting facts
    def _apply_observations(self, doc: dict[str, Any], observations: list[dict[str, Any]], statement: str):
        """Pure function of (doc, observations): returns (new_doc | None if unchanged, result lists).
        Observations name tripwires by stable id; ones that no longer exist, or are no longer armed (already
        handled by a concurrent/previous report), are skipped, so applying twice is a no-op."""
        by_id = {t["id"]: t for t in doc["tripwires"] if "id" in t}
        res: dict[str, list] = {"tripped": [], "maybe": [], "near": []}
        changed = False
        for o in observations:
            tw = by_id.get(o["tripwire_id"].rsplit("#", 1)[-1])
            if tw is None or tw["state"] != "armed":
                continue
            val = o["observed_value"]
            if _is_numeric(tw) and val is not None:
                # server-side truth: the model's verdict is ignored for numeric tripwires
                if evaluate(tw, val):
                    tw.update(state="tripped", observed=val, on=self._today().isoformat(), statement=statement[:200])
                    self._event(doc, f"Tripped: {tw['description']} (observed {_fmt(val)})")
                    res["tripped"].append((doc["title"], tw["description"], val))
                    changed = True
                elif tw["value"] and abs(val - tw["value"]) / abs(tw["value"]) <= NEAR:
                    res["near"].append((doc["title"], tw["description"], val, tw["value"]))
            elif not _is_numeric(tw) and o["verdict"] in ("tripped", "unsure"):
                tw.update(state="pending", statement=statement[:200], note=o.get("note", ""))
                res["maybe"].append((doc["title"], tw["description"]))
                changed = True
        if any(t["state"] == "tripped" for t in doc["tripwires"]) and doc["status"] == "active":
            doc["status"] = "revisit"
            changed = True
        return (doc if changed else None), res

    def report_fact(self, statement: str) -> dict[str, Any]:
        statement = ai._clean_text(statement or "")
        if len(statement) < 3:
            return {"ok": False, "spoken": "What did you want me to note?"}
        decisions = [d for d in self._all() if d["status"] in LIVE]
        items = [{"id": f"{d['key']}#{t['id']}", "decision": d["title"], **{k: t.get(k) for k in ("description", "metric", "op", "value", "unit")}}
                 for d in decisions for t in d["tripwires"] if t["state"] == "armed" and "id" in t]
        if not items:
            return {"ok": True, "tripped": [], "spoken": "Noted, but I'm not watching anything that this relates to."}
        try:
            obs, engine = ai.assess_fact(statement, items)
        except Exception:  # noqa: BLE001
            return {"ok": False, "spoken": "I couldn't check that against your tripwires just now. Please try again."}
        grouped: dict[str, list] = defaultdict(list)
        for o in obs:
            grouped[o["tripwire_id"].rsplit("#", 1)[0]].append(o)
        agg: dict[str, list] = {"tripped": [], "maybe": [], "near": []}
        for key, group in grouped.items():
            box: dict[str, Any] = {}

            def fn(doc, group=group, box=box):
                if doc is None or doc["status"] not in LIVE:
                    box["res"] = None
                    return None
                new, box["res"] = self._apply_observations(doc, group, statement)
                return new

            self.s.update(self.h, KIND, key, fn)
            for k, v in (box.get("res") or {}).items():
                agg[k].extend(v)
        parts = []
        for title, desc, val in agg["tripped"]:
            parts.append(f"That trips your {title} decision: {desc}, and it's now {_fmt(val)}.")
        for title, desc in agg["maybe"]:
            parts.append(f"That might trip your {title} tripwire, {desc}. Did it?")
        for title, desc, val, limit in agg["near"]:
            parts.append(f"Heads up: {title} is at {_fmt(val)}, close to your limit of {_fmt(limit)}.")
        if agg["tripped"]:
            parts.append("Want to keep it, change it, or adjust the limit?")
        return {"ok": True, "engine": engine, "tripped": [t[0] for t in agg["tripped"]],
                "needs_confirmation": [m[0] for m in agg["maybe"]], "close_to_limit": [n[0] for n in agg["near"]],
                "spoken": " ".join(parts) or "Noted. Nothing tripped."}

    def confirm_tripwire(self, topic: str, number: int, tripped: bool) -> dict[str, Any]:
        d, err = self._find(topic, LIVE)
        if err:
            return err

        def fn(doc):
            if self._bad_number(number, len(doc["tripwires"])):
                return None, {"ok": False, "spoken": f"{doc['title'].capitalize()} has {len(doc['tripwires'])} tripwires. Which number?"}
            tw = doc["tripwires"][number - 1]
            if tripped:
                tw.update(state="tripped", on=self._today().isoformat())
                doc["status"] = "revisit"
                self._event(doc, f"Confirmed tripped: {tw['description']}")
                spoken = f"Confirmed. {doc['title'].capitalize()} needs a second look. Keep it, change it, or adjust the limit?"
            else:
                tw["state"] = "armed"
                tw.pop("statement", None)
                self._event(doc, f"Dismissed false alarm: {tw['description']}")
                spoken = "Okay, still watching it."
            return doc, {"ok": True, "spoken": spoken}

        return self._edit(d["key"], LIVE, fn)

    # ------------------------------------------------------------------ revisiting
    def resolve_decision(self, topic: str, outcome: str, note: str | None = None, new_limit: float | None = None,
                         tripwire_number: int | None = None) -> dict[str, Any]:
        if outcome not in ("keep", "change", "close"):
            return {"ok": False, "spoken": "You can keep it, change it, or close it."}
        if new_limit is not None and ai._finite(new_limit) is None:
            return {"ok": False, "spoken": "That doesn't sound like a valid limit."}
        note = ai._clean_text(note)[:200] if note else None
        d, err = self._find(topic, LIVE)
        if err:
            return err

        def fn(doc):
            if outcome == "keep":
                moved = []
                if new_limit is not None:
                    tws = doc["tripwires"]
                    targets = [i for i, t in enumerate(tws, 1) if t["state"] in ("tripped", "pending") and t.get("value") is not None]
                    if tripwire_number is not None:
                        if self._bad_number(tripwire_number, len(tws)) or tws[tripwire_number - 1].get("value") is None:
                            return None, {"ok": False, "spoken": "That tripwire doesn't have a numeric limit I can move."}
                        targets = [tripwire_number]
                    if len(targets) != 1:  # never guess which limit to move
                        opts = "; ".join(f"{i}, {tws[i - 1]['description']}" for i in targets) or "none of them has a numeric limit"
                        return None, {"ok": False, "spoken": f"Which limit should I move? {opts}."}
                    tw = tws[targets[0] - 1]
                    tw["value"] = float(new_limit)
                    tw["description"] = re.sub(r"\$?\d[\d,]*(?:\.\d+)?", ("$" if "$" in tw["description"] else "") + _fmt(float(new_limit)), tw["description"], count=1)
                    moved.append(tw["description"])
                for tw in doc["tripwires"]:
                    if tw["state"] in ("tripped", "pending"):
                        tw["state"] = "armed"
                        for k in ("observed", "on", "statement", "note"):
                            tw.pop(k, None)
                doc["status"] = "active"
                if doc.get("revisit_on") and self._due_flag(doc):
                    doc["revisit_on"] = None
                self._event(doc, "Kept" + (f": {note}" if note else "") + (f"; limit moved ({moved[0]})" if moved else ""))
                spoken = f"Kept {doc['choice']}." + (f" New limit: {moved[0]}." if moved else " I'm watching again.")
            elif outcome == "change":
                doc["status"] = "changed"
                self._event(doc, "Changed" + (f": {note}" if note else ""))
                spoken = f"Okay, I've marked {doc['title']} as changed" + (f" to {note}." if note else ". Record the new decision whenever you're ready.")
            else:
                doc["status"] = "closed"
                self._event(doc, "Closed" + (f": {note}" if note else ""))
                spoken = f"Closed {doc['title']}. I'll stop watching it."
            return doc, {"ok": True, "status": doc["status"], "spoken": spoken}

        return self._edit(d["key"], LIVE, fn)

    # ------------------------------------------------------------------ reading back
    def whats_on_watch(self) -> dict[str, Any]:
        live = [d for d in self._all() if d["status"] in LIVE]
        if not live:
            return {"ok": True, "decisions": [], "spoken": "You're not watching any decisions yet."}
        attention, calm = [], []
        for d in live:
            if d["status"] == "revisit":
                trips = [t for t in d["tripwires"] if t["state"] in ("tripped", "pending")]
                attention.append((d["title"], f"{d['title']}, because {trips[0]['description']}" if trips else f"{d['title']}, it's time to revisit"))
            elif self._due(d):
                attention.append((d["title"], f"{d['title']}, its check-in date has arrived"))
            else:
                calm.append(d["title"])
        spoken = f"You're watching {len(live)} decision{'s' if len(live) != 1 else ''}."
        if attention:
            spoken += f" Needs attention: {_speak_list([a[1] for a in attention])}."
        if calm:
            spoken += f" All quiet on {_speak_list(calm)}."
        return {"ok": True, "needs_attention": [a[0] for a in attention], "decisions": [self._public(d) for d in live], "spoken": spoken}

    def why_did_we(self, topic: str) -> dict[str, Any]:
        d, err = self._find(topic)
        if err:
            return err
        spoken = f"On {_day(d['created_on'])} you chose {d['choice']}"
        spoken += f" because {d['reason']}." if d.get("reason") else "."
        if d.get("options"):
            spoken += f" You also considered {_speak_list(d['options'])}."
        if d["tripwires"]:
            spoken += f" You said you'd reconsider if {_speak_list([t['description'] for t in d['tripwires']])}."
        later = d["history"][1:]
        if later:
            spoken += " Since then: " + "; ".join(f"{_day(h['on'])}, {h['text'].lower()}" for h in later[-3:]) + "."
        spoken += f" Status: {d['status']}."
        return {"ok": True, "decision": self._public(d), "spoken": spoken}
