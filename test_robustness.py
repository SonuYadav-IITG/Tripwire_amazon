"""Robustness: concurrency & mutation safety, stale references, idempotence, timeouts/retries, circuit breaker,
truncated/garbage model output, backend failures, adversarial input, resource limits."""

import threading
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import boto3
import pytest
from botocore.stub import Stubber
from moto import mock_aws

from conftest import INTERNET
from tripwire import ai, service as svc_mod
from tripwire.service import Tripwire
from tripwire.store import ConflictError, DynamoStore, SqliteStore

MODEL = "anthropic.claude-test-v1:0"


# ------------------------------------------------------------------ concurrency / mutation safety
class RacyStore(SqliteStore):
    """Forces the worst case: both writers have READ the same version before either one WRITES."""

    def __init__(self, parties=2):
        super().__init__(":memory:")
        self.barrier = threading.Barrier(parties, timeout=5)
        self.armed = False
        self._seen = threading.local()

    def get_v(self, household, kind, key):
        out = super().get_v(household, kind, key)
        if self.armed and not getattr(self._seen, "done", False):
            self._seen.done = True
            self.barrier.wait()  # both threads now hold the same (soon stale) copy
        return out


def _race(*fns):
    ts = [threading.Thread(target=f) for f in fns]
    [t.start() for t in ts]
    [t.join(timeout=10) for t in ts]


def test_store_update_merges_two_writers_that_read_the_same_version():
    s = RacyStore()
    s.put("h", "c", "n", {"n": 0})
    s.armed = True
    _race(lambda: s.update("h", "c", "n", lambda d: {"n": d["n"] + 1}), lambda: s.update("h", "c", "n", lambda d: {"n": d["n"] + 10}))
    s.armed = False
    assert s.get("h", "c", "n") == {"n": 11}  # blind overwrite would leave 1 or 10


def test_store_update_never_loses_increments_under_heavy_contention():
    store = SqliteStore(":memory:")
    store.put("h", "c", "n", {"n": 0})

    def worker():
        for _ in range(25):
            store.update("h", "c", "n", lambda d: {"n": d["n"] + 1})

    _race(*[worker for _ in range(8)])
    assert store.get("h", "c", "n") == {"n": 200}


def test_two_concurrent_reports_on_one_decision_both_survive(clock):
    """The original lost-update bug: both requests read the decision, then both write. Both trips must persist."""
    store = RacyStore()
    a, b = Tripwire(store, "h", clock), Tripwire(store, "h", clock)
    a.record_decision(INTERNET)
    store.armed = True
    results = {}
    _race(lambda: results.update(p=a.report_fact("internet price to $55")),
          lambda: results.update(o=b.report_fact("we had three outages this month on the internet")))
    assert results["p"]["tripped"] == ["internet plan"] and results["o"]["tripped"] == ["internet plan"]
    store.armed = False
    states = [t["state"] for t in a.why_did_we("internet")["decision"]["tripwires"]]
    assert states == ["tripped", "tripped"]  # a blind overwrite would leave one 'armed'


def test_concurrent_edit_and_report_dont_clobber_each_other(clock):
    store = RacyStore()
    a, b = Tripwire(store, "h", clock), Tripwire(store, "h", clock)
    a.record_decision(INTERNET)
    store.armed = True
    _race(lambda: a.report_fact("internet price to $55"), lambda: b.add_tripwire("internet", "the fee goes above $9"))
    store.armed = False
    d = a.why_did_we("internet")["decision"]
    assert len(d["tripwires"]) == 3 and d["tripwires"][0]["state"] == "tripped"  # neither edit was lost


def test_conditional_put_rejects_stale_writer():
    s = SqliteStore(":memory:")
    s.put("h", "k", "a", {"x": 1})
    _, v = s.get_v("h", "k", "a")
    s.put("h", "k", "a", {"x": 2}, expected_ver=v)
    with pytest.raises(ConflictError):
        s.put("h", "k", "a", {"x": 3}, expected_ver=v)  # stale version
    with pytest.raises(ConflictError):
        s.put("h", "k", "a", {"x": 3}, expected_ver=0)  # "must not exist" but it does
    assert s.get("h", "k", "a") == {"x": 2}


def test_stale_tripwire_id_is_ignored_not_misapplied(svc, monkeypatch):
    """A verdict computed for tripwire X must never land on tripwire Y if the list changed in between."""
    svc.record_decision(INTERNET)
    real = ai.assess_offline

    def assess_then_edit(fact, items):
        out = real(fact, items)
        svc.remove_tripwire("internet", 1)  # someone deletes the PRICE tripwire while the model is "thinking"
        return out

    monkeypatch.setattr(ai, "assess_offline", assess_then_edit)
    r = svc.report_fact("internet price to $55")
    assert r["tripped"] == []  # the price tripwire is gone; the outage tripwire must NOT inherit its verdict
    tws = svc.why_did_we("internet")["decision"]["tripwires"]
    assert [t["state"] for t in tws] == ["armed"] and "outages" in tws[0]["description"]


def test_reporting_the_same_fact_twice_is_idempotent(svc):
    svc.record_decision(INTERNET)
    svc.report_fact("internet price to $55")
    again = svc.report_fact("internet price to $55")
    assert again["tripped"] == []  # already tripped, no double event
    events = [h["text"] for h in svc.why_did_we("internet")["decision"]["history"] if h["text"].startswith("Tripped")]
    assert len(events) == 1


def test_concurrent_record_decisions_get_unique_keys(store, clock):
    errs = []

    adjectives = ["cheap", "fast", "reliable", "fibre", "wireless", "business"]

    def rec(i):
        try:  # different content but the same derived title ("internet plan"): they all race for one key
            Tripwire(store, "h", clock).record_decision(
                f"We're going with the {adjectives[i]} internet plan because option {i} fits. We'd switch if the price goes above ${50 + i}.")
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=rec, args=(i,)) for i in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len({d["_key"] for d in store.list("h", "decision")}) == 6


@pytest.fixture
def dynamo():
    with mock_aws():
        c = boto3.client("dynamodb", region_name="us-east-1")
        c.create_table(TableName="Tripwire", BillingMode="PAY_PER_REQUEST",
                       KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
                       AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}, {"AttributeName": "sk", "AttributeType": "S"}])
        yield DynamoStore("Tripwire", client=c)


def test_dynamodb_conditional_writes_detect_races(dynamo):
    dynamo.put("h", "k", "a", {"x": 1}, expected_ver=0)
    _, v = dynamo.get_v("h", "k", "a")
    dynamo.put("h", "k", "a", {"x": 2}, expected_ver=v)
    with pytest.raises(ConflictError):
        dynamo.put("h", "k", "a", {"x": 9}, expected_ver=v)
    with pytest.raises(ConflictError):
        dynamo.put("h", "k", "a", {"x": 9}, expected_ver=0)
    n = []

    def bump(d):
        n.append(1)
        return {"x": d["x"] + 1}

    assert dynamo.update("h", "k", "a", bump) == {"x": 3}


def test_dynamodb_purge_ping_and_service_flow(dynamo, clock):
    dynamo.ping()
    t = Tripwire(dynamo, "demo", clock)
    t.record_decision(INTERNET)
    t.report_fact("internet price to $55")
    assert dynamo.purge("demo") == 1 and dynamo.list("demo", "decision") == []


# ------------------------------------------------------------------ timeouts & retries are configured
def test_bedrock_client_has_bounded_timeouts_and_retries():
    cfg = ai._bedrock_client().meta.config
    assert cfg.read_timeout <= 10 and cfg.connect_timeout <= 5
    assert cfg.retries["total_max_attempts"] == 2 and cfg.retries["mode"] == "standard"
    # worst-case wall time for one model call stays well under a minute (the botocore default is ~60s x 5)
    assert cfg.retries["total_max_attempts"] * cfg.read_timeout <= 20


def test_dynamodb_client_has_bounded_timeouts_and_retries():
    cfg = DynamoStore("T", region="us-east-1")._ddb.meta.config
    assert cfg.read_timeout <= 6 and cfg.connect_timeout <= 3 and cfg.retries["total_max_attempts"] <= 4


def test_timeout_env_overrides(monkeypatch):
    monkeypatch.setenv("TRIPWIRE_BEDROCK_READ_TIMEOUT", "3")
    assert ai._bedrock_client().meta.config.read_timeout == 3


# ------------------------------------------------------------------ Bedrock failure modes
def _client():
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _resp(payload, stop="tool_use", name="submit_decision"):
    return {"output": {"message": {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t", "name": name, "input": payload}}]}},
            "stopReason": stop, "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}


def test_truncated_model_answer_is_refused(monkeypatch):
    """stopReason=max_tokens can still yield a partial tool input (tripwires silently missing). Never accept it."""
    monkeypatch.setenv("BEDROCK_MODEL_ID", MODEL)
    c = _client()
    with Stubber(c) as stub:
        stub.add_response("converse", _resp({"title": "t", "choice": "c", "reason": "r", "tripwires": []}, stop="max_tokens"))
        with pytest.raises(RuntimeError, match="truncated"):
            ai.converse_tool("s", "u", "submit_decision", ai.EXTRACT_SCHEMA, client=c)


def test_temperature_rejected_by_model_is_retried_without_it(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", MODEL)
    c = _client()
    with Stubber(c) as stub:
        stub.add_client_error("converse", service_error_code="ValidationException",
                              service_message="This model does not support the temperature parameter")
        stub.add_response("converse", _resp({"title": "t", "choice": "c", "reason": "r", "tripwires": []}),
                          expected_params=ai._request("s", "u", "submit_decision", ai.EXTRACT_SCHEMA, MODEL, False))
        out = ai.converse_tool("s", "u", "submit_decision", ai.EXTRACT_SCHEMA, client=c)
        stub.assert_no_pending_responses()
    assert out["choice"] == "c"


def test_other_bedrock_errors_are_not_swallowed(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", MODEL)
    c = _client()
    with Stubber(c) as stub:
        stub.add_client_error("converse", service_error_code="ThrottlingException", service_message="slow down")
        with pytest.raises(Exception, match="Throttling"):
            ai.converse_tool("s", "u", "submit_decision", ai.EXTRACT_SCHEMA, client=c)


def test_non_finite_numbers_from_a_model_are_neutralised():
    inf = float("inf")
    assert ai.validate_assessment({"observations": [{"tripwire_id": "a#1", "observed_value": inf, "verdict": "tripped"}]}, {"a#1"})[0]["observed_value"] is None
    assert ai._clean_tripwire({"description": "x", "metric": "p", "op": ">", "value": inf})["metric"] is None
    assert ai._clean_tripwire({"description": "x", "metric": "p", "op": ">", "value": float("nan")})["metric"] is None
    d = ai.validate_extraction({"choice": "c", "tripwires": [], "revisit_in_days": inf})
    assert d["revisit_in_days"] is None


def test_infinite_observation_cannot_trip_anything(svc, monkeypatch):
    svc.record_decision(INTERNET)
    tid = f"{svc._all()[0]['key']}#{svc._all()[0]['tripwires'][0]['id']}"
    monkeypatch.setattr(ai, "assess_offline", lambda f, i: ai.validate_assessment(
        {"observations": [{"tripwire_id": tid, "observed_value": float("inf"), "verdict": "tripped"}]}, {tid}))
    assert svc.report_fact("internet price to infinity")["tripped"] == []


# ------------------------------------------------------------------ circuit breaker
def _count_calls(monkeypatch, fail=True):
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        if fail:
            raise TimeoutError("bedrock timed out")
        return {"title": "plan", "choice": "the plan", "reason": "cheap", "tripwires": []}

    monkeypatch.setattr(ai, "converse_tool", fake)
    return calls


def test_breaker_opens_after_repeated_failures_and_auto_mode_degrades_gracefully(monkeypatch, svc):
    monkeypatch.setenv("TRIPWIRE_AI", "auto")
    monkeypatch.setenv("TRIPWIRE_BREAKER_FAILS", "3")
    calls = _count_calls(monkeypatch)
    for _ in range(3):
        assert svc.record_decision(INTERNET)["engine"] == "offline-fallback"
    assert calls["n"] == 3
    assert svc.record_decision(INTERNET)["engine"] == "offline-fallback"
    assert calls["n"] == 3  # breaker open: Bedrock not even attempted, no more timeouts paid


def test_breaker_fails_fast_in_strict_mode_and_recovers(monkeypatch, svc):
    monkeypatch.setenv("TRIPWIRE_AI", "bedrock")
    monkeypatch.setenv("TRIPWIRE_BREAKER_FAILS", "2")
    monkeypatch.setenv("TRIPWIRE_BREAKER_COOLDOWN", "0.2")
    calls = _count_calls(monkeypatch)
    for _ in range(2):
        assert svc.record_decision(INTERNET)["ok"] is False
    assert calls["n"] == 2
    assert svc.record_decision(INTERNET)["ok"] is False and calls["n"] == 2  # failed fast
    time.sleep(0.3)  # cool-down elapses; Bedrock is healthy again
    monkeypatch.setattr(ai, "converse_tool", lambda *a, **k: {"title": "plan", "choice": "the plan", "reason": "cheap", "tripwires": []})
    r = svc.record_decision(INTERNET)
    assert r["ok"] and r["engine"] == "bedrock" and not ai.BREAKER.is_open()


def test_one_success_resets_the_failure_count(monkeypatch, svc):
    monkeypatch.setenv("TRIPWIRE_AI", "auto")
    monkeypatch.setenv("TRIPWIRE_BREAKER_FAILS", "3")
    _count_calls(monkeypatch)
    svc.record_decision(INTERNET); svc.record_decision(INTERNET)
    _count_calls(monkeypatch, fail=False)
    svc.record_decision(INTERNET)
    _count_calls(monkeypatch)
    svc.record_decision(INTERNET); svc.record_decision(INTERNET)
    assert not ai.BREAKER.is_open()


# ------------------------------------------------------------------ parsing edge cases
@pytest.mark.parametrize("phrase,expected", [
    ("the price goes above $1,200", 1200.0), ("the price goes above $1,200.50", 1200.5),
    ("the price goes above fifty", 50.0), ("the price goes over $2,000,000", 2000000.0),
])
def test_numbers_money_and_words_parse_correctly(phrase, expected):
    d = ai.extract_offline(f"We're going with the laptop because it's fast. We'd return it if {phrase}.")
    assert d["tripwires"][0]["value"] == expected


def test_curly_quotes_and_odd_whitespace_from_speech_to_text():
    d = ai.extract_offline("We’re  going with  the laptop because it’s fast.\n\nWe’d return it if the price goes above $900.")
    assert d["choice"] == "the laptop" and d["reason"] == "it's fast" and d["tripwires"][0]["value"] == 900.0


def test_money_limit_does_not_trip_on_a_small_number(svc):
    svc.record_decision("We're going with the laptop because it's fast. We'd return it if the price goes above $1,200.")
    assert svc.report_fact("the laptop price is $1,150")["tripped"] == []
    assert svc.report_fact("the laptop price is now $1,300")["tripped"] == ["laptop"]


# ------------------------------------------------------------------ adversarial & degenerate input
ADVERSARIAL = ["a " * 5000, "if " * 3000, "above 1 " * 2000, "we're going with " * 1500, "because " * 3000,
               "$" * 5000, "1," * 4000, "\x00\x01\x02", "😀" * 2000, "', DROP TABLE docs; --", "<script>alert(1)</script>", "\\" * 4000]


@pytest.mark.parametrize("text", ADVERSARIAL)
def test_hostile_input_never_crashes_or_hangs(svc, text):
    t0 = time.time()
    r1 = svc.record_decision(text)
    r2 = svc.report_fact(text)
    r3 = svc.why_did_we(text)
    assert time.time() - t0 < 3.0
    for r in (r1, r2, r3):
        assert isinstance(r["spoken"], str) and "Traceback" not in r["spoken"]


def test_empty_and_whitespace_inputs_get_friendly_replies(svc):
    for fn, args in [(svc.record_decision, ("",)), (svc.record_decision, ("   ",)), (svc.report_fact, ("",)),
                     (svc.add_tripwire, ("x", "")), (svc.record_decision, (None,))]:
        r = fn(*args)
        assert r["ok"] is False and r["spoken"]


def test_bad_numbers_and_types_are_rejected_politely(svc):
    svc.record_decision(INTERNET)
    svc.report_fact("internet price to $55")
    for kw in [dict(new_limit=float("nan")), dict(new_limit=float("inf")), dict(new_limit=60, tripwire_number=0),
               dict(new_limit=60, tripwire_number=99), dict(new_limit=60, tripwire_number=True)]:
        assert svc.resolve_decision("internet", "keep", **kw)["ok"] is False
    assert svc.resolve_decision("internet", "explode")["ok"] is False
    assert svc.remove_tripwire("internet", -1)["ok"] is False and svc.remove_tripwire("internet", True)["ok"] is False
    assert svc.confirm_tripwire("internet", 0, True)["ok"] is False
    assert svc.why_did_we("internet")["decision"]["status"] == "revisit"  # nothing was changed by the bad calls


def test_unicode_decisions_roundtrip(svc):
    r = svc.record_decision("We're going with the café plan ☕ because it's cheap. We'd switch if the price goes above $50.")
    assert r["ok"] and "café" in svc.why_did_we("café")["spoken"]


# ------------------------------------------------------------------ resource limits
def test_history_is_capped_but_keeps_the_original_decision(svc):
    svc.record_decision(INTERNET)
    for i in range(120):
        svc.resolve_decision("internet", "keep", note=f"review {i}")
    h = svc.why_did_we("internet")["decision"]["history"]
    assert len(h) <= svc_mod.MAX_HISTORY and h[0]["text"].startswith("Decided")


def test_tripwire_count_is_capped(svc):
    svc.record_decision(INTERNET)
    for i in range(12):
        svc.add_tripwire("internet", f"the fee goes above ${100 + i}")
    d = svc.why_did_we("internet")["decision"]
    assert len(d["tripwires"]) == svc_mod.MAX_TRIPWIRES
    assert svc.add_tripwire("internet", "the fee goes above $999")["ok"] is False


def test_decision_count_is_capped(svc, monkeypatch):
    monkeypatch.setattr(svc_mod, "MAX_DECISIONS", 3)
    for adj in ("cheap", "fast", "reliable"):
        assert svc.record_decision(f"We're going with the {adj} internet plan because it fits. We'd switch if the price goes above $50.")["ok"]
    r = svc.record_decision("We're going with the fibre internet plan because it's quick. We'd switch if the price goes above $70.")
    assert r["ok"] is False and "Close a few" in r["spoken"]


def test_prompt_size_is_bounded_and_keeps_the_relevant_tripwire():
    items = [{"id": f"d{i}#{i}", "decision": f"thing {i}", "description": f"widget {i} above 5", "metric": f"widget {i}", "op": ">", "value": 5.0, "unit": ""}
             for i in range(500)]
    items.append({"id": "target#1", "decision": "internet plan", "description": "price above $50", "metric": "price", "op": ">", "value": 50.0, "unit": ""})
    capped = ai._cap_items("the internet price is $55", items)
    assert len(capped) == ai.MAX_PROMPT_ITEMS and any(i["id"] == "target#1" for i in capped)


# ------------------------------------------------------------------ time handling
def test_timezone_setting_controls_today(monkeypatch):
    monkeypatch.setenv("TRIPWIRE_TZ", "Pacific/Kiritimati")  # UTC+14
    assert svc_mod.default_today() == datetime.now(ZoneInfo("Pacific/Kiritimati")).date()
    monkeypatch.setenv("TRIPWIRE_TZ", "Not/AZone")
    assert isinstance(svc_mod.default_today(), date)  # a bad name must not break requests


def test_purge_removes_only_that_household(store, clock):
    Tripwire(store, "a", clock).record_decision(INTERNET)
    Tripwire(store, "b", clock).record_decision(INTERNET)
    assert store.purge("a") == 1
    assert Tripwire(store, "a", clock).whats_on_watch()["decisions"] == []
    assert len(Tripwire(store, "b", clock).whats_on_watch()["decisions"]) == 1


# ------------------------------------------------------------------ retries from the client must be safe
def test_retried_record_decision_does_not_create_a_duplicate(svc):
    first = svc.record_decision(INTERNET)
    again = svc.record_decision(INTERNET)  # e.g. Alexa+ retried after a timeout
    assert again.get("duplicate") is True and "already have" in again["spoken"]
    assert len(svc.s.list("t", "decision")) == 1 and again["decision"]["title"] == first["decision"]["title"]
    assert svc.why_did_we("internet")["ok"]  # and the topic is still unambiguous


def test_a_genuinely_different_decision_is_not_mistaken_for_a_retry(svc):
    svc.record_decision(INTERNET)
    svc.record_decision("We're going with the 500 megabit internet plan because it's faster. We'd switch if the price goes above $80.")
    assert len(svc.s.list("t", "decision")) == 2


def test_retried_add_tripwire_is_a_noop(svc):
    svc.record_decision(INTERNET)
    svc.add_tripwire("internet", "the fee goes above $9")
    r = svc.add_tripwire("internet", "the fee goes above $9")
    assert r["ok"] and "already watching" in r["spoken"]
    assert len(svc.why_did_we("internet")["decision"]["tripwires"]) == 3


# ------------------------------------------------------------------ deployment config that tests with mocks can't see
def test_iam_policy_grants_every_dynamodb_action_the_store_actually_uses():
    """moto does not enforce IAM, so a missing permission would only surface in production. Check the template statically."""
    import pathlib
    import re

    src = (pathlib.Path(__file__).parent.parent / "src/tripwire/store.py").read_text()
    used = set(re.findall(r"self\._ddb\.(get_item|put_item|update_item|delete_item|query|scan|describe_table|batch_write_item)\(", src))
    needed = {"dynamodb:" + "".join(w.capitalize() for w in call.split("_")) for call in used}
    template = (pathlib.Path(__file__).parent.parent / "infra/template.yaml").read_text()
    missing = {a for a in needed if a not in template}
    assert not missing, f"IAM policy is missing {missing}"
    assert "dynamodb:UpdateItem" in needed  # sanity: the check really sees the CAS write path


def test_dockerfile_runs_unprivileged_and_has_a_healthcheck():
    import pathlib

    d = (pathlib.Path(__file__).parent.parent / "Dockerfile").read_text()
    assert "USER app" in d and "HEALTHCHECK" in d
    assert "tzdata" in (pathlib.Path(__file__).parent.parent / "pyproject.toml").read_text()
