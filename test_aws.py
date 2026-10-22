"""AWS integration tests.

* DynamoDB: the real DynamoStore runs against moto's DynamoDB emulation.
* Bedrock: a real boto3 'bedrock-runtime' client under botocore's Stubber, so every Converse request we
  build is validated against AWS's actual API model, and we can feed back good *and adversarial* model output.
"""

import boto3
import pytest
from botocore.stub import Stubber
from moto import mock_aws

from conftest import INTERNET
from tripwire import ai
from tripwire.service import Tripwire
from tripwire.store import DynamoStore

MODEL = "anthropic.claude-test-v1:0"


@pytest.fixture
def dynamo():
    with mock_aws():
        c = boto3.client("dynamodb", region_name="us-east-1")
        c.create_table(
            TableName="Tripwire",
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}, {"AttributeName": "sk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        yield DynamoStore("Tripwire", client=c)


def test_dynamo_store_crud_and_isolation(dynamo):
    dynamo.put("h1", "decision", "a", {"x": 1})
    dynamo.put("h1", "decision", "b", {"x": 2})
    dynamo.put("h2", "decision", "c", {"x": 3})
    assert dynamo.get("h1", "decision", "a") == {"x": 1}
    assert [d["_key"] for d in dynamo.list("h1", "decision")] == ["a", "b"]
    dynamo.delete("h1", "decision", "a")
    assert dynamo.get("h1", "decision", "a") is None


def test_decision_lifecycle_survives_new_service_objects_on_dynamodb(dynamo, clock):
    Tripwire(dynamo, "demo", clock).record_decision(INTERNET)
    r = Tripwire(dynamo, "demo", clock).report_fact("internet price to $55")  # e.g. a fresh container, weeks later
    assert r["tripped"] == ["internet plan"]


def _client():
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _tool_resp(name, payload):
    return {
        "output": {"message": {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": name, "input": payload}}]}},
        "stopReason": "tool_use",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        "metrics": {"latencyMs": 1},
    }


EXTRACTION = {
    "title": "Internet Plan", "choice": "the 200 Mbps plan", "options": ["100 Mbps plan"], "reason": "cheapest for our usage",
    "tripwires": [
        {"description": "price goes above $50", "metric": "price", "op": ">", "value": 50, "unit": "usd"},
        {"description": "outages more than twice a month", "metric": "outages per month", "op": ">", "value": 2},
        {"description": "we move house"},
    ],
    "revisit_in_days": 90,
}


def test_bedrock_extraction_request_matches_aws_schema_and_output_is_validated(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", MODEL)
    client = _client()
    with Stubber(client) as stub:
        stub.add_response("converse", _tool_resp("submit_decision", EXTRACTION))
        raw = ai.converse_tool(ai.EXTRACT_SYSTEM, INTERNET, "submit_decision", ai.EXTRACT_SCHEMA, client=client)
        stub.assert_no_pending_responses()
    d = ai.validate_extraction(raw)
    assert d["title"] == "internet plan" and d["revisit_in_days"] == 90
    assert d["tripwires"][0]["op"] == ">" and d["tripwires"][2]["metric"] is None  # qualitative stays qualitative


def test_extraction_rejects_garbage_and_sanitises_bad_numeric_specs():
    with pytest.raises(ValueError):
        ai.validate_extraction({"choice": "  ", "tripwires": []})
    d = ai.validate_extraction({"choice": "x", "tripwires": [{"description": "bad", "metric": "p", "op": "!!", "value": "NaN-ish"}]})
    assert d["tripwires"][0]["metric"] is None  # invalid op => treated as qualitative, never as a broken numeric check


def _bedrock_stub(monkeypatch):
    monkeypatch.setenv("TRIPWIRE_AI", "bedrock")
    monkeypatch.setenv("BEDROCK_MODEL_ID", MODEL)
    client = _client()
    stub = Stubber(client)
    stub.activate()
    monkeypatch.setattr(ai, "_bedrock_client", lambda: client)
    return stub


def tw_id(svc, n):
    """The real, stable id of the n-th tripwire (0-based) of the only decision, in the form the model is shown."""
    d = svc._all()[0]
    return f"{d['key']}#{d['tripwires'][n]['id']}"


def test_end_to_end_with_bedrock_record_then_report(monkeypatch, svc):
    stub = _bedrock_stub(monkeypatch)
    stub.add_response("converse", _tool_resp("submit_decision", EXTRACTION))
    rec = svc.record_decision(INTERNET)
    assert rec["engine"] == "bedrock" and len(rec["decision"]["tripwires"]) == 3
    stub.add_response("converse", _tool_resp("submit_assessment", {"observations": [
        {"tripwire_id": tw_id(svc, 0), "observed_value": 55, "verdict": "tripped", "note": "price rose"}]}))
    r = svc.report_fact("The provider raised our internet price to $55")
    stub.assert_no_pending_responses()
    assert r["engine"] == "bedrock" and r["tripped"] == ["internet plan"]


def test_a_lying_model_cannot_trip_a_numeric_tripwire(monkeypatch, svc):
    """The model says 'tripped' for a $45 price against a $50 limit; the server must override it."""
    stub = _bedrock_stub(monkeypatch)
    stub.add_response("converse", _tool_resp("submit_decision", EXTRACTION))
    svc.record_decision(INTERNET)
    stub.add_response("converse", _tool_resp("submit_assessment", {"observations": [
        {"tripwire_id": tw_id(svc, 0), "observed_value": 45, "verdict": "tripped"}]}))
    r = svc.report_fact("internet price is $45")
    stub.assert_no_pending_responses()  # proves the lie really reached the server (not ignored for a bad id)
    assert r["tripped"] == [] and svc.whats_on_watch()["needs_attention"] == []


def test_a_model_cannot_untrip_a_real_breach_or_invent_ids(monkeypatch, svc):
    stub = _bedrock_stub(monkeypatch)
    stub.add_response("converse", _tool_resp("submit_decision", EXTRACTION))
    svc.record_decision(INTERNET)
    stub.add_response("converse", _tool_resp("submit_assessment", {"observations": [
        {"tripwire_id": tw_id(svc, 0), "observed_value": 70, "verdict": "not_tripped"},
        {"tripwire_id": "nonexistent#deadbeef", "observed_value": 1, "verdict": "tripped"}]}))
    r = svc.report_fact("internet price is $70")
    stub.assert_no_pending_responses()
    assert r["tripped"] == ["internet plan"]


def test_assessment_prompt_lists_every_armed_tripwire():
    p = ai.assess_prompt("price is 55", [{"id": "k#0", "decision": "internet plan", "description": "price above $50", "metric": "price", "op": ">", "value": 50.0, "unit": "usd"}])
    assert "id=k#0" in p and "numeric: price > 50" in p and "internet plan" in p


def test_auto_mode_falls_back_and_says_so(monkeypatch, svc):
    monkeypatch.setenv("TRIPWIRE_AI", "auto")
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    assert svc.record_decision(INTERNET)["engine"] == "offline-fallback"


def test_bedrock_mode_never_hides_failure(monkeypatch, svc):
    monkeypatch.setenv("TRIPWIRE_AI", "bedrock")
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    r = svc.record_decision(INTERNET)
    assert r["ok"] is False and "Traceback" not in r["spoken"]
