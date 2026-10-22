"""Things going south: what a caller sees when the backing store fails. Uses a real server over real HTTP."""

import httpx
import pytest
from mcp import Client

from conftest import INTERNET
from test_mcp_e2e import _serve
from tripwire import server as srv
from tripwire.server import create_app
from tripwire.service import Tripwire
from tripwire.store import ConflictError, SqliteStore

SECRET = "arn:aws:dynamodb:us-east-1:123456789012:table/Tripwire AccessDenied secret-internal-detail"


class FlakyStore(SqliteStore):
    """A real store that can be told to fail in specific ways."""

    def __init__(self):
        super().__init__(":memory:")
        self.mode = None

    def list(self, *a, **k):
        if self.mode == "down":
            raise RuntimeError(SECRET)
        return super().list(*a, **k)

    def get_v(self, *a, **k):
        if self.mode == "down":
            raise RuntimeError(SECRET)
        return super().get_v(*a, **k)

    def put(self, *a, **k):
        if self.mode == "storm":
            raise ConflictError("always losing the race")
        return super().put(*a, **k)

    def ping(self):
        if self.mode == "down":
            raise RuntimeError(SECRET)
        super().ping()


@pytest.fixture
def flaky(monkeypatch, clock):
    monkeypatch.delenv("TRIPWIRE_TOKEN", raising=False)
    store = FlakyStore()
    yield store, *_serve_one(create_app(Tripwire(store, "f", clock)))


def _serve_one(app):
    gen = _serve(app)
    return next(gen), gen


async def _call(url, tool, args):
    async with Client(f"{url}/mcp") as c:
        r = await c.call_tool(tool, args)
        return r, "".join(getattr(x, "text", "") for x in r.content)


async def test_database_outage_gives_a_speakable_answer_and_leaks_nothing(flaky):
    store, url, _ = flaky
    r, _ = await _call(url, "record_decision", {"what_was_said": INTERNET})
    assert r.structured_content["ok"] is True
    store.mode = "down"
    for tool, args in [("whats_on_watch", {}), ("report_fact", {"statement": "internet price to $55"}),
                       ("why_did_we", {"topic": "internet"}), ("record_decision", {"what_was_said": INTERNET})]:
        r, text = await _call(url, tool, args)
        sc = r.structured_content
        assert sc["ok"] is False and sc["spoken"] == srv.BUSY and sc["error"] == "temporarily_unavailable"
        assert "arn:" not in text and "secret-internal-detail" not in text and "Traceback" not in text
    store.mode = None  # database comes back: the very next call works, no restart needed
    r, _ = await _call(url, "whats_on_watch", {})
    assert r.structured_content["ok"] is True and r.structured_content["decisions"]


async def test_write_conflict_storm_exhausts_retries_cleanly_and_corrupts_nothing(flaky):
    store, url, _ = flaky
    await _call(url, "record_decision", {"what_was_said": INTERNET})
    store.mode = "storm"
    r, _ = await _call(url, "resolve_decision", {"topic": "internet", "outcome": "close"})
    assert r.structured_content["spoken"] == srv.BUSY
    store.mode = None
    r, _ = await _call(url, "why_did_we", {"topic": "internet"})
    assert r.structured_content["decision"]["status"] == "active"  # the failed close changed nothing


def test_readiness_reflects_database_health_while_liveness_stays_up(flaky):
    store, url, _ = flaky
    assert httpx.get(f"{url}/healthz").status_code == 200 and httpx.get(f"{url}/readyz").json() == {"status": "ready"}
    store.mode = "down"
    r = httpx.get(f"{url}/readyz")
    assert r.status_code == 503 and "arn:" not in r.text and "secret" not in r.text
    assert httpx.get(f"{url}/healthz").status_code == 200  # process is alive; orchestrators can tell the difference


def test_probe_endpoints_stay_open_but_mcp_requires_the_token(monkeypatch, clock):
    monkeypatch.setenv("TRIPWIRE_TOKEN", "tok")
    url, _ = _serve_one(create_app(Tripwire(SqliteStore(":memory:"), "t", clock)))
    assert httpx.get(f"{url}/healthz").status_code == 200 and httpx.get(f"{url}/readyz").status_code == 200
    assert httpx.post(f"{url}/mcp", json={}).status_code == 401


def test_malformed_requests_are_rejected_without_crashing_the_server(flaky):
    store, url, _ = flaky
    h = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    for body in [b"", b"{not json", b"[]", b'{"jsonrpc":"2.0","id":1,"method":"nope"}', b"\xff\xfe\x00",
                 b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"report_fact","arguments":{"statement":123}}}',
                 b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"report_fact","arguments":{"statement":"' + b"x" * 5000 + b'"}}}',
                 b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"does_not_exist","arguments":{}}}']:
        r = httpx.post(f"{url}/mcp", content=body, headers=h)
        assert r.status_code < 500, (body[:40], r.status_code)
    assert httpx.get(f"{url}/healthz").status_code == 200  # still alive after all of that


async def test_oversized_and_wrongly_typed_arguments_are_refused_by_the_schema(flaky):
    store, url, _ = flaky
    async with Client(f"{url}/mcp") as c:
        for tool, args in [("report_fact", {"statement": "x" * 5000}), ("record_decision", {"what_was_said": "x" * 5000}),
                           ("confirm_tripwire", {"topic": "a", "number": 0, "tripped": True}),
                           ("resolve_decision", {"topic": "a", "outcome": "explode"})]:
            r = await c.call_tool(tool, args)
            assert r.is_error
