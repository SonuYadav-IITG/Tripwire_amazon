"""End-to-end: a real uvicorn server on a real socket, driven by the real MCP client over Streamable HTTP."""

import socket
import threading
import time

import httpx
import pytest
from conftest import INTERNET
import uvicorn
from mcp import Client

from tripwire.server import create_app
from tripwire.service import Tripwire
from tripwire.store import SqliteStore


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(monkeypatch, clock):
    monkeypatch.delenv("TRIPWIRE_TOKEN", raising=False)
    yield from _serve(create_app(Tripwire(SqliteStore(":memory:"), "e2e", clock)))


def _serve(app):
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    t.join(timeout=5)


async def test_full_voice_session_over_http(server):
    async with Client(f"{server}/mcp") as c:
        tools = {t.name for t in (await c.list_tools()).tools}
        assert {"record_decision", "report_fact", "whats_on_watch", "why_did_we", "resolve_decision"} <= tools
        r = await c.call_tool("record_decision", {"what_was_said": INTERNET})
        assert r.structured_content["ok"]
        r = await c.call_tool("report_fact", {"statement": "The provider raised our internet price to $55"})
        assert r.structured_content["tripped"] == ["internet plan"]
        r = await c.call_tool("resolve_decision", {"topic": "internet", "outcome": "keep", "new_limit": 60, "tripwire_number": 1})
        assert "price above $60" in r.structured_content["spoken"]


async def test_state_persists_across_separate_client_connections(server):
    async with Client(f"{server}/mcp") as c1:
        await c1.call_tool("record_decision", {"what_was_said": INTERNET})
    async with Client(f"{server}/mcp") as c2:  # a new Alexa+ conversation, days later
        r = await c2.call_tool("report_fact", {"statement": "internet price to $55"})
        assert r.structured_content["tripped"] == ["internet plan"]


def test_serves_mcp_spec_2025_11_25(server):
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
    r = httpx.post(f"{server}/mcp", json=body, headers={"Accept": "application/json, text/event-stream"})
    assert r.status_code == 200 and '"protocolVersion":"2025-11-25"' in r.text


def test_tool_schemas_have_annotations_and_descriptions(server):
    import anyio

    async def go():
        async with Client(f"{server}/mcp") as c:
            return (await c.list_tools()).tools

    tools = {t.name: t for t in anyio.run(go)}
    assert tools["whats_on_watch"].annotations.read_only_hint is True
    assert tools["remove_tripwire"].annotations.destructive_hint is True
    assert all(t.description for t in tools.values())


@pytest.fixture
def guarded_server(monkeypatch, clock):
    monkeypatch.setenv("TRIPWIRE_TOKEN", "s3cret")
    yield from _serve(create_app(Tripwire(SqliteStore(":memory:"), "auth", clock)))


def test_bearer_token_guard(guarded_server):
    base = guarded_server
    assert httpx.get(f"{base}/healthz").status_code == 200  # health check stays open
    assert httpx.post(f"{base}/mcp", json={}).status_code == 401
    assert httpx.post(f"{base}/mcp", json={}, headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = httpx.post(f"{base}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                    headers={"Authorization": "Bearer s3cret", "Accept": "application/json, text/event-stream"})
    assert ok.status_code != 401


@pytest.fixture
def hosted_server(monkeypatch, clock):
    monkeypatch.setenv("TRIPWIRE_ALLOWED_HOSTS", "demo.example.com")
    monkeypatch.delenv("TRIPWIRE_TOKEN", raising=False)
    yield from _serve(create_app(Tripwire(SqliteStore(":memory:"), "host", clock)))


def test_public_hostname_allowed_and_loopback_still_works_and_others_rejected(hosted_server):
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    h = {"Accept": "application/json, text/event-stream"}
    assert httpx.post(f"{hosted_server}/mcp", json=ping, headers={**h, "Host": "demo.example.com"}).status_code == 200
    assert httpx.post(f"{hosted_server}/mcp", json=ping, headers=h).status_code == 200  # 127.0.0.1:<port>
    assert httpx.post(f"{hosted_server}/mcp", json=ping, headers={**h, "Host": "evil.example.com"}).status_code == 421
