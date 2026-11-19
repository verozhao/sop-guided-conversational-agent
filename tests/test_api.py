"""HTTP API. Runs on the offline engine: no token is configured or sent."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from .conftest import DEMO_UTTERANCE


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SOP_INSPECTOR", raising=False)
    from server.app import app

    return TestClient(app)


def start(client: TestClient, scenario: str = "default") -> str:
    response = client.post("/api/session", json={"consent_scenario": scenario})
    assert response.status_code == 200
    return response.json()["session_id"]


def test_config_and_index(client: TestClient):
    config = client.get("/api/config").json()
    assert config["server_token_configured"] is False
    assert config["default_model"] == "claude-opus-5-5"
    assert "timeout" in config["consent_scenarios"]
    page = client.get("/")
    assert page.status_code == 200 and "Claims Support" in page.text
    assert client.get("/healthz").json() == {"status": "ok"}


def test_chat_round_trip_offline(client: TestClient):
    session_id = start(client)
    response = client.post("/api/chat", json={"session_id": session_id, "message": DEMO_UTTERANCE})
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "offline"
    assert "No API token is set" in body["notice"]
    assert "CL-2048" in body["reply"]
    assert body["state"]["phase"] == "PROCESS_CASE" and body["state"]["verified"] is True
    assert body["state"]["identity"]["provided"]["id_last4"] == "****"  # never echoed back
    assert body["trace"]["phase_path"] == ["VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE"]

    forced = client.post("/api/chat", json={"session_id": session_id, "message": "that's all", "engine": "offline"}).json()
    assert forced["notice"] == "" and forced["state"]["phase"] == "POST_PROCESS"

    snapshot = client.get(f"/api/session/{session_id}").json()
    assert snapshot["state"]["phase"] == "POST_PROCESS" and len(snapshot["history"]) == 5


def test_unknown_session_and_bad_input(client: TestClient):
    assert client.post("/api/chat", json={"session_id": "nope", "message": "hi"}).status_code == 404
    session_id = start(client)
    assert client.post("/api/chat", json={"session_id": session_id, "message": ""}).status_code == 422
    assert client.post("/api/chat", json={"session_id": session_id, "message": "hi", "engine": "magic"}).status_code == 422


def test_inspector_can_be_switched_off(client: TestClient, monkeypatch):
    monkeypatch.setenv("SOP_INSPECTOR", "off")
    session_id = start(client)
    body = client.post("/api/chat", json={"session_id": session_id, "message": "My name is Margaret Chen"}).json()
    assert "trace" not in body
    assert set(body["state"]) == {"session_id", "phase", "workflow", "verified", "outbox", "handoff"}
    assert "history" not in client.get(f"/api/session/{session_id}").json()
