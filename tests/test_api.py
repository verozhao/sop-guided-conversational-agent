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


def test_dotenv_loader_reads_only_known_names_and_never_overrides(tmp_path, monkeypatch):
    from server.app import load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "ANTHROPIC_API_KEY='from-file'\n"
        "export SOP_MODEL=claude-haiku-5-5\n"
        "SOP_EFFORT=medium\n"
        "AWS_SECRET_ACCESS_KEY=must-not-be-read\n"
        "not a setting\n"
    )
    monkeypatch.setenv("SOP_EFFORT", "low")  # a real environment variable wins
    monkeypatch.delenv("SOP_MODEL", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    assert load_dotenv(env_file) == []  # disabled for the test suite by SOP_SKIP_DOTENV

    monkeypatch.delenv("SOP_SKIP_DOTENV")
    assert load_dotenv(env_file) == ["ANTHROPIC_API_KEY", "SOP_MODEL"]
    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "from-file"
    assert os.environ["SOP_MODEL"] == "claude-haiku-5-5"
    assert os.environ["SOP_EFFORT"] == "low"
    assert "AWS_SECRET_ACCESS_KEY" not in os.environ
    assert load_dotenv(tmp_path / "missing.env") == []
    # The loader wrote these directly, so remove them directly: nothing may leak into other tests.
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("SOP_MODEL", None)
