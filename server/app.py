"""HTTP API and test UI for the insurance claims SOP agent.

    uvicorn server.app:app --port 8000

Configuration (environment variables, all optional):

    ANTHROPIC_API_KEY   token used when the browser does not send one
                        (also read from a .env file in the project root)
    SOP_MODEL           default model id            (claude-opus-5-5)
    SOP_EFFORT          low | medium | high | ...   (low)
    SOP_TODAY           YYYY-MM-DD or "real"        (2026-03-05, the fixtures' time frame)
    SOP_INSPECTOR       on | off                    (on). "off" hides harness internals from the API

The API token can also be pasted into the UI. It is then sent with each chat
request in the X-API-Key header, used for that request and never written to
disk or to a log.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from apps.insurance_claims import Agent
from apps.insurance_claims.agent import MAX_INPUT_CHARS
from apps.insurance_claims.state import Session
from sop_harness.llm import (
    DEFAULT_MODEL,
    EFFORT_LEVELS,
    SELECTABLE_MODELS,
    get_anthropic_llm,
)

STATIC_DIR = Path(__file__).parent / "static"
SESSION_TTL_SECONDS = 3 * 60 * 60
MAX_SESSIONS = 500
MODEL_ID_RE = re.compile(r"^claude-[a-z0-9.\-]{2,60}$")

DOTENV_NAMES = {"ANTHROPIC_API_KEY", "SOP_MODEL", "SOP_EFFORT", "SOP_TODAY", "SOP_INSPECTOR"}


def load_dotenv(path: Path | None = None) -> list[str]:
    """Read KEY=VALUE lines from .env in the project root.

    Only the variables this app uses are read, and real environment variables
    win. Returns the names that were set. Set SOP_SKIP_DOTENV=1 to disable
    (the test suite does, so tests can never pick up a real token).
    """
    if os.environ.get("SOP_SKIP_DOTENV"):
        return []
    path = path or Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text().splitlines():
        name, separator, value = line.strip().partition("=")
        name = name.strip().removeprefix("export ").strip()
        if not separator or name not in DOTENV_NAMES or name in os.environ:
            continue
        os.environ[name] = value.strip().strip("\"'")
        loaded.append(name)
    return loaded


load_dotenv()

app = FastAPI(title="SOP-guided insurance claims agent", docs_url="/api/docs", redoc_url=None)
agent = Agent()

_sessions: dict[str, Session] = {}
_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def _server_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def _default_model() -> str:
    value = os.environ.get("SOP_MODEL", "").strip()
    return value if MODEL_ID_RE.match(value) else DEFAULT_MODEL


def _effort() -> str | None:
    value = os.environ.get("SOP_EFFORT", "low").strip().lower()
    return value if value in EFFORT_LEVELS else None


def _inspector_on() -> bool:
    return os.environ.get("SOP_INSPECTOR", "on").strip().lower() not in {"off", "0", "false", "no"}


def _prune() -> None:
    now = time.time()
    stale = [sid for sid, s in _sessions.items() if now - s.updated_at > SESSION_TTL_SECONDS]
    if len(_sessions) - len(stale) > MAX_SESSIONS:
        oldest = sorted(_sessions.values(), key=lambda s: s.updated_at)
        stale += [s.id for s in oldest[: len(_sessions) - MAX_SESSIONS]]
    for sid in set(stale):
        _sessions.pop(sid, None)
        _locks.pop(sid, None)


def _get(session_id: str) -> tuple[Session, threading.Lock]:
    with _registry_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown or expired session. Start a new conversation.")
        return session, _locks[session_id]


def _state(session: Session) -> dict[str, Any]:
    full = agent.public_state(session)
    if _inspector_on():
        return full
    return {
        "session_id": full["session_id"],
        "phase": full["phase"],
        "workflow": full["workflow"],
        "verified": full["verified"],
        "outbox": full["outbox"],
        "handoff": (
            {"ticket_id": full["handoff"]["ticket_id"], "phase_at_handoff": full["handoff"].get("phase_at_handoff")}
            if full["handoff"]
            else None
        ),
    }


class NewSession(BaseModel):
    consent_scenario: str = Field(default="default", max_length=40)


class ChatMessage(BaseModel):
    session_id: str = Field(max_length=64)
    message: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    engine: str = Field(default="auto", pattern="^(auto|offline)$")
    model: str | None = Field(default=None, max_length=80)

    @field_validator("message", mode="before")
    @classmethod
    def _no_blank_message(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value  # spaces alone are an empty message


@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """One readable sentence instead of a list of validation records."""
    problems = []
    for error in exc.errors():
        kind = str(error.get("type", ""))
        field = str(error.get("loc", ["request"])[-1])
        if kind == "json_invalid":
            problems.append("The request body is not valid JSON.")
        elif kind == "missing":
            problems.append(f"{field} is required.")
        elif field == "message" and kind == "string_too_long":
            problems.append(f"The message is longer than {MAX_INPUT_CHARS} characters.")
        elif field == "message" and kind == "string_too_short":
            problems.append("The message is empty.")
        elif field == "message":
            problems.append("The message must be text.")
        else:
            problems.append(f"{field}: {error.get('msg', 'invalid value')}.")
    return JSONResponse(status_code=422, content={"detail": " ".join(dict.fromkeys(problems)) or "Invalid request."})


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict[str, Any]:
    models = list(SELECTABLE_MODELS)
    default = _default_model()
    if default not in models:
        models.insert(0, default)
    out: dict[str, Any] = {
        "server_token_configured": bool(_server_key()),
        "default_model": default,
        "models": models,
        "effort": _effort(),
        "today": agent.clock().isoformat(),
        "consent_scenarios": sorted(agent.data.consent_scenarios),
        "inspector": _inspector_on(),
        "max_message_chars": MAX_INPUT_CHARS,
    }
    if _inspector_on():
        out["tools"] = agent.tools.describe()  # harness internals, hidden with the inspector
    return out


@app.post("/api/session")
def new_session(body: NewSession) -> dict[str, Any]:
    session = agent.new_session(body.consent_scenario)
    with _registry_lock:
        _prune()
        _sessions[session.id] = session
        _locks[session.id] = threading.Lock()
    return {"session_id": session.id, "greeting": session.history[0]["text"], "state": _state(session)}


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session, _lock = _get(session_id)
    out: dict[str, Any] = {"state": _state(session)}
    if _inspector_on():
        out["history"] = session.history
    return out


@app.post("/api/chat")
def chat(body: ChatMessage, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    session, lock = _get(body.session_id)
    token = (x_api_key or "").strip() or _server_key()
    model = body.model if body.model and MODEL_ID_RE.match(body.model) else _default_model()

    llm = None
    if body.engine == "auto" and token:
        llm = get_anthropic_llm(token, model=model, effort=_effort())

    with lock:
        result = agent.handle(session, body.message, llm=llm)
        state = _state(session)

    trace = result["trace"]
    notice = ""
    if trace["llm_errors"]:
        first = trace["llm_errors"][0]
        if trace["engine"] == "offline":
            notice = f"Model call failed ({first['kind']}): {first['message']} This turn ran on the offline engine."
        elif first["stage"] == "understand":
            notice = (
                f"Model call failed while reading the message ({first['kind']}): {first['message']} "
                "The offline rules read it instead. The reply was still written by the model."
            )
        else:
            notice = (
                f"Model call failed while wording the reply ({first['kind']}): {first['message']} "
                "The reply for this turn is the built-in wording."
            )
    elif body.engine == "auto" and not token:
        notice = "No API token is set, so this turn ran on the offline rule engine."
    out: dict[str, Any] = {"reply": result["reply"], "state": state, "engine": trace["engine"], "notice": notice}
    if _inspector_on():
        out["trace"] = trace
    return out


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
