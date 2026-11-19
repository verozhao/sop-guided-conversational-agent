"""Shared test helpers. Every test runs offline: no network, no API token."""

from __future__ import annotations

import os

# Set before anything imports the server: no .env file is read during tests.
os.environ["SOP_SKIP_DOTENV"] = "1"

from datetime import date  # noqa: E402

import pytest  # noqa: E402

from apps.insurance_claims import Agent, AgentConfig

DEMO_DATE = date(2026, 3, 5)

DEMO_UTTERANCE = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied "
    "healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."
)

# Strings that exist only in the system of record. None may appear in a reply
# before the caller is verified.
PROTECTED = [
    "CL-2048", "CL-2011", "CL-1899", "CL-2102", "CL-3001",
    "pathology", "office note", "review file", "diagnosis report",
    "March 18", "2026-03-18", "January 12", "2026-01-12",
    "1,450", "1450", "780.00", "3,200", "425.00",
    "margaret@email.com", "m***@email.com", "2836", "POL-",
    "settled", "in progress",
]


class Convo:
    def __init__(self, agent: Agent, scenario: str = "default", llm=None):
        self.agent = agent
        self.s = agent.new_session(scenario)
        self.llm = llm
        self.last: dict | None = None

    def say(self, text: str) -> str:
        self.last = self.agent.handle(self.s, text, llm=self.llm)
        return self.last["reply"]

    def run(self, *lines: str) -> str:
        reply = ""
        for line in lines:
            reply = self.say(line)
        return reply

    @property
    def trace(self) -> dict:
        assert self.last is not None
        return self.last["trace"]

    @property
    def act(self) -> str:
        return self.trace["directive"]["act"]

    @property
    def tools(self) -> list[str]:
        return [t["name"] for t in self.trace["tools"]]


def assert_no_leak(reply: str, allow: tuple[str, ...] = ()) -> None:
    low = reply.lower()
    leaked = [item for item in PROTECTED if item not in allow and item.lower() in low]
    assert not leaked, f"protected data in reply: {leaked}\n{reply}"


@pytest.fixture(autouse=True)
def _no_real_token(monkeypatch):
    """No test may see an API token, whatever the developer's shell has set."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


@pytest.fixture
def agent() -> Agent:
    return Agent(config=AgentConfig(clock=lambda: DEMO_DATE))


@pytest.fixture
def convo(agent: Agent) -> Convo:
    return Convo(agent)


@pytest.fixture
def verified(agent: Agent) -> Convo:
    """A conversation that has just completed the demo utterance."""
    c = Convo(agent)
    c.say(DEMO_UTTERANCE)
    assert c.s.verified
    return c
