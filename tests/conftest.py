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
    """Drives one conversation in a test.

    Who is calling is always confirmed: a caller who gives three matching details and has not said
    "I'm the policyholder" is asked "are you the policyholder yourself?". Most tests are about what
    happens before or after that question. So when nobody else has come up in the conversation,
    this helper answers the question with a plain "yes", as a policyholder would, and returns the
    reply to that. Tests about the question itself pass ``answers_role=False`` or mention another
    person. Conversations with a scripted model are never answered for.
    """

    def __init__(self, agent: Agent, scenario: str = "default", llm=None, answers_role: bool = True):
        self.agent = agent
        self.s = agent.new_session(scenario)
        self.llm = llm
        self.answers_role = answers_role
        self.last: dict | None = None

    def say(self, text: str) -> str:
        self.last = self.agent.handle(self.s, text, llm=self.llm)
        memory = self.s.memory
        nobody_else = memory.caller_role == "unknown" and not memory.role_unclear  # no other person came up so far
        if self.answers_role and self.llm is None and self.act == "ROLE_CHECK" and self.s.role_checks == 0 and nobody_else:
            self.last = self.agent.handle(self.s, "yes", llm=self.llm)
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
    """No test may see an API token or a setting from the developer's shell."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "SOP_MODEL", "SOP_EFFORT", "SOP_TODAY", "SOP_INSPECTOR"):
        monkeypatch.delenv(name, raising=False)


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
