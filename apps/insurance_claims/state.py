"""Session state: the SOP position plus memory that crosses phase boundaries."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

VERIFY_ID = "VERIFY_ID"
RESOLVE_INTENT = "RESOLVE_INTENT"
PROCESS_CASE = "PROCESS_CASE"
POST_PROCESS = "POST_PROCESS"
CLOSED = "CLOSED"
ESCALATED = "ESCALATED"

WORKFLOW_PHASES = [VERIFY_ID, RESOLVE_INTENT, PROCESS_CASE, POST_PROCESS]
ALL_PHASES = WORKFLOW_PHASES + [CLOSED, ESCALATED]


@dataclass
class Memory:
    """Slots filled whenever the caller says something useful, in any phase.

    A slot that belongs to a later phase is stored now and consumed when the
    workflow gets there. ``captured`` logs where each slot came from so the
    inspector can show memory crossing a phase boundary.
    """

    identity: dict[str, str] = field(default_factory=dict)
    declined: list[str] = field(default_factory=list)
    caller_role: str = "unknown"
    role_guessed: bool = False         # the role was inferred from wording, not stated
    role_unclear: bool = False         # somebody else was mentioned and the caller has not said who they are
    representative_name: str = ""
    intent: str = "none"
    extra_intents: list[str] = field(default_factory=list)
    intent_consumed: bool = False
    case_hints: dict[str, Any] = field(default_factory=dict)
    hints_consumed: bool = False
    backlog_question: str = ""
    pending_topics: list[str] = field(default_factory=list)
    email_pref: str = "none"
    notes: list[str] = field(default_factory=list)
    last_emotion: str = "neutral"
    captured: dict[str, dict[str, Any]] = field(default_factory=dict)

    def remember(self, slot: str, turn: int, phase: str) -> None:
        self.captured[slot] = {"turn": turn, "phase": phase}


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    phase: str = VERIFY_ID
    turn: int = 0
    memory: Memory = field(default_factory=Memory)

    # identity and authorisation
    verified: bool = False
    identity_ok: bool = False
    party_id: str | None = None
    verification: dict[str, Any] = field(default_factory=dict)
    candidate_id: str | None = None    # the record the caller's details have been matching so far
    failed_attempts: int = 0
    name_misses: int = 0
    rep_name_refused: str = ""         # a representative name that was not on the policy
    role_checks: int = 0               # times "are you the policyholder yourself?" was asked
    rep_name_turns: int = 0            # turns at "what is your own name?" that brought no name
    rep_same_name: bool = False        # a representative gave the policyholder's name as their own once
    refusal_count: int = 0
    rep_denials: int = 0
    consent: dict[str, Any] = field(
        default_factory=lambda: {"scenario": "default", "status": None, "polls": 0, "exhausted": False, "timeouts": 0}
    )

    # conversation control
    pending: str = "identity"
    offer_reason: str = ""             # why a human was last offered: "dead_end" when nothing else can be done here
    oos_streak: int = 0
    oos_total: int = 0
    distress_streak: int = 0
    distress_offer_made: bool = False
    last_empathy: str = ""
    mixed_note_turn: int = -10
    answers_given: int = 0

    # case handling
    case_id: str | None = None
    last_candidates: list[str] = field(default_factory=list)
    unresolved_asks: int = 0
    last_intent: str = "none"
    discussed: list[dict[str, Any]] = field(default_factory=list)

    # post process
    email: dict[str, Any] = field(default_factory=lambda: {"decision": None, "reasks": 0, "new_since": False})
    outbox: list[dict[str, Any]] = field(default_factory=list)
    handoff: dict[str, Any] | None = None

    # facts already given to this verified caller in earlier turns (for the output guard)
    grounded: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def say(self, role: str, text: str) -> None:
        self.history.append({"role": role, "text": text, "turn": self.turn})
        self.updated_at = time.time()

    def recent_history(self, limit: int = 12) -> list[dict[str, Any]]:
        return self.history[-limit:]

    def caller_texts(self) -> list[str]:
        return [item["text"] for item in self.history if item["role"] == "caller"]
