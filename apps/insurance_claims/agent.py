"""The turn pipeline: understand -> SOP policy -> word the reply -> guard.

    caller text
        |  redact full SSNs
        v
    understand      model (structured output) or offline rules
        v
    SOP policy      deterministic: memory, gates, tools, counters, directive
        v
    word the reply  model, from the directive's facts only (or the template)
        v
    output guard    deterministic: grounded values, no premature claims
        v
    reply

If a model call fails, the turn continues on the offline engine. The policy
in the middle is the same either way.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from sop_harness.core import TurnTrace
from sop_harness.llm import LLM, LLMError

from .data import ClaimsData
from .guard import check_reply
from .nlg import llm_respond
from .nlu_llm import llm_understand
from .nlu_rules import rule_understand
from .normalize import mask_email, redact_ssn
from .sop import (
    MAX_FAILED_ATTEMPTS,
    MAX_PERSUASION,
    OOS_ESCALATE_AT,
    OOS_OFFER_AT,
    InsuranceSOP,
)
from .state import ESCALATED, WORKFLOW_PHASES, Session
from .tools import build_tools
from .verification import PII_FIELDS, REQUIRED_MATCHES

# Phones and word processors turn ' into a typographic apostrophe. "I’m her son" has to read
# exactly like "I'm her son", for the rules and for the checks the policy makes on the caller's words.
PLAIN_PUNCTUATION = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u02bc": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u00a0": " ",
})

# The fixtures are dated early 2026 (claims filed January to March, an appeal
# deadline of 2026-03-18). The demo clock defaults to a date inside that
# window so deadlines read sensibly. Set SOP_TODAY=real to use the system date.
DEFAULT_DEMO_DATE = date(2026, 3, 5)
MAX_INPUT_CHARS = 2000  # one caller message; the API rejects longer ones


def clock_from_env() -> Callable[[], date]:
    value = os.environ.get("SOP_TODAY", "").strip().lower()
    if value in {"real", "system", "now"}:
        return date.today
    if value:
        try:
            fixed = date.fromisoformat(value)
            return lambda: fixed
        except ValueError:
            pass
    return lambda: DEFAULT_DEMO_DATE


@dataclass
class AgentConfig:
    clock: Callable[[], date] | None = None
    max_input_chars: int = MAX_INPUT_CHARS
    max_traces: int = 60


class Agent:
    def __init__(self, data: ClaimsData | None = None, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.data = data or ClaimsData()
        self.clock = self.config.clock or clock_from_env()
        self.tools = build_tools(self.data, self.clock)
        self.sop = InsuranceSOP(self.data, self.tools, self.clock)

    # ---------------------------------------------------------------- sessions

    def new_session(self, consent_scenario: str = "default") -> Session:
        session = Session()
        if consent_scenario in self.data.consent_scenarios:
            session.consent["scenario"] = consent_scenario
        session.say("agent", self.sop.greeting())
        return session

    # -------------------------------------------------------------------- turn

    def handle(self, session: Session, text: str, llm: LLM | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        clean = redact_ssn((text or "").strip()[: self.config.max_input_chars].translate(PLAIN_PUNCTUATION))
        today = self.clock()
        session.turn += 1
        trace = TurnTrace(turn=session.turn, phase_before=session.phase, phase_path=[session.phase])
        session.say("caller", clean)

        # 1. understand
        understanding = None
        model_usable = llm is not None and session.phase != ESCALATED
        if model_usable:
            try:
                understanding, result = llm_understand(llm, session, clean, self.data, today)
                trace.llm_calls.append(result.usage("understand"))
            except LLMError as exc:
                trace.llm_errors.append({"stage": "understand", "kind": exc.kind, "message": exc.message})
                model_usable = not exc.fatal_for_turn
            except Exception as exc:  # a model call may fail in any way: the turn still completes offline
                trace.llm_errors.append({"stage": "understand", "kind": "unexpected", "message": type(exc).__name__})
                model_usable = False
        if understanding is None:
            understanding = rule_understand(session, clean, self.data, today)
        trace.understanding = understanding.to_dict()

        # 2. decide (deterministic)
        directive = self.sop.step(session, understanding, trace, clean)
        trace.directive = directive.summary()
        if session.verified:
            # What a verified caller has been told stays grounded for the rest of the call.
            session.grounded.append(json.dumps(directive.facts, ensure_ascii=False))
            del session.grounded[:-20]

        # 3. word the reply, 4. guard it
        reply = directive.fallback
        trace.reply_source = "template"
        trace.guard = {"ok": True, "violations": [], "checked": "deterministic template"}
        if model_usable and not directive.fixed:
            try:
                result = llm_respond(llm, session, directive, clean)
                trace.llm_calls.append(result.usage("respond"))
                verdict = check_reply(result.text, session=session, directive=directive, data=self.data)
                trace.guard = {**verdict.to_dict(), "checked": "model reply"}
                if verdict.ok:
                    reply = result.text.strip()
                    trace.reply_source = "model"
                else:
                    trace.guard["rejected_reply"] = result.text
                    trace.reply_source = "template (the guard rejected the model reply)"
            except LLMError as exc:
                trace.llm_errors.append({"stage": "respond", "kind": exc.kind, "message": exc.message})
            except Exception as exc:  # the SOP step is already applied, so the template reply goes out
                trace.llm_errors.append({"stage": "respond", "kind": "unexpected", "message": type(exc).__name__})

        used_model = understanding.source == "llm" or trace.reply_source == "model"
        trace.engine = getattr(llm, "name", "llm") if (llm is not None and used_model) else "offline"
        session.say("agent", reply)
        trace.latency_ms = int((time.perf_counter() - started) * 1000)
        record = trace.to_dict()
        session.traces.append(record)
        del session.traces[: -self.config.max_traces]
        return {"reply": reply, "trace": record}

    # ------------------------------------------------------------- inspection

    def public_state(self, session: Session) -> dict[str, Any]:
        """State for the test UI's inspector. Not something a caller would see."""
        memory = session.memory
        party = self.data.party(session.party_id) if session.verified else None
        verification = session.verification or {}
        provided = {
            name: ("****" if name == "id_last4" else value) for name, value in memory.identity.items()
        }
        return {
            "session_id": session.id,
            "phase": session.phase,
            "workflow": WORKFLOW_PHASES,
            "turn": session.turn,
            "today": self.clock().isoformat(),
            "pending": session.pending,
            "verified": session.verified,
            "identity": {
                "required_matches": REQUIRED_MATCHES,
                "fields": list(PII_FIELDS),
                "provided": provided,
                "matched": verification.get("matched", []),
                "conflicts": verification.get("conflicts", []),
                "incomplete": verification.get("incomplete", []),
                "declined": list(memory.declined),
            },
            "caller": {
                "role": memory.caller_role,
                "representative_name": memory.representative_name,
                "policyholder": party.name if party else "",
                "policy_number": party.policy_number if party else "",
                "email_on_file": mask_email(party.email) if party else "",
            },
            "consent": dict(session.consent),
            "memory": {
                "intent": memory.intent,
                "intent_consumed": memory.intent_consumed,
                "case_hints": dict(memory.case_hints),
                "hints_consumed": memory.hints_consumed,
                "saved_question": memory.backlog_question,
                "pending_topics": list(memory.pending_topics),
                "email_preference": memory.email_pref,
                "notes": list(memory.notes),
                "last_emotion": memory.last_emotion,
                "captured": dict(memory.captured),
            },
            "case_id": session.case_id,
            "discussed": list(session.discussed),
            "counters": {
                "failed_verification_attempts": [session.failed_attempts, MAX_FAILED_ATTEMPTS],
                "verification_pushbacks": [session.refusal_count, MAX_PERSUASION + 1],
                "out_of_scope_streak": [session.oos_streak, OOS_ESCALATE_AT],
                "out_of_scope_offer_at": OOS_OFFER_AT,
                "distress_streak": session.distress_streak,
            },
            "email": dict(session.email),
            "outbox": list(session.outbox),
            "handoff": session.handoff,
            "allowed_tools": self.tools.allowed_in(session.phase),
        }
