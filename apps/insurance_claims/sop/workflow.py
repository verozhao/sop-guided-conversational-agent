"""The insurance claims SOP as a deterministic policy.

    VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE -> POST_PROCESS

``InsuranceSOP.step`` takes the structured reading of one caller message and
returns a ``Directive``: what the reply must do and the only facts it may
use. Phase order, safety gates, tool calls, counters and escalation are all
decided here in code. A model only reads the message beforehand and words the
reply afterwards.

How much freedom each phase gives the model:

* VERIFY_ID      strict. No claim data exists in the turn. The reply may only
                 work towards verification, explain why, or hand off.
* RESOLVE_INTENT guided. The model interprets messy language into a bounded
                 intent and claim choice. Code validates both against the
                 caller's own claims.
* PROCESS_CASE   freest. The model answers follow-up questions in its own
                 words, from tool results for the chosen workflow path only.
* POST_PROCESS   strict again. An email is sent only on an explicit choice.
"""

from __future__ import annotations

from datetime import date
from typing import Callable

from sop_harness.core import Directive, PhaseGraph, ToolRegistry, TurnTrace

from ..data import ClaimsData
from ..state import (
    CLOSED,
    ESCALATED,
    POST_PROCESS,
    PROCESS_CASE,
    RESOLVE_INTENT,
    VERIFY_ID,
    Session,
)
from ..understanding import Understanding
from .common import (
    DISTRESS_OFFER_AT,
    EMPATHY,
    GREETING,
    REPEAT_CUE,
    REPEAT_EMPATHY,
    TONE,
    TurnCtx,
    describe_hints,
)
from .post_process import PostProcessPhase
from .process_case import ProcessCasePhase
from .resolve_intent import ResolveIntentPhase
from .verify_id import VerifyIdPhase

# Which phase may follow which. Entry gates are added in ``InsuranceSOP.__init__``.
EDGES: dict[str, set[str]] = {
    VERIFY_ID: {RESOLVE_INTENT, ESCALATED},
    RESOLVE_INTENT: {PROCESS_CASE, POST_PROCESS, ESCALATED},
    PROCESS_CASE: {RESOLVE_INTENT, POST_PROCESS, ESCALATED},
    POST_PROCESS: {RESOLVE_INTENT, PROCESS_CASE, CLOSED, ESCALATED},
    CLOSED: {RESOLVE_INTENT, ESCALATED},
    ESCALATED: set(),
}


class InsuranceSOP(VerifyIdPhase, ResolveIntentPhase, ProcessCasePhase, PostProcessPhase):
    def __init__(self, data: ClaimsData, tools: ToolRegistry, clock: Callable[[], date]):
        self.data = data
        self.tools = tools
        self.clock = clock

        def verified_gate(session: Session) -> tuple[bool, str]:
            return session.verified, "identity is not verified"

        def case_gate(session: Session) -> tuple[bool, str]:
            if not session.verified:
                return False, "identity is not verified"
            if not session.case_id or data.claim_for_party(session.party_id, session.case_id) is None:
                return False, "no claim of the verified caller is selected"
            return True, ""

        self.graph = PhaseGraph(
            EDGES,
            gates={
                RESOLVE_INTENT: verified_gate,
                PROCESS_CASE: case_gate,
                POST_PROCESS: verified_gate,
                CLOSED: verified_gate,
            },
        )
        self._handlers = {
            VERIFY_ID: self._verify,
            RESOLVE_INTENT: self._resolve,
            PROCESS_CASE: self._process,
            POST_PROCESS: self._post,
            CLOSED: self._closed,
        }

    # ------------------------------------------------------------------ entry

    def greeting(self) -> str:
        return GREETING

    def step(self, s: Session, u: Understanding, trace: TurnTrace, text: str = "") -> Directive:
        t = TurnCtx(trace=trace, text=text)
        self._absorb(s, u, t)
        self._set_tone(s, u, t)

        if s.phase == ESCALATED:
            return self._after_handoff(s, t)

        accepted_offer = u.affirmation == "yes" and u.scope != "out_of_scope" and (
            s.pending == "human_offer"
            or (s.human_offer_open and s.pending in {"identity", "consent", "open"} and not t.new_identity)
        )
        if u.wants_human or accepted_offer:
            reason = "caller asked for a human representative" if u.wants_human else "caller accepted the offer of a human representative"
            return self._escalate(s, t, reason, lead="Of course.", lead_acknowledges=False)
        if s.pending == "human_offer":
            # The offer was not taken up. Go back to where the workflow was.
            s.pending = self._default_pending(s)
            u.affirmation = "none"
            u.done = False
        s.human_offer_open = False

        if u.scope == "out_of_scope":
            return self._out_of_scope(s, t)
        s.oos_streak = 0
        t.mixed_scope = u.scope == "mixed"
        t.offer_human = (
            s.distress_streak >= DISTRESS_OFFER_AT and not s.distress_offer_made and s.phase != VERIFY_ID
        )

        for _ in range(6):
            directive = self._handlers[s.phase](s, u, t)
            if directive is not None:
                return directive
            t.entered = True
        raise RuntimeError("SOP policy did not settle on a directive")

    # ------------------------------------------------------------------ memory

    def _absorb(self, s: Session, u: Understanding, t: TurnCtx) -> None:
        """Store everything useful the caller said, whatever the current phase."""
        m, turn, phase = s.memory, s.turn, s.phase
        captured = t.trace.memory_captured

        if not s.verified:
            for name, value in u.identity.items():
                if m.identity.get(name) != value:
                    m.identity[name] = value
                    t.new_identity.append(name)
                    m.remember(f"identity.{name}", turn, phase)
                    captured.append(f"identity.{name}")
                if name in m.declined:
                    m.declined.remove(name)
            for name in u.declined_fields:
                if name not in m.declined:
                    m.declined.append(name)
                    captured.append(f"declined.{name}")
                if name in m.identity and name not in u.identity:
                    del m.identity[name]  # the caller withdrew a value given earlier
            if u.caller_role != "unknown" and u.caller_role != m.caller_role:
                if m.caller_role == "representative":
                    # Someone who said they were calling for the policyholder cannot
                    # drop the consent requirement by saying otherwise later.
                    note = "Caller first said they were a representative, then said they were the policyholder."
                    if note not in m.notes:
                        m.notes.append(note)
                else:
                    m.caller_role = u.caller_role
                    captured.append(f"caller_role={u.caller_role}")
            if u.representative_name and u.representative_name != m.representative_name:
                m.representative_name = u.representative_name
                captured.append("representative_name")

        if u.intent != "none":
            m.intent = u.intent
            m.intent_consumed = False
            m.remember("intent", turn, phase)
            captured.append(f"intent={u.intent}")
        if u.question and u.intent != "none" and phase == VERIFY_ID:
            m.backlog_question = u.question
            m.remember("question", turn, phase)
            captured.append("question")
        if phase != PROCESS_CASE:
            for topic in u.followup_topics:
                if topic not in m.pending_topics:
                    m.pending_topics.append(topic)
                    captured.append(f"topic={topic}")
        if u.case_hints:
            t.turn_hints = dict(u.case_hints)
            if s.case_id is None:
                m.case_hints.update(u.case_hints)
                m.hints_consumed = False
                m.remember("case_hints", turn, phase)
                captured.append(f"case_hints={describe_hints(u.case_hints)}")
        if u.email_summary_choice != "none" and s.pending != "email_offer" and s.email["decision"] is None:
            if m.email_pref != u.email_summary_choice:
                m.email_pref = u.email_summary_choice
                m.remember("email_pref", turn, phase)
                captured.append(f"email_pref={u.email_summary_choice}")
        for note in u.notes:
            if note not in m.notes and len(m.notes) < 10:
                m.notes.append(note)
                captured.append("note")

        m.last_emotion = u.emotion
        s.distress_streak = s.distress_streak + 1 if u.emotion in {"angry", "frustrated"} else 0

    def _set_tone(self, s: Session, u: Understanding, t: TurnCtx) -> None:
        if u.emotion not in TONE:
            return
        t.tone = TONE[u.emotion]
        options = list(EMPATHY[u.emotion])
        if u.emotion == "frustrated" and REPEAT_CUE.search(t.text):
            options.insert(0, REPEAT_EMPATHY)
        t.empathy = next((line for line in options if line != s.last_empathy), options[0])
        s.last_empathy = t.empathy
