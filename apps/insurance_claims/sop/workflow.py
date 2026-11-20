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

import re
from datetime import date
from typing import Callable

from sop_harness.core import Directive, PhaseGraph, ToolRegistry, TurnTrace

from ..data import ClaimsData
from ..nlu_rules import states_third_party
from ..state import (
    CLOSED,
    ESCALATED,
    POST_PROCESS,
    PROCESS_CASE,
    RESOLVE_INTENT,
    VERIFY_ID,
    Session,
)
from ..understanding import MENTIONS_OTHER_RE, OWN_DETAILS_RE, Understanding
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

        if s.verified and u.caller_role == "representative" and not self._is_rep(s):
            # The gate opened for the policyholder. Someone who now says they are calling
            # for the policyholder has not been through listing and consent, so disclosure stops.
            return self._escalate(
                s, t, "caller was verified as the policyholder, then said they are calling on someone else's behalf",
                lead=(
                    "Thank you for telling me. Claim details can only go to the policyholder or to a "
                    "representative they have approved, so a person needs to take it from here."
                ),
            )

        # A human is brought in on a clear request, or on a yes to a question that offered
        # only that. A yes to "verify, or a human?" is not a request for a human.
        accepted_offer = s.pending == "human_offer" and u.affirmation == "yes" and not u.done and u.scope != "out_of_scope"
        if u.wants_human or accepted_offer:
            reason = "caller asked for a human representative" if u.wants_human else "caller accepted the offer of a human representative"
            return self._escalate(s, t, reason, lead="Of course.", lead_acknowledges=False)
        if s.pending == "human_offer":
            # The offer was not taken up. Go back to where the workflow was.
            # Saying no to a person ends the conversation only where nothing else could be
            # done (no claims on file, not an authorized representative). After an
            # out-of-scope or "can't do that here" offer, the workflow simply goes on.
            t.declined_human = (u.affirmation == "no" or u.done) and s.offer_reason == "dead_end"
            s.pending = self._default_pending(s)
            s.offer_reason = ""
            u.affirmation = "none"

        if u.scope == "out_of_scope":
            return self._out_of_scope(s, u, t)
        # One relevant message takes one off the count. It does not wipe it, so a caller
        # cannot keep asking unrelated questions by slipping a relevant line in between.
        s.oos_streak = max(0, s.oos_streak - 1)
        t.mixed_scope = u.scope == "mixed"
        t.offer_human = s.distress_streak >= DISTRESS_OFFER_AT and not s.distress_offer_made

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
            already_matched = set(s.verification.get("matched", []))
            holder = (m.identity.get("full_name") or "").lower()
            if u.source == "llm" and u.caller_role == "representative" and not u.representative_name and not states_third_party(t.text):
                # The model concluded this from the wording ("my husband told me to call"). The
                # caller did not say it, so it is settled with the role question, not assumed.
                u.guessed.append("caller_role")
            # Has the caller acted as a representative: given a name of their own that is not
            # the policyholder's, or been told they are not listed?
            acted_as_rep = bool(
                (m.representative_name and m.representative_name.lower() != holder) or s.rep_denials or s.rep_name_refused
            )
            if m.caller_role == "representative" and m.role_guessed and not acted_as_rep and u.caller_role != "representative":
                if OWN_DETAILS_RE.search(t.text):
                    # The role was read from a turn of phrase, and now they give details as their
                    # own ("my date of birth is ..."). Treat the role as open and ask later.
                    m.caller_role, m.role_guessed, m.role_unclear = "unknown", False, True
                    if u.representative_name and "full_name" not in u.identity:
                        u.identity["full_name"] = u.representative_name
                    u.representative_name = ""
            for name, value in u.identity.items():
                stored = m.identity.get(name, "")
                if name == "full_name" and " " not in value and stored and " " not in stored and stored.lower() != value.lower():
                    # "Margaret" in one message and "Chen" in the next are one name
                    first = re.search(r"\bfirst\s+name\b", t.text, re.I)
                    value = f"{value} {stored}" if first else f"{stored} {value}"
                if name in already_matched and m.identity.get(name) != value:
                    continue  # a detail that already matched is not replaced by a later reading
                if m.identity.get(name) != value:
                    m.identity[name] = value
                    t.new_identity.append(name)
                    m.remember(f"identity.{name}", turn, phase)
                    captured.append(f"identity.{name}")
                if name in m.declined:
                    m.declined.remove(name)
            if len(u.declined_fields) >= 4:
                # A blanket "I'm not giving you anything" is a refusal to verify, which the
                # persuasion ladder handles. It is not a list of fields to work around.
                u.refuses_verification = True
                u.declined_fields = []
            for name in u.declined_fields:
                if name not in m.declined:
                    m.declined.append(name)
                    captured.append(f"declined.{name}")
                if name in m.identity and name not in u.identity and name not in already_matched:
                    del m.identity[name]  # the caller withdrew a value given earlier
            # The cue is taken from the caller's own words, so the model path gets it too.
            if (u.mentions_other or (u.caller_role == "unknown" and MENTIONS_OTHER_RE.search(t.text))) and m.caller_role == "unknown":
                m.role_unclear = True
            if u.caller_role != "unknown":
                m.role_unclear = False
            if u.caller_role != "unknown" and u.caller_role != m.caller_role:
                inferred = "caller_role" in u.guessed
                if m.caller_role == "representative" and (not m.role_guessed or acted_as_rep):
                    # Someone who said they were calling for the policyholder cannot
                    # drop the consent requirement by saying otherwise later.
                    note = "Caller first said they were a representative, then said they were the policyholder."
                    if note not in m.notes:
                        m.notes.append(note)
                elif m.caller_role == "policyholder" and inferred:
                    pass  # "I'm the policyholder" outweighs a role inferred from a turn of phrase
                else:
                    # A role that was only inferred gives way to what the caller says outright.
                    m.caller_role = u.caller_role
                    m.role_guessed = inferred
                    captured.append(f"caller_role={u.caller_role}")
            if u.representative_name and u.representative_name != m.representative_name:
                m.representative_name = u.representative_name
                captured.append("representative_name")

        if u.intent != "none":
            # "the claim from 2025" is read as a general question. It must not replace a
            # specific question the caller asked earlier and is still waiting on.
            specific_waiting = m.intent not in {"none", "general_claim_question"} and not m.intent_consumed
            waiting_phase = phase in {VERIFY_ID, RESOLVE_INTENT}
            if u.intent == "general_claim_question" and specific_waiting:
                pass
            elif specific_waiting and waiting_phase and u.intent != m.intent:
                # A second question before the first could be answered: keep both, in order.
                m.extra_intents = list(dict.fromkeys([*m.extra_intents, u.intent, *u.extra_intents]))[:3]
                if u.question and u.question not in m.backlog_question:
                    m.backlog_question = f"{m.backlog_question} {u.question}".strip()
                captured.append(f"intent+={u.intent}")
            else:
                m.intent = u.intent
                m.extra_intents = [i for i in u.extra_intents if i != u.intent]
                m.intent_consumed = False
                m.remember("intent", turn, phase)
                captured.append(f"intent={u.intent}")
                if u.question and waiting_phase:
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
                corrected = any(
                    key in m.case_hints and m.case_hints[key] != value
                    for key, value in u.case_hints.items()
                    if key != "exclude_current"
                )
                if corrected:
                    m.case_hints = {}  # "sorry, I meant my auto claim": start again from what they say now
                m.case_hints.update({k: v for k, v in u.case_hints.items() if k != "exclude_current"})
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
