"""Shared parts of the claims policy: limits, wording, and what every phase can do.

``PolicyBase`` holds the three things no phase handler may do differently:
building a directive (which attaches the disclosure rules for the caller's
verification state), handing off to a human, and declining out-of-scope
requests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from sop_harness.core import Directive, PhaseGraph, ToolRegistry, TurnTrace

from ..data import ClaimsData
from ..normalize import MONTH_NAMES, join_items, long_date
from ..state import (
    ESCALATED,
    POST_PROCESS,
    PROCESS_CASE,
    RESOLVE_INTENT,
    VERIFY_ID,
    Session,
)
from ..understanding import INTENT_LABELS, Understanding
from ..verification import REQUIRED_MATCHES

# ---- SOP limits --------------------------------------------------------------

MAX_FAILED_ATTEMPTS = 3     # mismatching identity attempts before a human takes over
MAX_PERSUASION = 2          # refusals met with empathy and explanation before handing off
OOS_OFFER_AT = 3            # consecutive out-of-scope messages before offering a human
OOS_ESCALATE_AT = 4         # ...and before transferring
DISTRESS_OFFER_AT = 3       # consecutive upset turns before proactively offering a human
MAX_EMAIL_REASKS = 2        # unclear answers to the email offer before defaulting to "skip"

# ---- rules attached to every directive -------------------------------------------

PRE_VERIFICATION_RULES = [
    "Do not state, confirm, deny or hint at anything about a claim or policy: not its existence, status, reason, amounts, documents or dates.",
    "Do not say or imply that the caller is verified.",
    "Do not reveal what is on file. Refer only to details the caller has given.",
]
POST_VERIFICATION_RULES = [
    "State only what is in facts. If the caller asks for something facts do not cover, say you don't have that information here and offer a human claims representative.",
    "Do not promise outcomes such as approval, payment or timing beyond what facts say.",
    "Do not take or promise any action other than explaining, sending the email summary when asked, or transferring to a human.",
]

# ---- grounded facts about the process itself (safe to say before verification) ---

# Between "the policyholder's details check out" and "the policyholder has consented".
REP_GATE_RULES = [
    "Do not state, confirm, deny or hint at anything about a claim: not its existence, status, reason, amounts, documents or dates.",
    "Do not say the caller is verified or may now hear claim details.",
    "You may say that the details given check out and that a consent request went to the policyholder. Reveal nothing else that is on file.",
]

PROCESS_FACTS = {
    "why_verification_is_required": (
        "Claim files hold private health and financial information. Verifying identity makes sure nobody "
        "else can see or act on the caller's claim. It is required before any claim detail can be discussed "
        "and it cannot be skipped."
    ),
    "what_is_accepted": (
        "Any three of: full name, date of birth, phone number on the policy, email address on the policy, "
        "last four digits of the SSN or national ID. The caller chooses which."
    ),
    "what_is_never_needed": "A full SSN, passwords or payment card numbers are never needed.",
    "what_it_prevents": (
        "Without this check, anyone who knows a policyholder's name could contact support and be told "
        "private medical and financial details."
    ),
    "how_long_it_takes": "Two or three short answers.",
    "alternatives": "The caller can pick different details, or ask for a human representative at any time.",
    "after_verification": "Once verified, the assistant can pull up the claim and answer questions straight away.",
    "who_you_are": "An automated virtual assistant for claims support.",
}
CONSENT_FACTS = {
    "why_consent_is_required": (
        "The claim belongs to the policyholder. A representative can be given claim details only when they are "
        "listed on the policy and the policyholder approves the request. This protects the policyholder's "
        "private information."
    ),
    "how_consent_works": (
        "A consent request goes to the phone number on file for the policyholder. The assistant checks its "
        "status and continues as soon as it is approved."
    ),
    "alternatives": "The policyholder can call directly, or a human representative can help.",
}

# ---- wording ----------------------------------------------------------------------

OWN_LABELS = {
    "full_name": "your full name",
    "dob": "your date of birth",
    "phone": "the phone number on your policy",
    "email": "the email address on your policy",
    "id_last4": "the last four digits of your SSN or national ID",
    "policy_number": "your policy number",
}
THEIR_LABELS = {
    "full_name": "the policyholder's full name",
    "dob": "their date of birth",
    "phone": "the phone number on the policy",
    "email": "the email address on the policy",
    "id_last4": "the last four digits of their SSN or national ID",
    "policy_number": "the policy number",
}
SHORT_LABELS = {
    "full_name": "full name",
    "dob": "date of birth",
    "phone": "phone number",
    "email": "email address",
    "id_last4": "last four digits of the SSN or national ID",
    "policy_number": "policy number",
}

EMPATHY = {
    "frustrated": [
        "I hear you, and I'm sorry this has been frustrating.",
        "I understand, and I'm sorry for the hassle.",
    ],
    "angry": [
        "I understand you're upset, and I'm sorry for the trouble this has caused you.",
        "You have every right to be upset, and I do want to get this sorted out for you.",
    ],
    "anxious": [
        "I can hear this is worrying you, and I'll take you through it one step at a time.",
        "I understand this is stressful, and we'll take it step by step.",
    ],
    "confused": [
        "No problem at all, let's take it one piece at a time.",
        "That's a fair thing to ask.",
    ],
    "sad": [
        "I'm really sorry you're going through this.",
        "I'm so sorry, that sounds really hard.",
    ],
}
TONE = {
    "frustrated": (
        "The caller sounds frustrated. Open with one sincere sentence that acknowledges it and responds to what "
        "they actually said. No defensiveness and no scripted apology."
    ),
    "angry": (
        "The caller is angry. Stay calm and respectful. Acknowledge the anger and the inconvenience in one "
        "sentence before anything else. Do not argue, and do not tell them to calm down."
    ),
    "anxious": (
        "The caller sounds worried. Reassure them briefly and concretely, for example that you will take them "
        "through it step by step, before continuing."
    ),
    "confused": "The caller is confused. Slow down, use plain words and explain the current step in one simple sentence.",
    "sad": "The caller is upset or grieving. Offer one brief, genuine sentence of sympathy before continuing.",
}

REPEAT_EMPATHY = "I understand, and I know repeating yourself is the last thing you want."
REPEAT_CUE = re.compile(r"already|again|how many times|\bi said\b|told you|for the (?:second|third|last) time", re.I)

GREETING = (
    "Hi, thanks for contacting claims support. I'm a virtual assistant and I can help with questions about "
    "your insurance claims. Before we look at anything I need to verify your identity. Could you give me any "
    "three of these: your full name, your date of birth, the phone number on your policy, the email address "
    "on your policy, or the last four digits of your SSN or national ID?"
)


@dataclass
class TurnCtx:
    trace: TurnTrace
    text: str = ""
    entered: bool = False            # the current handler runs because of a transition made this turn
    new_identity: list[str] = field(default_factory=list)
    turn_hints: dict[str, Any] = field(default_factory=dict)
    just_verified: bool = False
    case_just_resolved: bool = False
    same_case: bool = False
    resolved_by: str = ""
    hint_text: str = ""
    status_mismatch: str = ""        # the caller described the claim with a status it does not have
    exclude_case: str = ""           # "the other claim": anything but this one
    uncovered: bool = False          # the question maps to no workflow path
    declined_human: bool = False     # the caller just said no to an offered human
    mixed_scope: bool = False
    tone: str = ""
    empathy: str = ""
    offer_human: bool = False


def usd(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def count_phrase(n: int) -> str:
    return {1: "one more detail", 2: "two more details", 3: "three details"}.get(n, f"{n} more details")


def describe_hints(hints: dict[str, Any]) -> str:
    parts = [str(hints["status"])] if hints.get("status") else []
    if hints.get("claim_type") and hints["claim_type"] != "other":
        parts.append(str(hints["claim_type"]))
    text = " ".join(parts + ["claim"])
    if hints.get("claim_type") == "other":
        text += " of that kind"
    month, year = hints.get("month"), hints.get("year")
    if month and year:
        text += f" from {MONTH_NAMES[month - 1]} {year}"
    elif month:
        text += f" from {MONTH_NAMES[month - 1]}"
    elif year:
        text += f" from {year}"
    if hints.get("claim_id"):
        text += f" {hints['claim_id']}"
    return text


def filter_claims(claims: list[dict[str, Any]], hints: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for claim in claims:
        year, month, _day = claim["filed"].split("-")
        if hints.get("claim_id") and claim["claim_id"] != hints["claim_id"]:
            continue
        if hints.get("claim_type") and claim["type"] != hints["claim_type"]:
            continue
        if hints.get("status") and claim["status"] != hints["status"]:
            continue
        if hints.get("month") and int(month) != hints["month"]:
            continue
        if hints.get("year") and int(year) != hints["year"]:
            continue
        out.append(claim)
    return out


def claim_phrase(claim: dict[str, Any]) -> str:
    return (
        f"{article(claim['status'])} {claim['status']} {claim['type']} claim filed "
        f"{long_date(claim['filed'])} ({claim['claim_id']})"
    )


class PolicyBase:
    """Capabilities shared by all phase handlers."""

    data: ClaimsData
    tools: ToolRegistry
    clock: Callable[[], date]
    graph: PhaseGraph

    # ----------------------------------------------------------------- helpers

    def _today(self) -> str:
        return self.clock().isoformat()

    def _is_rep(self, s: Session) -> bool:
        return s.memory.caller_role == "representative"

    def _first_name(self, s: Session) -> str:
        m = s.memory
        if self._is_rep(s) and m.representative_name:
            return m.representative_name.split()[0]
        party = self.data.party(s.party_id) if s.verified else None
        return party.first_name if party else ""

    def _labels(self, s: Session, fields: list[str], conjunction: str = "and") -> str:
        table = THEIR_LABELS if self._is_rep(s) else OWN_LABELS
        return join_items([table[f] for f in fields], conjunction)

    def _default_pending(self, s: Session) -> str:
        if s.phase == VERIFY_ID:
            return "consent" if s.consent.get("status") == "pending" else "identity"
        if s.phase == RESOLVE_INTENT:
            return "choose_case" if s.last_candidates else "open"
        if s.phase == PROCESS_CASE:
            return "anything_else"
        if s.phase == POST_PROCESS:
            return "email_offer"
        return "none"

    def _directive(
        self,
        s: Session,
        t: TurnCtx,
        act: str,
        goal: str,
        fallback: str,
        *,
        facts: dict[str, Any] | None = None,
        cover: list[str] | None = None,
        must_not: list[str] | None = None,
        fixed: bool = False,
        rules: list[str] | None = None,
    ) -> Directive:
        facts = dict(facts or {})
        cover = list(cover or [])
        base_rules = rules if rules is not None else (POST_VERIFICATION_RULES if s.verified else PRE_VERIFICATION_RULES)
        rules = list(base_rules) + list(must_not or [])
        if s.verified:
            facts.setdefault("today", self._today())
            name = self._first_name(s)
            if name:
                facts.setdefault("caller_first_name", name)
        text = fallback.strip()
        heard = t.trace.memory_captured
        if s.phase not in {POST_PROCESS, ESCALATED} and not fixed:
            if "email_pref=send" in heard:
                facts["email_summary_request"] = (
                    "The caller asked for an email summary of the call. That is noted. A summary is offered at the "
                    "end of the call, once their questions are answered, and goes to the email address on file. "
                    "Nothing is sent now."
                )
                cover.append("Tell the caller their request for an email summary is noted and will be taken up at the end of the call.")
                text = "I've noted that you'd like an email summary, and I'll come back to it at the end of our conversation. " + text
            elif "email_pref=skip" in heard:
                facts["email_summary_request"] = "The caller said they do not want an email summary. That is noted and none will be sent."
                cover.append("Confirm that no email summary will be sent.")
                text = "I've noted that you don't want an email summary, so I won't send one. " + text
        if t.mixed_scope and s.phase != ESCALATED and s.mixed_note_turn != s.turn - 1:
            s.mixed_note_turn = s.turn
            cover.append("Say briefly that the unrelated part of the message is outside what you can help with here.")
            text = "I can only help with insurance claims, so I'll leave the other part of your message aside. " + text
        if t.empathy:
            text = f"{t.empathy} {text}"
        if t.offer_human and s.phase in {RESOLVE_INTENT, PROCESS_CASE} and not fixed:
            cover.append(
                "The caller has been upset for several turns. Add one sentence saying they can ask for a human "
                "representative at any point. Make it a statement, not a question."
            )
            text += " If you'd rather speak with a person at any point, just say so and I'll connect you."
            s.distress_offer_made = True
        return Directive(
            act=act, goal=goal, fallback=text, facts=facts, must_cover=cover, must_not=rules, tone=t.tone, fixed=fixed
        )

    def _explain_line(self, s: Session, t: TurnCtx) -> str:
        low = t.text.lower()
        if re.search(r"\b(?:bot|robot|ai|human|machine|computer)\b|real person|who are you|what are you|who am i", low):
            return "I'm an automated virtual assistant for claims support, and I can bring in a human representative whenever you ask."
        if s.verified and (s.phase == POST_PROCESS or re.search(r"summary|e-?mail", low)):
            return (
                "The summary lists what we discussed, your claim status and your next steps, and it can only "
                "go to the email address on file."
            )
        if re.search(r"consent|approv", low) or s.consent.get("status") == "pending":
            return (
                "The claim belongs to the policyholder, so I can only share details with someone else once "
                "they've approved it. It protects their private information."
            )
        return (
            "I ask because claim files hold private health and financial information, and verifying makes sure "
            "only you, or someone you've authorized, can get into yours. I never need a full SSN, only the last "
            "four digits, and you choose which details to give."
        )

    def _holder(self, s: Session) -> str:
        """How to refer to the policyholder when talking to a representative."""
        name = s.memory.identity.get("full_name") or ""
        return name.split()[0] if name else "the policyholder"

    def _verified_intro(self, s: Session, t: TurnCtx) -> str:
        if not t.just_verified:
            return ""
        name = self._first_name(s)
        if self._is_rep(s):
            holder = self._holder(s)
            holder = holder[0].upper() + holder[1:]
            return f"Thank you, {name}. {holder}'s approval has come through, so I can go over the claim with you. "
        return f"Thank you, {name}. You're verified. " if name else "Thank you. You're verified. "

    # ------------------------------------------------- hand-off and scope control

    def _escalate(
        self, s: Session, t: TurnCtx, reason: str, lead: str, act: str = "ESCALATED", lead_acknowledges: bool = True
    ) -> Directive:
        was_verified = s.verified
        if lead_acknowledges:
            t.empathy = ""
        out = self.tools.call("transfer_to_human", s, t.trace, reason=reason)
        self.graph.move(s, ESCALATED, reason, t.trace)
        s.pending = "none"
        t.offer_human = False
        t.mixed_scope = False
        ticket = s.handoff or {}
        passed_along = {
            "identity_details_given": ticket.get("identity_details_provided") or [],
            "what_the_caller_asked_about": "" if ticket.get("intent") in (None, "none") else INTENT_LABELS.get(ticket["intent"], ""),
            "claim": ticket.get("claim") or describe_hints(ticket["claim_hints"]) if ticket.get("claim") or ticket.get("claim_hints") else "",
            "topics_already_covered": len(ticket.get("topics_discussed") or []),
            "notes": len(ticket.get("notes") or []),
        }
        has_context = any(passed_along.values())
        if has_context:
            text = (
                f"{lead} I'm connecting you with a human representative now. I've passed along what you've told "
                f"me so you won't have to repeat yourself, and your reference is {out['ticket_id']}."
            )
            handover = "Say what they told you has been passed along (see facts.handoff.passed_along), so they will not have to repeat it."
        else:
            text = (
                f"{lead} I'm connecting you with a human representative now. I've noted why I'm transferring "
                f"you, and your reference is {out['ticket_id']}."
            )
            handover = "Say the reason for the transfer has been noted. Do not say they will not have to repeat themselves: nothing else was captured."
        cover = [
            "Tell the caller you are connecting them with a human representative now.",
            handover,
            f"Give the reference {out['ticket_id']}.",
        ]
        if not was_verified:
            text += " For your protection, they will also need to verify your identity before discussing any claim details."
            cover.append("Mention that the representative will also need to verify identity before discussing claim details.")
        return self._directive(
            s, t, act,
            goal=f"Hand the conversation to a human representative. Reason: {reason}.",
            fallback=text,
            facts={
                "handoff": {
                    "ticket_id": out["ticket_id"],
                    "reason": reason,
                    "identity_verified": was_verified,
                    "passed_along": passed_along if has_context else "only the reason for the transfer",
                }
            },
            cover=cover,
            must_not=["Do not keep persuading or ask further questions."],
        )

    def _after_handoff(self, s: Session, t: TurnCtx) -> Directive:
        if s.handoff is not None:
            s.handoff["messages_after_handoff"].append(t.text)
        ticket = s.handoff["ticket_id"] if s.handoff else ""
        t.empathy, t.tone = "", ""
        return self._directive(
            s, t, "ESCALATED_WAIT",
            goal="The conversation has been handed to a human representative.",
            fallback=(
                f"You're in the queue for a human representative (reference {ticket}). "
                "I've added your message to the notes for them."
            ),
            facts={"handoff": {"ticket_id": ticket}},
            fixed=True,
        )

    def _out_of_scope(self, s: Session, u: Understanding, t: TurnCtx) -> Directive:
        s.oos_streak += 1
        s.oos_total += 1
        t.empathy, t.tone = "", ""
        if s.phase == VERIFY_ID and u.refuses_verification:
            # An instruction-override or "I'm already verified" message is pressure to skip
            # the gate. It counts towards knowing when to stop persuading.
            s.refusal_count += 1
            if s.refusal_count > MAX_PERSUASION:
                return self._escalate(
                    s, t, "caller declined identity verification",
                    lead="I understand, and I won't keep pressing you.",
                )
        if s.oos_streak >= OOS_ESCALATE_AT:
            return self._escalate(
                s, t, "repeated out-of-scope requests",
                lead="These questions are outside what I'm able to help with, so a person will be better placed to assist you.",
            )
        redirect = self._redirect(s)
        facts = {
            "scope": "This line only handles insurance claims support.",
            "out_of_scope_count": s.oos_streak,
            "where_the_call_stands": redirect,
        }
        if s.oos_streak >= OOS_OFFER_AT:
            s.pending = "human_offer"
            return self._directive(
                s, t, "OOS_OFFER_HUMAN",
                goal="The caller keeps asking for things outside insurance claims support. Decline again and offer a human representative.",
                fallback=(
                    "I'm still not able to help with that, since it's outside insurance claims support. If "
                    "there's something you need that I'm not picking up on, I can connect you with a human "
                    "representative. Would you like me to do that?"
                ),
                facts=facts,
                cover=["Decline politely without answering the question.", "Ask whether they would like a human representative."],
                must_not=["Do not answer the out-of-scope question, even partly."],
                fixed=True,
            )
        opener = (
            "I'm sorry, that's outside what I can help with here. I can only assist with insurance claims and related questions."
            if s.oos_streak == 1
            else "I'm not able to help with that either, I'm afraid, as I can only assist with insurance claims."
        )
        return self._directive(
            s, t, "OOS_DECLINE",
            goal="The caller asked for something outside insurance claims support. Decline politely and steer back to the current step.",
            fallback=f"{opener} {redirect}",
            facts=facts,
            cover=["Decline politely without answering the question.", "Bring the caller back to where the call stands."],
            must_not=["Do not answer the out-of-scope question, even partly."],
            fixed=True,
        )

    def _redirect(self, s: Session) -> str:
        if s.phase == VERIFY_ID:
            if s.consent.get("status") == "pending":
                return "I'm still waiting for the policyholder's approval before we can continue."
            if s.identity_ok:
                if s.pending == "rep_name":
                    return "To continue, I still need your own full name."
                return "I can't share anything about the claim without the policyholder's approval."
            matched = s.verification.get("matched", [])
            if matched:
                return f"To continue, I still need {count_phrase(REQUIRED_MATCHES - len(matched))} to verify your identity."
            return "If you have a question about a claim, I'll first need to verify your identity with any three of the details I mentioned."
        if s.phase == RESOLVE_INTENT:
            return "Which claim can I help you with today?"
        if s.phase == PROCESS_CASE:
            return f"Is there anything else you'd like to know about claim {s.case_id}?"
        if s.phase == POST_PROCESS:
            return "Would you like me to send the email summary, or skip it?"
        return "If you have another question about a claim, I'm happy to help."
