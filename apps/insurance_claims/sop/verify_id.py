"""Phase 1, VERIFY_ID: strict.

Nothing about a claim can be said in this phase, because no claim data exists
in the turn: the claim tools are not allowed here and the directive's facts
hold only the rules of verification itself. The model is free to talk
naturally, answer questions about the process, handle partial answers and
refusals, and offer alternate identity fields. It cannot advance the phase.

The gate opens in exactly one place, ``_grant``, when the identity matcher
reports three matching details (and, for a representative, when they are
listed on the policy and the policyholder has approved).
"""

from __future__ import annotations

from typing import Any

from sop_harness.core import Directive

from ..normalize import join_items
from ..state import RESOLVE_INTENT, Session
from ..understanding import Understanding
from ..verification import PII_FIELDS, REQUIRED_MATCHES, verify_identity
from .common import (
    CONSENT_FACTS,
    MAX_FAILED_ATTEMPTS,
    MAX_PERSUASION,
    PROCESS_FACTS,
    SHORT_LABELS,
    PolicyBase,
    TurnCtx,
    count_phrase,
)


class VerifyIdPhase(PolicyBase):
    def _grant(self, s: Session, t: TurnCtx, reason: str) -> None:
        """The only place a session becomes verified."""
        if not (s.identity_ok and s.party_id):
            raise RuntimeError("verification cannot be granted without a matching record")
        s.verified = True
        s.pending = "open"
        t.just_verified = True
        self.graph.move(s, RESOLVE_INTENT, reason, t.trace)

    def _ask_identity(self, s: Session, matched: list[str], remaining: list[str]) -> str:
        needed = REQUIRED_MATCHES - len(matched)
        options = self._labels(s, remaining, "or")
        if not matched:
            return f"Could you give me any three of these: {options}?"
        if needed == 1:
            return f"I just need one more detail to finish. It can be {options}."
        return f"I need {count_phrase(needed)} to finish. You can choose from {options}."

    def _verify(self, s: Session, u: Understanding, t: TurnCtx) -> Directive | None:
        m = s.memory
        result = verify_identity(self.data, m.identity)
        s.verification = result.summary()
        s.identity_ok = result.verified
        s.party_id = result.record.party_id if result.verified and result.record else None
        is_rep = self._is_rep(s)

        if result.verified:
            if not is_rep:
                self._grant(s, t, "three identity details matched one policyholder record")
                return None
            return self._rep_gate(s, u, t)

        matched = result.matched
        needed = REQUIRED_MATCHES - len(matched)
        remaining = [f for f in PII_FIELDS if f not in matched and f not in m.declined]
        alternatives = [f for f in remaining if f not in result.conflicts and f not in result.incomplete]
        new_pii = [f for f in t.new_identity if f in PII_FIELDS]
        who = "the policyholder's identity" if is_rep else "your identity"
        facts: dict[str, Any] = {
            "verification": {
                "rule": "At least three of five details must match: full name, date of birth, phone number, email address, last four digits of the SSN or national ID.",
                "details_confirmed_so_far": [SHORT_LABELS[f] for f in matched],
                "more_details_needed": needed,
                "caller_can_choose_from": [SHORT_LABELS[f] for f in remaining],
                "caller_declined_to_give": [SHORT_LABELS[f] for f in m.declined],
            },
            "process": PROCESS_FACTS,
            "caller_is_a_representative": is_rep,
        }
        if m.intent != "none" or m.backlog_question:
            facts["caller_request_noted_for_later"] = (
                "The caller has said what they are calling about. It is saved and will be picked up right "
                "after verification, so do not ask them to repeat it and do not discuss it yet."
            )
        rep_note = ""
        if is_rep:
            facts["representative"] = CONSENT_FACTS
            if not s.consent.get("notice"):
                s.consent["notice"] = True
                rep_note = (
                    " Since you're calling on someone else's behalf, I'll verify the policyholder's details "
                    "first, and then I'll need their consent before I can share anything about the claim."
                )
        s.pending = "identity"

        # 1. Something the caller gave does not match. Count the attempt and
        #    discard the wrong value so it can be corrected or replaced.
        if result.conflicts:
            for name in result.conflicts:
                m.identity.pop(name, None)
            s.failed_attempts += 1
            left = MAX_FAILED_ATTEMPTS - s.failed_attempts
            if left <= 0:
                return self._escalate(
                    s, t, "identity could not be verified after several attempts",
                    lead="I'm sorry, I wasn't able to verify your identity with the details provided.",
                    act="VERIFY_LOCKED",
                )
            bad = [SHORT_LABELS[f] for f in result.conflicts]
            if result.record is None:
                text = "I wasn't able to find a match with the details you gave me."
            elif matched:
                text = (
                    f"I've confirmed {self._labels(s, matched)}, but I wasn't able to match the "
                    f"{join_items(bad)} you gave me with our records."
                )
            else:
                text = f"I wasn't able to match the {join_items(bad)} you gave me with our records."
            text += " Could you double-check that"
            if alternatives:
                text += f", or give me a different detail instead, such as {self._labels(s, alternatives, 'or')}"
            text += "?"
            if left == 1:
                text += " For your security, I can try this one more time before I pass you to a representative."
            facts["verification"]["details_that_did_not_match"] = bad
            facts["verification"]["attempts_left"] = left
            return self._directive(
                s, t, "VERIFY_MISMATCH",
                goal="Some of the details the caller gave do not match the records. Say which ones, without revealing what is on file, and ask them to re-check or use a different detail.",
                fallback=text,
                facts=facts,
                cover=[
                    "Name the details that did not match.",
                    "Invite the caller to double-check them or to give a different accepted detail.",
                ] + (["Say that one more attempt is possible before a representative has to take over."] if left == 1 else []),
            )

        # 2. The caller has ruled out too many details to ever reach three.
        if len(matched) + len(remaining) < REQUIRED_MATCHES:
            s.refusal_count += 1
            if s.refusal_count > MAX_PERSUASION:
                return self._escalate(
                    s, t, "caller declined the identity details needed for verification",
                    lead="I understand, and I won't keep pressing you.",
                )
            s.human_offer_open = True
            return self._directive(
                s, t, "VERIFY_IMPOSSIBLE",
                goal="With the details the caller has declined, three matching details can no longer be reached. Explain that kindly and offer the two ways forward.",
                fallback=(
                    f"I understand you'd rather not share some of those. I do need three details in total to "
                    f"verify {who}, and with what's left I can't get there. Would you be willing to share one "
                    f"of them after all, or would you like me to connect you with a human representative?"
                ),
                facts=facts,
                cover=[
                    "Respect the caller's reluctance.",
                    "Explain that three details are required and why.",
                    "Offer the choice: reconsider one of the declined details, or a human representative.",
                ],
            )

        # 3. The caller refuses or pushes back, and gave nothing new.
        if u.refuses_verification and not new_pii:
            s.refusal_count += 1
            if s.refusal_count > MAX_PERSUASION:
                return self._escalate(
                    s, t, "caller declined identity verification",
                    lead="I understand, and I won't keep pressing you.",
                )
            ask = self._ask_identity(s, matched, remaining)
            if s.refusal_count == 1:
                credit = f"You're right that you've given me {self._labels(s, matched)}, thank you. " if matched else ""
                text = (
                    f"{credit}Your claim file holds private health and financial information, so I have to "
                    f"confirm it's really you before I can share anything from it, including the reason for a "
                    f"decision. It only takes a moment. {ask}"
                )
                cover = [
                    "Acknowledge the caller's feelings first.",
                    "Credit any identity details they have already given.",
                    "Explain that claim details are protected and verification has to come first, and why that protects them.",
                    "Offer the accepted ways to verify and ask for what is still needed.",
                ]
            else:
                s.human_offer_open = True
                text = (
                    f"I do want to get you an answer today. I'm not able to skip this step, because it's what "
                    f"stops anyone else from getting into your claim. You can use whichever details you're most "
                    f"comfortable with: {self._labels(s, remaining, 'or')}. Or, if you'd rather, I can connect "
                    f"you with a human representative. Which would you prefer?"
                )
                cover = [
                    "Acknowledge the caller's feelings first.",
                    "Say plainly, without lecturing, that this step cannot be skipped and what it protects.",
                    "Let them choose which accepted details to use.",
                    "Offer a human representative as the alternative and ask which they prefer.",
                ]
            facts["verification"]["times_caller_has_pushed_back"] = s.refusal_count
            return self._directive(
                s, t, f"VERIFY_PERSUADE_{s.refusal_count}",
                goal="The caller is pushing back on verification. De-escalate, explain why the step matters, and keep them moving towards completing it without giving anything away.",
                fallback=text,
                facts=facts,
                cover=cover,
                must_not=["Do not answer the caller's claim question yet, not even in general terms."],
            )

        # 4. A question about the process itself.
        if u.process_question and not new_pii:
            ask = self._ask_identity(s, matched, remaining)
            return self._directive(
                s, t, "VERIFY_EXPLAIN",
                goal="Answer the caller's question about the process from facts.process (and facts.representative if relevant), then continue with verification.",
                fallback=f"{self._explain_line(s, t)} {ask}",
                facts=facts,
                cover=["Answer the process question directly.", "Then ask for the identity details still needed."],
            )

        # 5. Progress, or a specific detail was declined.
        if new_pii or u.declined_fields or result.incomplete:
            opener = "Thank you."
            if u.has_request:
                opener += " I can help with that as soon as we've finished verifying."
            if u.declined_fields:
                opener = f"That's fine, you don't need to give me {self._labels(s, u.declined_fields, 'or')}."
            if matched:
                opener += f" I've confirmed {self._labels(s, matched)}."
            if result.incomplete:
                opener += " For the name to count I'll need the full first and last name."
            cover = ["Acknowledge what the caller just gave or declined.", "Ask for what is still needed, offering the remaining choices."]
            if u.has_request:
                cover.insert(1, "Tell the caller you will get to their request as soon as verification is done.")
            if u.emotion in {"angry", "frustrated"}:
                # An upset caller who is cooperating still deserves the reason for the step.
                opener += " Claim files hold private information, so I do have to finish this before I can share anything."
                cover.insert(1, "Say in one sentence why verification has to come first: the claim file is private.")
            if result.incomplete:
                cover.append("Ask for the full first and last name, because a first name alone does not count.")
            return self._directive(
                s, t, "VERIFY_PROGRESS",
                goal="Verification is under way but not complete. Acknowledge progress and ask for the remaining details.",
                fallback=f"{opener}{rep_note} {self._ask_identity(s, matched, remaining)}",
                facts=facts,
                cover=cover,
            )

        # 6. Nothing usable yet: ask (again), acknowledging any request we saved.
        ask = self._ask_identity(s, matched, remaining)
        if u.has_request:
            text = f"I'd be glad to help with that. Before I can look at any claim, I need to verify {who}.{rep_note} {ask}"
            cover = [
                "Tell the caller you can help with their request once verification is done.",
                "Ask for the identity details still needed.",
            ]
        else:
            text = f"To continue, I need to verify {who}.{rep_note} {ask}"
            cover = ["Ask for the identity details still needed."]
        return self._directive(
            s, t, "VERIFY_ASK",
            goal="Identity is not verified. Keep the conversation natural and ask for the details needed to verify.",
            fallback=text,
            facts=facts,
            cover=cover,
        )

    def _rep_gate(self, s: Session, u: Understanding, t: TurnCtx) -> Directive | None:
        """Policyholder details match. A representative also needs listing and consent."""
        m = s.memory
        holder = (m.identity.get("full_name") or "the policyholder").split()[0]
        facts: dict[str, Any] = {
            "representative": CONSENT_FACTS,
            "policyholder_details_match": True,
            "policyholder_first_name_as_given_by_caller": holder,
        }
        if not m.representative_name:
            s.pending = "rep_name"
            return self._directive(
                s, t, "ASK_REP_NAME",
                goal="The policyholder's details match. Ask the caller for their own full name so it can be checked against the authorized representatives.",
                fallback="Thank you, those details match. Since you're calling on someone else's behalf, may I have your own full name?",
                facts=facts,
                cover=["Say the details match.", "Ask for the caller's own full name."],
            )

        rep_first = m.representative_name.split()[0]
        if self.data.find_representative(s.party_id, m.representative_name) is None:
            s.rep_denials += 1
            if s.rep_denials >= 2:
                return self._escalate(
                    s, t, "caller is not listed as an authorized representative",
                    lead="I understand this is frustrating.",
                )
            s.pending = "human_offer"
            facts["caller_is_listed_as_authorized_representative"] = False
            facts.pop("policyholder_details_match", None)
            return self._directive(
                s, t, "REP_NOT_AUTHORIZED",
                goal="The caller is not listed as an authorized representative on this policy, so nothing about the claim can be shared. Explain kindly and offer the ways forward.",
                fallback=(
                    f"Thank you, {rep_first}. I don't see you listed as an authorized representative, so I'm "
                    f"not able to discuss this policy or any claim with you. "
                    f"{holder} can call us directly, or I can connect you with a human representative who can "
                    f"explain how to be added. Would you like me to do that?"
                ),
                facts=facts,
                cover=[
                    "Say you cannot discuss the claim because they are not listed as an authorized representative.",
                    "Offer the alternatives: the policyholder calls directly, or a human representative.",
                ],
            )

        facts["caller_is_listed_as_authorized_representative"] = True
        status = s.consent.get("status")
        first_request = status is None
        if first_request:
            status = self.tools.call("request_consent", s, t.trace)["status"]
        elif status == "pending":
            status = self.tools.call("check_consent_status", s, t.trace)["status"]
        facts["consent_request"] = {"status": status, "sent_to": "the phone number on file for the policyholder"}

        if status == "approved":
            self._grant(s, t, "policyholder details matched and the policyholder approved the representative")
            return None

        if status == "pending":
            s.pending = "consent"
            if first_request:
                text = (
                    f"Thank you, {rep_first}. Those details match, and I see you're listed as an authorized "
                    f"representative. Because the claim belongs to {holder}, I need their approval before I can "
                    f"share details, so I've just sent a consent request to the phone number on file for them. "
                    f"Could you let me know once they've approved it?"
                )
                cover = [
                    "Confirm the details match and that the caller is a listed representative.",
                    "Explain that the policyholder's approval is needed and why.",
                    "Say a consent request was sent to the phone number on file, and ask the caller to say when it is approved.",
                ]
            else:
                explain = f"{self._explain_line(s, t)} " if u.process_question else ""
                text = (
                    f"{explain}I've checked again and the approval hasn't come through yet. The request went to "
                    f"the phone number on file for {holder}. Let me know when it's done and I'll check right away."
                )
                cover = ["Say the approval has not arrived yet.", "Ask the caller to say when it has been approved."]
                if u.process_question:
                    cover.insert(0, "Answer the caller's question about consent from facts.representative.")
            return self._directive(
                s, t, "CONSENT_REQUESTED" if first_request else "CONSENT_PENDING",
                goal="A representative is calling. Claim details stay closed until the policyholder approves the consent request.",
                fallback=text,
                facts=facts,
                cover=cover,
            )

        # expired or denied
        s.consent["timeouts"] = s.consent.get("timeouts", 0) + 1
        s.pending = "human_offer"
        happened = "was declined" if status == "denied" else "has expired without an answer"
        return self._directive(
            s, t, "CONSENT_NOT_GIVEN",
            goal="The policyholder's consent was not obtained, so nothing about the claim can be shared. Explain and offer the ways forward.",
            fallback=(
                f"I'm sorry, the consent request {happened}, so I'm not able to share claim details. "
                f"{holder} can call us directly, or I can connect you with a human representative to work out "
                f"another way to confirm consent. Would you like me to transfer you?"
            ),
            facts=facts,
            cover=[
                f"Say the consent request {happened} and that claim details cannot be shared without it.",
                "Offer the alternatives: the policyholder calls directly, or a human representative.",
            ],
        )
