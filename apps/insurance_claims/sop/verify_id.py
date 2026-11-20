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

import re
from typing import Any

from sop_harness.core import Directive

from ..nlu_rules import role_answer
from ..normalize import join_items
from ..state import RESOLVE_INTENT, Session
from ..understanding import Understanding
from ..verification import PII_FIELDS, REQUIRED_MATCHES, verify_identity
from .common import (
    CONSENT_FACTS,
    MAX_FAILED_ATTEMPTS,
    MAX_NAME_MISSES,
    MAX_PERSUASION,
    PROCESS_FACTS,
    REP_GATE_RULES,
    SHORT_LABELS,
    PolicyBase,
    TurnCtx,
    count_phrase,
)

# A detail given only in part is neither accepted nor counted as wrong. The caller is told what is missing.
PARTIAL_NOTES = {
    "dob": " For the date of birth I'll need the full date, with the day, the month and the year.",
    "dob_or_id": (
        " I wasn't sure whether that number is a year of birth or the last four digits of an SSN or ID, so I "
        "haven't used it. A date of birth needs the full date, and for the ID just tell me it's the last four digits."
    ),
    "phone": " That phone number looks incomplete, so please give the full number with the area code.",
}
PARTIAL_LABELS = {
    "dob": "a date of birth without the full day, month and year",
    "dob_or_id": "a four-digit number that could be a year of birth or the last four digits of an ID",
    "phone": "a phone number with digits missing",
}
REP_NAME_WHY = "Only a representative who is listed on the policy can be approved, so I need your own name to check that."


# "she approved it", "it's done": the caller reports that the approval was given
APPROVAL_SAID_RE = re.compile(
    r"\b(?:has|have|just|already)\s+(?:approved|accepted|confirmed|consented)\b|\b(?:approved|accepted|confirmed)\s+(?:it|the|that|my)\b|"
    r"\bsaid\s+(?:yes|ok|okay)\b|\b(?:it's|it\s+is|that's|all)\s+(?:done|approved|sorted)\b|"
    r"\b(?:she|he|they)\s+(?:did|approved|accepted|consented)\b|^\W*(?:done|approved)\W*$",
    re.I,
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
        guess = "full_name" in u.guessed and "full_name" in t.new_identity
        if guess and s.name_misses >= MAX_NAME_MISSES:
            # The allowance for guessed names is spent: a guess is no longer tried at all,
            # so it can neither probe the records nor count against the caller.
            m.identity.pop("full_name", None)
            t.new_identity = [f for f in t.new_identity if f != "full_name"]
            guess = False
        result = verify_identity(self.data, m.identity, prefer=s.candidate_id)
        if guess and "full_name" in result.conflicts:
            # A short reply the offline rules took for a name. It matches nobody, so it was
            # probably not a name: drop it without comment. The allowance is limited.
            s.name_misses += 1
            m.identity.pop("full_name", None)
            t.new_identity = [f for f in t.new_identity if f != "full_name"]
            result = verify_identity(self.data, m.identity, prefer=s.candidate_id)
        if result.record is not None:
            s.candidate_id = result.record.party_id
        s.verification = result.summary()
        s.identity_ok = result.verified
        s.party_id = result.record.party_id if result.verified and result.record else None
        holder_name = (m.identity.get("full_name") or "").lower()
        own_name = bool(m.representative_name) and m.representative_name.lower() != holder_name
        if s.pending == "role_check":
            # The answer to "are you the policyholder yourself?". The caller's own words decide,
            # whichever engine read them. Only a yes that points at nobody else opens the gate:
            # "yes, I'm her son" and "yes, for my mother" are not that yes.
            # A model's yes is never enough on its own, and a model's "representative" always counts.
            answer = role_answer(t.text, m.identity.get("full_name") or "")
            if answer == "other" or u.affirmation == "no" or own_name:
                m.caller_role, m.role_guessed, m.role_unclear = "representative", False, False
            elif answer == "self":
                m.caller_role, m.role_guessed, m.role_unclear = "policyholder", False, False
            elif u.caller_role == "representative":
                m.caller_role, m.role_guessed, m.role_unclear = "representative", False, False
            else:
                # Something stood next to the yes that cannot be placed, or nothing clear was said.
                # It is asked once more, and a second such answer goes to a person.
                m.caller_role, m.role_guessed, m.role_unclear = "unknown", False, True
        is_rep = self._is_rep(s)

        if result.verified:
            if is_rep and m.role_guessed and not own_name and not s.rep_denials:
                # Taken for a representative from wording alone: ask before treating them as one.
                return self._role_check(s, t)
            if not is_rep:
                if m.role_unclear and m.caller_role == "unknown":
                    return self._role_check(s, t)
                self._grant(s, t, "three identity details matched one policyholder record")
                return None
            return self._rep_gate(s, u, t)

        matched = result.matched
        needed = REQUIRED_MATCHES - len(matched)
        remaining = [f for f in PII_FIELDS if f not in matched and f not in m.declined]
        alternatives = [f for f in remaining if f not in result.conflicts and f not in result.incomplete]
        # A third party is never shown which of the details they gave were right. For them the
        # choice is always the whole list, and nothing is named as noted or confirmed.
        offer = [f for f in PII_FIELDS if f not in m.declined] if is_rep else remaining
        shown = [] if is_rep else matched
        new_pii = [f for f in t.new_identity if f in PII_FIELDS]
        who = "the policyholder's identity" if is_rep else "your identity"
        facts: dict[str, Any] = {
            "verification": {
                "rule": "At least three of five details must match: full name, date of birth, phone number, email address, last four digits of the SSN or national ID.",
                "details_confirmed_so_far": [SHORT_LABELS[f] for f in matched],
                "more_details_needed": needed,
                "caller_can_choose_from": [SHORT_LABELS[f] for f in offer],
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
        rep_cover = (
            ["Say that once the policyholder's details are confirmed, the policyholder will also be asked to approve before anything about the claim can be shared."]
            if rep_note
            else []
        )
        rep_rules: list[str] = []
        if is_rep:
            # A third party is not told which of the details they supplied were right.
            facts["verification"].pop("details_confirmed_so_far")
            rep_rules = [
                "The details asked for are the policyholder's. Do not say whether the details given so far match the records. Say only how many more are needed."
            ]
        s.pending = "identity"

        # 1. Something the caller gave does not match. Count the attempt and
        #    discard the wrong value so it can be corrected or replaced.
        if result.conflicts:
            for name in result.conflicts:
                m.identity.pop(name, None)
            others = [f for f in PII_FIELDS if f != "full_name" and f in m.identity]
            if result.record is None and result.conflicts == ["full_name"] and not others and s.name_misses < MAX_NAME_MISSES:
                # Names are not secrets, and a short reply can be misread as one. Nothing to
                # count the first few times: ask for the name as it appears on the policy.
                s.name_misses += 1
                facts["verification"]["name_not_found"] = True
                return self._directive(
                    s, t, "VERIFY_NAME_NOT_FOUND",
                    goal="The name given matches no policy. Say so without guessing why, and ask for the full name as it appears on the policy or for other details.",
                    fallback=(
                        f"I couldn't find a policy under that name. Could you give me the full name as it "
                        f"appears on the policy, or any three of these: {self._labels(s, list(PII_FIELDS), 'or')}?"
                    ),
                    facts=facts,
                    cover=[
                        "Say you could not find a policy under that name.",
                        "Ask for the full name as it appears on the policy, or for other accepted details.",
                    ],
                )
            s.failed_attempts += 1
            left = MAX_FAILED_ATTEMPTS - s.failed_attempts
            if left <= 0:
                return self._escalate(
                    s, t, "identity could not be verified after several attempts",
                    lead="I'm sorry, I wasn't able to verify your identity with the details provided.",
                    act="VERIFY_LOCKED",
                )
            bad = [SHORT_LABELS[f] for f in result.conflicts]
            if is_rep:
                text = "One or more of the details you gave don't match our records."
            elif result.record is None:
                text = "I wasn't able to find a match with the details you gave me."
            elif matched:
                text = (
                    f"I've confirmed {self._labels(s, matched)}, but I wasn't able to match the "
                    f"{join_items(bad)} you gave me with our records."
                )
            else:
                text = f"I wasn't able to match the {join_items(bad)} you gave me with our records."
            if is_rep:
                text += " Could you double-check them, or give me a different detail instead?"
            else:
                text += " Could you double-check that"
                if alternatives:
                    text += f", or give me a different detail instead, such as {self._labels(s, alternatives, 'or')}"
                text += "?"
            if left == 1:
                text += " For your security, I can try this one more time before I pass you to a representative."
            facts["verification"]["attempts_left"] = left
            if is_rep:
                goal = "One or more details the caller gave do not match the records. Say so without saying which, and ask them to re-check or use a different detail."
                first_point = "Say that one or more details did not match. Do not say which matched and which did not."
            else:
                facts["verification"]["details_that_did_not_match"] = bad
                goal = "Some of the details the caller gave do not match the records. Say which ones, without revealing what is on file, and ask them to re-check or use a different detail."
                first_point = "Name the details that did not match."
            return self._directive(
                s, t, "VERIFY_MISMATCH",
                goal=goal,
                fallback=text,
                facts=facts,
                cover=[
                    first_point,
                    "Invite the caller to double-check them or to give a different accepted detail.",
                ] + (["Say that one more attempt is possible before a representative has to take over."] if left == 1 else []),
                must_not=rep_rules,
            )

        # 2. The caller has ruled out too many details to ever reach three.
        if len(matched) + len(remaining) < REQUIRED_MATCHES:
            if u.process_question and not u.declined_fields and not u.refuses_verification:
                return self._directive(
                    s, t, "VERIFY_EXPLAIN",
                    goal="Answer the caller's question about the process from facts.process. Then say that three details are still needed and that what is left is not enough, and offer the two ways forward.",
                    fallback=(
                        f"{self._explain_line(s, t)} I do need three details in total, and with what's left I "
                        f"can't get there. You can share one you set aside after all, or I can connect you with a "
                        f"human representative."
                    ),
                    facts=facts,
                    cover=[
                        "Answer the process question directly.",
                        "Say three details are needed and the remaining ones are not enough.",
                        "Offer the choice: reconsider a declined detail, or a human representative.",
                    ],
                    must_not=rep_rules,
                )
            s.refusal_count += 1
            if s.refusal_count > MAX_PERSUASION:
                return self._escalate(
                    s, t, "caller declined the identity details needed for verification",
                    lead="I understand, and I won't keep pressing you.",
                )
            shortfall = REQUIRED_MATCHES - len(matched) - len(remaining)
            how_many = "one" if shortfall == 1 else f"at least {['', 'one', 'two', 'three'][min(shortfall, 3)]}"
            facts["verification"]["declined_details_needed_after_all"] = shortfall
            return self._directive(
                s, t, "VERIFY_IMPOSSIBLE",
                goal="With the details the caller has declined, three matching details can no longer be reached. Explain that kindly and offer the two ways forward.",
                fallback=(
                    f"I understand you'd rather not share some of those. I do need three details in total to "
                    f"verify {who}, and with what's left I can't get there. Would you be willing to share {how_many} "
                    f"of them after all, or would you like me to connect you with a human representative?"
                ),
                facts=facts,
                cover=[
                    "Respect the caller's reluctance.",
                    "Explain that three details are required and why.",
                    "Offer the choice: reconsider one of the declined details, or a human representative.",
                ],
                must_not=rep_rules,
            )

        # 3. The caller refuses or pushes back, and gave nothing new.
        if u.refuses_verification and not new_pii:
            s.refusal_count += 1
            if s.refusal_count > MAX_PERSUASION:
                return self._escalate(
                    s, t, "caller declined identity verification",
                    lead="I understand, and I won't keep pressing you.",
                )
            ask = self._ask_identity(s, matched, offer)
            if s.refusal_count == 1:
                credit = f"You're right that you've given me {self._labels(s, shown)}, thank you. " if shown else ""
                if not t.empathy:
                    credit = "I understand, and it's sensible to be careful with personal details. " + credit
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
                text = (
                    f"I do want to get you an answer today. I'm not able to skip this step, because it's what "
                    f"stops anyone else from getting into your claim. You can use whichever details you're most "
                    f"comfortable with: {self._labels(s, offer, 'or')}. Or, if you'd rather, I can connect "
                    f"you with a human representative. Which would you prefer?"
                )
                cover = [
                    "Acknowledge the caller's feelings first.",
                    "Say plainly, without lecturing, that this step cannot be skipped and what it protects. Use a different angle from the last explanation (facts.process has several).",
                    "Let them choose which accepted details to use.",
                    "Offer a human representative as the alternative and ask which they prefer.",
                ]
            if is_rep:
                text += " The policyholder can also call us directly."
                cover.append(
                    "Explain that a representative can be given claim details only when listed on the policy and "
                    "approved by the policyholder, and that the policyholder can also call directly."
                )
            facts["verification"]["times_caller_has_pushed_back"] = s.refusal_count
            return self._directive(
                s, t, f"VERIFY_PERSUADE_{s.refusal_count}",
                goal="The caller is pushing back on verification. De-escalate, explain why the step matters, and keep them moving towards completing it without giving anything away.",
                fallback=text,
                facts=facts,
                cover=cover,
                must_not=["Do not answer the caller's claim question yet, not even in general terms."] + rep_rules,
            )

        # 4. A question about the process itself.
        if u.process_question and not new_pii:
            ask = self._ask_identity(s, matched, offer)
            return self._directive(
                s, t, "VERIFY_EXPLAIN",
                goal="Answer the caller's question about the process from facts.process (and facts.representative if relevant), then continue with verification.",
                fallback=f"{self._explain_line(s, t)} {ask}",
                facts=facts,
                cover=["Answer the process question directly.", "Then ask for the identity details still needed."],
                must_not=rep_rules,
            )

        # 5. Progress, or a specific detail was declined.
        incomplete_now = [f for f in result.incomplete if f in t.new_identity]
        if new_pii or u.declined_fields or incomplete_now or u.partial:
            opener = "Thank you."
            if u.has_request:
                opener += " I can help with that as soon as we've finished verifying."
            if u.declined_fields:
                opener = f"That's fine, you don't need to give me {self._labels(s, u.declined_fields, 'or')}."
            if shown:
                opener += f" I've confirmed {self._labels(s, shown)}."
            if result.incomplete:
                opener += " For the name to count I'll need the full first and last name."
            for kind in u.partial:
                opener += PARTIAL_NOTES[kind]
            if u.partial:
                facts["verification"]["given_incompletely"] = [PARTIAL_LABELS[kind] for kind in u.partial]
            cover = ["Acknowledge what the caller just gave or declined.", "Ask for what is still needed, offering the remaining choices."]
            if u.partial:
                cover.insert(1, "Say what was incomplete about the detail they gave (facts.verification.given_incompletely) and what is needed instead.")
            cover += rep_cover
            if u.process_question:
                opener = f"{self._explain_line(s, t)} {opener}"
                cover.insert(0, "Answer the caller's question about the process first, from facts.process (and facts.representative if relevant).")
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
                fallback=f"{opener}{rep_note} {self._ask_identity(s, matched, offer)}",
                facts=facts,
                cover=cover,
                must_not=rep_rules,
            )

        # 6. The caller is leaving. Let them go without another request for details.
        if u.done and not u.has_request:
            return self._directive(
                s, t, "VERIFY_GOODBYE",
                goal="The caller is ending the conversation before verifying. Let them go politely. Nothing was looked up and nothing can be said about any claim.",
                fallback="No problem. Whenever you'd like to pick this up again, I'll just need to verify your identity first. Take care.",
                facts=facts,
                cover=["Say goodbye politely.", "Say they can come back any time and that verification is the first step."],
                must_not=["Do not ask for identity details again."],
            )

        # 7. Nothing usable yet: ask (again), acknowledging any request we saved.
        ask = self._ask_identity(s, matched, offer)
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
            cover=cover + rep_cover,
            must_not=rep_rules,
        )

    def _role_check(self, s: Session, t: TurnCtx) -> Directive:
        """The details match, but the caller mentioned somebody else and never said who is calling.

        A policyholder answers yes and goes on. Anyone else goes through the representative gate.
        """
        asked = s.pending == "role_check"
        if asked:
            s.role_checks += 1
            if s.role_checks >= 2:
                return self._escalate(
                    s, t, "it could not be established whether the caller is the policyholder",
                    lead="I'm sorry, I couldn't tell whether you are the policyholder, so a person needs to take it from here.",
                )
        s.pending = "role_check"
        text = (
            "I ask because claim details can only go to the policyholder or to someone the policyholder has approved, "
            "and I couldn't tell which you are. Are you the policyholder yourself? A yes or no is enough."
            if asked
            else "Thank you, those details check out. One question before I go on: are you the policyholder yourself?"
        )
        return self._directive(
            s, t, "ROLE_CHECK",
            goal="The details match, but it is unclear whether the caller is the policyholder or someone calling for them. Ask, and say nothing about any claim.",
            fallback=text,
            facts={"why_this_is_asked": "Claim details go to the policyholder, or to a representative the policyholder has approved."},
            cover=["Say the details check out.", "Ask one question: whether the caller is the policyholder themselves."],
            rules=REP_GATE_RULES,
        )

    def _rep_gate(self, s: Session, u: Understanding, t: TurnCtx) -> Directive | None:
        """The policyholder's details check out. A representative also needs listing and consent.

        Until consent is approved the reply works under ``REP_GATE_RULES``: it may say the
        details check out and that a consent request went out, and nothing else that is on file.
        """
        m = s.memory
        same_person = "self_is_holder" in u.guessed or (
            bool(m.representative_name) and m.representative_name.lower() == (m.identity.get("full_name") or "").lower()
        )
        if m.role_guessed and same_person and not s.rep_denials:
            # They were taken for a representative from a turn of phrase, and the name they give
            # as their own is the policyholder's. Ask, instead of refusing them as not listed.
            m.caller_role, m.role_guessed, m.role_unclear, m.representative_name = "unknown", False, True, ""
            return self._role_check(s, t)
        holder = self._holder(s)
        Holder = holder[0].upper() + holder[1:]
        facts: dict[str, Any] = {
            "representative": CONSENT_FACTS,
            "how_to_refer_to_the_policyholder": holder,
        }
        if t.declined_human or (u.done and s.consent.get("status") != "pending"):
            # They were offered a human and said no. Nothing more can be done here.
            s.pending = "none"
            return self._directive(
                s, t, "REP_GOODBYE",
                goal="The caller declined a human representative and nothing can be shared.",
                fallback=f"No problem. {Holder} is welcome to contact us directly at any time. Take care.",
                facts=facts,
                rules=REP_GATE_RULES,
                fixed=True,
            )
        if len(m.representative_name.split()) < 2:
            # No name yet, or only a first name ("I'm David, her son"). The list of authorized
            # representatives holds full names, so a first name is asked about, never refused.
            asked_before = s.pending == "rep_name"
            s.pending = "rep_name"
            facts["representative"] = {**CONSENT_FACTS, "why_the_callers_name_is_needed": REP_NAME_WHY}
            pushing = u.refuses_verification or "full_name" in u.declined_fields or (asked_before and u.affirmation == "no")
            if pushing:
                s.refusal_count += 1
                if s.refusal_count > MAX_PERSUASION:
                    return self._escalate(
                        s, t, "representative declined to give their own name",
                        lead="I understand, and I won't keep pressing you.",
                    )
                return self._directive(
                    s, t, "REP_NAME_PERSUADE",
                    goal="The caller does not want to give their own name. Explain why it is needed, without pressure, and offer the alternatives.",
                    fallback=(
                        f"I understand. {REP_NAME_WHY} Without it I can't share anything about the claim. "
                        f"{Holder} can also call us directly, or I can connect you with a human representative. "
                        f"Could I have your full name?"
                    ),
                    facts=facts,
                    cover=[
                        "Acknowledge the caller's reluctance.",
                        "Explain why their own name is needed (facts.representative.why_the_callers_name_is_needed).",
                        "Mention the alternatives: the policyholder can call directly, or a human representative.",
                        "Ask once more for their full name.",
                    ],
                    rules=REP_GATE_RULES,
                )
            if u.process_question:
                return self._directive(
                    s, t, "ASK_REP_NAME",
                    goal="The caller asked about this step. Answer from facts.representative, then ask for their own full name.",
                    fallback=f"{self._explain_line(s, t)} May I have your full name?",
                    facts=facts,
                    cover=["Answer the caller's question about this step.", "Ask for the caller's own full name."],
                    rules=REP_GATE_RULES,
                )
            if asked_before and re.fullmatch(r"\W*[A-Za-z][A-Za-z\-]{1,20}\W*", t.text) and u.affirmation == "none":
                text = "Thank you. I need your first and last name, please."
            elif asked_before:
                text = "I still need your own full name before I can go on. What is it, please?"
            else:
                text = "Thank you. Since you're calling on someone else's behalf, may I have your own full name as well?"
            return self._directive(
                s, t, "ASK_REP_NAME",
                goal="The caller is calling for someone else and has not given their own name. Ask for it.",
                fallback=text,
                facts=facts,
                cover=["Ask for the caller's own full name, first and last."],
                rules=REP_GATE_RULES,
            )

        rep_first = m.representative_name.split()[0]
        if s.rep_name_refused and s.rep_name_refused.lower() != m.representative_name.lower():
            # Told they are not on the policy, the caller now gives another name.
            return self._escalate(
                s, t, "caller gave a different name after not being found as an authorized representative",
                lead="Thank you. Since the name has changed, a person needs to confirm who is calling before anything can be shared.",
            )
        if self.data.find_representative(s.party_id, m.representative_name) is None:
            s.rep_name_refused = m.representative_name
            s.rep_denials += 1
            if s.rep_denials >= 2:
                return self._escalate(
                    s, t, "caller is not listed as an authorized representative",
                    lead="I understand this is frustrating.",
                )
            s.pending = "human_offer"
            s.offer_reason = "dead_end"
            facts["caller_is_an_authorized_representative"] = False
            return self._directive(
                s, t, "REP_NOT_AUTHORIZED",
                goal="The caller is not an authorized representative for this policyholder, so nothing can be shared. Explain kindly and offer the ways forward.",
                fallback=(
                    f"Thank you, {rep_first}. I don't see you listed as an authorized representative, so I'm "
                    f"not able to discuss this policy or any claim with you. "
                    f"{Holder} can call us directly, or I can connect you with a human representative who can "
                    f"explain how to be added. Would you like me to do that?"
                ),
                facts=facts,
                cover=[
                    "Say you cannot discuss anything because they are not listed as an authorized representative.",
                    "Offer the alternatives: the policyholder calls directly, or a human representative.",
                ],
                must_not=["Do not say whether the details they gave matched."],
                rules=REP_GATE_RULES,
            )

        status = s.consent.get("status")
        first_request = status is None
        # A question about the step ("why do you need her consent?") is answered before the status
        # is checked again, unless the caller says the approval has been given.
        asks_first = status == "pending" and u.process_question and not APPROVAL_SAID_RE.search(t.text)
        if first_request:
            status = self.tools.call("request_consent", s, t.trace)["status"]
        elif status == "pending" and not asks_first:
            status = self.tools.call("check_consent_status", s, t.trace)["status"]
        facts["consent_request"] = {"status": status, "sent_to": "the policyholder"}

        if status == "approved":
            self._grant(s, t, "policyholder details matched and the policyholder approved the representative")
            return None

        alternatives = f"If {holder} can't approve it right now, they can also call us directly, or you can ask me for a human representative."
        if status == "pending":
            s.pending = "consent"
            if first_request:
                text = (
                    f"Thank you, {rep_first}. Those details check out. Because the claim belongs to {holder}, I "
                    f"need their approval before I can share anything about it, which protects their private "
                    f"information. I've just sent {holder} a consent request. Could you let me know once they've "
                    f"approved it? {alternatives}"
                )
                if u.process_question:
                    text = f"{self._explain_line(s, t)} {text}"
                cover = [
                    "Say the details the caller gave check out.",
                    "Explain that the policyholder's approval is needed before anything about the claim can be shared, and why.",
                    "Say a consent request was sent to the policyholder and ask the caller to say when it is approved.",
                    "Mention the alternatives if the policyholder cannot approve now: they can call directly, or a human representative can help.",
                ]
            else:
                explain = f"{self._explain_line(s, t)} " if u.process_question else ""
                waiting = "I'm still waiting for the approval." if asks_first else "I've checked again and the approval hasn't come through yet."
                text = (
                    f"{explain}{waiting} The request went to "
                    f"{holder} directly. Let me know when it's done and I'll check right away. {alternatives}"
                )
                cover = [
                    "Say the approval has not arrived yet.",
                    "Ask the caller to say when it has been approved.",
                    "Mention the alternatives: the policyholder can call directly, or a human representative can help.",
                ]
                if u.process_question:
                    cover.insert(0, "Answer the caller's question about consent from facts.representative.")
            return self._directive(
                s, t, "CONSENT_REQUESTED" if first_request else "CONSENT_PENDING",
                goal="A representative is calling. Claim details stay closed until the policyholder approves the consent request.",
                fallback=text,
                facts=facts,
                cover=cover,
                rules=REP_GATE_RULES,
            )

        # expired or denied
        s.consent["timeouts"] = s.consent.get("timeouts", 0) + 1
        if s.consent["timeouts"] >= 2:
            return self._escalate(
                s, t, "the policyholder's consent was not received",
                lead="I'm not able to request consent again from here, so a person will take this forward.",
            )
        s.pending = "human_offer"
        s.offer_reason = "dead_end"
        happened = "was declined" if status == "denied" else "has expired without an answer"
        return self._directive(
            s, t, "CONSENT_NOT_GIVEN",
            goal="The policyholder's consent was not obtained, so nothing about the claim can be shared. Explain and offer the ways forward.",
            fallback=(
                f"I'm sorry, the consent request {happened}, so I'm not able to share claim details. "
                f"{Holder} can call us directly, or I can connect you with a human representative to work out "
                f"another way to confirm consent. Would you like me to transfer you?"
            ),
            facts=facts,
            cover=[
                f"Say the consent request {happened} and that claim details cannot be shared without it.",
                "Offer the alternatives: the policyholder calls directly, or a human representative.",
            ],
            rules=REP_GATE_RULES,
        )
