"""Phase 3, PROCESS_CASE: the freest phase.

The intent selects a workflow path, the path fixes which tools run, and the
tool results are the facts for the reply. A message that asks about two
things ("how much was it for, and when is the deadline?") takes the union of
both paths. Within those facts the model answers follow-up questions in its
own words. It cannot reach data outside the paths, and the output guard
checks every identifier, amount and date in its reply against the facts.
"""

from __future__ import annotations

import re
from typing import Any

from sop_harness.core import Directive

from ..normalize import join_items, long_date
from ..state import POST_PROCESS, PROCESS_CASE, RESOLVE_INTENT, Session
from ..understanding import INTENT_LABELS, Understanding
from .common import (
    DISTRESS_OFFER_AT,
    PROCESS_FACTS,
    PolicyBase,
    TurnCtx,
    article,
    claim_phrase,
    filter_claims,
    lower_first,
    usd,
)

# Workflow paths. An intent selects a path; a path fixes which tools may run.
PATHS: dict[str, list[str]] = {
    "claim_status": ["get_claim_summary"],
    "general_claim_question": ["get_claim_summary"],
    "denial_explanation": ["get_claim_summary", "get_denial_details"],
    "document_requirements": ["get_required_documents"],
    "appeal_or_next_steps": ["get_denial_details", "get_required_documents"],
    "payment_or_amounts": ["get_claim_summary", "get_payment_details"],
}
DEFAULT_TOPICS: dict[str, list[str]] = {
    "document_requirements": ["submission_method"],
    "appeal_or_next_steps": ["submission_timing", "submission_method"],
}

# The check-in after an answer, varied so three answers in a row do not end the same way.
CLOSERS = [
    "Is there anything else you'd like to know about this claim?",
    "What else can I help you with on this claim?",
    "Would you like me to go over anything else on this claim?",
]


def paths_for(primary: str, extra: list[str], has_topics: bool) -> list[str]:
    """The bounded set of paths one message may take: the main intent plus what else it asks."""
    if primary == "none":
        primary = "document_requirements" if has_topics else "general_claim_question"
    intents = [primary] + [i for i in extra if i != primary and i in PATHS]
    if len(intents) > 1:
        intents = [i for i in intents if i != "general_claim_question"]  # adds nothing next to a specific one
    return intents[:3]


class ProcessCasePhase(PolicyBase):
    def _is_switch(self, s: Session, u: Understanding, t: TurnCtx) -> bool:
        """Does this message point at a different claim than the one in focus?"""
        if u.selected_case_id and u.selected_case_id != s.case_id:
            return True
        hints = t.turn_hints
        if not hints or not s.case_id:
            return False
        if hints.get("claim_id") and hints["claim_id"] != s.case_id:
            return True
        if hints.get("exclude_current"):
            return True
        strong = {k: v for k, v in hints.items() if k in {"claim_type", "month", "year"}}
        if not strong:
            return False  # a status word alone ("was it denied?") is about the current claim
        claim = self.data.claim_for_party(s.party_id, s.case_id)
        if claim is None:
            return True
        current = {"claim_id": claim["case_id"], "type": claim["case_type"], "filed": claim["created_at"], "status": claim["status"]}
        return not filter_claims([current], strong)

    def _has_request(self, s: Session, u: Understanding, t: TurnCtx) -> bool:
        """Did this message ask for something about a claim (the current one or another)?"""
        if u.intent != "none" or u.asks_about_claim or u.followup_topics:
            return True
        if s.case_id is None:
            return bool(t.turn_hints) or bool(u.selected_case_id)
        return self._is_switch(s, u, t)

    def _start_switch(self, s: Session, t: TurnCtx) -> None:
        if t.turn_hints.get("exclude_current") and s.case_id:
            t.exclude_case = s.case_id
        s.memory.case_hints = {k: v for k, v in t.turn_hints.items() if k != "exclude_current"}
        s.memory.hints_consumed = False
        s.memory.remember("case_hints", s.turn, s.phase)
        s.case_id = None

    def _process(self, s: Session, u: Understanding, t: TurnCtx) -> Directive | None:
        m = s.memory
        if not t.entered:
            if self._is_switch(s, u, t):
                self._start_switch(s, t)
                self.graph.move(s, RESOLVE_INTENT, "the caller asked about a different claim", t.trace)
                return None
            wants_more = u.intent != "none" or u.asks_about_claim or bool(u.followup_topics)
            lost = not wants_more and u.emotion == "confused" and not u.done and not u.more and not u.process_question
            if lost:
                # "I don't understand any of this": go over where the claim stands again.
                t.reexplain = True
                intents, question, topics = ["general_claim_question"], "", []
            elif not wants_more:
                if u.done or (s.pending == "anything_else" and u.affirmation == "no"):
                    self.graph.move(s, POST_PROCESS, "the caller needs nothing else on the claim", t.trace)
                    return None
                s.pending = "anything_else"
                if u.process_question:
                    return self._directive(
                        s, t, "PROCESS_EXPLAIN",
                        goal="Answer the caller's question about the process from facts.process, then ask if there is anything else about the claim.",
                        fallback=f"{self._explain_line(s, t)} Is there anything else you'd like to know about claim {s.case_id}?",
                        facts={"process": PROCESS_FACTS, "claim_in_focus": s.case_id},
                    )
                if u.emotion in {"angry", "frustrated"} and u.affirmation == "none" and not u.more:
                    t.offer_human = False
                    s.distress_offer_made = True
                    if s.distress_streak >= DISTRESS_OFFER_AT:
                        s.pending = "human_offer"
                        return self._directive(
                            s, t, "LISTEN_OFFER_HUMAN",
                            goal="The caller has been upset for several turns without asking anything. Do not go round in circles: ask whether they would like a human representative.",
                            fallback="I don't want to keep you going round in circles. Would you like me to connect you with a human representative?",
                            facts={"claim_in_focus": s.case_id},
                            cover=["Acknowledge that the caller is upset.", "Ask one question: whether they would like a human representative."],
                        )
                    return self._directive(
                        s, t, "LISTEN",
                        goal="The caller is upset and has not asked a specific question. Acknowledge it, ask what was missed or what they need, and say a human representative is available.",
                        fallback=(
                            "Tell me what I've missed or what you need, and I'll go over it. If you'd rather "
                            "speak with a person, just say so and I'll connect you."
                        ),
                        facts={"claim_in_focus": s.case_id},
                        cover=[
                            "Acknowledge that the caller is upset.",
                            "Ask what you missed or what they need.",
                            "Say they can ask for a human representative. Make it a statement, not a question.",
                        ],
                    )
                if u.affirmation == "yes" or u.more:
                    return self._directive(
                        s, t, "ASK_WHAT_ELSE",
                        goal="The caller has another question but has not asked it yet. Invite it.",
                        fallback="Of course. What else would you like to know?",
                        facts={"claim_in_focus": s.case_id},
                    )
                return self._directive(
                    s, t, "ASK_ANYTHING_ELSE",
                    goal="Nothing new was asked. Check whether the caller needs anything else on this claim.",
                    fallback=f"Is there anything else you'd like to know about claim {s.case_id}?",
                    facts={"claim_in_focus": s.case_id},
                )
            else:
                t.uncovered = u.intent == "none" and not u.followup_topics
                intents = paths_for(u.intent, u.extra_intents, bool(u.followup_topics))
                question = u.question
                topics = list(u.followup_topics)
        else:
            # Entered this turn. A specific question asked now wins. Otherwise the question
            # the caller asked earlier (in any phase) is the one to answer.
            specific_now = u.intent not in {"none", "general_claim_question"}
            remembered = m.intent != "none" and not m.intent_consumed
            if specific_now or not remembered:
                primary, extra = u.intent, u.extra_intents
                question = u.question or m.backlog_question
            else:
                primary, extra = m.intent, m.extra_intents
                question = m.backlog_question or u.question
                source = m.captured.get("intent")
                if source and (source["turn"] != s.turn or source["phase"] != PROCESS_CASE):
                    t.trace.memory_used.append(f"intent '{primary}' (captured in {source['phase']}, turn {source['turn']})")
            topics = list(dict.fromkeys(m.pending_topics + u.followup_topics))
            if m.pending_topics:
                t.trace.memory_used.append(f"follow-up topics {m.pending_topics} noted earlier")
            intents = paths_for(primary, extra, bool(topics))
        m.backlog_question = ""
        m.pending_topics = []
        intent = intents[0]

        # Run the tools of every path the message takes. Nothing outside those paths is fetched.
        claim = self.tools.call("get_claim_summary", s, t.trace, case_id=s.case_id)
        plan: list[str] = []
        for path in intents:
            wanted = [name for name in PATHS[path] if name != "get_claim_summary"]
            if path in {"general_claim_question", "claim_status"} and claim["status"] == "denied":
                wanted.append("get_denial_details")
            plan += [name for name in wanted if name not in plan]
        results: dict[str, Any] = {"get_claim_summary": claim}
        for name in plan:
            results[name] = self.tools.call(name, s, t.trace, case_id=s.case_id)
        asked_topics = bool(topics)
        if not topics:
            topics = list(dict.fromkeys(topic for path in intents for topic in DEFAULT_TOPICS.get(path, [])))
        if topics:
            results["get_followup_guidance"] = self.tools.call(
                "get_followup_guidance", s, t.trace, case_id=s.case_id, topics=topics
            )

        s.pending = "anything_else"
        s.last_intent = intent
        m.intent_consumed = True
        m.extra_intents = []
        s.email["new_since"] = True
        for position, path in enumerate(intents):
            entry = {"case_id": s.case_id, "intent": path, "topics": topics if position == 0 else [], "turn": s.turn}
            if not any(d["case_id"] == entry["case_id"] and d["intent"] == path and d["topics"] == entry["topics"] for d in s.discussed):
                s.discussed.append(entry)

        # No question was asked yet ("I have a question about my healthcare claim" ... "the first one"):
        # give the overview and ask what they want to know, instead of a generic check-in.
        overview = intents == ["general_claim_question"] and not t.uncovered
        nothing_asked_yet = overview and t.entered and u.intent == "none"
        closer = "What would you like to know about it?" if nothing_asked_yet else CLOSERS[s.answers_given % len(CLOSERS)]
        if t.reexplain:
            closer = "Which part would you like me to go over again?"
        first_answer = s.answers_given == 0
        s.answers_given += 1

        facts: dict[str, Any] = {
            "claim_in_focus": s.case_id,
            "workflow_path": intent,
            "caller_question": question,
            "data": results,
        }
        if len(intents) > 1:
            facts["also_covering"] = intents[1:]
        cover = []
        if t.just_verified:
            cover.append("Confirm briefly that verification is complete.")
        if t.case_just_resolved and not t.same_case:
            line = "Say which claim you pulled up (type, filed date, claim number) so the caller can correct you if it is the wrong one."
            if t.status_mismatch:
                line += (
                    f" They described it as their {t.status_mismatch}, but its status on file is {claim['status']}."
                    " Point that out gently."
                )
            elif t.hint_text:
                line += f" They described it earlier as their {t.hint_text}. Use that, and do not ask them to repeat it."
            cover.append(line)
        # The caller's words stay in facts.caller_question. They are never pasted into an instruction.
        if t.uncovered:
            cover.append(
                "The caller's request is in facts.caller_question, in their own words. If facts.data answers it, "
                "answer from there. If it does not, or it asks you to do something beyond explaining, say plainly "
                "that you can't do or look that up here, and ask whether they would like a human claims representative."
            )
        elif t.reexplain:
            cover.append(
                "The caller did not follow the last answer. Explain where this claim stands again in plainer words, "
                "one point at a time, from facts.data only, and ask which part to go over."
            )
        elif overview:
            cover.append("Give a short overview of where this claim stands, from facts.data only.")
        elif question:
            parts = join_items([INTENT_LABELS[path] for path in intents])
            cover.append(
                f"Answer the caller's question (facts.caller_question, in their own words) from facts.data only. It is about {parts}."
            )
        else:
            cover.append(f"Tell the caller about {join_items([INTENT_LABELS[path] for path in intents])}, from facts.data only.")
        denial = results.get("get_denial_details") or {}
        if denial.get("appeal_deadline"):
            cover.append("Keep the appeal deadline and any document timing separate, exactly as the facts state them.")
        if m.notes:
            # What the caller has told us is not record data, but the reply should respect it.
            facts["caller_said_earlier"] = list(m.notes[-5:])
            cover.append("Take into account what the caller has already said (facts.caller_said_earlier): do not suggest an option they have ruled out.")
        if t.uncovered or t.reexplain:
            pass  # the reply ends on its own question
        elif nothing_asked_yet:
            cover.append("End by asking what they would like to know about this claim.")
        elif first_answer:
            cover.append("End by inviting any other question about this claim.")
        else:
            cover.append(
                "Close naturally. If you invite more questions, use different words from your earlier replies, "
                "or offer one specific further detail that the facts contain."
            )
        return self._directive(
            s, t, "ANSWER",
            goal=f"Answer the caller about claim {s.case_id}. Workflow path: {', '.join(intents)}.",
            fallback=self._answer_text(s, t, intents, results, question, asked_topics, closer),
            facts=facts,
            cover=cover,
        )

    def _answer_text(
        self,
        s: Session,
        t: TurnCtx,
        intents: list[str],
        results: dict[str, Any],
        question: str = "",
        asked_topics: bool = False,
        closer: str = CLOSERS[0],
    ) -> str:
        claim = results["get_claim_summary"]
        denial = results.get("get_denial_details") or {}
        documents = results.get("get_required_documents") or {}
        payment = results.get("get_payment_details")
        followup = results.get("get_followup_guidance")
        parts: list[str] = []

        intro = self._verified_intro(s, t).strip()
        if intro:
            parts.append(intro)
        if t.case_just_resolved and not t.same_case:
            filed = long_date(claim["filed"])
            if t.status_mismatch:
                found = {"claim_id": claim["claim_id"], "type": claim["type"], "filed": claim["filed"], "status": claim["status"]}
                parts.append(
                    f"I don't see {article(t.status_mismatch)} {t.status_mismatch} on this policy, "
                    f"but I did find {claim_phrase(found)}."
                )
            elif t.resolved_by == "hints" and t.hint_text:
                parts.append(f"I found the {t.hint_text} you mentioned: {claim['claim_id']}, filed {filed}.")
            elif t.resolved_by == "only_claim":
                parts.append(f"I see one claim on this policy: {claim['claim_id']}, {article(claim['type'])} {claim['type']} claim filed {filed}.")
            else:
                parts.append(f"Got it, claim {claim['claim_id']}, the {claim['type']} claim filed {filed}.")

        if t.uncovered and not t.case_just_resolved:
            # One question only, and it is a yes or no: would they like a person.
            s.pending = "human_offer"
            parts.append(
                "I'm sorry, that's not something I can do or look up from here. A human claims representative "
                "can help with it. Would you like me to connect you?"
            )
            return " ".join(parts)

        said: set[str] = set()

        def say(key: str, sentence: str) -> None:
            if sentence and key not in said:
                said.add(key)
                parts.append(sentence)

        denied = bool(denial.get("denied"))
        needed = denial.get("documents_needed") or documents.get("documents_needed") or []
        status_line = f"The current status is {claim['status']} ({lower_first(claim['summary'])})."

        def say_deadline() -> None:
            if not denial.get("appeal_deadline"):
                return
            when = long_date(denial["appeal_deadline"])
            if denial.get("deadline_passed"):
                say("deadline", f"The appeal deadline was {when}, which has already passed, so a claims representative would need to review your options.")
                return
            days = denial.get("days_until_deadline", 0)
            span = "today" if days == 0 else "tomorrow" if days == 1 else f"{days} days from today"
            say("deadline", f"The appeal deadline is {when}, which is {span}.")

        def say_denial() -> None:
            if not denied:
                say("status", status_line if t.status_mismatch else f"This claim wasn't denied. {status_line}")
                return
            say("reason", f"It was denied because {denial['denial_reason']}.")
            if needed:
                say("documents", f"To have it reconsidered, we need the {join_items(needed)}.")
            say_deadline()

        for intent in intents:
            if intent in {"claim_status", "general_claim_question"}:
                if denied:
                    say_denial()
                else:
                    say("status", status_line)
            elif intent == "denial_explanation":
                say_denial()
            elif intent == "document_requirements":
                asks_which = bool(re.search(r"\b(?:what|which)\b.*\b(?:documents?|paperwork|need|missing|required)", question.lower()))
                if not needed:
                    say("documents", "There are no documents requested for this claim right now.")
                elif asks_which or not asked_topics:
                    say("documents", f"For this claim we need the {join_items(needed)}.")
            elif intent == "appeal_or_next_steps":
                if denied and needed:
                    say("documents", f"The next step is to send us the {join_items(needed)}.")
                    say_deadline()
                elif denied:
                    say_denial()
                else:
                    say("status", f"{status_line} Our records show no documents requested from you for this claim.")
            elif intent == "payment_or_amounts" and payment:
                amounts = payment["amounts"]
                allowed, paid = amounts.get("allowed_max_amount"), amounts.get("net_pay")
                say(
                    "payment",
                    f"For claim {claim['claim_id']}, the allowed maximum is {usd(allowed)}, which is the most the plan "
                    f"pays for the covered service. The expected reimbursement is "
                    f"{usd(amounts.get('expected_reimbursement_amount'))}, and the amount paid is {usd(paid)}.",
                )
                try:
                    gap = round(float(allowed) - float(paid), 2)
                    if gap > 0 and float(paid) > 0:
                        say("gap", f"That is {usd(gap)} less than the allowed maximum.")
                except (TypeError, ValueError):
                    pass
                say("payment_status", f"The claim's status is {payment['status']}.")

        if followup:
            for item in followup.get("guidance", []):
                parts.append(item["text"])
            if not followup.get("guidance") and followup.get("note") and "document_requirements" not in intents:
                parts.append(followup["note"])

        parts.append(closer)
        return " ".join(parts)
