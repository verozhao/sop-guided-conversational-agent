"""Phase 3, PROCESS_CASE: the freest phase.

The intent selects a workflow path, the path fixes which tools run, and the
tool results are the facts for the reply. Within those facts the model
answers follow-up questions in its own words. It cannot reach data outside
the path, and the output guard checks every identifier, amount and date in
its reply against the facts.
"""

from __future__ import annotations

import re
from typing import Any

from sop_harness.core import Directive

from ..normalize import join_items, long_date
from ..state import POST_PROCESS, PROCESS_CASE, RESOLVE_INTENT, Session
from ..understanding import INTENT_LABELS, Understanding
from .common import (
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
    "appeal_or_next_steps": ["submission_method", "processing_time_after_submission"],
}


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
        if u.intent != "none" or u.question or u.followup_topics:
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
            wants_more = u.intent != "none" or bool(u.question) or bool(u.followup_topics)
            if not wants_more:
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
                if u.affirmation == "yes":
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
            intent = u.intent
            if intent == "none":
                intent = "document_requirements" if u.followup_topics else "general_claim_question"
                t.uncovered = not u.followup_topics
            question = u.question
            topics = list(u.followup_topics)
        else:
            if u.intent != "none":
                intent = u.intent
            elif m.intent != "none" and not m.intent_consumed:
                intent = m.intent
                source = m.captured.get("intent")
                if source and (source["turn"] != s.turn or source["phase"] != PROCESS_CASE):
                    t.trace.memory_used.append(f"intent '{intent}' (captured in {source['phase']}, turn {source['turn']})")
            else:
                intent = "general_claim_question"
            question = u.question or m.backlog_question
            topics = list(dict.fromkeys(m.pending_topics + u.followup_topics))
            if m.pending_topics:
                t.trace.memory_used.append(f"follow-up topics {m.pending_topics} noted earlier")
        m.backlog_question = ""
        m.pending_topics = []

        claim = self.tools.call("get_claim_summary", s, t.trace, case_id=s.case_id)
        plan = [name for name in PATHS[intent] if name != "get_claim_summary"]
        if intent in {"general_claim_question", "claim_status"} and claim["status"] == "denied":
            plan.append("get_denial_details")
        results: dict[str, Any] = {"get_claim_summary": claim}
        for name in plan:
            results[name] = self.tools.call(name, s, t.trace, case_id=s.case_id)
        asked_topics = bool(topics)
        topics = topics or list(DEFAULT_TOPICS.get(intent, []))
        if topics:
            results["get_followup_guidance"] = self.tools.call(
                "get_followup_guidance", s, t.trace, case_id=s.case_id, topics=topics
            )

        s.pending = "anything_else"
        s.last_intent = intent
        m.intent_consumed = True
        s.email["new_since"] = True
        entry = {"case_id": s.case_id, "intent": intent, "topics": topics, "turn": s.turn}
        if not any(d["case_id"] == entry["case_id"] and d["intent"] == intent and d["topics"] == topics for d in s.discussed):
            s.discussed.append(entry)

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
        if t.uncovered:
            cover.append(
                f"The caller asked: {question} If facts.data answers it, answer from there. If it does not, or if it "
                "asks you to do something beyond explaining, say plainly that you can't do or look that up here "
                "and offer a human claims representative."
            )
        else:
            cover.append(
                f"Answer the caller's question from facts.data only: {question}" if question
                else f"Tell the caller about {INTENT_LABELS[intent]}, from facts.data only."
            )
        cover.append("End by asking whether there is anything else they would like to know about this claim.")
        return self._directive(
            s, t, "ANSWER",
            goal=f"Answer the caller about claim {s.case_id}. Workflow path: {intent} ({INTENT_LABELS[intent]}).",
            fallback=self._answer_text(s, t, intent, results, question, asked_topics),
            facts={
                "claim_in_focus": s.case_id,
                "workflow_path": intent,
                "caller_question": question,
                "data": results,
            },
            cover=cover,
        )

    def _answer_text(
        self, s: Session, t: TurnCtx, intent: str, results: dict[str, Any], question: str = "", asked_topics: bool = False
    ) -> str:
        claim = results["get_claim_summary"]
        denial = results.get("get_denial_details")
        documents = results.get("get_required_documents")
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

        status_line = f"The current status is {claim['status']} ({lower_first(claim['summary'])})."
        if t.uncovered and not t.case_just_resolved:
            parts.append(
                "I'm sorry, that's not something I can do or look up from here. A human claims representative "
                "can help with it, and I can connect you if you'd like."
            )
            parts.append(f"Is there anything else you'd like to know about claim {claim['claim_id']}?")
            s.human_offer_open = True
            return " ".join(parts)

        def denial_lines(with_documents: bool = True) -> list[str]:
            if not denial or not denial.get("denied"):
                return [status_line] if t.status_mismatch else [f"This claim wasn't denied. {status_line}"]
            lines = [f"It was denied because {denial['denial_reason']}."]
            if with_documents and denial.get("documents_needed"):
                lines.append(f"To have it reconsidered, we need the {join_items(denial['documents_needed'])}.")
            lines.extend(deadline_lines())
            return lines

        def deadline_lines() -> list[str]:
            if not denial or not denial.get("appeal_deadline"):
                return []
            when = long_date(denial["appeal_deadline"])
            if denial.get("deadline_passed"):
                return [f"The appeal deadline was {when}, which has already passed, so a claims representative would need to review your options."]
            days = denial.get("days_until_deadline", 0)
            span = "today" if days == 0 else "tomorrow" if days == 1 else f"{days} days from today"
            return [f"The appeal deadline is {when}, which is {span}."]

        if intent in {"claim_status", "general_claim_question"}:
            parts.extend(denial_lines() if denial and denial.get("denied") else [status_line])
        elif intent == "denial_explanation":
            parts.extend(denial_lines())
        elif intent == "document_requirements":
            needed = (documents or {}).get("documents_needed", [])
            asks_which = bool(re.search(r"\b(?:what|which)\b.*\b(?:documents?|paperwork|need|missing|required)", question.lower()))
            if not needed:
                parts.append("There are no documents requested for this claim right now.")
            elif asks_which or not asked_topics:
                parts.append(f"For this claim we need the {join_items(needed)}.")
        elif intent == "appeal_or_next_steps":
            needed = (documents or {}).get("documents_needed", [])
            if denial and denial.get("denied") and needed:
                parts.append(f"The next step is to send us the {join_items(needed)}.")
                parts.extend(deadline_lines())
            elif denial and denial.get("denied"):
                parts.extend(denial_lines())
            else:
                parts.append(f"{status_line} There's nothing you need to send us for this claim right now.")
        elif intent == "payment_or_amounts" and payment:
            amounts = payment["amounts"]
            parts.append(
                f"For claim {claim['claim_id']}, the allowed maximum is {usd(amounts.get('allowed_max_amount'))}, "
                f"the expected reimbursement is {usd(amounts.get('expected_reimbursement_amount'))}, and the "
                f"amount paid so far is {usd(amounts.get('net_pay'))}. The claim's status is {payment['status']}."
            )

        if followup:
            for item in followup.get("guidance", []):
                parts.append(item["text"])
            if not followup.get("guidance") and followup.get("note") and intent != "document_requirements":
                parts.append(followup["note"])

        parts.append("Is there anything else you'd like to know about this claim?")
        return " ".join(parts)
