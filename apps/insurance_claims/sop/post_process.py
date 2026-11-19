"""Phase 4, POST_PROCESS: strict again.

The email summary is offered once the claim questions are done. It is sent
only on an explicit choice by the customer, only to the address on file, and
its content is composed in code. A preference stated earlier in the call is
remembered: "send" is confirmed before sending, "skip" is honoured directly.
"""

from __future__ import annotations

import re
from typing import Any

from sop_harness.core import Directive

from ..email_summary import summary_sections
from ..state import CLOSED, POST_PROCESS, PROCESS_CASE, RESOLVE_INTENT, Session
from ..tools import email_target
from ..understanding import Understanding
from .common import (
    MAX_EMAIL_REASKS,
    PolicyBase,
    TurnCtx,
)

EMAIL_IN_TEXT = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


class PostProcessPhase(PolicyBase):
    def _route_request(self, s: Session, u: Understanding, t: TurnCtx, origin: str) -> None:
        """A new claim request arrived after wrap-up started. Go back to work."""
        if origin == POST_PROCESS and s.case_id and not self._is_switch(s, u, t):
            t.same_case = True
            self.graph.move(s, PROCESS_CASE, "the caller has another question on the same claim", t.trace)
            return
        if s.case_id and not self._is_switch(s, u, t):
            s.memory.case_hints = {"claim_id": s.case_id}
            t.same_case = True
        else:
            self._start_switch(s, t)
        s.case_id = None
        self.graph.move(s, RESOLVE_INTENT, "the caller has another request", t.trace)

    def _close(self, s: Session, t: TurnCtx, act: str, lead: str, facts: dict[str, Any] | None = None, cover: list[str] | None = None) -> Directive:
        name = self._first_name(s)
        self.graph.move(s, CLOSED, "post-process complete", t.trace)
        s.pending = "none"
        thanks = f"Thank you for contacting claims support, {name}. Take care." if name else "Thank you for contacting claims support. Take care."
        return self._directive(
            s, t, act,
            goal="Wrap up the conversation warmly.",
            fallback=f"{lead} {thanks}".strip(),
            facts=facts,
            cover=(cover or []) + ["Thank the caller and say goodbye."],
            must_not=["Do not ask another question."],
        )

    def _post(self, s: Session, u: Understanding, t: TurnCtx) -> Directive | None:
        m = s.memory
        address = email_target(s, self.data)
        whose = "the policyholder's email address on file" if self._is_rep(s) else "your email address on file"
        target = f"{whose} ({address})" if self._is_rep(s) else address
        offer_facts = {
            "email_summary": {
                "would_be_sent_to": f"{whose} ({address})",
                "contents": ["what was discussed", "the claim status and outcome", "follow-up items and next steps"],
                "the_customer_decides": "send it or skip it",
                "address_rule": "The summary can only go to the email address on file. Changing that address needs a human representative.",
            }
        }

        preview = summary_sections(s, self.data, self.clock())
        offer_facts["email_summary"]["would_contain"] = {
            "what_we_discussed": preview["what_we_discussed"],
            "follow_up_items_and_next_steps": preview["follow_up_items_and_next_steps"],
        }

        if t.entered:
            s.email["reasks"] = 0
            if s.email["decision"] is not None and not s.email["new_since"]:
                return self._close(s, t, "CLOSING", lead="")
            if m.email_pref == "skip" and s.email["decision"] is None:
                s.email["decision"] = "skip"
                t.trace.memory_used.append("email preference 'skip' stated earlier in the call")
                return self._close(
                    s, t, "EMAIL_SKIPPED",
                    lead="As you asked earlier, I won't send an email summary.",
                    facts={"email_summary": {"sent": False, "reason": "the customer said earlier they did not want one"}},
                    cover=["Say that, as they asked earlier, no email summary will be sent."],
                )
            s.pending = "email_offer"
            if m.email_pref == "send" and s.email["decision"] is None:
                t.trace.memory_used.append("email preference 'send' stated earlier in the call")
                offer_facts["email_summary"]["earlier_request"] = "The customer asked earlier in the call for an email summary."
                return self._directive(
                    s, t, "EMAIL_OFFER",
                    goal="The claim questions are done. The customer asked earlier for an email summary. Confirm before sending, and leave room to skip it.",
                    fallback=(
                        f"Earlier you asked for a summary by email. I can send it to {target} now, covering what "
                        f"we discussed, the claim status and the next steps. Shall I send it, or would you rather skip it?"
                    ),
                    facts=offer_facts,
                    cover=["Recall that they asked for an email summary earlier.", "Say where it would go and what it contains.", "Ask whether to send it now or skip it."],
                )
            again = "an updated summary" if s.email["decision"] is not None else "a summary"
            return self._directive(
                s, t, "EMAIL_OFFER",
                goal="The claim questions are done. Offer the email summary and let the customer choose to send or skip it.",
                fallback=(
                    f"Before we wrap up, would you like me to email you {again} of today's conversation? It "
                    f"covers what we discussed, the claim status and the next steps, and it would go to "
                    f"{target}. I can send it or we can skip it, whichever you prefer."
                ),
                facts=offer_facts,
                cover=["Offer the email summary.", "Say what it contains and where it would go.", "Make clear they can choose to send it or skip it."],
                must_not=["Do not send or claim to have sent anything yet."],
            )

        choice = u.email_summary_choice
        if choice == "none" and s.pending == "email_offer" and u.affirmation != "none":
            choice = "send" if u.affirmation == "yes" else "skip"
        party = self.data.party(s.party_id)
        typed = EMAIL_IN_TEXT.search(t.text)
        other_address = u.identity.get("email", "") or (typed.group(0).lower() if typed else "")
        on_file = {party.email.lower(), *(a.lower() for a in party.email_aliases)} if party else set()
        if other_address and other_address.lower() not in on_file and choice != "skip":
            s.pending = "email_offer"
            return self._directive(
                s, t, "EMAIL_ADDRESS_FIXED",
                goal="The caller wants the summary sent to a different address. That is not allowed. Explain, and ask whether to send it to the address on file or skip it.",
                fallback=(
                    f"I can only send the summary to the email address on file, which is {address}. Changing "
                    f"the address on file needs a human representative. Would you like me to send it to "
                    f"{address}, or shall we skip it?"
                ),
                facts=offer_facts,
                cover=[
                    "Say the summary can only go to the email address on file, and give the masked address.",
                    "Say that changing the address needs a human representative.",
                    "Ask whether to send it to the address on file or skip it.",
                ],
                must_not=["Do not send or claim to have sent anything yet.", "Do not repeat the address the caller gave."],
            )
        claim_request = u.intent != "none" or bool(u.followup_topics) or self._is_switch(s, u, t)
        if claim_request or (choice == "none" and bool(u.question) and not u.process_question):
            if choice != "none":
                m.email_pref = choice
                m.remember("email_pref", s.turn, s.phase)
            self._route_request(s, u, t, POST_PROCESS)
            return None
        if choice == "send":
            s.email["decision"] = "send"
            out = self.tools.call("send_email_summary", s, t.trace)
            return self._close(
                s, t, "EMAIL_SENT",
                lead=f"Done. I've sent the summary to {target}.",
                facts={"email_sent": out},
                cover=[
                    f"Confirm the summary was sent to {target}.",
                    "If you say what it contains, stay within facts.email_sent.contains.",
                ],
            )
        if choice == "skip":
            s.email["decision"] = "skip"
            return self._close(
                s, t, "EMAIL_SKIPPED",
                lead="No problem, I won't send a summary.",
                facts={"email_summary": {"sent": False, "reason": "the customer chose to skip it"}},
                cover=["Confirm that no email will be sent."],
            )

        if not u.process_question:
            s.email["reasks"] += 1  # answering a question about the summary is not an unclear answer
        if s.email["reasks"] > MAX_EMAIL_REASKS:
            s.email["decision"] = "skip"
            return self._close(
                s, t, "EMAIL_SKIPPED",
                lead="I'll leave the email summary out, since I didn't get a clear yes.",
                facts={"email_summary": {"sent": False, "reason": "no clear choice was given, and nothing is sent without one"}},
                cover=["Say the email summary will not be sent because there was no clear yes."],
            )
        s.pending = "email_offer"
        explain = f"{self._explain_line(s, t)} " if u.process_question else ""
        return self._directive(
            s, t, "EMAIL_OFFER_REPEAT",
            goal="The customer has not clearly chosen whether to send the email summary. Answer any question about it from facts, then ask again.",
            fallback=f"{explain}Would you like me to send the summary to {target}, or shall we skip it?",
            facts=offer_facts,
            cover=["Answer any question about the summary.", "Ask whether to send it or skip it."],
            must_not=["Do not send or claim to have sent anything yet."],
        )

    def _closed(self, s: Session, u: Understanding, t: TurnCtx) -> Directive | None:
        target = email_target(s, self.data)
        if u.email_summary_choice == "send" and not s.outbox:
            s.email["decision"] = "send"
            out = self.tools.call("send_email_summary", s, t.trace)
            return self._directive(
                s, t, "EMAIL_SENT",
                goal="The customer changed their mind and now wants the summary. Confirm it was sent.",
                fallback=f"Sure. I've sent the summary to {target}. Take care.",
                facts={"email_sent": out},
                cover=[f"Confirm the summary was sent to {target}."],
            )
        if s.verified and self._has_request(s, u, t):
            self._route_request(s, u, t, CLOSED)
            return None
        if u.emotion != "neutral" or u.process_question:
            s.pending = "human_offer"
            return self._directive(
                s, t, "CLOSED_CONCERN",
                goal="The conversation was wrapped up, and the caller has come back unhappy or with a concern. Acknowledge it and ask whether they would like a human representative.",
                fallback="If you'd like to take this further, I can connect you with a human representative. Would you like me to do that?",
                cover=["Acknowledge what the caller said.", "Ask one question: whether they would like a human representative."],
            )
        t.empathy = ""
        return self._directive(
            s, t, "CLOSED_ACK",
            goal="The conversation is finished.",
            fallback="You're welcome. Take care!" if u.done or u.affirmation != "none" or not t.text.strip().endswith("?")
            else "This conversation has been wrapped up. If you have another question about a claim, just ask.",
            fixed=True,
        )
