"""Phase-gated tools: the only path from the system of record into a reply.

Every tool takes the session first. The party is always read from the
session (set by verification), never from a model or from caller text, so a
caller can only ever reach their own claims.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any, Callable

from sop_harness.core import ToolDenied, ToolRegistry

from .data import ClaimsData
from .email_summary import compose_summary
from .normalize import join_items, mask_email
from .state import (
    CLOSED,
    POST_PROCESS,
    PROCESS_CASE,
    RESOLVE_INTENT,
    VERIFY_ID,
    Session,
)
from .verification import FIELD_LABELS

AMOUNT_FIELDS = ["allowed_max_amount", "expected_reimbursement_amount", "net_pay", "net_fee"]


def build_tools(data: ClaimsData, clock: Callable[[], date]) -> ToolRegistry:
    registry = ToolRegistry()
    guide = data.guidelines

    def text(node: Any) -> str:
        return node.get("en", "") if isinstance(node, dict) else ""

    def own_claim(session: Session, case_id: str) -> dict[str, Any]:
        claim = data.claim_for_party(session.party_id, case_id)
        if claim is None:
            raise ToolDenied("claim is not on the verified caller's policy")
        return claim

    def doc_guidance(table: str, document: str) -> str:
        rows = guide.get(table, {})
        wanted = document.lower()
        for key, node in rows.items():
            if key != "default" and (wanted in key.lower() or key.lower() in wanted):
                return text(node)
        return ""

    # -- consent (representative callers) ------------------------------------

    def consent_sequence(session: Session) -> list[str]:
        scenarios = data.consent_scenarios
        scenario = scenarios.get(session.consent.get("scenario") or "default") or scenarios["default"]
        return list(scenario["status_sequence"])

    def step_consent(session: Session, index: int) -> str:
        sequence = consent_sequence(session)
        status = sequence[min(index, len(sequence) - 1)]
        if status == "pending" and index >= len(sequence) - 1:
            status = "expired"  # the request ran out without an answer
        session.consent.update(status=status, polls=index + 1, exhausted=status == "expired")
        return status

    def request_consent(session: Session) -> dict[str, Any]:
        if not session.identity_ok:
            raise ToolDenied("consent can only be requested after the policyholder's details match")
        return {"status": step_consent(session, 0), "sent_to": "the phone number on file for the policyholder"}

    def check_consent_status(session: Session) -> dict[str, Any]:
        if session.consent.get("status") is None:
            raise ToolDenied("no consent request is open")
        return {"status": step_consent(session, session.consent.get("polls", 1))}

    # -- claims ---------------------------------------------------------------

    def list_claims(session: Session) -> list[dict[str, Any]]:
        return [
            {"claim_id": c["case_id"], "type": c["case_type"], "filed": c["created_at"], "status": c["status"]}
            for c in data.claims_for(session.party_id)
        ]

    def get_claim_summary(session: Session, case_id: str) -> dict[str, Any]:
        claim = own_claim(session, case_id)
        return {
            "claim_id": claim["case_id"],
            "type": claim["case_type"],
            "filed": claim["created_at"],
            "status": claim["status"],
            "summary": claim["summary"],
        }

    def get_denial_details(session: Session, case_id: str) -> dict[str, Any]:
        claim = own_claim(session, case_id)
        if claim["status"] != "denied":
            return {"claim_id": claim["case_id"], "denied": False, "status": claim["status"]}
        today = clock()
        out: dict[str, Any] = {
            "claim_id": claim["case_id"],
            "denied": True,
            "denial_reason": claim.get("denial_reason", ""),
            "documents_needed": claim.get("documents_needed", []),
            "today": today.isoformat(),
        }
        deadline = claim.get("appeal_deadline")
        if deadline:
            days = (date.fromisoformat(deadline) - today).days
            out["appeal_deadline"] = deadline
            out["deadline_passed"] = days < 0
            if days >= 0:
                out["days_until_deadline"] = days
            else:
                out["note"] = "The appeal deadline has passed. A human claims representative has to review the options."
        return out

    def get_required_documents(session: Session, case_id: str) -> dict[str, Any]:
        claim = own_claim(session, case_id)
        needed = claim.get("documents_needed", [])
        if not needed:
            return {
                "claim_id": claim["case_id"],
                "documents_needed": [],
                "note": "No documents are currently requested for this claim.",
            }
        return {
            "claim_id": claim["case_id"],
            "documents_needed": needed,
            "how_to_submit": text(guide.get("default_guidance")),
            "claim_type_guidance": text(guide.get("case_type_guidance", {}).get(claim["case_type"])),
            "documents": [
                {
                    "name": document,
                    "requirements": doc_guidance("document_guidance", document),
                    "if_unavailable": doc_guidance("document_alternative_guidance", document)
                    or text(guide.get("document_alternative_guidance", {}).get("default")),
                }
                for document in needed
            ],
        }

    def get_followup_guidance(session: Session, case_id: str, topics: list[str]) -> dict[str, Any]:
        claim = own_claim(session, case_id)
        needed = claim.get("documents_needed", [])
        if not needed:
            return {
                "claim_id": claim["case_id"],
                "guidance": [],
                "note": "No documents are requested for this claim, so document follow-up guidance does not apply.",
            }
        settings = guide.get("claim_followup_settings", {})
        values = {
            "case_id": claim["case_id"],
            "documents": "the " + join_items(needed),
            "average_processing_time_after_submission": text(settings.get("average_processing_time_after_submission")),
        }
        rows = {row["topic"]: row for row in guide.get("claim_followup_guidance", [])}
        guidance = [
            {"topic": topic, "text": text(rows[topic]).format(**values)}
            for topic in dict.fromkeys(topics)
            if topic in rows
        ]
        out: dict[str, Any] = {"claim_id": claim["case_id"], "guidance": guidance}
        if "missing_required_material_alternatives" in topics:
            out["if_no_alternative_exists"] = text(settings.get("human_review_after_document_alternatives_exhausted"))
        out["if_the_question_is_not_covered_above"] = text(guide.get("claim_followup_fallback"))
        return out

    def get_payment_details(session: Session, case_id: str) -> dict[str, Any]:
        claim = own_claim(session, case_id)
        meanings = data.claim_schema.get("field_descriptions", {})
        return {
            "claim_id": claim["case_id"],
            "status": claim["status"],
            "currency": "USD",
            "amounts": {name: claim[name] for name in AMOUNT_FIELDS if name in claim},
            "field_meanings": {name: meanings.get(name, {}).get("description", "") for name in AMOUNT_FIELDS},
        }

    # -- post process and escalation -----------------------------------------

    def send_email_summary(session: Session) -> dict[str, Any]:
        if session.email.get("decision") != "send":
            raise ToolDenied("the customer has not chosen to receive the email summary")
        message = compose_summary(session, data, clock())
        session.outbox.append(message)
        session.email["new_since"] = False
        return {"sent": True, "to": message["to_masked"], "subject": message["subject"]}

    def transfer_to_human(session: Session, reason: str) -> dict[str, Any]:
        memory = session.memory
        party = data.party(session.party_id) if session.verified else None
        ticket = {
            "ticket_id": "HO-" + uuid.uuid4().hex[:6].upper(),
            "reason": reason,
            "phase_at_handoff": session.phase,
            "identity_verified": session.verified,
            "caller_role": memory.caller_role,
            "caller_name_as_stated": memory.representative_name or memory.identity.get("full_name", ""),
            "policy_number": party.policy_number if party else "",
            "identity_details_provided": [FIELD_LABELS[f] for f in memory.identity if f in FIELD_LABELS],
            "intent": memory.intent,
            "claim": session.case_id or "",
            "claim_hints": dict(memory.case_hints),
            "caller_question": memory.backlog_question,
            "last_emotion": memory.last_emotion,
            "topics_discussed": [dict(item) for item in session.discussed],
            "notes": list(memory.notes),
            "messages_after_handoff": [],
        }
        session.handoff = ticket
        return {"ticket_id": ticket["ticket_id"], "identity_verified": session.verified}

    every_phase = {VERIFY_ID, RESOLVE_INTENT, PROCESS_CASE, POST_PROCESS, CLOSED}
    registry.register(
        "request_consent", request_consent, phases={VERIFY_ID}, requires_verified=False,
        description="Ask the policyholder to approve a representative. Needs matching policyholder details first.",
    )
    registry.register(
        "check_consent_status", check_consent_status, phases={VERIFY_ID}, requires_verified=False,
        description="Poll the open consent request.",
    )
    registry.register(
        "list_claims", list_claims, phases={RESOLVE_INTENT, PROCESS_CASE}, requires_verified=True,
        description="List the verified caller's claims (id, type, filed date, status).",
    )
    for name, fn, description in [
        ("get_claim_summary", get_claim_summary, "Status and summary of one claim."),
        ("get_denial_details", get_denial_details, "Denial reason, documents needed and appeal deadline."),
        ("get_required_documents", get_required_documents, "Requested documents with submission guidance."),
        ("get_followup_guidance", get_followup_guidance, "Guidance for document follow-up topics."),
        ("get_payment_details", get_payment_details, "Amounts on the claim and what each field means."),
    ]:
        registry.register(name, fn, phases={PROCESS_CASE}, requires_verified=True, description=description)
    registry.register(
        "send_email_summary", send_email_summary, phases={POST_PROCESS, CLOSED}, requires_verified=True,
        description="Send the conversation summary to the email on file. Needs the customer's explicit choice.",
    )
    registry.register(
        "transfer_to_human", transfer_to_human, phases=every_phase, requires_verified=False,
        description="Hand the conversation to a human representative with a context ticket.",
    )
    return registry


def email_target(session: Session, data: ClaimsData) -> str:
    party = data.party(session.party_id)
    return mask_email(party.email) if party else "the email address on file"
