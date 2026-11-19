"""Email summary of the conversation.

The summary is composed in code from the session log and the system of
record. No model writes it, so it cannot contain anything the tools did not
return for this caller.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from .data import ClaimsData
from .normalize import join_items, long_date, mask_email
from .state import Session
from .understanding import INTENT_LABELS, TOPIC_LABELS


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def compose_summary(session: Session, data: ClaimsData, today: date) -> dict[str, Any]:
    party = data.party(session.party_id)
    memory = session.memory
    guide = data.guidelines
    case_ids = list(dict.fromkeys(item["case_id"] for item in session.discussed if item.get("case_id")))
    claims = [c for c in (data.claim_for_party(session.party_id, cid) for cid in case_ids) if c]

    discussed: list[str] = []
    for item in session.discussed:
        label = INTENT_LABELS.get(item.get("intent", ""), "your claim")
        line = f"{_sentence(label)[:-1]} (claim {item['case_id']})" if item.get("case_id") else _sentence(label)[:-1]
        if line not in discussed:
            discussed.append(line)
        for topic in item.get("topics", []):
            topic_line = f"{_sentence(TOPIC_LABELS.get(topic, topic))[:-1]} (claim {item['case_id']})"
            if topic_line not in discussed:
                discussed.append(topic_line)
    if not discussed:
        discussed.append("We did not review a specific claim on this call")

    status_lines: list[str] = []
    next_steps: list[str] = []
    for claim in claims:
        head = f"Claim {claim['case_id']} ({claim['case_type']}, filed {long_date(claim['created_at'])})"
        line = f"{head}: {claim['status'].capitalize()}. {_sentence(claim['summary'])}"
        if claim["status"] == "denied" and claim.get("denial_reason"):
            line += f" Reason for the denial: {claim['denial_reason']}."
        status_lines.append(line)

        needed = claim.get("documents_needed", [])
        if needed:
            next_steps.append(f"Claim {claim['case_id']}: send us the {join_items(needed)}.")
            deadline = claim.get("appeal_deadline")
            if deadline:
                days = (date.fromisoformat(deadline) - today).days
                if days >= 0:
                    next_steps.append(
                        f"Appeal deadline for claim {claim['case_id']}: {long_date(deadline)} "
                        f"({days} day{'s' if days != 1 else ''} from {long_date(today.isoformat())})."
                    )
                else:
                    next_steps.append(
                        f"The appeal deadline for claim {claim['case_id']} ({long_date(deadline)}) has passed. "
                        "Please contact a claims representative to review your options."
                    )
            how = guide.get("default_guidance", {}).get("en", "")
            if how:
                next_steps.append("How to submit: " + how)
            timing = guide.get("claim_followup_settings", {}).get("average_processing_time_after_submission", {}).get("en")
            if timing:
                next_steps.append(
                    f"After we receive the documents the claim goes back into review. Average processing time is {timing}."
                )
        elif claim["status"] == "open":
            next_steps.append(f"Claim {claim['case_id']} is in progress. No action is needed from you right now.")
        else:
            next_steps.append(f"Claim {claim['case_id']} is {claim['status']}. No further action is needed.")
    if session.handoff:
        next_steps.append(f"A human representative will follow up (reference {session.handoff['ticket_id']}).")
    if not next_steps:
        next_steps.append("No follow-up is needed from you at this time.")

    first_name = party.first_name if party else "there"
    lines = [
        f"Hello {first_name},",
        "",
        f"Thank you for contacting claims support on {long_date(today.isoformat())}. "
        "Here is a summary of our conversation.",
    ]
    if memory.caller_role == "representative" and memory.representative_name:
        lines += ["", f"The call was made on your behalf by {memory.representative_name}, with your consent."]
    lines += ["", "WHAT WE DISCUSSED"] + [f"- {line}" for line in discussed]
    if status_lines:
        lines += ["", "CLAIM STATUS AND OUTCOME"] + [f"- {line}" for line in status_lines]
    lines += ["", "FOLLOW-UP ITEMS AND NEXT STEPS"] + [f"- {line}" for line in next_steps]
    lines += [
        "",
        "If anything here looks wrong, please contact claims support.",
        "",
        "Claims Support (automated summary)",
    ]

    subject = "Summary of your claims support conversation"
    if len(claims) == 1:
        subject += f" (claim {claims[0]['case_id']})"
    return {
        "to_masked": mask_email(party.email) if party else "",
        "subject": subject,
        "body": "\n".join(lines),
        "sent_on": today.isoformat(),
        "delivery": "simulated",
    }
