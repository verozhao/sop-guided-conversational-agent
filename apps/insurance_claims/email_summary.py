"""Email summary of the conversation.

The summary is composed in code from the session log and the system of
record. No model writes it, so it cannot contain anything the tools did not
return for this caller. Its follow-up items follow what was actually
discussed: guidance the caller asked about is repeated in writing.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from .data import ClaimsData
from .guidance import human_review_note, if_unavailable, render_followup
from .normalize import join_items, long_date, mask_email
from .state import Session
from .understanding import INTENT_LABELS, TOPIC_LABELS

ALTERNATIVES_TOPIC = "missing_required_material_alternatives"


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def _topic_line(item: dict[str, Any], claim: dict[str, Any] | None) -> str:
    """One 'what we discussed' line for an answered question."""
    intent = item.get("intent", "")
    denied = bool(claim) and claim["status"] == "denied"
    if intent in {"general_claim_question", "claim_status"} and denied:
        label = "where the claim stands and why it was denied"
    else:
        label = INTENT_LABELS.get(intent, "your claim")
    line = _sentence(label)[:-1]
    return f"{line} (claim {item['case_id']})" if item.get("case_id") else line


def summary_sections(session: Session, data: ClaimsData, today: date) -> dict[str, list[str]]:
    """The three required parts of the summary, as plain lines."""
    guide = data.guidelines
    case_ids = list(dict.fromkeys(item["case_id"] for item in session.discussed if item.get("case_id")))
    claims = {cid: data.claim_for_party(session.party_id, cid) for cid in case_ids}
    claims = {cid: claim for cid, claim in claims.items() if claim}

    discussed: list[str] = []
    for item in session.discussed:
        line = _topic_line(item, claims.get(item.get("case_id", "")))
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
    for cid, claim in claims.items():
        head = f"Claim {cid} ({claim['case_type']}, filed {long_date(claim['created_at'])})"
        if claim["status"] == "denied" and claim.get("denial_reason"):
            status_lines.append(f"{head}: Denied. Reason: {claim['denial_reason']}.")
        else:
            status_lines.append(f"{head}: {claim['status'].capitalize()}. {_sentence(claim['summary'])}")

        needed = claim.get("documents_needed", [])
        if not needed:
            next_steps.append(
                f"Claim {cid} is {'open and in progress' if claim['status'] == 'open' else claim['status']}. "
                "Our records show no documents requested from you for it at this time."
            )
            continue

        next_steps.append(f"Claim {cid}: to have the denial reconsidered, send us the {join_items(needed)}.")
        deadline = claim.get("appeal_deadline")
        if deadline:
            days = (date.fromisoformat(deadline) - today).days
            if days >= 0:
                next_steps.append(
                    f"Appeal deadline for claim {cid}: {long_date(deadline)} "
                    f"({days} day{'s' if days != 1 else ''} from {long_date(today.isoformat())})."
                )
            else:
                next_steps.append(
                    f"The appeal deadline for claim {cid} ({long_date(deadline)}) has passed. "
                    "Please contact a claims representative to review your options."
                )

        # Guidance in writing: how to submit always, plus every topic the caller asked about.
        asked = [t for item in session.discussed if item.get("case_id") == cid for t in item.get("topics", [])]
        topics = list(dict.fromkeys(["submission_method", *[t for t in asked if t != ALTERNATIVES_TOPIC]]))
        for entry in render_followup(guide, claim, topics):
            next_steps.append(entry["text"])
        if ALTERNATIVES_TOPIC in asked:
            for document in needed:
                next_steps.append(f"If you cannot get the {document}: {if_unavailable(guide, document)}")
            note = human_review_note(guide)
            if note:
                next_steps.append(note)

    if session.handoff:
        next_steps.append(f"A human representative will follow up (reference {session.handoff['ticket_id']}).")
    if not next_steps:
        next_steps.append("No follow-up items came up on this call.")
    return {"what_we_discussed": discussed, "claim_status_and_outcome": status_lines, "follow_up_items_and_next_steps": next_steps}


def compose_summary(session: Session, data: ClaimsData, today: date) -> dict[str, Any]:
    party = data.party(session.party_id)
    memory = session.memory
    sections = summary_sections(session, data, today)
    case_ids = list(dict.fromkeys(item["case_id"] for item in session.discussed if item.get("case_id")))

    first_name = party.first_name if party else "there"
    lines = [
        f"Hello {first_name},",
        "",
        f"Thank you for contacting claims support on {long_date(today.isoformat())}. "
        "Here is a summary of our conversation.",
    ]
    if memory.caller_role == "representative" and memory.representative_name:
        lines += ["", f"The call was made on your behalf by {memory.representative_name}, with your consent."]
    lines += ["", "WHAT WE DISCUSSED"] + [f"- {line}" for line in sections["what_we_discussed"]]
    if sections["claim_status_and_outcome"]:
        lines += ["", "CLAIM STATUS AND OUTCOME"] + [f"- {line}" for line in sections["claim_status_and_outcome"]]
    lines += ["", "FOLLOW-UP ITEMS AND NEXT STEPS"] + [f"- {line}" for line in sections["follow_up_items_and_next_steps"]]
    lines += [
        "",
        "If anything here looks wrong, please contact claims support.",
        "",
        "Claims Support (automated summary)",
    ]

    subject = "Summary of your claims support conversation"
    if len(case_ids) == 1:
        subject += f" (claim {case_ids[0]})"
    return {
        "to_masked": mask_email(party.email) if party else "",
        "subject": subject,
        "body": "\n".join(lines),
        "sections": sections,
        "sent_on": today.isoformat(),
        "delivery": "simulated",
    }
