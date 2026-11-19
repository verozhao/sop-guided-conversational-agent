"""Rendering of the document guideline fixture.

Shared by the claim tools (what the reply may say) and the email summary
(what the customer gets in writing), so both use exactly the same text.
"""

from __future__ import annotations

from typing import Any

from .normalize import join_items


def text(node: Any) -> str:
    return node.get("en", "") if isinstance(node, dict) else ""


def document_guidance(guide: dict[str, Any], table: str, document: str) -> str:
    """Guidance for one requested document. Names are matched loosely, because
    the claim says "pathology report" where the guideline says "original pathology report"."""
    wanted = document.lower()
    for key, node in guide.get(table, {}).items():
        if key != "default" and (wanted in key.lower() or key.lower() in wanted):
            return text(node)
    return ""


def if_unavailable(guide: dict[str, Any], document: str) -> str:
    return document_guidance(guide, "document_alternative_guidance", document) or text(
        guide.get("document_alternative_guidance", {}).get("default")
    )


def render_followup(guide: dict[str, Any], claim: dict[str, Any], topics: list[str]) -> list[dict[str, str]]:
    """The follow-up guidance texts for the given topics, filled in for this claim, in the order asked."""
    needed = claim.get("documents_needed", [])
    if not needed:
        return []
    settings = guide.get("claim_followup_settings", {})
    values = {
        "case_id": claim["case_id"],
        "documents": "the " + join_items(needed),
        "average_processing_time_after_submission": text(settings.get("average_processing_time_after_submission")),
    }
    rows = {row["topic"]: row for row in guide.get("claim_followup_guidance", [])}

    def fill(template: str) -> str:
        out = template.format(**values)
        if len(needed) == 1:
            # The guideline sentences are written for several documents. With one, the verb agrees.
            for plural, singular in ((" stay ", " stays "), (" are ", " is "), (" were ", " was ")):
                out = out.replace(values["documents"] + plural, values["documents"] + singular)
        return out

    return [{"topic": topic, "text": fill(text(rows[topic]))} for topic in dict.fromkeys(topics) if topic in rows]


def human_review_note(guide: dict[str, Any]) -> str:
    return text(guide.get("claim_followup_settings", {}).get("human_review_after_document_alternatives_exhausted"))
