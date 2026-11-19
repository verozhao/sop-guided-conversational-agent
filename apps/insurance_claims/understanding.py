"""The structured reading of one caller message.

Both understanding engines (model and offline rules) produce this object.
The SOP policy consumes it and nothing else, so the policy behaves the same
whichever engine produced the reading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from .normalize import (
    norm_claim_id,
    norm_dob,
    norm_email,
    norm_last4,
    norm_name,
    norm_phone,
    norm_policy,
)
from .verification import PII_FIELDS

SCOPES = ["in_scope", "out_of_scope", "mixed"]
ROLES = ["policyholder", "representative", "unknown"]
INTENTS = [
    "none",
    "claim_status",
    "denial_explanation",
    "document_requirements",
    "appeal_or_next_steps",
    "payment_or_amounts",
    "general_claim_question",
]
CLAIM_TYPES = ["none", "healthcare", "dental", "auto", "other"]
CLAIM_STATUSES = ["none", "denied", "open", "closed"]
EMOTIONS = ["neutral", "frustrated", "angry", "anxious", "confused", "sad"]
TOPICS = [
    "missing_required_material_alternatives",
    "submission_timing",
    "processing_time_after_submission",
    "submission_method",
    "file_format_requirements",
    "receipt_confirmation",
]
AFFIRMATIONS = ["none", "yes", "no"]
EMAIL_CHOICES = ["none", "send", "skip"]

INTENT_LABELS = {
    "claim_status": "the status of the claim",
    "denial_explanation": "why the claim was denied",
    "document_requirements": "which documents are needed and how to provide them",
    "appeal_or_next_steps": "the appeal and next steps",
    "payment_or_amounts": "the payment amounts on the claim",
    "general_claim_question": "general questions about the claim",
}
TOPIC_LABELS = {
    "missing_required_material_alternatives": "what to do if a requested document is not available",
    "submission_timing": "when the documents need to be submitted",
    "processing_time_after_submission": "how long the review takes after submission",
    "submission_method": "how to submit the documents",
    "file_format_requirements": "file format and quality requirements",
    "receipt_confirmation": "how receipt of the documents is confirmed",
}


@dataclass
class Understanding:
    scope: str = "in_scope"
    caller_role: str = "unknown"
    identity: dict[str, str] = field(default_factory=dict)
    representative_name: str = ""
    declined_fields: list[str] = field(default_factory=list)
    refuses_verification: bool = False
    intent: str = "none"
    case_hints: dict[str, Any] = field(default_factory=dict)
    selected_case_id: str = ""
    question: str = ""
    followup_topics: list[str] = field(default_factory=list)
    process_question: bool = False
    emotion: str = "neutral"
    wants_human: bool = False
    affirmation: str = "none"
    email_summary_choice: str = "none"
    done: bool = False
    notes: list[str] = field(default_factory=list)
    source: str = "rules"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def has_request(self) -> bool:
        """The caller asked for something about a claim in this message."""
        return self.intent != "none" or bool(self.question) or bool(self.case_hints) or bool(self.selected_case_id)


def _string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _enum(values: list[str], description: str) -> dict[str, Any]:
    return {"type": "string", "enum": values, "description": description}


# JSON schema for structured output. Every property is required and no extra
# properties are allowed, so the model's reply always parses into the same
# shape. Empty string, "none", 0 and [] mean "not stated".
UNDERSTANDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scope": _enum(SCOPES, "Whether the message belongs to insurance claims support."),
        "caller_role": _enum(ROLES, "Who the caller says they are."),
        "identity": {
            "type": "object",
            "additionalProperties": False,
            "description": "Identity details stated or corrected in THIS message. Empty string when not stated.",
            "properties": {
                "full_name": _string("The POLICYHOLDER's name as stated."),
                "dob": _string("Policyholder date of birth as YYYY-MM-DD."),
                "phone": _string("Phone number, digits only."),
                "email": _string("Email address."),
                "id_last4": _string("Last four digits of the SSN or national ID, four digits only."),
                "policy_number": _string("Policy number as stated."),
                "representative_name": _string("The caller's own name when calling for someone else."),
            },
            "required": [
                "full_name", "dob", "phone", "email", "id_last4", "policy_number", "representative_name",
            ],
        },
        "declined_fields": {
            "type": "array",
            "items": {"type": "string", "enum": list(PII_FIELDS)},
            "description": "Identity fields the caller explicitly refuses or says they cannot give.",
        },
        "refuses_verification": {
            "type": "boolean",
            "description": "The caller refuses or pushes back on verifying, or demands claim information without it.",
        },
        "intent": _enum(INTENTS, "What the caller wants about a claim, if stated in this message."),
        "case_hints": {
            "type": "object",
            "additionalProperties": False,
            "description": "Clues in THIS message about which claim is meant.",
            "properties": {
                "claim_id": _string("Claim number if stated, for example CL-2048."),
                "claim_type": _enum(CLAIM_TYPES, "Kind of claim."),
                "status": _enum(CLAIM_STATUSES, "Claim status the caller mentions."),
                "month": {"type": "integer", "description": "Month the claim was filed, 1 to 12. 0 if not stated."},
                "year": {"type": "integer", "description": "Year the claim was filed. 0 if not stated."},
            },
            "required": ["claim_id", "claim_type", "status", "month", "year"],
        },
        "selected_case_id": _string(
            "Only when claims_on_file is listed in the context and the caller clearly means exactly one of them: that claim id."
        ),
        "question": _string("The caller's question or request about a claim, restated in one short sentence."),
        "followup_topics": {
            "type": "array",
            "items": {"type": "string", "enum": TOPICS},
            "description": "Document follow-up topics the message asks about.",
        },
        "process_question": {
            "type": "boolean",
            "description": "The caller asks about the process itself, not about a claim.",
        },
        "emotion": _enum(EMOTIONS, "Dominant emotion in the message."),
        "wants_human": {"type": "boolean", "description": "The caller asks for a human representative."},
        "affirmation": _enum(AFFIRMATIONS, "Whether the message answers the agent's last question with yes or no."),
        "email_summary_choice": _enum(EMAIL_CHOICES, "Whether the caller wants the email summary sent or skipped."),
        "done": {"type": "boolean", "description": "The caller signals they need nothing more."},
        "notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Other facts worth remembering for later in the call, each as one short sentence.",
        },
    },
    "required": [
        "scope", "caller_role", "identity", "declined_fields", "refuses_verification", "intent",
        "case_hints", "selected_case_id", "question", "followup_topics", "process_question",
        "emotion", "wants_human", "affirmation", "email_summary_choice", "done", "notes",
    ],
}


def _pick(value: Any, allowed: list[str], default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default


def normalize_identity(raw: dict[str, Any], today: date | None = None) -> dict[str, str]:
    """Reduce stated identity values to canonical form. Unusable values are dropped."""
    out: dict[str, str] = {}
    name = norm_name(str(raw.get("full_name") or ""))
    if name:
        out["full_name"] = name
    dob = norm_dob(str(raw.get("dob") or ""), today)
    if dob:
        out["dob"] = dob
    phone = norm_phone(str(raw.get("phone") or ""))
    if phone:
        out["phone"] = phone
    email = norm_email(str(raw.get("email") or ""))
    if email:
        out["email"] = email
    last4 = norm_last4(str(raw.get("id_last4") or ""))
    if last4:
        out["id_last4"] = last4
    policy = norm_policy(str(raw.get("policy_number") or ""))
    if policy:
        out["policy_number"] = policy
    return out


def normalize_hints(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    claim_id = norm_claim_id(str(raw.get("claim_id") or ""))
    if claim_id:
        out["claim_id"] = claim_id
    claim_type = _pick(raw.get("claim_type"), CLAIM_TYPES, "none")
    if claim_type != "none":
        out["claim_type"] = claim_type
    status = _pick(raw.get("status"), CLAIM_STATUSES, "none")
    if status != "none":
        out["status"] = status
    try:
        month = int(raw.get("month") or 0)
    except (TypeError, ValueError):
        month = 0
    if 1 <= month <= 12:
        out["month"] = month
    try:
        year = int(raw.get("year") or 0)
    except (TypeError, ValueError):
        year = 0
    if 1990 <= year <= 2100:
        out["year"] = year
    return out


def understanding_from_json(raw: dict[str, Any], today: date | None = None) -> Understanding:
    """Validate and normalise a model's JSON reading. Unknown values fall back to defaults."""
    identity_raw = raw.get("identity") if isinstance(raw.get("identity"), dict) else {}
    hints_raw = raw.get("case_hints") if isinstance(raw.get("case_hints"), dict) else {}
    return Understanding(
        scope=_pick(raw.get("scope"), SCOPES, "in_scope"),
        caller_role=_pick(raw.get("caller_role"), ROLES, "unknown"),
        identity=normalize_identity(identity_raw, today),
        representative_name=norm_name(str(identity_raw.get("representative_name") or "")),
        declined_fields=[f for f in raw.get("declined_fields") or [] if f in PII_FIELDS],
        refuses_verification=bool(raw.get("refuses_verification")),
        intent=_pick(raw.get("intent"), INTENTS, "none"),
        case_hints=normalize_hints(hints_raw),
        selected_case_id=norm_claim_id(str(raw.get("selected_case_id") or "")),
        question=str(raw.get("question") or "").strip()[:300],
        followup_topics=[t for t in raw.get("followup_topics") or [] if t in TOPICS],
        process_question=bool(raw.get("process_question")),
        emotion=_pick(raw.get("emotion"), EMOTIONS, "neutral"),
        wants_human=bool(raw.get("wants_human")),
        affirmation=_pick(raw.get("affirmation"), AFFIRMATIONS, "none"),
        email_summary_choice=_pick(raw.get("email_summary_choice"), EMAIL_CHOICES, "none"),
        done=bool(raw.get("done")),
        notes=[str(n).strip()[:200] for n in (raw.get("notes") or []) if str(n).strip()][:5],
        source="llm",
    )
