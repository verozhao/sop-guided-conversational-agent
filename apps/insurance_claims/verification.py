"""Identity verification: at least three matching PII fields, checked in code.

A model extracts what the caller said. This module alone decides whether it
matches a policyholder record. Record values never leave this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .data import ClaimsData, Policyholder
from .normalize import name_tokens, names_match, norm_email, norm_phone, norm_policy

PII_FIELDS = ("full_name", "dob", "phone", "email", "id_last4")
REQUIRED_MATCHES = 3

FIELD_LABELS = {
    "full_name": "full name",
    "dob": "date of birth",
    "phone": "phone number",
    "email": "email address",
    "id_last4": "last four digits of the SSN or national ID",
    "policy_number": "policy number",
}


@dataclass
class VerificationResult:
    record: Policyholder | None = None
    matched: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    incomplete: list[str] = field(default_factory=list)
    verified: bool = False

    def summary(self) -> dict[str, object]:
        return {
            "matched": list(self.matched),
            "conflicts": list(self.conflicts),
            "incomplete": list(self.incomplete),
            "verified": self.verified,
            "matches_needed": max(0, REQUIRED_MATCHES - len(self.matched)),
        }


def field_matches(record: Policyholder, name: str, value: str) -> bool:
    if name == "full_name":
        return any(names_match(value, candidate) for candidate in (record.name, *record.name_aliases))
    if name == "dob":
        return value == record.dob
    if name == "phone":
        wanted = norm_phone(value)
        return len(wanted) == 10 and any(
            wanted == norm_phone(candidate) for candidate in (record.phone, *record.phone_aliases)
        )
    if name == "email":
        wanted = norm_email(value)
        return bool(wanted) and any(
            wanted == candidate.lower() for candidate in (record.email, *record.email_aliases)
        )
    if name == "id_last4":
        return value == record.id_last4
    return False


def verify_identity(data: ClaimsData, claimed: dict[str, str]) -> VerificationResult:
    """Compare what the caller has claimed so far with the records.

    Verified means: one record matches at least ``REQUIRED_MATCHES`` of the
    five PII fields. The policy number helps find the record but does not
    count as PII. Values that do not match are reported as ``conflicts``; the
    policy counts them as failed attempts and discards them.
    """
    result = VerificationResult()
    usable: dict[str, str] = {}
    for name in PII_FIELDS:
        value = claimed.get(name)
        if not value:
            continue
        if name == "full_name" and len(name_tokens(value)) < 2:
            result.incomplete.append(name)  # a first name alone is not a full name
            continue
        usable[name] = value

    candidates = list(data.policyholders)
    policy = claimed.get("policy_number") or ""
    pinned = False
    if policy:
        by_policy = [p for p in candidates if norm_policy(p.policy_number) == norm_policy(policy)]
        if by_policy:
            candidates, pinned = by_policy, True
        else:
            result.conflicts.append("policy_number")

    if not usable:
        return result

    scored = [(sum(field_matches(p, n, v) for n, v in usable.items()), p) for p in candidates]
    best_score = max(score for score, _ in scored)
    best = [p for score, p in scored if score == best_score]

    if pinned:
        record = candidates[0]
    elif best_score == 0:
        result.conflicts.extend(usable)  # nothing the caller gave matches anything
        return result
    elif len(best) > 1:
        # Ambiguous between records: more information is needed. A value that
        # matches none of the tied records is still wrong and counts as such,
        # so a tie cannot be used to guess without limit.
        result.conflicts.extend(
            name for name, value in usable.items() if not any(field_matches(p, name, value) for p in best)
        )
        return result
    else:
        record = best[0]

    result.record = record
    for name, value in usable.items():
        (result.matched if field_matches(record, name, value) else result.conflicts).append(name)
    result.verified = len(result.matched) >= REQUIRED_MATCHES
    return result
