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


def verify_identity(data: ClaimsData, claimed: dict[str, str], prefer: str | None = None) -> VerificationResult:
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

    # The five details decide which record this is. The policy number only breaks a
    # tie. A wrong policy number is reported as wrong; it never turns the caller's
    # own correct details into mismatches.
    policy = norm_policy(claimed.get("policy_number") or "")
    by_policy = [p for p in data.policyholders if norm_policy(p.policy_number) == policy] if policy else []

    if not usable:
        if policy and not by_policy:
            result.conflicts.append("policy_number")
        return result

    scored = [(sum(field_matches(p, n, v) for n, v in usable.items()), p) for p in data.policyholders]
    best_score = max(score for score, _ in scored)
    best = [p for score, p in scored if score == best_score]

    if best_score == 0:
        result.conflicts.extend(usable)  # nothing the caller gave matches anything
        return result
    if len(best) > 1:
        pinned = [p for p in best if p in by_policy]
        earlier = [p for p in best if prefer and p.party_id == prefer]
        if len(pinned) == 1:
            record = pinned[0]
        elif len(earlier) == 1:
            # The caller's details were matching this record already. A value that belongs
            # to somebody else is a mismatch, and what matched before still stands.
            record = earlier[0]
        else:
            # Ambiguous between records: more information is needed. A value that
            # matches none of the tied records is still wrong and counts as such,
            # so a tie cannot be used to guess without limit.
            result.conflicts.extend(
                name for name, value in usable.items() if not any(field_matches(p, name, value) for p in best)
            )
            if policy and not any(p in by_policy for p in best):
                result.conflicts.append("policy_number")
            return result
    else:
        record = best[0]
    if policy and record not in by_policy:
        result.conflicts.append("policy_number")

    result.record = record
    for name, value in usable.items():
        (result.matched if field_matches(record, name, value) else result.conflicts).append(name)
    result.verified = len(result.matched) >= REQUIRED_MATCHES
    return result


__all__ = ["FIELD_LABELS", "PII_FIELDS", "REQUIRED_MATCHES", "VerificationResult", "field_matches", "verify_identity"]
