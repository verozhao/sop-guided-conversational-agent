"""Output guard: the last deterministic check before a reply reaches the caller.

A model-written reply is released only if every sensitive value in it is
grounded in the turn's facts (or was said by the caller) and it does not
claim an action or status the workflow has not reached. A failed reply is
replaced by the directive's deterministic fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from sop_harness.core import Directive

from .data import ClaimsData
from .normalize import find_dates, norm_claim_id, norm_policy
from .state import Session

MAX_REPLY_CHARS = 1800

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<![\w\-])(\+?\d[\d\s().\-]{8,16}\d)(?![\w\-])")
_CLAIM_RE = re.compile(r"\bCL[\s\-]?\d{3,8}\b", re.I)
_POLICY_RE = re.compile(r"\bPOL[\s\-]?\d{3,8}\b", re.I)
_MONEY_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)|(?<![\d\-/.])(\d[\d,]*\.\d{2})(?![\d\-/])")
_VERIFIED_RE = re.compile(
    r"\byou(?:'re| are|'ve been| have been)\s+(?:now |all |fully |successfully )*(?:verified|authenticated)\b|"
    r"\byour identity (?:is|has been|has now been|is now)\s+(?:verified|confirmed)\b|"
    r"\b(?:verified|confirmed) your identity\b|"
    r"\bverification (?:is|has been|is now) (?:complete|completed|done|successful)\b",
    re.I,
)
# Words before such a phrase that make it a condition or a negation, not a claim:
# "once you're verified", "until your identity is verified", "I haven't verified your identity".
_NOT_A_CLAIM_RE = re.compile(
    r"(?:\b(?:once|when|whenever|until|till|after|before|as soon as|if|unless|whether)\b[^.!?;]*|"
    r"(?:\bnot|n't|\bnever|\bcannot)\s+(?:\w+\s+){0,2})$",
    re.I,
)


def claims_verified(text: str) -> bool:
    for match in _VERIFIED_RE.finditer(text):
        if not _NOT_A_CLAIM_RE.search(text[max(0, match.start() - 80): match.start()]):
            return True
    return False
_SENT_RE = re.compile(
    r"\b(?:i|we)(?:'ve| have)?\s+(?:just\s+|now\s+|already\s+)?(?:sent|emailed)\b|"
    r"\bhas been (?:sent|emailed)\b|\bis on its way\b",
    re.I,
)
_TRANSFER_RE = re.compile(
    r"\b(?:i'm|i am|i'll be|we're)\s+(?:now\s+)?(?:transferring|connecting|handing)\b|"
    r"\bi(?:'ve| have)\s+(?:transferred|connected|escalated|handed)\b",
    re.I,
)


@dataclass
class GuardResult:
    ok: bool = True
    violations: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.ok = False
        self.violations.append(reason)

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "violations": self.violations}


def _amounts(text: str) -> set[float]:
    values: set[float] = set()
    for match in _MONEY_RE.finditer(text):
        raw = (match.group(1) or match.group(2)).replace(",", "")
        try:
            values.add(round(float(raw), 2))
        except ValueError:
            continue
    return values


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def check_reply(reply: str, *, session: Session, directive: Directive, data: ClaimsData) -> GuardResult:
    result = GuardResult()
    text = (reply or "").strip()
    if not text:
        result.fail("empty reply")
        return result
    if len(text) > MAX_REPLY_CHARS:
        result.fail("reply too long")

    facts_blob = json.dumps(directive.facts, ensure_ascii=False) + "\n" + directive.fallback
    caller_blob = "\n".join(session.caller_texts())
    allowed = facts_blob + "\n" + caller_blob
    allowed_low = allowed.lower()
    low = text.lower()

    # ---- identifiers must be grounded --------------------------------------
    allowed_claims = {norm_claim_id(m.group(0)) for m in _CLAIM_RE.finditer(allowed)}
    for match in _CLAIM_RE.finditer(text):
        if norm_claim_id(match.group(0)) not in allowed_claims:
            result.fail(f"ungrounded claim id {match.group(0)}")

    allowed_policies = {norm_policy(m.group(0)) for m in _POLICY_RE.finditer(allowed)}
    for match in _POLICY_RE.finditer(text):
        if norm_policy(match.group(0)) not in allowed_policies:
            result.fail("ungrounded policy number")

    for match in _EMAIL_RE.finditer(text):
        if match.group(0).lower() not in allowed_low:
            result.fail("ungrounded email address")

    allowed_digits = _digits(allowed)
    for match in _PHONE_RE.finditer(text):
        digits = _digits(match.group(1))
        if len(digits) >= 10 and digits[-10:] not in allowed_digits:
            result.fail("ungrounded phone number")

    # ---- amounts and dates must be grounded ---------------------------------
    # An amount is grounded if it appears in the facts, or is one arithmetic step
    # from two amounts that do ("the gap between the two is $25.00").
    allowed_amounts = _amounts(allowed)
    derived = {round(abs(a - b), 2) for a in allowed_amounts for b in allowed_amounts}
    derived |= {round(a + b, 2) for a in allowed_amounts for b in allowed_amounts}
    for amount in _amounts(text):
        if amount not in allowed_amounts and amount not in derived:
            result.fail(f"ungrounded amount {amount:.2f}")

    allowed_dates = find_dates(allowed)
    full_dates = {(y, m, d) for y, m, d, _ in allowed_dates if y is not None}
    month_days = {(m, d) for _, m, d, _ in allowed_dates}
    for year, month, day, _span in find_dates(text):
        if year is None:
            if (month, day) not in month_days:
                result.fail(f"ungrounded date {month:02d}-{day:02d}")
        elif (year, month, day) not in full_dates:
            result.fail(f"ungrounded date {year:04d}-{month:02d}-{day:02d}")

    # ---- values held in the system of record --------------------------------
    for record in data.policyholders:
        if re.search(rf"(?<!\d){re.escape(record.id_last4)}(?!\d)", text) and record.id_last4 not in allowed:
            result.fail("identity number from the records")
        if record.name.lower() in low and record.name.lower() not in allowed_low:
            result.fail("another person's name from the records")
    for claim in data.claims:
        for key in ("denial_reason", "summary"):
            value = str(claim.get(key) or "").lower()
            if value and value in low and value not in allowed_low:
                result.fail(f"claim {key} that was not provided this turn")
        for document in claim.get("documents_needed", []):
            if document.lower() in low and document.lower() not in allowed_low:
                result.fail(f"requested document '{document}' that was not provided this turn")

    # ---- status and action claims --------------------------------------------
    if not session.verified and claims_verified(text):
        result.fail("claims the caller is verified")
    if _SENT_RE.search(text) and not (directive.facts.get("email_sent") or directive.facts.get("consent_request")):
        result.fail("claims something was sent")
    if _TRANSFER_RE.search(text) and not directive.facts.get("handoff"):
        result.fail("claims a transfer that did not happen")
    return result
