"""Output guard: the last deterministic check before a model-written reply reaches the caller.

The first line of defence is structural: before verification the model is
given no record data, and after it only the tool results of the current
workflow path. The guard is the second line. It catches a model that states
a value it was not given, or claims a status or action the workflow has not
reached. It works on patterns, so it covers the common ways of writing an
identifier, amount, date or claim, not every possible paraphrase.

A reply that fails is replaced by the directive's deterministic reply.

Grounded means: in this turn's facts, said by the caller, or (after
verification) in the facts of an earlier turn of the same call.
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
_NUMBER = r"\d[\d,]*(?:\.\d{1,2})?"
_MONEY_RE = re.compile(
    rf"\$\s?({_NUMBER})|(?<![\d\-/.])(\d[\d,]*\.\d{{2}})(?![\d\-/])|\b({_NUMBER})\s*(?:dollars|bucks|USD)\b|\bUSD\s*({_NUMBER})",
    re.I,
)
_DAYS_RE = re.compile(r"(?<![\d.,$])(\d{1,4})\s+(?:more\s+|business\s+|calendar\s+)?days?\b", re.I)

# Words just before a phrase that make it a condition, a negation or a doubt, not a claim:
# "once you're verified", "I haven't sent anything", "I can't say whether it will be approved".
_SOFTENED_RE = re.compile(
    r"(?:\b(?:once|when|whenever|until|till|after|before|as soon as|if|unless|whether|may|might|could)\b[^.!?;]*|"
    r"(?:\bnot|n't|\bnever|\bcannot|\bno|\bnothing)\s+(?:\w+\s+){0,3})$",
    re.I,
)
_VERIFIED_RE = re.compile(
    r"\byou(?:'re| are|'ve been| have been)\s+(?:now |all |fully |successfully )*(?:verified|authenticated)\b|"
    r"\byour identity (?:is|has been|has now been|is now)\s+(?:verified|confirmed)\b|"
    r"\b(?:verified|confirmed) your identity\b|\bverified you\b|\bidentity checks out\b|"
    r"\bverification (?:is|has been|is now) (?:complete|completed|done|successful)\b|"
    r"\bverification (?:passed|succeeded)\b|\b(?:passed|cleared) verification\b",
    re.I,
)
# Before verification the assistant cannot have looked at anything.
_SEES_RECORD_RE = re.compile(
    r"\bI(?:'m| am)? (?:can |do )?(?:see|found|pulled up|looking at)\b[^.?!]{0,60}\b(?:claim|policy|file|record)s?\b|"
    r"\bour records (?:show|say|indicate)\b",
    re.I,
)
_CLAIM_STATUS_RE = re.compile(
    r"\b(?:your|the|that|this) claim (?:was|is|has been|got) (denied|approved|paid|closed|open|pending|settled|rejected)\b", re.I
)
_SENT_RE = re.compile(
    r"\b(?:i|we)(?:'ve| have)?\s+(?:just\s+|now\s+|already\s+)?(?:sent|emailed)\b|"
    r"\bhas been (?:sent|emailed)\b|\bis on its way\b",
    re.I,
)
_TRANSFER_RE = re.compile(
    r"\b(?:i'm|i am|i'll be|we're)\s+(?:now\s+)?(?:transferring|connecting|handing)\b|"
    r"\bi(?:'ve| have)\s+(?:transferred|connected|escalated|handed)\b|"
    r"\byou(?:'re| are) (?:now )?being (?:transferred|connected)\b",
    re.I,
)
# First-person offers of an action the assistant has no means to take ("I can help arrange a fax").
_ACTION_OFFER_RE = re.compile(
    r"\bI(?:'ll| will| can| could)\s+(?:also\s+|just\s+|go ahead and\s+|help\s+(?:you\s+)?(?:to\s+)?|get\s+(?:that|this|it)\s+)?"
    r"(?:arrange|set up|schedule|submit|file|upload|approve|reopen|re-open|expedite|waive|extend|process|"
    r"update|change|cancel|fax|mail|resend|fix|reverse|overturn)\b",
    re.I,
)
# Outcomes nobody on this line can promise.
_PROMISE_RE = re.compile(
    r"\b(?:has|have) (?:now )?been approved\b|\bis (?:now )?approved\b|"
    r"\bwill be (?:approved|paid|reimbursed|overturned|reversed)\b|\bpaid in full\b|\bguarantee[ds]?\b",
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


def _asserted(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """First match of ``pattern`` that is stated as fact, not as a condition, negation or doubt."""
    for match in pattern.finditer(text):
        if not _SOFTENED_RE.search(text[max(0, match.start() - 80): match.start()]):
            return match
    return None


def claims_verified(text: str) -> bool:
    return _asserted(_VERIFIED_RE, text) is not None


def _amounts(text: str) -> set[float]:
    values: set[float] = set()
    for match in _MONEY_RE.finditer(text):
        raw = next(group for group in match.groups() if group).replace(",", "")
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
    text = re.sub(r"[‐-―−]", "-", text)  # typographic dashes read as hyphens

    facts_blob = json.dumps(directive.facts, ensure_ascii=False) + "\n" + directive.fallback
    caller_blob = "\n".join(session.caller_texts())
    # Grounded: this turn's facts, what the caller said, and facts already given to this
    # caller in earlier turns after verification (empty before verification).
    earlier = "\n".join(session.grounded) if session.verified else ""
    allowed = facts_blob + "\n" + caller_blob + "\n" + earlier
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

    # ---- amounts, dates and day counts must be grounded ----------------------
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

    allowed_numbers = set(re.findall(r"(?<![\d.])\d{1,4}(?![\d.])", allowed))
    for match in _DAYS_RE.finditer(text):
        count = match.group(1)
        if count not in allowed_numbers and not (count == "7" and "week" in allowed_low):
            result.fail(f"ungrounded number of days: {count}")

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
                result.fail(f"claim {key} that was not provided")
        for document in claim.get("documents_needed", []):
            if document.lower() in low and document.lower() not in allowed_low:
                result.fail(f"requested document '{document}' that was not provided")

    # ---- status and action claims --------------------------------------------
    if not session.verified:
        if claims_verified(text):
            result.fail("claims the caller is verified")
        if _asserted(_SEES_RECORD_RE, text):
            result.fail("claims to have looked at a record before verification")
        status = _asserted(_CLAIM_STATUS_RE, text)
        if status and status.group(1).lower() not in caller_blob.lower():
            result.fail(f"states a claim status before verification: {status.group(1)}")
    sent = _asserted(_SENT_RE, text)
    if sent and not any(directive.facts.get(key) for key in ("email_sent", "consent_request", "handoff")):
        result.fail("claims something was sent")
    if _asserted(_TRANSFER_RE, text) and not directive.facts.get("handoff"):
        result.fail("claims a transfer that did not happen")
    offer = _ACTION_OFFER_RE.search(text)
    if offer:
        result.fail(f"offers an action the assistant cannot take: '{offer.group(0)}'")
    promise = _asserted(_PROMISE_RE, text)
    if promise and promise.group(0).lower() not in allowed_low:
        result.fail(f"promises an outcome: '{promise.group(0)}'")
    return result
