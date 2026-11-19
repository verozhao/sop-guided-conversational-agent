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
_SHORT_PHONE_RE = re.compile(r"(?<![\d\-.])\d{3}[\-.\s]\d{4}(?![\d\-])")
_URL_RE = re.compile(r"(?<![@\w])(?:https?://|www\.)[^\s,;)]+|(?<![@\w.*])[a-z0-9\-]+\.(?:com|org|net|gov|io)\b(?!@)", re.I)
_CLAIM_RE = re.compile(r"\bCL[\s\-]?\d{3,8}\b", re.I)
_POLICY_RE = re.compile(r"\bPOL[\s\-]?\d{3,8}\b", re.I)
_NUMBER = r"\d[\d,]*(?:\.\d{1,2})?"
_MONEY_RE = re.compile(
    rf"\$\s?({_NUMBER})|(?<![\d\-/.])(\d[\d,]*\.\d{{2}})(?![\d\-/])|\b({_NUMBER})\s*(?:dollars|bucks|USD)\b|\bUSD\s*({_NUMBER})",
    re.I,
)
_DURATION_RE = re.compile(
    r"(?<![\d.,$])(\d{1,4})\s+(?:more\s+|business\s+|calendar\s+|working\s+)?(?:days?|hours?|weeks?|months?|minutes?|years?)\b", re.I
)
_PERCENT_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s?(?:%|percent\b)", re.I)
# "until the 28th", "by the 3rd": a day of the month on its own
_DAY_ONLY_RE = re.compile(r"\b(?:until|by|on|before|after|due|is)\s+the\s+(\d{1,2})(?:st|nd|rd|th)\b(?!\s+of\b)", re.I)
_WEEKDAY_RE = re.compile(
    r"\b(?:next|this|by|on|until|before|every)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
    r"(?<![\w:])\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)(?![\w])",
    re.I,
)

# Words that make the phrase after them a negation, a doubt or a condition, not a claim.
# They have to stand right before the phrase, in the same clause, with no punctuation between:
# "nothing has been sent", "I can't say whether it will be approved", "once you're verified".
# "After checking those details, you're verified" is still a claim.
_WORDS = r"(?:[\w'$%\-]+(?:[.,]\d+)*\s+)"
_NEGATED_RE = re.compile(rf"(?:\bnot|n't|\bnever|\bcannot|\bno|\bnothing|\bunable to)\s+{_WORDS}{{0,5}}$", re.I)
_DOUBTED_RE = re.compile(rf"(?:\bwhether\s+{_WORDS}{{0,4}}|\bhow much\s+{_WORDS}{{0,2}}|\bif and when\s+{_WORDS}{{0,2}})$", re.I)
_CONDITIONAL_RE = re.compile(
    rf"\b(?:once|when|whenever|until|till|after|before|as soon as|if|unless)\s+{_WORDS}{{0,2}}$", re.I
)

_VERIFIED_RE = re.compile(
    r"\byou(?:'re| are|'ve been| have been)\s+(?:now |all |fully |successfully )*(?:verified|authenticated)\b|"
    r"\byour identity (?:is|has been|has now been|is now)\s+(?:verified|confirmed)\b|"
    r"\b(?:verified|confirmed) your identity\b|\bverified you\b|\b(?:was|were) able to verify\b|\bidentity checks out\b|"
    r"\bverification (?:is |has been |is now )?(?:complete|completed|done|successful)\b|"
    r"\bverification (?:passed|succeeded)\b|\b(?:passed|cleared) verification\b|\bconfirmed who you are\b|"
    r"\bpassed (?:the |our )?(?:security|identity|verification) check\b",
    re.I,
)
# "Two more details and you're verified" states a requirement, not a result.
_REQUIREMENT_RE = re.compile(r"\b(?:need|more|left|remaining|last|just)\b[^.!?]*\b(?:and|then)\s*$", re.I)
# Before verification the assistant cannot have looked at anything.
_SEES_RECORD_RE = re.compile(
    r"\bI(?:'m| am|'ve| have)? (?:do |just |now )?(?:see|found|pulled up|located|looking at)\s+(?:that\s+)?"
    r"(?:your|the|a|an|that|this|one|two|three|four|\d+)\s+(?:[\w\-]+\s+){0,3}?(?:claim|policy|file|record)s?\b|"
    r"\bI can see (?:that )?(?:your|the|a|an) (?:[\w\-]+\s+){0,3}?(?:claim|policy|file|record)s? (?:was|is|has|shows|on|here)\b|"
    r"\bour records (?:show|say|indicate)\b|\b(?:there(?:'s| is)|I (?:do )?have) an? (?:[\w\-]+\s+){0,3}?claim on file\b",
    re.I,
)
_STATUSES = r"denied|approved|paid|closed|open|pending|settled|rejected|under review|in progress|being reviewed|on file"
_CLAIM_STATUS_RE = re.compile(
    rf"\b(?:your|the|that|this) (?:[\w\-]+\s+)?claim (?:was|is|has been|got) (?:still |now |currently )?({_STATUSES})\b", re.I
)
# "Yes, your appeal deadline is March 18." before verification confirms what the caller only proposed.
_CONFIRMS_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:yes|yep|yeah|correct|that's right|that is right|right|indeed|exactly|i can confirm|confirmed)\b[^.!?]*"
    r"\b(?:claim|deadline|appeal|payment|policy)\b[^.!?]*\b(?:is|was|were|are|has been)\b",
    re.I,
)
# After verification a status has to be one the facts give.
_STATUS_OF_CLAIM_RE = re.compile(
    r"\b(?:claim|appeal)\b[^.?!,;]{0,30}?\b(?:has been|was|is|got)\s+(?:now\s+|still\s+|currently\s+)?"
    r"(approved|denied|paid|closed|open|pending|settled|rejected)\b",
    re.I,
)
_SENT_RE = re.compile(
    r"\b(?:i|we)(?:'ve| have)?\s+(?:just\s+|now\s+|already\s+|gone ahead and\s+|went ahead and\s+)*(?:sent|emailed)\b|"
    r"\bhas (?:just |now |already )?been (?:sent|emailed|delivered)\b|\bis on its way\b|"
    r"\b(?:e-?mail|summary|message)\s+(?:was|were)\s+(?:just\s+)?(?:sent|emailed|delivered)\b|"
    r"\b(?:e-?mail|summary|message)\s+(?:just\s+)?(?:sent|delivered)(?=\s*[.!,]|\s*$)|"
    r"\b(?:e-?mail|summary|message)\s+(?:went|has gone|is going)\s+out\b|\bin your inbox\b",
    re.I,
)
_TRANSFER_RE = re.compile(
    r"\b(?:i'm|i am|i'll be|we're)\s+(?:now\s+)?(?:transferring|connecting|handing)\b|"
    r"\bi(?:'ve| have)\s+(?:transferred|connected|escalated|handed)\b|"
    r"\byou(?:'re| are) (?:now )?being (?:transferred|connected)\b|"
    r"\b(?:let me|i'll|i will)\s+(?:now\s+)?(?:transfer you|connect you|put you through)\b[^.?!]*"
    r"\b(?:now|right away|right now|please hold|one moment|hold on)\b",
    re.I,
)
_ACTIONS = (
    r"arrange|set up|schedule|submit|file|upload|approve|reopen|re-open|expedite|waive|extend|process|"
    r"update|change|cancel|fax|mail|resend|fix|reverse|overturn"
)
# First-person offers of an action the assistant has no means to take ("I can help arrange a fax").
_ACTION_OFFER_RE = re.compile(
    rf"\b(?:I(?:'ll| will| can| could|'m going to| am going to)|we(?:'ll| will)|let me)\s+"
    rf"(?:also\s+|just\s+|go ahead and\s+|help\s+(?:you\s+)?(?:to\s+)?|get\s+(?:that|this|it)\s+)?(?:{_ACTIONS})\b",
    re.I,
)
# The same actions reported as done ("I've reopened the claim", "I went ahead and extended the deadline").
_ACTION_DONE_RE = re.compile(
    r"\bI(?:'ve| have)?\s+(?:just\s+|already\s+|now\s+|gone ahead and\s+|went ahead and\s+)*"
    r"(?:submitted|filed|uploaded|approved|reopened|re-opened|expedited|waived|extended|processed|updated|changed|"
    r"cancell?ed|faxed|mailed|resent|fixed|reversed|overturned|arranged|scheduled|set up)\b",
    re.I,
)
# Outcomes nobody on this line can promise.
_PROMISE_RE = re.compile(
    r"\bwill be (?:approved|paid|reimbursed|overturned|reversed)\b|\bguarantee[ds]?\b|"
    r"\b(?:you'll|you will)\s+(?:definitely\s+|certainly\s+)?(?:get|be)\s+(?:paid|reimbursed|approved|covered)\b|"
    r"\b(?:definitely|certainly|surely)\s+(?:be\s+|get\s+)?(?:approved|paid|reimbursed|covered|overturned)\b|"
    r"\bmake sure\b[^.?!]{0,40}\b(?:approved|paid|overturned|reversed)\b|\bgets? approved\b",
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


def _asserted(pattern: re.Pattern[str], text: str, conditions_soften: bool = True) -> re.Match[str] | None:
    """First match of ``pattern`` that is stated as fact.

    A match right after a negation or a doubt is not a statement. Right after a
    condition ("once", "until", "if") it is not one either, except for promises:
    "if you send it today it will be approved" is exactly the promise to stop.
    """
    for match in pattern.finditer(text):
        before = text[max(0, match.start() - 90): match.start()]
        if _NEGATED_RE.search(before) or _DOUBTED_RE.search(before):
            continue
        if conditions_soften and _CONDITIONAL_RE.search(before):
            continue
        return match
    return None


def claims_verified(text: str) -> bool:
    for match in _VERIFIED_RE.finditer(text):
        before = text[max(0, match.start() - 90): match.start()]
        if _NEGATED_RE.search(before) or _CONDITIONAL_RE.search(before) or _REQUIREMENT_RE.search(before):
            continue
        return True
    return False


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
    # typographic dashes, apostrophes and quotes read as their plain forms
    text = re.sub(r"[‐-―−]", "-", text)
    text = text.translate({0x2018: "'", 0x2019: "'", 0x02BC: "'", 0x201C: '"', 0x201D: '"'})

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
    for match in _SHORT_PHONE_RE.finditer(text):
        if _digits(match.group(0)) not in allowed_digits:
            result.fail("ungrounded phone number")
    for match in _URL_RE.finditer(text):
        if match.group(0).lower().rstrip(".") not in allowed_low:
            result.fail("ungrounded web address")

    # ---- amounts, dates and durations must be grounded -----------------------
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
    for match in _DAY_ONLY_RE.finditer(text):
        if int(match.group(1)) not in {d for _, d in month_days}:
            result.fail(f"ungrounded day of the month: {match.group(1)}")
    for match in _WEEKDAY_RE.finditer(text):
        if match.group(0).lower() not in allowed_low:
            result.fail(f"ungrounded day or time: '{match.group(0)}'")

    allowed_numbers = set(re.findall(r"(?<![\d.])\d{1,4}(?![\d.])", allowed))
    for match in _DURATION_RE.finditer(text):
        count = match.group(1)
        if count not in allowed_numbers and not (count == "7" and "week" in allowed_low):
            result.fail(f"ungrounded length of time: '{match.group(0)}'")
    for match in _PERCENT_RE.finditer(text):
        if match.group(1) not in allowed_numbers and match.group(0).lower() not in allowed_low:
            result.fail(f"ungrounded percentage: {match.group(0)}")

    # ---- values held in the system of record --------------------------------
    for record in data.policyholders:
        if re.search(rf"(?<!\d){re.escape(record.id_last4)}(?!\d)", text) and record.id_last4 not in allowed:
            result.fail("identity number from the records")
        if record.name.lower() in low and record.name.lower() not in allowed_low:
            result.fail("another person's name from the records")
    for claim in data.claims:
        # A claim's one-line summary is generic wording ("auto claim in progress"), so it
        # only counts as a leak before verification, when nothing may be said at all.
        keys = ("denial_reason",) if session.verified else ("denial_reason", "summary")
        for key in keys:
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
        if _CONFIRMS_RE.search(text):
            result.fail("confirms a claim detail before verification")
    else:
        status = _asserted(_STATUS_OF_CLAIM_RE, text)
        if status and status.group(1).lower() not in allowed_low:
            result.fail(f"states a claim status the facts do not give: {status.group(1)}")

    sent = _asserted(_SENT_RE, text)
    if sent:
        said_now = any(directive.facts.get(key) for key in ("email_sent", "consent_request", "handoff"))
        earlier_email = bool(session.outbox) and bool(re.search(r"summary|e-?mail", low))
        consent_request = session.consent.get("status") is not None and bool(re.search(r"consent|request|approv", low))
        if not (said_now or earlier_email or consent_request):
            result.fail("claims something was sent")
    if _asserted(_TRANSFER_RE, text) and not directive.facts.get("handoff"):
        result.fail("claims a transfer that did not happen")
    offer = _asserted(_ACTION_OFFER_RE, text, conditions_soften=False)
    if offer:
        result.fail(f"offers an action the assistant cannot take: '{offer.group(0)}'")
    done = _asserted(_ACTION_DONE_RE, text)
    if done:
        result.fail(f"claims an action the assistant cannot take: '{done.group(0)}'")
    promise = _asserted(_PROMISE_RE, text, conditions_soften=False)
    if promise and promise.group(0).lower() not in allowed_low:
        result.fail(f"promises an outcome: '{promise.group(0)}'")
    return result
