"""Normalisation of caller-supplied values and small text utilities.

Identity is compared in code, never by a model, so every field is reduced to
a canonical form here first.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

MONTHS: dict[str, int] = {}
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
for _index, _name in enumerate(MONTH_NAMES, start=1):
    MONTHS[_name.lower()] = _index
    MONTHS[_name.lower()[:3]] = _index
MONTHS["sept"] = 9

MONTH_ALT = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)

_ISO_RE = re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
_NUM_RE = re.compile(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})(?!\d)")
_MDY_RE = re.compile(rf"\b({MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_DMY_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTH_ALT})\.?,?\s+(\d{{4}})\b", re.I)
_MD_RE = re.compile(rf"\b({MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _expand_year(year: int) -> int:
    if year >= 100:
        return year
    return 1900 + year if year > 30 else 2000 + year


def find_dates(text: str) -> list[tuple[int | None, int, int, tuple[int, int]]]:
    """Find date mentions. Returns (year or None, month, day, span) tuples.

    Month and day without a year are returned with ``year=None``.
    """
    found: list[tuple[int | None, int, int, tuple[int, int]]] = []
    taken: list[tuple[int, int]] = []

    def free(span: tuple[int, int]) -> bool:
        return all(span[1] <= a or span[0] >= b for a, b in taken)

    def add(year: int | None, month: int, day: int, span: tuple[int, int]) -> None:
        if not free(span):
            return
        if year is None:
            if _safe_date(2000, month, day) is None:
                return
        elif _safe_date(year, month, day) is None:
            return
        taken.append(span)
        found.append((year, month, day, span))

    for m in _ISO_RE.finditer(text):
        add(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.span())
    for m in _MDY_RE.finditer(text):
        add(int(m.group(3)), MONTHS[m.group(1).lower().rstrip(".")], int(m.group(2)), m.span())
    for m in _DMY_RE.finditer(text):
        add(int(m.group(3)), MONTHS[m.group(2).lower().rstrip(".")], int(m.group(1)), m.span())
    for m in _NUM_RE.finditer(text):
        add(_expand_year(int(m.group(3))), int(m.group(1)), int(m.group(2)), m.span())
    for m in _MD_RE.finditer(text):
        add(None, MONTHS[m.group(1).lower().rstrip(".")], int(m.group(2)), m.span())
    return sorted(found, key=lambda item: item[3][0])


def parse_date(text: str) -> date | None:
    """First complete date in ``text``. Numeric dates are read as month/day/year."""
    for year, month, day, _span in find_dates(text or ""):
        if year is not None:
            return _safe_date(year, month, day)
    return None


def norm_dob(text: str, today: date | None = None) -> str:
    parsed = parse_date(text or "")
    if parsed is None:
        return ""
    limit = (today or date.today()).year
    return parsed.isoformat() if 1900 <= parsed.year <= limit else ""


_HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "mx"}


def name_tokens(text: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", text or "")
    plain = "".join(ch for ch in plain if not unicodedata.combining(ch)).lower()
    plain = re.sub(r"[^a-z\s]", " ", plain)
    return [tok for tok in plain.split() if tok not in _HONORIFICS]


def names_match(a: str, b: str) -> bool:
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb:
        return False
    return sorted(ta) == sorted(tb) or "".join(ta) == "".join(tb)


def norm_name(text: str) -> str:
    return " ".join(tok.capitalize() for tok in name_tokens(text))


def norm_phone(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    return digits[-10:] if len(digits) >= 10 else digits


def norm_email(text: str) -> str:
    value = (text or "").strip().strip(".,;:<>()[]\"'").lower()
    return value if re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", value) else ""


def norm_last4(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    return digits[-4:] if len(digits) >= 4 else ""


def norm_policy(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    return f"POL-{digits}" if digits else ""


def norm_claim_id(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    return f"CL-{digits}" if digits else ""


def long_date(iso: str) -> str:
    try:
        parsed = date.fromisoformat(iso)
    except (TypeError, ValueError):
        return iso
    return f"{MONTH_NAMES[parsed.month - 1]} {parsed.day}, {parsed.year}"


def mask_email(email: str) -> str:
    local, _, domain = (email or "").partition("@")
    if not local or not domain:
        return "the email address on file"
    return f"{local[0]}***@{domain}"


_SSN_RE = re.compile(r"(?<![\d-])\d{3}[- ]?\d{2}[- ]?(\d{4})(?![\d-])")


def redact_ssn(text: str) -> str:
    """Replace a full SSN with a marker that keeps only the last four digits.

    Applied to every caller message before it is stored, logged or sent to a
    model, so a full SSN typed by mistake never leaves the request.
    """
    return _SSN_RE.sub(lambda m: f"[SSN ending {m.group(1)}]", text or "")


def join_items(items: list[str], conjunction: str = "and") -> str:
    items = [item for item in items if item]
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return f"{', '.join(items[:-1])}, {conjunction} {items[-1]}"
