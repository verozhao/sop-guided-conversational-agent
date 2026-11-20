"""Fixture-backed system of record for the demo.

In production these lookups would be service calls. The rest of the app only
touches this module through the phase-gated tools in ``tools.py`` and the
identity matcher in ``verification.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .normalize import name_tokens, names_match, norm_claim_id

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class Policyholder:
    party_id: str
    name: str
    policy_number: str
    dob: str
    id_type: str
    id_last4: str
    phone: str
    email: str
    name_aliases: tuple[str, ...] = field(default_factory=tuple)
    phone_aliases: tuple[str, ...] = field(default_factory=tuple)
    email_aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def first_name(self) -> str:
        return self.name.split()[0]


def _with_middle_name(listed: str, given: str) -> bool:
    """The same first and last name with a middle name or initial in between: "David J. Chen" is David Chen."""
    a, b = name_tokens(listed), name_tokens(given)
    return len(a) >= 2 and len(b) > len(a) and b[0] == a[0] and b[-1] == a[-1]


class ClaimsData:
    def __init__(self, fixtures_dir: Path | str = FIXTURES_DIR):
        root = Path(fixtures_dir)

        def load(name: str) -> Any:
            with open(root / name, encoding="utf-8") as handle:
                return json.load(handle)

        self.policyholders: list[Policyholder] = [
            Policyholder(
                party_id=row["party_id"],
                name=row["name"],
                policy_number=row["policy_number"],
                dob=row["dob"],
                id_type=row.get("id_type", "ssn_last4"),
                id_last4=row["id_last4"],
                phone=row["phone"],
                email=row["email"],
                name_aliases=tuple(row.get("name_aliases", [])),
                phone_aliases=tuple(row.get("phone_aliases", [])),
                email_aliases=tuple(row.get("email_aliases", [])),
            )
            for row in load("policyholders.json")
        ]
        self.claims: list[dict[str, Any]] = load("claims.json")
        self.representatives: list[dict[str, Any]] = load("representatives.json")
        self.guidelines: dict[str, Any] = load("required_document_guideline.json")
        self.claim_schema: dict[str, Any] = load("claim_schema.json")
        self.consent_scenarios: dict[str, Any] = load("consent_scenarios.json")

    # -- lookups ------------------------------------------------------------

    def party(self, party_id: str | None) -> Policyholder | None:
        return next((p for p in self.policyholders if p.party_id == party_id), None)

    def claims_for(self, party_id: str | None) -> list[dict[str, Any]]:
        rows = [c for c in self.claims if c["party_id"] == party_id]
        return sorted(rows, key=lambda c: c["created_at"], reverse=True)

    def claim_for_party(self, party_id: str | None, case_id: str) -> dict[str, Any] | None:
        """Return a claim only if it belongs to the given party."""
        wanted = norm_claim_id(case_id)
        for claim in self.claims_for(party_id):
            if norm_claim_id(claim["case_id"]) == wanted:
                return claim
        return None

    def find_representative(self, party_id: str | None, rep_name: str) -> dict[str, Any] | None:
        for rep in self.representatives:
            if rep["buyer_party_id"] == party_id and (names_match(rep["rep_name"], rep_name) or _with_middle_name(rep["rep_name"], rep_name)):
                return rep
        return None

    def followup_topics(self) -> list[str]:
        return [row["topic"] for row in self.guidelines.get("claim_followup_guidance", [])]
