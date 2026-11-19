"""Phase 2, RESOLVE_INTENT: guided.

The model has interpreted the caller's words into a bounded intent and a set
of hints about which claim is meant (or picked one from the caller's own
list). This handler validates both against the verified caller's claims and
either settles on one claim, asks which one, or says nothing matches. Hints
remembered from before verification are used here, so the caller is not asked
to repeat them.
"""

from __future__ import annotations

from typing import Any

from sop_harness.core import Directive

from ..state import POST_PROCESS, PROCESS_CASE, Session
from ..understanding import INTENT_LABELS, Understanding
from .common import (
    PolicyBase,
    TurnCtx,
    article,
    claim_phrase,
    describe_hints,
    filter_claims,
)


class ResolveIntentPhase(PolicyBase):
    def _resolve(self, s: Session, u: Understanding, t: TurnCtx) -> Directive | None:
        m = s.memory
        claims = self.tools.call("list_claims", s, t.trace)
        intro = self._verified_intro(s, t)
        intro_cover = ["Confirm briefly that verification is complete."] if t.just_verified else []

        if not claims:
            if not t.entered and (u.done or (s.pending == "anything_else" and u.affirmation == "no")):
                self.graph.move(s, POST_PROCESS, "no claim to process and the caller needs nothing else", t.trace)
                return None
            s.pending = "anything_else"
            return self._directive(
                s, t, "NO_CLAIMS",
                goal="There are no claims on the caller's policy. Say so and offer a human representative for anything beyond that.",
                fallback=(
                    f"{intro}I don't see any claims on file under this policy. If you were expecting one, or "
                    f"want to start a new claim, I can connect you with a human representative. Is there "
                    f"anything else I can help you with?"
                ),
                facts={"claims_on_file": []},
                cover=intro_cover + ["Say there are no claims on file.", "Offer a human representative, and ask if there is anything else."],
            )

        if not t.entered and u.done and not self._has_request(s, u, t):
            self.graph.move(s, POST_PROCESS, "the caller needs nothing else", t.trace)
            return None

        by_id = {c["claim_id"]: c for c in claims}
        if t.exclude_case:
            claims = [c for c in claims if c["claim_id"] != t.exclude_case]
        chosen: dict[str, Any] | None = None
        explicit = u.selected_case_id or m.case_hints.get("claim_id", "")
        hints = {k: v for k, v in m.case_hints.items() if k not in {"claim_id", "exclude_current"}}

        if explicit and explicit in by_id:
            chosen = by_id[explicit]
            t.resolved_by = "explicit"
        elif explicit:
            m.case_hints = {}
            s.last_candidates = [c["claim_id"] for c in claims]
            s.pending = "choose_case"
            listing = "; ".join(claim_phrase(c) for c in claims)
            return self._directive(
                s, t, "CLAIM_ID_NOT_FOUND",
                goal="The claim number the caller gave is not on their policy. Say so without speculating about it, list the claims that are on file, and ask which one they mean.",
                fallback=f"{intro}I don't see claim {explicit} under this policy. What I do see is: {listing}. Which one would you like to look at?",
                facts={"claim_number_given_by_caller": explicit, "found_on_this_policy": False, "claims_on_file": claims},
                cover=intro_cover + ["Say that claim number is not on this policy.", "List the claims on file and ask which one they mean."],
                must_not=["Do not suggest the claim number might belong to someone else."],
            )
        else:
            candidates = filter_claims(claims, hints)
            if hints and not candidates and t.turn_hints and t.turn_hints != hints:
                only_now = {k: v for k, v in t.turn_hints.items() if k != "claim_id"}
                retry = filter_claims(claims, only_now)
                if retry:
                    candidates, hints = retry, only_now
                    m.case_hints = dict(only_now)
            described_as = describe_hints(hints)
            if not candidates and hints.get("status"):
                # Callers are often wrong about a status, or it is part of their
                # question ("why was my auto claim denied?"). Match on the rest.
                relaxed = {k: v for k, v in hints.items() if k != "status"}
                loose = filter_claims(claims, relaxed) if relaxed else []
                if loose:
                    t.status_mismatch = described_as
                    candidates, hints = loose, relaxed
            if len(candidates) == 1 and (hints or t.exclude_case or len(claims) == 1):
                chosen = candidates[0]
                t.resolved_by = "hints" if hints else "explicit" if t.exclude_case else "only_claim"
            elif not candidates:
                described = describe_hints(hints)
                m.case_hints = {}
                s.last_candidates = [c["claim_id"] for c in claims]
                s.pending = "choose_case"
                listing = "; ".join(claim_phrase(c) for c in claims)
                return self._directive(
                    s, t, "CLAIM_NOT_MATCHED",
                    goal="No claim on the policy matches what the caller described. Say so, list what is on file, and ask which one they mean.",
                    fallback=(
                        f"{intro}I couldn't find {article(described)} {described} on this policy. What I do "
                        f"see is: {listing}. Which one would you like to look at?"
                    ),
                    facts={"caller_described": described, "claims_on_file": claims},
                    cover=intro_cover + ["Say nothing on file matches the description.", "List the claims on file and ask which one they mean."],
                )
            else:
                s.last_candidates = [c["claim_id"] for c in candidates]
                s.pending = "choose_case"
                listing = "; ".join(claim_phrase(c) for c in candidates)
                if t.status_mismatch:
                    text = (
                        f"{intro}I don't see {article(t.status_mismatch)} {t.status_mismatch} on this policy, "
                        f"but these could be the one you mean: {listing}. Which one is it?"
                    )
                    goal = "Nothing matches the status the caller mentioned, but other details match several claims. Say so, list them and ask which one they mean."
                elif hints:
                    described = describe_hints(hints)
                    text = f"{intro}I found {len(candidates)} claims that could be the {described} you mean: {listing}. Which one is it?"
                    goal = "More than one claim matches what the caller described. List the matches and ask which one they mean."
                else:
                    text = f"{intro}I see {len(candidates)} claims on this policy: {listing}. Which one can I help you with?"
                    goal = "The caller has not said which claim they mean. List the claims on file and ask."
                facts: dict[str, Any] = {"matching_claims": candidates}
                if hints:
                    facts["caller_described"] = describe_hints(hints)
                if m.intent != "none" and not m.intent_consumed:
                    facts["caller_wants"] = INTENT_LABELS.get(m.intent, "")
                return self._directive(
                    s, t, "ASK_WHICH_CLAIM",
                    goal=goal,
                    fallback=text,
                    facts=facts,
                    cover=intro_cover + ["List the claims with type, filed date and claim number.", "Ask which one the caller means."],
                    must_not=["Do not go into the details of any claim until the caller has chosen one."],
                )

        # A single claim is identified.
        source = m.captured.get("case_hints")
        if t.resolved_by == "hints":
            t.hint_text = describe_hints(hints)
            if source and (source["turn"] != s.turn or source["phase"] != s.phase):
                t.trace.memory_used.append(
                    f"case hints '{t.hint_text}' (captured in {source['phase']}, turn {source['turn']})"
                )
        s.case_id = chosen["claim_id"]
        s.last_candidates = []
        m.hints_consumed = True
        t.case_just_resolved = True
        self.graph.move(s, PROCESS_CASE, f"claim {s.case_id} identified ({t.resolved_by})", t.trace)
        return None
