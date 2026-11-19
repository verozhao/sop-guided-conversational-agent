"""The insurance claims SOP as a deterministic policy.

    VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE -> POST_PROCESS

``workflow``        phase graph, the per-turn loop, cross-phase memory
``verify_id``       phase 1 (strict)
``resolve_intent``  phase 2 (guided)
``process_case``    phase 3 (flexible, grounded)
``post_process``    phase 4 (strict)
``common``          limits, wording, hand-off and scope control
"""

from .common import (
    DISTRESS_OFFER_AT,
    EMPATHY,
    GREETING,
    MAX_EMAIL_REASKS,
    MAX_FAILED_ATTEMPTS,
    MAX_PERSUASION,
    OOS_ESCALATE_AT,
    OOS_OFFER_AT,
    TurnCtx,
    claim_phrase,
    describe_hints,
    filter_claims,
)
from .process_case import DEFAULT_TOPICS, PATHS
from .workflow import EDGES, InsuranceSOP

__all__ = [
    "DEFAULT_TOPICS",
    "DISTRESS_OFFER_AT",
    "EDGES",
    "EMPATHY",
    "GREETING",
    "InsuranceSOP",
    "MAX_EMAIL_REASKS",
    "MAX_FAILED_ATTEMPTS",
    "MAX_PERSUASION",
    "OOS_ESCALATE_AT",
    "OOS_OFFER_AT",
    "PATHS",
    "TurnCtx",
    "claim_phrase",
    "describe_hints",
    "filter_claims",
]
