"""Domain-agnostic building blocks for SOP-guided conversational agents.

The harness separates three concerns:

* ``core``  - what the workflow allows (phase graph, gates, phase-gated tools)
* ``llm``   - how language is understood and phrased (model access)
* an app (see ``apps/``) - the concrete SOP: slots, policies, tools, wording
"""

from .core import (
    Directive,
    IllegalTransition,
    PhaseGraph,
    SOPViolation,
    ToolDenied,
    ToolRegistry,
    ToolSpec,
    TurnTrace,
)

__all__ = [
    "Directive",
    "IllegalTransition",
    "PhaseGraph",
    "SOPViolation",
    "ToolDenied",
    "ToolRegistry",
    "ToolSpec",
    "TurnTrace",
]
