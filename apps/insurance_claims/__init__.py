"""Insurance claims support agent built on the SOP harness.

Workflow: VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE -> POST_PROCESS
"""

from .agent import Agent, AgentConfig

__all__ = ["Agent", "AgentConfig"]
