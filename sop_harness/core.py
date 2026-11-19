"""Workflow primitives: phase graph, gated tools, turn directives and traces.

Everything in this module is deterministic code. The language model never
decides a phase change, never calls a tool directly and never sees data that
a tool did not return for the current turn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


class SOPViolation(Exception):
    """Raised when code tries to do something the SOP does not allow."""


class IllegalTransition(SOPViolation):
    pass


class ToolDenied(SOPViolation):
    pass


Gate = Callable[[Any], "tuple[bool, str]"]


class PhaseGraph:
    """Allowed phase order plus entry gates.

    ``edges`` lists which phases may follow which. ``gates`` are predicates
    that must hold before a phase can be entered, whatever the calling code
    believes. A buggy or manipulated policy handler therefore cannot move a
    session past a safety gate.
    """

    def __init__(self, edges: dict[str, set[str]], gates: dict[str, Gate] | None = None):
        self.edges = {k: set(v) for k, v in edges.items()}
        self.gates = dict(gates or {})

    def can_move(self, src: str, dst: str) -> bool:
        return dst in self.edges.get(src, set())

    def move(self, session: Any, dst: str, reason: str, trace: "TurnTrace | None" = None) -> None:
        src = session.phase
        if not self.can_move(src, dst):
            raise IllegalTransition(f"{src} -> {dst} is not an allowed transition")
        gate = self.gates.get(dst)
        if gate is not None:
            ok, why = gate(session)
            if not ok:
                raise IllegalTransition(f"gate for {dst} is closed: {why}")
        session.phase = dst
        if trace is not None:
            trace.phase_path.append(dst)
            trace.transitions.append({"from": src, "to": dst, "reason": reason})


@dataclass(frozen=True)
class ToolSpec:
    name: str
    fn: Callable[..., Any]
    phases: frozenset[str]
    requires_verified: bool
    description: str


class ToolRegistry:
    """Tools are the only way claim data enters a turn.

    Each tool declares the phases in which it may run and whether the session
    must be verified. Calls outside those bounds raise ``ToolDenied`` and are
    logged in the trace as denied.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        phases: set[str] | frozenset[str],
        requires_verified: bool,
        description: str,
    ) -> None:
        self._tools[name] = ToolSpec(name, fn, frozenset(phases), requires_verified, description)

    def spec(self, name: str) -> ToolSpec:
        return self._tools[name]

    def allowed_in(self, phase: str) -> list[str]:
        return sorted(n for n, t in self._tools.items() if phase in t.phases)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "phases": sorted(t.phases),
                "requires_verified": t.requires_verified,
                "description": t.description,
            }
            for t in self._tools.values()
        ]

    def call(self, name: str, session: Any, trace: "TurnTrace | None" = None, **args: Any) -> Any:
        spec = self._tools.get(name)
        entry: dict[str, Any] = {"name": name, "args": args, "phase": getattr(session, "phase", "?")}
        if trace is not None:
            trace.tools.append(entry)
        if spec is None:
            entry.update(allowed=False, error="unknown tool")
            raise ToolDenied(f"unknown tool {name}")
        if session.phase not in spec.phases:
            entry.update(allowed=False, error=f"not allowed in phase {session.phase}")
            raise ToolDenied(f"tool {name} is not allowed in phase {session.phase}")
        if spec.requires_verified and not session.verified:
            entry.update(allowed=False, error="identity not verified")
            raise ToolDenied(f"tool {name} requires a verified caller")
        try:
            result = spec.fn(session, **args)
        except ToolDenied as exc:
            entry.update(allowed=False, error=str(exc))
            raise
        entry.update(allowed=True, result=result)
        return result


@dataclass
class Directive:
    """What the SOP decided this turn's reply must do.

    ``facts`` is the complete set of data the reply may use. ``fallback`` is a
    deterministic reply that satisfies the directive; it is used verbatim in
    offline mode and whenever the model's reply fails the output guard.
    """

    act: str
    goal: str
    fallback: str
    facts: dict[str, Any] = field(default_factory=dict)
    must_cover: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)
    tone: str = ""
    fixed: bool = False  # True: send ``fallback`` verbatim, never rephrase

    def summary(self) -> dict[str, Any]:
        return {
            "act": self.act,
            "goal": self.goal,
            "must_cover": self.must_cover,
            "must_not": self.must_not,
            "tone": self.tone,
            "facts": self.facts,
            "fixed": self.fixed,
        }


@dataclass
class TurnTrace:
    """Everything that happened in one turn, for the inspector and for tests."""

    turn: int
    engine: str = "offline"
    phase_before: str = ""
    phase_path: list[str] = field(default_factory=list)
    transitions: list[dict[str, str]] = field(default_factory=list)
    understanding: dict[str, Any] = field(default_factory=dict)
    memory_used: list[str] = field(default_factory=list)
    memory_captured: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    directive: dict[str, Any] = field(default_factory=dict)
    guard: dict[str, Any] = field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    llm_errors: list[dict[str, str]] = field(default_factory=list)
    reply_source: str = "template"
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
