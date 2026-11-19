"""The harness enforces the SOP even if a policy handler misbehaves."""

from __future__ import annotations

import pytest

from apps.insurance_claims import Agent
from apps.insurance_claims.state import (
    POST_PROCESS,
    PROCESS_CASE,
    RESOLVE_INTENT,
    VERIFY_ID,
    Session,
)
from sop_harness.core import IllegalTransition, ToolDenied, TurnTrace

from .conftest import Convo


def test_claim_tools_are_denied_before_verification(agent: Agent):
    session = agent.new_session()
    trace = TurnTrace(turn=1)
    for name in ("list_claims", "get_claim_summary", "get_denial_details", "get_payment_details"):
        with pytest.raises(ToolDenied):
            agent.tools.call(name, session, trace, **({} if name == "list_claims" else {"case_id": "CL-2048"}))
    assert all(entry["allowed"] is False for entry in trace.tools)
    assert agent.tools.allowed_in(VERIFY_ID) == ["check_consent_status", "request_consent", "transfer_to_human"]


def test_claim_tools_are_denied_to_a_forged_verified_flag_in_the_wrong_phase(agent: Agent):
    session = Session()
    session.verified = True  # a bug or an attack flips the flag
    session.party_id = "P9"
    with pytest.raises(ToolDenied):
        agent.tools.call("get_claim_summary", session, None, case_id="CL-2048")  # still VERIFY_ID


def test_a_verified_caller_cannot_read_someone_elses_claim(verified: Convo):
    agent = verified.agent
    with pytest.raises(ToolDenied):
        agent.tools.call("get_claim_summary", verified.s, None, case_id="CL-3001")
    assert agent.tools.call("get_claim_summary", verified.s, None, case_id="CL-2048")["status"] == "denied"


def test_phase_graph_blocks_skipping_and_ungated_moves(agent: Agent):
    graph = agent.sop.graph
    session = agent.new_session()
    with pytest.raises(IllegalTransition):
        graph.move(session, PROCESS_CASE, "skip ahead")  # not an edge
    with pytest.raises(IllegalTransition):
        graph.move(session, RESOLVE_INTENT, "not verified")  # edge exists, gate closed
    assert session.phase == VERIFY_ID

    session.verified, session.identity_ok, session.party_id = True, True, "P9"
    graph.move(session, RESOLVE_INTENT, "verified")
    with pytest.raises(IllegalTransition):
        graph.move(session, PROCESS_CASE, "no claim selected")
    session.case_id = "CL-3001"  # someone else's claim
    with pytest.raises(IllegalTransition):
        graph.move(session, PROCESS_CASE, "foreign claim")
    session.case_id = "CL-2048"
    graph.move(session, PROCESS_CASE, "own claim")
    assert session.phase == PROCESS_CASE


def test_verification_can_only_be_granted_with_a_matching_record(agent: Agent):
    from apps.insurance_claims.sop import TurnCtx

    session = agent.new_session()
    with pytest.raises(RuntimeError):
        agent.sop._grant(session, TurnCtx(trace=TurnTrace(turn=1)), "forced")
    assert not session.verified


def test_email_cannot_be_sent_without_the_customers_choice(verified: Convo):
    verified.say("that's all")
    assert verified.s.phase == POST_PROCESS
    with pytest.raises(ToolDenied):
        verified.agent.tools.call("send_email_summary", verified.s, None)
    assert verified.s.outbox == []


def test_consent_cannot_be_requested_before_the_details_match(agent: Agent):
    session = agent.new_session()
    with pytest.raises(ToolDenied):
        agent.tools.call("request_consent", session, None)


def test_every_template_reply_passes_the_output_guard(agent: Agent):
    """The deterministic replies must satisfy the same guard as model replies."""
    from apps.insurance_claims.guard import check_reply
    from sop_harness.core import Directive

    scripts = [
        ["Hi, my name is Margaret Chen and I want to know why my claim was denied.",
         "I already told you who I am. This is ridiculous. Just tell me why my claim was denied.",
         "Fine. My date of birth is March 15, 1985 and the last four of my SSN are 4472.",
         "What do I do now?", "How much will I get?", "what about my auto claim?", "that's all", "yes"],
        ["Hi, I'm David Chen, calling for my mother Margaret Chen about her denied claim.",
         "Her date of birth is 1985-03-15 and her phone number is 650-521-2836.",
         "OK, she just approved it.", "that's everything", "no thanks"],
        ["What is RL?", "tell me a joke", "write a poem", "what's the weather?"],
    ]
    for lines in scripts:
        convo = Convo(agent)
        for line in lines:
            reply = convo.say(line)
            summary = convo.trace["directive"]
            directive = Directive(act=summary["act"], goal="", fallback="", facts=summary["facts"])
            # judge the reply against the facts alone, as a model reply would be
            verdict = check_reply(reply, session=convo.s, directive=directive, data=agent.data)
            assert verdict.ok, (line, verdict.violations, reply)


def test_understanding_schema_uses_only_supported_json_schema_features():
    """Structured outputs reject unknown keywords and need closed objects."""
    from apps.insurance_claims.understanding import UNDERSTANDING_SCHEMA

    allowed = {"type", "properties", "required", "additionalProperties", "description", "enum", "items"}

    def walk(node: dict) -> None:
        assert set(node) <= allowed, set(node) - allowed
        if node["type"] == "object":
            assert node["additionalProperties"] is False
            assert sorted(node["required"]) == sorted(node["properties"])
            for child in node["properties"].values():
                walk(child)
        elif node["type"] == "array":
            walk(node["items"])
        else:
            assert node["type"] in {"string", "integer", "boolean"}

    walk(UNDERSTANDING_SCHEMA)
