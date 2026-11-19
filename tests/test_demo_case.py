"""The demo test case from the brief, end to end."""

from __future__ import annotations

from apps.insurance_claims.state import (
    CLOSED,
    POST_PROCESS,
    PROCESS_CASE,
    RESOLVE_INTENT,
    VERIFY_ID,
)

from .conftest import DEMO_UTTERANCE, Convo, assert_no_leak


def test_demo_utterance_verifies_resolves_and_answers(convo: Convo):
    reply = convo.say(DEMO_UTTERANCE)

    # identity extracted and verified on three PII fields
    assert convo.s.verified
    assert convo.s.party_id == "P9"
    assert set(convo.s.verification["matched"]) == {"full_name", "dob", "id_last4"}

    # phases were walked in order, in one turn
    assert convo.trace["phase_path"] == [VERIFY_ID, RESOLVE_INTENT, PROCESS_CASE]

    # the hint from the same utterance resolved the claim; the caller was not asked again
    assert convo.s.case_id == "CL-2048"
    assert convo.act == "ANSWER"
    assert "which one" not in reply.lower() and "which claim" not in reply.lower()

    # the answer is grounded in tool data
    assert "CL-2048" in reply
    assert "pathology report" in reply
    assert "March 18, 2026" in reply
    assert "get_denial_details" in convo.tools


def test_nothing_about_the_claim_is_disclosed_before_verification(convo: Convo):
    lines = [
        "I'm calling about my denied healthcare claim from January.",
        "Why was it denied? Just give me the reason.",
        "My name is Margaret Chen.",
        "Can you at least tell me the claim number or how much it was for?",
        "DOB is 1985-03-15",
    ]
    for line in lines:
        reply = convo.say(line)
        assert convo.s.phase == VERIFY_ID
        assert not convo.s.verified
        assert_no_leak(reply)
        # no claim tool ran, and none is even available in this phase
        assert not [t for t in convo.tools if t.startswith(("get_", "list_"))]
        assert "data" not in convo.trace["directive"]["facts"]


def test_hint_given_during_verification_is_used_after_it(convo: Convo):
    convo.say("Hi, I'm calling about my denied healthcare claim from January.")
    memory = convo.s.memory
    assert convo.s.phase == VERIFY_ID
    assert memory.intent != "none"
    assert memory.case_hints == {"claim_type": "healthcare", "status": "denied", "month": 1}
    assert memory.captured["case_hints"]["phase"] == VERIFY_ID

    convo.say("Margaret Chen, born March 15 1985")
    assert convo.s.phase == VERIFY_ID

    reply = convo.say("4472")
    assert convo.s.verified
    assert convo.s.phase == PROCESS_CASE
    assert convo.s.case_id == "CL-2048"
    assert "CL-2048" in reply and "pathology report" in reply
    assert any("captured in VERIFY_ID, turn 1" in used for used in convo.trace["memory_used"])


def test_full_workflow_ends_with_an_email_summary_the_customer_chose(convo: Convo):
    convo.say(DEMO_UTTERANCE)
    convo.say("What documents do I need and how do I send them?")
    assert convo.s.phase == PROCESS_CASE
    assert "get_required_documents" in convo.tools

    reply = convo.say("No, that's all.")
    assert convo.s.phase == POST_PROCESS
    assert convo.act == "EMAIL_OFFER"
    assert "m***@email.com" in reply and "skip" in reply.lower()
    assert convo.s.outbox == []  # nothing is sent before the customer chooses

    reply = convo.say("Yes, please send it.")
    assert convo.s.phase == CLOSED
    assert convo.act == "EMAIL_SENT"
    assert len(convo.s.outbox) == 1
    body = convo.s.outbox[0]["body"]
    for section in ("WHAT WE DISCUSSED", "CLAIM STATUS AND OUTCOME", "FOLLOW-UP ITEMS AND NEXT STEPS"):
        assert section in body
    assert "CL-2048" in body and "Denied" in body and "pathology report" in body
    assert "March 18, 2026" in body


def test_customer_can_skip_the_email(verified: Convo):
    verified.say("that's all")
    verified.say("skip it")
    assert verified.s.phase == CLOSED
    assert verified.act == "EMAIL_SKIPPED"
    assert verified.s.outbox == []
