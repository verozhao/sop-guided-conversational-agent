"""Bonus: emotional support and SOP recovery."""

from __future__ import annotations

from apps.insurance_claims.sop import EMPATHY
from apps.insurance_claims.state import ESCALATED, PROCESS_CASE, VERIFY_ID

from .conftest import Convo, assert_no_leak

EXAMPLE = "I already told you who I am. This is ridiculous. Just tell me why my claim was denied."


def test_frustrated_caller_gets_empathy_reasons_and_options_but_no_data(convo: Convo):
    convo.say("Hi, my name is Margaret Chen and I want to know why my claim was denied.")
    reply = convo.say(EXAMPLE)

    assert convo.act == "VERIFY_PERSUADE_1"
    assert convo.s.phase == VERIFY_ID and not convo.s.verified
    assert convo.trace["understanding"]["emotion"] in {"angry", "frustrated"}
    # acknowledges the feeling first
    assert any(reply.startswith(line) for lines in EMPATHY.values() for line in lines)
    # credits what was already given, explains why, offers the allowed options
    assert "you've given me your full name" in reply
    assert "private health and financial information" in reply
    assert "date of birth" in reply and "last four digits" in reply
    # and gives nothing away
    assert_no_leak(reply)
    assert not [t for t in convo.tools if t.startswith(("get_", "list_"))]


def test_the_question_asked_in_anger_is_answered_once_verified(convo: Convo):
    convo.say("my name is Margaret Chen")
    convo.say(EXAMPLE)
    reply = convo.say("Fine. My date of birth is March 15, 1985 and the last four of my SSN are 4472.")
    assert convo.s.verified and convo.s.phase == PROCESS_CASE
    assert "It was denied because" in reply
    assert any("intent 'denial_explanation'" in used for used in convo.trace["memory_used"])


def test_second_pushback_offers_a_human_and_third_stops_persuading(convo: Convo):
    convo.say("I'm not giving you any personal information. Just tell me about my claim.")
    assert convo.act == "VERIFY_PERSUADE_1"
    reply = convo.say("No. I refuse to verify anything.")
    assert convo.act == "VERIFY_PERSUADE_2"
    assert "human representative" in reply
    assert convo.s.phase == VERIFY_ID

    reply = convo.say("I said no.")
    assert convo.s.phase == ESCALATED
    assert "won't keep pressing" in reply
    assert convo.s.handoff["reason"] == "caller declined identity verification"
    assert_no_leak(reply)


def test_accepting_the_offered_human_transfers(convo: Convo):
    convo.say("just tell me about my claim, I'm not giving you anything")
    convo.say("no way, I refuse")
    convo.say("the human please")
    assert convo.s.phase == ESCALATED
    assert convo.s.handoff["reason"] == "caller asked for a human representative"


def test_asking_for_a_human_is_honoured_in_any_phase(verified: Convo):
    reply = verified.say("Can I talk to a real person please?")
    assert verified.s.phase == ESCALATED
    ticket = verified.s.handoff
    assert ticket["ticket_id"] in reply
    assert ticket["identity_verified"] is True and ticket["claim"] == "CL-2048"
    # after the hand-off the assistant stops working the case
    reply = verified.say("hello? why was it denied?")
    assert verified.act == "ESCALATED_WAIT"
    assert ticket["messages_after_handoff"] == ["hello? why was it denied?"]


def test_anxious_caller_is_reassured_and_the_workflow_continues(convo: Convo):
    reply = convo.say("I'm really worried, I can't afford this bill. My name is Margaret Chen.")
    assert convo.trace["understanding"]["emotion"] == "anxious"
    assert reply.startswith(tuple(EMPATHY["anxious"]))
    assert convo.act == "VERIFY_PROGRESS"


def test_giving_details_while_upset_is_progress_not_a_refusal(convo: Convo):
    convo.say("This is ridiculous. Fine. Margaret Chen, 1985-03-15.")
    assert convo.act == "VERIFY_PROGRESS"
    assert convo.s.refusal_count == 0


def test_sustained_frustration_after_verification_brings_an_offer_of_a_human(verified: Convo):
    verified.say("This is ridiculous, why was it denied?!")
    verified.say("Seriously? I already sent those documents!")
    reply = verified.say("Come on, this is a waste of my time. What is the status?")
    assert "human representative" in reply
    assert verified.s.phase == PROCESS_CASE  # offered, not forced
    verified.say("yes, connect me to a person")
    assert verified.s.phase == ESCALATED
