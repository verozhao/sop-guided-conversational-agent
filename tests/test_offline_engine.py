"""The offline rule engine: common phrasings are read correctly.

These guard the experience of someone testing without an API token, and the
fallback path when a model call fails.
"""

from __future__ import annotations

from datetime import date

import pytest

from apps.insurance_claims import Agent
from apps.insurance_claims.nlu_rules import rule_understand
from apps.insurance_claims.state import POST_PROCESS, PROCESS_CASE, VERIFY_ID

from .conftest import DEMO_DATE, Convo


def read(agent: Agent, text: str, phase: str = VERIFY_ID, pending: str = "identity"):
    session = agent.new_session()
    session.phase, session.pending = phase, pending
    if phase != VERIFY_ID:
        session.verified, session.party_id, session.case_id = True, "P9", "CL-2048"
    return rule_understand(session, text, agent.data, DEMO_DATE)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("margaret chen", {"full_name": "Margaret Chen"}),
        ("It's Margaret Chen.", {"full_name": "Margaret Chen"}),
        ("this is margaret chen calling", {"full_name": "Margaret Chen"}),
        ("My name's Margaret Chen, DOB 3/15/1985", {"full_name": "Margaret Chen", "dob": "1985-03-15"}),
        ("born on the 15th of March 1985", {"dob": "1985-03-15"}),
        ("my birthday is March 15th, 1985", {"dob": "1985-03-15"}),
        ("you can reach me at 650.521.2836", {"phone": "6505212836"}),
        ("email: Margaret@Email.com", {"email": "margaret@email.com"}),
        ("last 4 of social 4472", {"id_last4": "4472"}),
        ("ssn ends in 4472", {"id_last4": "4472"}),
        ("it's 4472", {"id_last4": "4472"}),
        ("policy number 9921", {"policy_number": "POL-9921"}),
        ("Margaret Chen, 1985-03-15, 4472. It's about my auto claim.", {"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"}),
    ],
)
def test_identity_extraction(agent: Agent, text: str, expected: dict):
    assert read(agent, text).identity == expected


@pytest.mark.parametrize(
    "text",
    [
        "THIS IS UNACCEPTABLE I WANT ANSWERS NOW",
        "I'm calling about my claim",
        "This is ridiculous",
        "I'm really upset about this",
        "DOB is 1985-03-15",  # a date of birth must not become a claim hint
        "why do you need my name?",
        "explain reinforcement learning",
        "tell me something",
        "go away now",
    ],
)
def test_no_name_is_invented(agent: Agent, text: str):
    u = read(agent, text)
    assert "full_name" not in u.identity
    assert "month" not in u.case_hints


def test_questions_about_the_process_are_not_refusals_or_transfers(agent: Agent):
    u = read(agent, "is this a real person?")
    assert u.process_question and not u.wants_human
    u = read(agent, "why do you need my social security number?")
    assert u.process_question and not u.refuses_verification and not u.declined_fields
    u = read(agent, "what do you need from me")
    assert u.process_question and u.intent == "none"
    assert read(agent, "I'm her representative").wants_human is False
    assert read(agent, "I'm her representative").caller_role == "representative"


def test_declined_fields_and_refusals(agent: Agent):
    assert read(agent, "I don't want to give my social").declined_fields == ["id_last4"]
    assert read(agent, "I don't have an email").declined_fields == ["email"]
    assert read(agent, "can't you just look it up with my name?").refuses_verification
    assert read(agent, "can't you just look it up with my name?").declined_fields == []
    assert read(agent, "I'm not giving you any personal information").refuses_verification


@pytest.mark.parametrize(
    "text, scope",
    [
        ("what is the capital of France", "out_of_scope"),
        ("can you recommend a good doctor?", "out_of_scope"),
        ("should I get the surgery?", "out_of_scope"),
        ("2 + 2?", "out_of_scope"),
        ("explain reinforcement learning", "out_of_scope"),
        ("hello, I need help with a claim", "in_scope"),
        ("what's my deductible?", "in_scope"),
        ("I'm so stressed, the hospital keeps calling me about the bill", "in_scope"),
        ("thanks", "in_scope"),
    ],
)
def test_scope_before_verification(agent: Agent, text: str, scope: str):
    assert read(agent, text).scope == scope


@pytest.mark.parametrize(
    "text, scope",
    [
        ("why?", "in_scope"),
        ("and the one from last year?", "in_scope"),
        ("can you approve it now?", "in_scope"),
        ("how will I know you received it", "in_scope"),
        ("what is RL", "out_of_scope"),
        ("write me a poem", "out_of_scope"),
    ],
)
def test_scope_after_verification_uses_context(agent: Agent, text: str, scope: str):
    assert read(agent, text, PROCESS_CASE, "anything_else").scope == scope


@pytest.mark.parametrize(
    "text, intent, topic",
    [
        ("why was it denied", "denial_explanation", None),
        ("why?", "denial_explanation", None),
        ("what do you need from me", "document_requirements", None),
        ("where do I send the pathology report", "document_requirements", "submission_method"),
        ("can I fax it?", "document_requirements", "submission_method"),
        ("is a photo of the report ok?", "document_requirements", "file_format_requirements"),
        ("how will I know you received it", "document_requirements", "receipt_confirmation"),
        ("I can't get the office note, the clinic closed", "document_requirements", "missing_required_material_alternatives"),
        ("when is the deadline", "appeal_or_next_steps", None),
        ("how much did you pay", "payment_or_amounts", None),
        ("what's the status", "claim_status", None),
    ],
)
def test_intents_and_topics(agent: Agent, text: str, intent: str, topic: str | None):
    u = read(agent, text, PROCESS_CASE, "anything_else")
    assert u.intent == intent
    if topic:
        assert topic in u.followup_topics
    assert "file_format_requirements" not in read(agent, "I won't share personal information").followup_topics


@pytest.mark.parametrize(
    "text, choice",
    [("sure", "send"), ("yes please", "send"), ("ok", "send"), ("no thanks", "skip"), ("nah", "skip"), ("skip", "skip"),
     ("what's in it?", "none"), ("send it to my work email bob@work.com", "none")],
)
def test_email_choice(agent: Agent, text: str, choice: str):
    assert read(agent, text, POST_PROCESS, "email_offer").email_summary_choice == choice


def test_summary_only_goes_to_the_address_on_file(verified: Convo):
    verified.say("that's all")
    reply = verified.say("send it to my work email bob@work.com")
    assert verified.act == "EMAIL_ADDRESS_FIXED"
    assert verified.s.outbox == []
    assert "m***@email.com" in reply and "bob@work.com" not in reply
    verified.say("ok, send it there")
    assert len(verified.s.outbox) == 1
    assert verified.s.outbox[0]["to_masked"] == "m***@email.com"


def test_the_other_claim_excludes_the_current_one(verified: Convo):
    verified.say("and the one from last year?")
    assert verified.act == "ASK_WHICH_CLAIM"  # two claims were filed in 2025
    assert verified.s.last_candidates == ["CL-1899", "CL-2011"]
    verified.say("the first one")
    assert verified.s.case_id == "CL-1899"

    verified.say("back to my healthcare claim from January 2026 please")
    assert verified.s.case_id == "CL-2048"
    reply = verified.say("what about the other healthcare claim?")
    assert verified.s.case_id == "CL-2011"
    assert "closed" in reply


def test_a_request_no_path_covers_is_declined_with_an_offer_of_a_human(verified: Convo):
    reply = verified.say("can you approve it now?")
    assert verified.s.phase == PROCESS_CASE
    assert "not something I can do or look up" in reply and "human claims representative" in reply
    verified.say("yes, connect me to a representative")
    assert verified.s.phase == "ESCALATED"


def test_a_bare_why_repeats_the_reason(verified: Convo):
    reply = verified.say("why?")
    assert "It was denied because" in reply


def test_dates_follow_the_demo_clock():
    assert DEMO_DATE == date(2026, 3, 5)
