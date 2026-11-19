"""RESOLVE_INTENT and PROCESS_CASE: bounded paths, grounded answers."""

from __future__ import annotations

from datetime import date

from apps.insurance_claims import Agent, AgentConfig
from apps.insurance_claims.state import POST_PROCESS, PROCESS_CASE, RESOLVE_INTENT

from .conftest import DEMO_UTTERANCE, Convo

VERIFY_ONLY = "Margaret Chen, born 1985-03-15, phone 650-521-2836"


def test_ambiguous_hint_lists_candidates_and_an_ordinal_picks_one(convo: Convo):
    convo.say(VERIFY_ONLY)
    reply = convo.say("I have a question about my healthcare claim")
    assert convo.act == "ASK_WHICH_CLAIM"
    assert convo.s.phase == RESOLVE_INTENT
    assert "CL-2048" in reply and "CL-2011" in reply
    assert "CL-1899" not in reply  # the dental claim does not match the hint
    assert "pathology" not in reply  # no claim details until one is chosen

    reply = convo.say("the first one")
    assert convo.s.phase == PROCESS_CASE
    assert convo.s.case_id == "CL-2048"


def test_a_follow_up_hint_narrows_the_choice(convo: Convo):
    convo.say(VERIFY_ONLY)
    convo.say("it's about my healthcare claim")
    convo.say("the one from 2025")
    assert convo.s.case_id == "CL-2011"
    assert convo.s.phase == PROCESS_CASE


def test_description_that_matches_nothing_lists_what_is_on_file(convo: Convo):
    convo.say(VERIFY_ONLY)
    reply = convo.say("I'm calling about my dental claim from March")
    assert convo.act == "CLAIM_NOT_MATCHED"
    assert convo.s.phase == RESOLVE_INTENT
    assert "CL-1899" in reply and "CL-2048" in reply


def test_a_wrong_status_word_does_not_hide_the_claim(convo: Convo):
    convo.say(VERIFY_ONLY)
    reply = convo.say("I'm calling about my denied dental claim")
    assert convo.s.case_id == "CL-1899"
    assert "don't see a denied dental claim" in reply
    assert "closed dental claim" in reply


def test_a_claim_of_another_policyholder_is_never_reachable(convo: Convo):
    convo.say(VERIFY_ONLY)
    reply = convo.say("Tell me about claim CL-3001")
    assert convo.act == "CLAIM_ID_NOT_FOUND"
    assert convo.s.case_id is None
    assert "diagnosis" not in reply.lower()
    assert "CL-2048" in reply  # the caller's own claims are offered instead


def test_switching_to_another_claim_goes_back_through_resolve(verified: Convo):
    reply = verified.say("What about my dental claim, how much was paid?")
    assert verified.trace["phase_path"] == [PROCESS_CASE, RESOLVE_INTENT, PROCESS_CASE]
    assert verified.s.case_id == "CL-1899"
    assert "get_payment_details" in verified.tools
    assert "$425.00" in reply and "$450.00" in reply
    assert "$1,450.00" not in reply


def test_a_status_word_alone_does_not_switch_claims(verified: Convo):
    verified.say("what about my dental claim?")
    assert verified.s.case_id == "CL-1899"
    verified.say("was this claim denied?")
    assert verified.s.case_id == "CL-1899"
    assert verified.trace["phase_path"] == [PROCESS_CASE]


def test_each_path_calls_only_its_own_tools(verified: Convo):
    verified.say("How much will I be reimbursed?")
    assert verified.trace["directive"]["facts"]["workflow_path"] == "payment_or_amounts"
    assert set(verified.tools) == {"get_claim_summary", "get_payment_details"}

    reply = verified.say("How do I submit the documents?")
    assert verified.trace["directive"]["facts"]["workflow_path"] == "document_requirements"
    assert set(verified.tools) == {"get_claim_summary", "get_required_documents", "get_followup_guidance"}
    assert "member portal" in reply

    reply = verified.say("What are my next steps to appeal?")
    assert verified.trace["directive"]["facts"]["workflow_path"] == "appeal_or_next_steps"
    assert "March 18, 2026" in reply


def test_follow_up_topic_guidance_is_rendered_from_the_guideline(verified: Convo):
    reply = verified.say("How long will it take after I submit them?")
    assert "usually less than a week" in reply
    reply = verified.say("I can't get the pathology report, the lab lost it. What can I do?")
    topics = verified.trace["understanding"]["followup_topics"]
    assert "missing_required_material_alternatives" in topics
    assert "replacement copy" in reply


def test_denial_path_on_a_claim_that_was_not_denied(verified: Convo):
    reply = verified.say("And my auto claim, why was it denied?")
    assert verified.s.case_id == "CL-2102"
    assert "don't see a denied auto claim" in reply and "open auto claim" in reply
    reply = verified.say("so why was it denied?")
    assert verified.s.case_id == "CL-2102"
    assert "wasn't denied" in reply


def test_policyholder_without_claims(convo: Convo):
    reply = convo.run("Ava Lopez", "08/21/1990", "ava.lopez@email.com")
    assert convo.act == "NO_CLAIMS"
    assert convo.s.phase == RESOLVE_INTENT
    assert "don't see any claims" in reply
    convo.say("no, that's all")
    assert convo.s.phase == POST_PROCESS


def test_deadline_in_the_past_is_reported_as_passed():
    agent = Agent(config=AgentConfig(clock=lambda: date(2026, 11, 18)))
    convo = Convo(agent)
    reply = convo.say(DEMO_UTTERANCE)
    assert "has already passed" in reply
    assert "days from today" not in reply


def test_discussion_log_feeds_the_summary(verified: Convo):
    verified.say("How much will I be reimbursed?")
    verified.say("What about my auto claim, what's the status?")
    cases = {item["case_id"] for item in verified.s.discussed}
    intents = {item["intent"] for item in verified.s.discussed}
    assert cases == {"CL-2048", "CL-2102"}
    assert {"payment_or_amounts", "claim_status"} <= intents
