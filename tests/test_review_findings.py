"""Regression tests for the findings of the pre-submission code review.

Each test fixes one behaviour that a reviewer reproduced as wrong: an email
sent on a refusal, a third party verified as the policyholder, a short reply
taken for a name, a clarification question declined as out of scope, a
transfer the caller did not ask for, a claim switched by accident.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest

from apps.insurance_claims import Agent
from apps.insurance_claims.agent import DEFAULT_DEMO_DATE, clock_from_env
from apps.insurance_claims.guard import check_reply
from apps.insurance_claims.nlg import NLG_SYSTEM
from apps.insurance_claims.nlu_llm import NLU_SYSTEM
from apps.insurance_claims.nlu_rules import rule_understand
from apps.insurance_claims.normalize import redact_ssn
from apps.insurance_claims.state import CLOSED, ESCALATED, POST_PROCESS, PROCESS_CASE, RESOLVE_INTENT, VERIFY_ID
from apps.insurance_claims.understanding import UNDERSTANDING_SCHEMA
from apps.insurance_claims.verification import verify_identity
from sop_harness.core import Directive
from sop_harness.llm import AnthropicLLM, LLMError, ScriptedLLM

from .conftest import DEMO_DATE, DEMO_UTTERANCE, Convo, assert_no_leak
from .test_llm_path import reading

THREE = "Margaret Chen, 1985-03-15, 4472"


def at_email_offer(agent: Agent) -> Convo:
    c = Convo(agent)
    c.run(DEMO_UTTERANCE, "No, that's all.")
    assert c.s.phase == POST_PROCESS and c.s.pending == "email_offer"
    return c


# ---------------------------------------------------------------- the email is sent only on a clear yes


@pytest.mark.parametrize(
    "text",
    ["absolutely not", "definitely not", "of course not", "ok no", "yeah no", "never send me emails",
     "don't send it", "I don't want it", "no thanks", "skip it", "that's all, bye"],
)
def test_a_refusal_at_the_email_offer_never_sends(agent: Agent, text: str):
    c = at_email_offer(agent)
    c.say(text)
    assert c.act == "EMAIL_SKIPPED" and not c.s.outbox and "send_email_summary" not in c.tools


@pytest.mark.parametrize(
    "text",
    ["yes... actually no", "no, send it", "no don't skip, send it please", "hmm",
     "I can't afford this and you want to send me an email??"],
)
def test_mixed_or_unclear_answers_at_the_email_offer_are_asked_again(agent: Agent, text: str):
    c = at_email_offer(agent)
    c.say(text)
    assert c.act == "EMAIL_OFFER_REPEAT" and not c.s.outbox and c.s.phase == POST_PROCESS


@pytest.mark.parametrize("text", ["yes please", "send it", "ok", "sure go ahead", "can you send it?", "don't skip it"])
def test_a_clear_yes_at_the_email_offer_sends(agent: Agent, text: str):
    c = at_email_offer(agent)
    c.say(text)
    assert c.act == "EMAIL_SENT" and len(c.s.outbox) == 1


@pytest.mark.parametrize("text", ["what will be in it?", "can you send it to my work email instead?", "ok but not to that address"])
def test_questions_about_the_summary_are_answered_at_the_offer(agent: Agent, text: str):
    c = at_email_offer(agent)
    reply = c.say(text)
    assert c.act == "EMAIL_OFFER_REPEAT" and not c.s.outbox
    assert "address on file" in reply and "denied because" not in reply


def test_another_question_at_the_email_offer_goes_back_to_the_claim(agent: Agent):
    c = at_email_offer(agent)
    c.say("yes I have another question")
    assert c.act == "ASK_WHAT_ELSE" and c.s.phase == PROCESS_CASE and not c.s.outbox
    assert "1,450.00" in c.say("how much was it for?")
    c.say("no that's all")
    assert c.act == "EMAIL_OFFER"  # the offer comes back, nothing was decided for the caller
    c.say("yes please")
    assert c.act == "EMAIL_SENT" and len(c.s.outbox) == 1


def test_a_question_asked_with_a_yes_is_answered_before_anything_is_sent(agent: Agent):
    c = at_email_offer(agent)
    reply = c.say("sure, but first, how do I appeal?")
    assert c.s.phase == PROCESS_CASE and not c.s.outbox and "March 18, 2026" in reply


def test_no_email_address_during_verification_is_not_a_no_to_the_summary(convo: Convo):
    convo.run("My name is Margaret Chen", "no email, sorry")
    assert convo.s.memory.email_pref == "none" and "email" in convo.s.memory.declined
    convo.run("1985-03-15, 4472", "the denied one", "no that's all")
    assert convo.act == "EMAIL_OFFER"


def test_after_closing_another_question_reopens_the_claim(agent: Agent):
    c = at_email_offer(agent)
    c.say("no thanks")
    assert c.s.phase == CLOSED
    assert c.say("wait, I had another question") == "Of course. What would you like to know?"
    assert "March 18, 2026" in c.say("how do I appeal?") and c.s.phase == PROCESS_CASE


def test_after_closing_only_a_request_for_the_summary_sends_it(agent: Agent):
    for text, sent in (("can you email me the form?", 0), ("send me the documents list", 0), ("actually, send it", 1), ("I'd like the summary after all", 1)):
        c = at_email_offer(agent)
        c.say("no thanks")
        assert c.s.phase == CLOSED and not c.s.outbox
        c.say(text)
        assert len(c.s.outbox) == sent, text


def test_an_upset_message_after_closing_is_not_met_with_youre_welcome(agent: Agent):
    c = at_email_offer(agent)
    c.say("no thanks")
    reply = c.say("You people are useless! I want to make a complaint about how this was handled.")
    assert c.act == "CLOSED_CONCERN" and "welcome" not in reply.lower() and c.s.pending == "human_offer"
    c.say("yes")
    assert c.s.phase == ESCALATED


# ---------------------------------------------------------------- who is calling


@pytest.mark.parametrize(
    "text",
    [
        "I'm Margaret Chen's husband, calling about her denied healthcare claim. Born 1985-03-15, SSN last four 4472.",
        "I'm Margaret Chen's son, her DOB is 1985-03-15 and her phone is 650-521-2836",
        "I'm calling for Margaret Chen, DOB 1985-03-15, SSN last four 4472",
        "I am the husband of Margaret Chen. Her date of birth is 1985-03-15 and her SSN ends in 4472.",
    ],
)
def test_a_third_party_with_three_correct_details_is_not_verified_as_the_policyholder(convo: Convo, text: str):
    reply = convo.say(text)
    assert not convo.s.verified and convo.s.phase == VERIFY_ID
    assert convo.s.memory.caller_role == "representative" and convo.act == "ASK_REP_NAME"
    assert_no_leak(reply)


@pytest.mark.parametrize(
    "text",
    [
        "My husband told me to call about my claim. I'm Margaret Chen, born 1985-03-15, last four 4472.",
        "I'm the policyholder, my name is Margaret Chen, born 1985-03-15, last four 4472, calling about my son's dental claim",
    ],
)
def test_a_policyholder_who_mentions_a_relative_is_still_the_policyholder(convo: Convo, text: str):
    convo.say(text)
    assert convo.s.verified and convo.s.memory.caller_role != "representative"


def test_saying_you_act_for_the_policyholder_after_verification_stops_disclosure(verified: Convo):
    reply = verified.say("Actually I'm her husband, I'm calling on her behalf")
    assert verified.s.phase == ESCALATED and "approved" in reply


# ---------------------------------------------------------------- names


@pytest.mark.parametrize(
    "text",
    ["its margaret", "fair enough", "makes sense", "uh huh", "oh boy", "seriously though", "Speak English",
     "ok fine, I'll verify", "go away", "not today", "Cell Number", "what now"],
)
def test_a_short_reply_is_not_taken_for_a_name(convo: Convo, text: str):
    convo.say(text)
    assert "full_name" not in convo.s.memory.identity and convo.s.failed_attempts == 0


@pytest.mark.parametrize(
    "text, field",
    [("cell number 650-521-2836", "phone"), ("id number 4472", "id_last4"), ("mobile number 6505212836", "phone")],
)
def test_a_labelled_detail_is_read_as_that_detail_only(convo: Convo, text: str, field: str):
    convo.say(text)
    assert convo.s.verification["matched"] == [field] and convo.s.failed_attempts == 0


def test_thats_plus_a_name_is_a_name_and_a_first_name_alone_asks_for_the_rest(convo: Convo):
    convo.say("That's Margaret Chen")
    assert convo.s.verification["matched"] == ["full_name"]
    other = Convo(convo.agent)
    reply = other.say("It's Margaret")
    assert other.s.verification["incomplete"] == ["full_name"] and "first and last name" in reply


def test_a_chat_reply_cannot_undo_a_confirmed_name_or_count_as_an_attempt(convo: Convo):
    convo.run("My name is Margaret Chen", "makes sense", "fair enough")
    assert convo.s.verification["matched"] == ["full_name"] and convo.s.failed_attempts == 0


def test_names_that_match_no_policy_are_forgiven_only_a_few_times(convo: Convo):
    convo.run("jane doe", "john smith", "My name is Mary Jones")
    assert convo.s.failed_attempts == 0 and convo.s.name_misses == 3
    convo.say("My name is Peter Pan")
    assert convo.s.failed_attempts == 1  # the allowance is spent, so names cannot be probed without limit
    convo.run("My name is Wendy Darling", "My name is John Darling")
    assert convo.s.phase == ESCALATED and not convo.s.verified


def test_a_lower_case_name_still_verifies(convo: Convo):
    convo.say("margaret chen, 1985-03-15, 4472")
    assert convo.s.verified


# ---------------------------------------------------------------- clarification questions before verification


@pytest.mark.parametrize(
    "text, expect",
    [
        ("what do you mean?", "three details"),
        ("which three?", "three details"),
        ("how many do you need?", "three details"),
        ("can you repeat that?", "three details"),
        ("what?", "three details"),
        ("can I use my address instead?", "can't be used"),
        ("is my driver's license ok?", "can't be used"),
        ("what happens if I get it wrong?", "human representative"),
        ("will you store my information?", "only used to confirm identity"),
        ("are you there?", "still here"),
        ("give me a minute", "take your time"),
        ("why do you need my SSN?", "private health and financial"),
    ],
)
def test_clarification_questions_are_answered_not_declined(convo: Convo, text: str, expect: str):
    reply = convo.say(text)
    assert convo.act == "VERIFY_EXPLAIN" and convo.s.oos_streak == 0
    assert expect in reply and "full name" in reply
    assert_no_leak(reply)


@pytest.mark.parametrize(
    "text",
    ["I got a bill from the hospital, can you help me with that?", "why did you reject my MRI?",
     "can you look up my bill?", "do you handle dental?", "do you like pizza?", "how are you?"],
)
def test_an_unclear_question_before_verification_is_never_counted_as_out_of_scope(convo: Convo, text: str):
    reply = convo.say(text)
    assert convo.act == "VERIFY_ASK" and convo.s.oos_streak == 0 and "verify your identity" in reply


def test_four_clarifications_in_a_row_do_not_transfer_the_caller(convo: Convo):
    convo.run("what do you mean?", "which three?", "can I use my address instead?", "how many do you need?")
    assert convo.s.phase == VERIFY_ID and convo.s.handoff is None


def test_a_question_sent_with_a_detail_is_answered(convo: Convo):
    reply = convo.say("My name is Margaret Chen. Why do you need my SSN?")
    assert convo.act == "VERIFY_PROGRESS" and "private health and financial" in reply and "confirmed your full name" in reply


def test_plain_no_to_the_identity_question_is_pushback_not_silence(convo: Convo):
    convo.say("no")
    assert convo.act == "VERIFY_PERSUADE_1"
    convo.say("I'd rather not")
    assert convo.act == "VERIFY_PERSUADE_2"


def test_goodbye_before_verifying_is_accepted(convo: Convo):
    for text in ("never mind, goodbye", "thanks, bye"):
        c = Convo(convo.agent)
        reply = c.say(text)
        assert c.act == "VERIFY_GOODBYE" and "three" not in reply and c.s.phase == VERIFY_ID


# ---------------------------------------------------------------- scope after verification


@pytest.mark.parametrize(
    "text",
    ["how do I bake sourdough bread?", "why is the sky blue?", "do you know any good pizza places?",
     "how tall is Mount Everest?", "can you fix my computer?", "can you help me with my taxes"],
)
def test_unrelated_questions_after_verification_are_declined_and_counted(verified: Convo, text: str):
    reply = verified.say(text)
    assert verified.act == "OOS_DECLINE" and verified.s.oos_streak == 1 and "claims" in reply


@pytest.mark.parametrize(
    "text",
    ["can you approve it?", "when will I hear back?", "can you call me back?", "is there a phone number I can call?",
     "can I put my husband down as my representative?", "was this decision made by an actual person?"],
)
def test_claim_requests_no_path_covers_get_an_offer_of_a_person_not_a_decline(verified: Convo, text: str):
    reply = verified.say(text)
    assert verified.act == "ANSWER" and verified.s.oos_streak == 0
    assert verified.s.pending == "human_offer" and reply.endswith("Would you like me to connect you?")


@pytest.mark.parametrize("text", ["03/15/1985", "1985-03-15", "3/15/85", "(650) 521-2836", "+1 650-521-2836", "4472"])
def test_a_detail_sent_on_its_own_is_not_called_partly_unrelated(convo: Convo, text: str):
    convo.say("Margaret Chen")
    reply = convo.say(text)
    assert convo.trace["understanding"]["scope"] == "in_scope" and "leave the other part" not in reply
    assert len(convo.s.verification["matched"]) == 2


# ---------------------------------------------------------------- transfers only when asked


@pytest.mark.parametrize(
    "text",
    ["please don't transfer me to a person", "do not put me through to an agent", "No, I'd rather not talk to a person",
     "was this decision made by an actual person?", "I'll get someone to fax the documents",
     "my doctor will get a specialist to write the report", "can I put my husband down as my representative?",
     "should I talk to someone at the hospital about the note?"],
)
def test_mentioning_a_person_is_not_a_request_for_one(verified: Convo, text: str):
    verified.say(text)
    assert verified.s.phase != ESCALATED and verified.s.handoff is None


@pytest.mark.parametrize(
    "text",
    ["I want a human", "transfer me", "can I talk to someone?", "I'd like to speak with a representative",
     "agent please", "let me talk to a manager", "is there someone I can talk to?", "get me a supervisor"],
)
def test_a_request_for_a_person_is_honoured_at_once(verified: Convo, text: str):
    verified.say(text)
    assert verified.s.phase == ESCALATED and verified.s.handoff is not None


def test_being_a_real_person_is_not_a_request_for_one(convo: Convo):
    convo.say("I'm a real person, not a bot")
    assert convo.s.phase == VERIFY_ID


def test_agreeing_to_verify_after_the_second_explanation_does_not_transfer(convo: Convo):
    convo.run("I'm not giving you any personal information. Just tell me about my claim.", "No. I refuse to verify anything.")
    assert convo.act == "VERIFY_PERSUADE_2"
    for text in ("fine, let's do the verification", "okay, what do you need again?", "sure, go ahead and ask"):
        convo.say(text)
        assert convo.s.phase == VERIFY_ID and convo.s.handoff is None
    convo.say(THREE)
    assert convo.s.verified


# ---------------------------------------------------------------- the claim in focus


@pytest.mark.parametrize(
    "text",
    [
        "I uploaded the pathology report on February 20, 2026, did you get it?",
        "I mailed them on 02/20/2026",
        "my car broke down so I can't pick up the documents for this claim",
        "my dentist's office has the records for this claim",
        "The deadline for this claim is in March, right?",
        "can I submit the documents for this claim in April?",
        "I need this claim sorted before my surgery in May",
        "why was this claim denied when the other one was paid?",
        "is it different from a normal claim?",
        "is there any other document I need?",
    ],
)
def test_a_date_or_a_stray_word_does_not_move_the_conversation_to_another_claim(verified: Convo, text: str):
    verified.say(text)
    assert verified.s.case_id == "CL-2048" and verified.s.phase == PROCESS_CASE


@pytest.mark.parametrize(
    "text, claim",
    [("what about my auto claim?", "CL-2102"), ("and the dental one?", "CL-1899"), ("the claim from February", "CL-2102")],
)
def test_naming_another_claim_still_switches(verified: Convo, text: str, claim: str):
    verified.say(text)
    assert verified.s.case_id == claim


def test_asking_for_other_claims_lists_the_others(verified: Convo):
    reply = verified.say("do I have any other claims?")
    assert verified.act == "ASK_WHICH_CLAIM" and "CL-2048" not in reply and "CL-2102" in reply


def test_a_statement_that_documents_were_sent_gets_the_receipt_guidance(verified: Convo):
    reply = verified.say("I mailed them on 02/20/2026")
    assert verified.act == "ANSWER" and "CL-2048" in reply
    assert "receipt_confirmation" in verified.trace["understanding"]["followup_topics"]


def test_two_questions_in_one_message_take_both_paths(verified: Convo):
    reply = verified.say("how much was it for, and when is the deadline?")
    assert "1,450.00" in reply and "March 18, 2026" in reply
    assert {"get_payment_details", "get_denial_details"} <= set(verified.tools)


def test_the_question_asked_first_leads_the_answer(verified: Convo):
    verified.say("why was this claim denied when the other one was paid?")
    assert verified.trace["understanding"]["intent"] == "denial_explanation"


# ---------------------------------------------------------------- after "anything else?"


@pytest.mark.parametrize("text", ["no wait", "don't hang up", "do not close this", "thank you, one more thing", "I have another question"])
def test_not_finished_yet_keeps_the_claim_open(verified: Convo, text: str):
    reply = verified.say(text)
    assert verified.act == "ASK_WHAT_ELSE" and verified.s.phase == PROCESS_CASE and "What else" in reply


@pytest.mark.parametrize("text", ["no I don't understand any of this", "nah I'm still confused", "what does that mean?"])
def test_a_confused_caller_gets_the_explanation_again(verified: Convo, text: str):
    reply = verified.say(text)
    assert verified.act == "ANSWER" and verified.s.phase == PROCESS_CASE
    assert "denied because" in reply and reply.endswith("Which part would you like me to go over again?")


@pytest.mark.parametrize("text", ["No!! This is wrong", "no that can't be right"])
def test_disagreeing_with_the_outcome_leads_to_the_next_steps(verified: Convo, text: str):
    reply = verified.say(text)
    assert verified.s.phase == PROCESS_CASE and "March 18, 2026" in reply


def test_an_upset_caller_who_asks_nothing_is_heard_and_then_offered_a_person(verified: Convo):
    reply = verified.say("no, you're not listening to me")
    assert verified.act == "LISTEN" and "what I've missed" in reply and verified.s.phase == PROCESS_CASE
    verified.say("this is ridiculous")
    reply = verified.say("ugh, I already told you")
    assert verified.act == "LISTEN_OFFER_HUMAN" and verified.s.pending == "human_offer"
    verified.say("yes")
    assert verified.s.phase == ESCALATED


def test_a_passing_goodbye_inside_a_sentence_is_not_a_closing(verified: Convo):
    verified.say("I have to say goodbye to my savings then")
    assert verified.s.phase == PROCESS_CASE


@pytest.mark.parametrize("text", ["no", "no thanks", "No, that's all. Thanks!", "that's all I needed to know, thank you", "nope, I'm all set"])
def test_a_real_closing_starts_the_wrap_up(verified: Convo, text: str):
    verified.say(text)
    assert verified.act == "EMAIL_OFFER" and verified.s.phase == POST_PROCESS


# ---------------------------------------------------------------- finding the claim


def test_a_corrected_claim_hint_replaces_the_first_one(convo: Convo):
    convo.run(THREE, "it's about my healthcare claim")
    assert convo.act == "ASK_WHICH_CLAIM"
    convo.say("sorry, I meant my auto claim")
    assert convo.s.case_id == "CL-2102"


def test_a_question_asked_before_verification_survives_choosing_a_claim(convo: Convo):
    convo.run("How much was my claim for?", THREE)
    assert convo.act == "ASK_WHICH_CLAIM"
    reply = convo.say("the denied one")
    assert convo.s.case_id == "CL-2048" and "1,450.00" in reply


def test_a_question_asked_in_anger_before_verification_is_still_answered_after_it(convo: Convo):
    convo.say("My name is Margaret Chen and I want to know how much you paid on my claim.")
    convo.say("I already told you who I am. This is ridiculous. Just tell me!")
    assert convo.act == "VERIFY_PERSUADE_1" and convo.s.memory.intent == "payment_or_amounts"
    assert convo.s.memory.backlog_question.startswith("My name is Margaret Chen and I want to know how much")
    convo.say("Fine. 1985-03-15 and 4472.")
    reply = convo.say("the denied one")
    assert "1,450.00" in reply and "get_payment_details" in convo.tools
    assert convo.trace["directive"]["facts"]["caller_question"].startswith("My name is Margaret Chen")


def test_a_second_question_before_verification_is_added_to_the_first(convo: Convo):
    convo.run("Why was my January healthcare claim denied?", "Also, how much was it for? I'm Margaret Chen.")
    reply = convo.say("1985-03-15, 4472")
    assert "denied because" in reply and "1,450.00" in reply


@pytest.mark.parametrize("text", ["none of those", "I want to file a new claim", "it's a new one, not on your list"])
def test_what_is_not_on_file_is_not_answered_with_the_same_list(convo: Convo, text: str):
    convo.say("Margaret Chen, 1985-03-15, 650-521-2836")
    reply = convo.say(text)
    assert convo.act == "NOT_ON_FILE" and "CL-" not in reply and convo.s.pending == "human_offer"
    convo.say("no thanks")
    assert convo.s.phase == POST_PROCESS  # declined the person: wrap up, do not list again


def test_last_year_is_a_year_not_the_last_claim_in_the_list(agent: Agent):
    session = agent.new_session()
    session.phase, session.pending, session.verified, session.party_id = RESOLVE_INTENT, "choose_case", True, "P9"
    session.last_candidates = ["CL-2102", "CL-2048", "CL-1899", "CL-2011"]
    u = rule_understand(session, "the one from last year", agent.data, DEMO_DATE)
    assert u.selected_case_id == "" and u.case_hints == {"year": 2025}
    assert rule_understand(session, "the last one", agent.data, DEMO_DATE).selected_case_id == "CL-2011"


# ---------------------------------------------------------------- identity matching


def test_a_wrong_policy_number_cannot_turn_correct_details_into_mismatches(convo: Convo):
    reply = convo.say("Hi, policy POL-8836. My name is Margaret Chen.")
    assert convo.act == "VERIFY_MISMATCH" and "policy number" in reply and convo.s.failed_attempts == 1
    convo.say("Sorry, wrong number. I'm Margaret Chen, born 1985-03-15, SSN last four 4472.")
    assert convo.s.verified and convo.s.failed_attempts == 1


def test_identity_matching_is_decided_by_the_five_details(agent: Agent):
    result = verify_identity(agent.data, {"policy_number": "POL-8836", "full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"})
    assert result.verified and result.record.name == "Margaret Chen" and result.conflicts == ["policy_number"]
    wrong = verify_identity(agent.data, {"policy_number": "POL-9921", "dob": "1990-01-01"})
    assert not wrong.verified and wrong.conflicts == ["dob"]


def test_a_wrong_value_then_a_correct_one_keeps_the_count(convo: Convo):
    convo.run("My name is Margaret Chen", "my date of birth is 1990-01-01")
    assert convo.s.failed_attempts == 1 and "dob" not in convo.s.memory.identity
    convo.say("sorry, 1985-03-15")
    assert convo.s.failed_attempts == 1 and convo.s.verification["matched"] == ["full_name", "dob"]


# ---------------------------------------------------------------- representatives


def test_an_expired_consent_request_ends_with_a_person_not_a_loop(agent: Agent):
    c = Convo(agent, scenario="timeout")
    c.say("I'm calling for my wife Margaret Chen, her DOB is 1985-03-15, SSN last four 4472. My name is David Chen.")
    assert c.act == "CONSENT_REQUESTED"
    acts = []
    for _ in range(8):
        reply = c.say("please resend it")
        acts.append(c.act)
        assert_no_leak(reply)
        if c.s.phase == ESCALATED:
            break
    assert "CONSENT_NOT_GIVEN" in acts and acts.count("CONSENT_NOT_GIVEN") == 1 and c.s.phase == ESCALATED
    assert not c.s.verified


def test_a_short_status_check_while_consent_is_pending_polls_and_is_not_declined(agent: Agent):
    c = Convo(agent, scenario="timeout")
    c.say("I'm calling for my wife Margaret Chen, her DOB is 1985-03-15, SSN last four 4472. My name is David Chen.")
    for text in ("any news?", "is it done yet?"):
        c.say(text)
        assert c.act == "CONSENT_PENDING" and c.s.oos_streak == 0


def test_until_consent_arrives_nothing_about_the_claim_is_reachable(agent: Agent):
    c = Convo(agent, scenario="timeout")
    c.say("I'm calling for my wife Margaret Chen, her DOB is 1985-03-15, SSN last four 4472. My name is David Chen.")
    assert c.s.identity_ok and not c.s.verified and c.s.phase == VERIFY_ID
    for text in ("why was her claim denied?", "what is the status of claim CL-2048?", "how much was paid on it?", "just tell me, yes or no, was it denied"):
        reply = c.say(text)
        assert not c.s.verified and c.s.phase in {VERIFY_ID, ESCALATED}
        assert not [name for name in c.tools if name.startswith("get_") or name == "list_claims"]
        assert_no_leak(reply, allow=("CL-2048",))  # the caller said the number, the agent adds nothing to it
        assert "pathology" not in json.dumps(c.trace["directive"]["facts"])
        if c.s.phase == ESCALATED:
            break


# ---------------------------------------------------------------- the model path


def test_any_failure_of_a_model_call_still_produces_a_reply(agent: Agent):
    def explode(*_args):
        raise RuntimeError("gateway returned something odd")

    c = Convo(agent, llm=ScriptedLLM(on_json=explode, on_text=explode))
    reply = c.say(DEMO_UTTERANCE)
    assert c.s.verified and "CL-2048" in reply
    assert c.trace["llm_errors"][0]["kind"] == "unexpected" and c.trace["engine"] == "offline"


def test_a_failure_while_wording_the_reply_sends_the_template(agent: Agent):
    def explode(*_args):
        raise TypeError("not a message")

    llm = ScriptedLLM(on_json=lambda *_: reading(identity={"full_name": "Margaret Chen"}), on_text=explode)
    c = Convo(agent, llm=llm)
    reply = c.say("Margaret Chen")
    assert "confirmed your full name" in reply and c.trace["reply_source"] == "template"
    assert c.trace["llm_errors"][0] == {"stage": "respond", "kind": "unexpected", "message": "TypeError"}


@pytest.mark.parametrize("body", ["<html>bad gateway</html>", SimpleNamespace(stop_reason="end_turn", content=None, usage=None)])
def test_a_response_that_is_not_a_message_is_a_failed_call(body):
    endpoint = SimpleNamespace(create=lambda **_kwargs: body)
    client = SimpleNamespace(messages=endpoint, beta=SimpleNamespace(messages=endpoint))
    llm = AnthropicLLM(api_key="test-key", client=client)
    with pytest.raises(LLMError) as caught:
        llm.complete_text(system="s", user="u")
    assert caught.value.kind == "bad_output"


def test_this_claim_keeps_the_focus_even_when_a_model_reads_a_date_as_a_hint(agent: Agent):
    c = Convo(agent)
    c.say(DEMO_UTTERANCE)
    c.llm = ScriptedLLM(
        on_json=lambda *_: reading(intent="appeal_or_next_steps", case_hints={"month": 3, "year": 2026}, question="Is the deadline in March?"),
        on_text=lambda *_: "Yes, the appeal deadline is March 18, 2026.",
    )
    c.say("The deadline for this claim is in March, right?")
    assert c.s.case_id == "CL-2048" and c.act == "ANSWER" and c.s.phase == PROCESS_CASE


def test_the_model_can_report_more_to_ask(agent: Agent):
    c = at_email_offer(agent)
    c.llm = ScriptedLLM(on_json=lambda *_: reading(more=True), on_text=lambda *_: "Of course, what else would you like to know?")
    c.say("hang on, there's something else")
    assert c.act == "ASK_WHAT_ELSE" and c.s.phase == PROCESS_CASE and not c.s.outbox


def test_no_value_from_the_records_appears_in_any_prompt_or_schema(agent: Agent):
    text = (NLU_SYSTEM + NLG_SYSTEM + json.dumps(UNDERSTANDING_SCHEMA)).lower()
    values: set[str] = set()
    for record in agent.data.policyholders:
        values |= {record.name, record.dob, record.email, record.phone, record.id_last4, record.policy_number}
    for claim in agent.data.claims:
        values |= {claim["case_id"], claim["created_at"], str(claim.get("denial_reason") or ""), str(claim.get("appeal_deadline") or "")}
    found = sorted(value for value in values if value and value.lower() in text)
    assert not found, f"record values in static prompt text: {found}"


# ---------------------------------------------------------------- the output guard, one check at a time


def _guard(verified: Convo, reply: str, facts: dict | None = None, fallback: str = "ok"):
    directive = Directive(act="ANSWER", goal="", fallback=fallback, facts=facts or {})
    return check_reply(reply, session=verified.s, directive=directive, data=verified.agent.data)


@pytest.mark.parametrize(
    "reply, problem",
    [
        ("Your claim CL-9999 is open.", "ungrounded claim id"),
        ("We paid 780 dollars on it.", "ungrounded amount"),
        ("The deadline is the 2nd of April.", "ungrounded date"),
        ("You have 45 days to respond.", "ungrounded number of days"),
        ("I've sent the summary to you.", "claims something was sent"),
        ("I'm transferring you now.", "claims a transfer"),
        ("It will be approved once we get the report.", "promises an outcome"),
        ("I can expedite that for you.", "offers an action"),
        ("Call us on 415-555-0199.", "ungrounded phone number"),
        ("Write to help@example.org.", "ungrounded email address"),
    ],
)
def test_guard_rejects_each_kind_of_ungrounded_statement(verified: Convo, reply: str, problem: str):
    result = _guard(verified, reply)
    assert not result.ok and any(problem in violation for violation in result.violations), result.violations


def test_guard_accepts_grounded_values_written_differently(verified: Convo):
    facts = {"appeal_deadline": "2026-03-18", "days_until_deadline": 13, "allowed": "$1,450.00", "paid": "$1,425.00"}
    reply = "The deadline is the 18th of March, 13 days away. The gap is 25 dollars, and I haven't sent anything yet."
    assert _guard(verified, reply, facts).ok


def test_guard_accepts_a_fact_given_earlier_in_the_same_verified_call(verified: Convo):
    verified.say("how much was it for?")
    assert _guard(verified, "As I said, the allowed maximum is $1,450.00.").ok
    assert not _guard(verified, "As I said, the allowed maximum is $9,999.00.").ok


@pytest.mark.parametrize(
    "reply",
    ["I can see your claim was denied in January.", "Our records show a denial.", "I've verified you, thanks.",
     "Your identity checks out.", "Verification passed."],
)
def test_guard_rejects_claims_of_access_before_verification(convo: Convo, reply: str):
    directive = Directive(act="VERIFY_ASK", goal="", fallback="ok", facts={})
    assert not check_reply(reply, session=convo.s, directive=directive, data=convo.agent.data).ok


# ---------------------------------------------------------------- small things


@pytest.mark.parametrize(
    "text, redacted",
    [
        ("my ssn is 123-45-4472", "my ssn is [SSN ending 4472]"),
        ("123.45.4472", "[SSN ending 4472]"),
        ("123 45 4472", "[SSN ending 4472]"),
        ("123/45/4472", "[SSN ending 4472]"),
        ("social 123454472", "social [SSN ending 4472]"),
        ("my phone is 6505212836", "my phone is 6505212836"),
        ("order 123454472 arrived", "order 123454472 arrived"),
    ],
)
def test_full_ssns_are_masked_whatever_the_separator(text: str, redacted: str):
    assert redact_ssn(text) == redacted


def test_guidance_for_a_single_document_reads_correctly(agent: Agent):
    c = Convo(agent)
    c.say("ma tian, 1964-09-10, 6688")
    assert c.s.case_id == "CL-3001"
    reply = c.say("how do I submit it, and how will I know you got it?")
    assert "the diagnosis report stays attached" in reply and "the diagnosis report was attached correctly" in reply


def test_the_scripted_scenarios_meet_their_outcome_checks_offline():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    run = subprocess.run(
        [sys.executable, "scripts/live_check.py", "--offline", "--scenarios", "all", "--quiet"],
        cwd=root, capture_output=True, text=True, timeout=120, env={"SOP_SKIP_DOTENV": "1", "PATH": ""},
    )
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert "all checks passed" in run.stdout and "turns 36" in run.stdout


def test_the_demo_clock_follows_the_environment(monkeypatch):
    monkeypatch.delenv("SOP_TODAY", raising=False)
    assert clock_from_env()() == DEFAULT_DEMO_DATE
    monkeypatch.setenv("SOP_TODAY", "2026-03-17")
    assert clock_from_env()() == date(2026, 3, 17)
    monkeypatch.setenv("SOP_TODAY", "real")
    assert clock_from_env()() == date.today()
    monkeypatch.setenv("SOP_TODAY", "not a date")
    assert clock_from_env()() == DEFAULT_DEMO_DATE
