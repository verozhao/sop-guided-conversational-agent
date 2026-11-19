"""Regression tests for the second pre-submission review.

The second round looked for regressions from the first round of fixes and for
what those fixes had missed: role detection in any letter case, the output
guard in both directions, scope before and after verification, safer choices
at the email offer and at an offer of a person, and partial identity answers.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.insurance_claims import Agent
from apps.insurance_claims.guard import check_reply
from apps.insurance_claims.state import CLOSED, ESCALATED, POST_PROCESS, PROCESS_CASE, RESOLVE_INTENT, VERIFY_ID
from apps.insurance_claims.verification import verify_identity
from sop_harness.core import Directive
from sop_harness.llm import AnthropicLLM, LLMError, ScriptedLLM

from .conftest import DEMO_UTTERANCE, Convo, assert_no_leak
from .test_llm_path import _client, _Endpoint, _response, reading

THREE = "My name is Margaret Chen, DOB is 1985-03-15, SSN last four is 4472."
LISTED = "Margaret Chen, born 1985-03-15, phone 650-521-2836"
AVA = "My name is Ava Lopez, date of birth 1990-08-21, last four 9180."


def at_email_offer(agent: Agent) -> Convo:
    c = Convo(agent)
    c.run(DEMO_UTTERANCE, "No, that's all.")
    assert c.s.pending == "email_offer"
    return c


# ---------------------------------------------------------------- who is calling


@pytest.mark.parametrize(
    "first",
    [
        "i am margaret chen's husband",
        "I am the policyholder's son",
        "I'm the caregiver for Margaret Chen",
        "margaret chen is my mother, i'm calling for her",
        "this is her husband",
        "It's my wife's policy, not mine",
    ],
)
def test_a_third_party_is_recognised_however_it_is_typed(convo: Convo, first: str):
    convo.say(first)
    assert convo.s.memory.caller_role == "representative"
    reply = convo.say("margaret chen, 1985-03-15, 4472")
    assert not convo.s.verified and convo.act == "ASK_REP_NAME"
    assert_no_leak(reply)


def test_a_representative_who_gives_their_own_name_beside_the_relationship_is_not_asked_again(convo: Convo):
    convo.say("I'm Margaret Chen's son David Chen. DOB 1985-03-15, SSN last four 4472.")
    assert convo.s.memory.representative_name == "David Chen" and convo.act == "CONSENT_REQUESTED"


def test_repeating_the_policyholders_name_does_not_make_it_the_representatives_name(convo: Convo):
    convo.say("I'm calling for my mother Margaret Chen")
    convo.say("Margaret Chen, born 1985-03-15, phone 650-521-2836")
    assert convo.act == "ASK_REP_NAME" and convo.s.memory.representative_name == ""
    convo.say("David Chen")
    assert convo.act == "CONSENT_REQUESTED"


@pytest.mark.parametrize(
    "text",
    [
        "Margaret Chen. The hospital said their claim department sent it to you",
        "My husband told me to call about my claim. I'm Margaret Chen",
        "I'm Margaret Chen. My friend Jane Doe said I should call you",
    ],
)
def test_talking_about_someone_else_does_not_make_the_policyholder_a_representative(convo: Convo, text: str):
    convo.say(text)
    convo.say("1985-03-15, 4472")
    if convo.act == "ROLE_CHECK":
        convo.say("yes")
    assert convo.s.verified and convo.s.memory.caller_role != "representative"


# ---------------------------------------------------------------- the role question


@pytest.mark.parametrize(
    "lines",
    [
        ["Calling for Margaret. I'm David, the husband.", "Margaret Chen, 1985-03-15, 4472"],
        ["My wife asked me to call about a claim that was denied.", "Margaret Chen, 1985-03-15, 4472"],
        ["hi, my mom can't come to the phone", "Margaret Chen, 1985-03-15, 4472"],
        ["she asked me to call", "margaret chen, 1985-03-15, 4472"],
    ],
)
def test_nobody_who_mentions_someone_else_passes_as_the_policyholder_unasked(convo: Convo, lines: list[str]):
    reply = ""
    for line in lines:
        reply = convo.say(line)
        assert_no_leak(reply)
    assert not convo.s.verified and convo.act in {"ROLE_CHECK", "ASK_REP_NAME"}
    if convo.act == "ROLE_CHECK":
        reply = convo.say("no, I'm calling for her")
        assert convo.s.memory.caller_role == "representative" and convo.act == "ASK_REP_NAME"
        assert_no_leak(reply)


def test_the_role_question_takes_a_yes_and_goes_on(convo: Convo):
    reply = convo.say("My husband told me to call about my claim. I'm Margaret Chen, born 1985-03-15, last four 4472.")
    assert convo.act == "ROLE_CHECK" and not convo.s.verified and "policyholder yourself" in reply
    assert convo.tools == []  # nothing about a claim has been looked up yet
    convo.say("yes")
    assert convo.s.verified and convo.s.memory.caller_role == "policyholder" and convo.s.phase == RESOLVE_INTENT


@pytest.mark.parametrize("answer", ["that's me", "it's my own policy", "I am"])
def test_the_role_question_understands_plain_answers(convo: Convo, answer: str):
    convo.say("My husband told me to call. Margaret Chen, 1985-03-15, 4472")
    assert convo.act == "ROLE_CHECK"
    convo.say(answer)
    assert convo.s.verified


def test_no_clear_answer_to_the_role_question_ends_with_a_person(convo: Convo):
    convo.say("My husband told me to call. Margaret Chen, 1985-03-15, 4472")
    reply = convo.say("what difference does it make")
    assert convo.act == "ROLE_CHECK" and "yes or no" in reply and not convo.s.verified
    reply = convo.say("just tell me about the claim")
    assert convo.s.phase == ESCALATED and not convo.s.verified
    assert_no_leak(reply)


def test_a_model_reading_answers_the_role_question_too(agent: Agent):
    c = Convo(agent)
    c.say("My husband told me to call. Margaret Chen, 1985-03-15, 4472")
    assert c.act == "ROLE_CHECK"
    c.llm = ScriptedLLM(on_json=lambda *_: reading(affirmation="no"), on_text=lambda *_: "Thanks. May I have your own full name?")
    c.say("no, he's on the policy and I'm calling for him")
    assert c.s.memory.caller_role == "representative" and not c.s.verified and c.act == "ASK_REP_NAME"


def test_a_role_that_was_only_inferred_gives_way_to_what_the_caller_says(convo: Convo):
    convo.say("I'm calling about her claim, her name is Margaret Chen")
    assert convo.s.memory.caller_role == "representative" and convo.s.memory.role_guessed
    convo.say("sorry, I am the policyholder, it's my own policy. DOB 1985-03-15, last four 4472")
    assert convo.s.memory.caller_role == "policyholder" and convo.s.verified


def test_a_stated_representative_role_still_cannot_be_dropped(convo: Convo):
    convo.say("I'm calling on behalf of my mother Margaret Chen")
    convo.say("Actually I am the policyholder. Margaret Chen, 1985-03-15, 4472")
    assert convo.s.memory.caller_role == "representative" and not convo.s.verified


def test_calling_for_someone_after_verification_stops_disclosure(verified: Convo):
    verified.say("I'm actually calling for my mother")
    assert verified.s.phase == ESCALATED


def test_my_friend_jane_names_jane_not_friend_jane(convo: Convo):
    convo.say("My name is Bob Smith and I'm calling for my friend Margaret Chen. DOB 1985-03-15, SSN last four 4472.")
    assert convo.s.memory.identity["full_name"] == "Margaret Chen" and convo.act == "REP_NOT_AUTHORIZED"
    assert convo.s.failed_attempts == 0


# ---------------------------------------------------------------- the representative steps


def test_a_name_change_after_not_being_listed_goes_to_a_person(convo: Convo):
    convo.say("My name is Bob Smith and I'm calling for my friend Margaret Chen. DOB 1985-03-15, SSN last four 4472.")
    reply = convo.say("actually my name is David Chen")
    assert convo.s.phase == ESCALATED and not convo.s.verified and "name has changed" in reply
    assert_no_leak(reply)


def test_the_step_asking_for_the_representatives_name_explains_persuades_and_stops(convo: Convo):
    convo.say("I'm calling for my mother Margaret Chen. Her DOB is 1985-03-15 and her SSN last four is 4472.")
    assert convo.act == "ASK_REP_NAME"
    assert "listed on the policy" in convo.say("why do you need my name?") and convo.act == "ASK_REP_NAME"
    reply = convo.say("I'd rather not give my name")
    assert convo.act == "REP_NAME_PERSUADE" and "call us directly" in reply
    assert "first and last name" in convo.say("David")
    convo.say("no")
    assert convo.act == "REP_NAME_PERSUADE"
    convo.say("I refuse to give my name")
    assert convo.s.phase == ESCALATED and not convo.s.verified


def test_a_representative_is_never_told_which_details_matched(convo: Convo):
    convo.say("I'm calling on behalf of my mother, Margaret Chen. My name is David Chen.")
    convo.say("Her date of birth is 1985-03-15")
    reply = convo.say("her SSN last four is 1111")
    assert convo.act == "VERIFY_MISMATCH" and convo.s.failed_attempts == 1
    assert "confirmed" not in reply and "date of birth" not in reply and "last four" not in reply
    rules = " ".join(convo.trace["directive"]["must_not"])
    assert "Do not say whether the details given so far match" in rules
    assert "details_that_did_not_match" not in convo.trace["directive"]["facts"]["verification"]


# ---------------------------------------------------------------- asking for a person


@pytest.mark.parametrize(
    "text",
    [
        "I don't want to talk to a bot. Get me a human.",
        "I'm not talking to a robot, give me a real person",
        "Stop. I do not want to chat with an AI. I want a human.",
        "I don't want to keep going in circles, transfer me",
        "I don't want a bot I want a human",
    ],
)
def test_a_request_for_a_person_beside_a_complaint_about_the_bot_is_honoured(verified: Convo, text: str):
    verified.say(text)
    assert verified.s.phase == ESCALATED


@pytest.mark.parametrize("text", ["do I need to talk to a person?", "should I speak to a representative about this?", "I don't need a human"])
def test_asking_whether_a_person_is_needed_is_not_a_request_for_one(verified: Convo, text: str):
    verified.say(text)
    assert verified.s.phase == PROCESS_CASE and verified.s.handoff is None


@pytest.mark.parametrize("text", ["ok thanks bye", "okay, thank you, goodbye", "alright, I'll call back later", "sure, thanks anyway"])
def test_a_goodbye_that_starts_with_ok_does_not_accept_an_offer_of_a_person(convo: Convo, text: str):
    convo.say(AVA)
    assert convo.act == "NO_CLAIMS" and convo.s.pending == "human_offer"
    convo.say(text)
    assert convo.s.phase == POST_PROCESS and convo.s.handoff is None  # leaving, not asking to be connected


def test_a_plain_yes_still_accepts_the_offer(convo: Convo):
    for text in ("yes", "yes please connect me", "sure, a person please"):
        c = Convo(convo.agent)
        c.say(AVA)
        c.say(text)
        assert c.s.phase == ESCALATED, text


def test_a_model_reading_of_yes_and_done_does_not_transfer(agent: Agent):
    c = Convo(agent)
    c.say(AVA)
    c.llm = ScriptedLLM(on_json=lambda *_: reading(affirmation="yes", done=True), on_text=lambda *_: "Take care.")
    c.say("ok thanks, bye")
    assert c.s.phase != ESCALATED


def test_declining_a_person_after_unrelated_questions_goes_back_to_the_claims(convo: Convo):
    convo.run(THREE, "what is RL?", "who won the world cup?", "tell me a joke")
    assert convo.act == "OOS_OFFER_HUMAN"
    reply = convo.say("no")
    assert convo.act == "ASK_WHICH_CLAIM" and convo.s.phase == RESOLVE_INTENT and "CL-2048" in reply
    convo.say("the denied one")
    assert convo.s.case_id == "CL-2048"


def test_declining_a_person_after_cant_do_that_stays_on_the_claim(verified: Convo):
    verified.say("can you approve it?")
    assert verified.s.pending == "human_offer"
    verified.say("fine, forget it")
    assert verified.s.phase == PROCESS_CASE and verified.s.handoff is None


# ---------------------------------------------------------------- scope


def test_short_reactions_after_an_answer_are_never_declined(verified: Convo):
    for text in ("are you sure?", "how so?", "since when?"):
        reply = verified.say(text)
        assert verified.act == "ANSWER" and verified.s.oos_streak == 0 and "denied because" in reply, text
    assert verified.s.phase == PROCESS_CASE


@pytest.mark.parametrize("text", ["what was the date of service?", "What does coinsurance mean?", "what was the procedure code?"])
def test_claim_questions_with_no_answer_on_file_get_an_honest_reply(verified: Convo, text: str):
    reply = verified.say(text)
    assert verified.s.oos_streak == 0 and "not something I can do or look up" in reply


def test_when_was_it_filed_is_answered_from_the_claim(verified: Convo):
    assert "filed on January 12, 2026" in verified.say("When was it filed?")


def test_after_verification_what_happens_next_is_about_the_claim(verified: Convo):
    reply = verified.say("what happens next?")
    assert "March 18, 2026" in reply and "private health" not in reply
    reply = verified.say("how does this work?")
    assert verified.act == "PROCESS_EXPLAIN" and "why a decision was made" in reply and "last four digits" not in reply


@pytest.mark.parametrize(
    "text, expect",
    [
        ("Which three do I have to give?", "three details"),
        ("how long will this take", "two or three short answers"),
        ("Is my policy number enough?", "doesn't count as one of the three"),
        ("what is this for?", "private health and financial"),
        ("what's the point of this?", "private health and financial"),
    ],
)
def test_more_questions_about_the_step_get_their_own_answer(convo: Convo, text: str, expect: str):
    reply = convo.say(text)
    assert convo.act == "VERIFY_EXPLAIN" and expect in reply and convo.s.oos_streak == 0


@pytest.mark.parametrize(
    "text",
    ["Please approve my claim", "Can you expedite my claim?", "Can you update my phone number?", "Can you pay me today?", "I want to file a new claim"],
)
def test_requests_to_do_something_get_an_offer_of_a_person_not_the_denial_reason(verified: Convo, text: str):
    reply = verified.say(text)
    assert verified.s.pending == "human_offer" and "denied because" not in reply and reply.endswith("Would you like me to connect you?")


# ---------------------------------------------------------------- picking a claim


@pytest.mark.parametrize(
    "text, claim",
    [("the november one", "CL-1899"), ("February 28", "CL-2102"), ("the one filed in February", "CL-2102"), ("November", "CL-1899")],
)
def test_a_month_on_its_own_answers_which_claim(convo: Convo, text: str, claim: str):
    convo.say(LISTED)
    convo.say(text)
    assert convo.s.case_id == claim


@pytest.mark.parametrize("text", ["the 2026 one", "the newer one", "the more recent one"])
def test_a_year_or_newer_picks_between_two_claims(convo: Convo, text: str):
    convo.run(LISTED, "the healthcare one")
    assert convo.act == "ASK_WHICH_CLAIM"
    convo.say(text)
    assert convo.s.case_id == "CL-2048"


def test_unclear_answers_do_not_turn_into_not_on_file(convo: Convo):
    convo.say(LISTED)
    convo.say("the blue one")
    assert convo.act == "ASK_WHICH_CLAIM"
    reply = convo.say("you know which one")
    assert convo.act == "ASK_WHICH_CLAIM" and "claim number" in reply
    reply = convo.say("come on")
    assert convo.act == "CLAIM_UNCLEAR" and "isn't one of the claims" not in reply
    convo.say("no")
    assert convo.act == "ASK_WHICH_CLAIM" and convo.s.phase == RESOLVE_INTENT  # not pushed to the wrap-up


@pytest.mark.parametrize("text", ["no, I meant the denied one", "go back to the denied one"])
def test_correcting_the_claim_by_its_status_switches(convo: Convo, text: str):
    convo.run(LISTED, "the dental claim")
    assert convo.s.case_id == "CL-1899"
    convo.say(text)
    assert convo.s.case_id == "CL-2048" and convo.s.phase == PROCESS_CASE


@pytest.mark.parametrize("text", ["it's not about my auto claim", "I also have a dental claim but that's fine"])
def test_a_claim_mentioned_only_to_set_it_aside_does_not_switch(verified: Convo, text: str):
    verified.say(text)
    assert verified.s.case_id == "CL-2048"


def test_the_other_claim_when_there_is_only_one(convo: Convo):
    convo.say("My name is Ma Tian, date of birth 1964-09-10, last four 6688.")
    reply = convo.say("what about my other claim?")
    assert convo.s.case_id == "CL-3001" and "CL-3001 is the only claim on this policy" in reply


# ---------------------------------------------------------------- the email offer


@pytest.mark.parametrize("text", ["ok bye", "okay, thanks, bye", "alright, goodbye", "fine. I need to go", "I don't have email"])
def test_someone_leaving_at_the_email_offer_is_not_sent_an_email(agent: Agent, text: str):
    c = at_email_offer(agent)
    c.say(text)
    assert c.act == "EMAIL_SKIPPED" and not c.s.outbox and c.s.phase == CLOSED


@pytest.mark.parametrize("text", ["send it later", "yes, later", "ok, maybe tomorrow"])
def test_later_is_not_a_yes_for_now(agent: Agent, text: str):
    c = at_email_offer(agent)
    c.say(text)
    assert not c.s.outbox


@pytest.mark.parametrize(
    "text", ["yes, but send it to my work email", "ok but use my new email address", "sure, to a different address though"]
)
def test_a_yes_with_another_address_is_not_a_yes(agent: Agent, text: str):
    c = at_email_offer(agent)
    reply = c.say(text)
    assert c.act == "EMAIL_ADDRESS_FIXED" and not c.s.outbox and "address on file" in reply


def test_another_address_is_refused_whatever_the_model_reads(agent: Agent):
    c = at_email_offer(agent)
    c.llm = ScriptedLLM(on_json=lambda *_: reading(email_summary_choice="send", affirmation="yes"), on_text=lambda *_: "Sent!")
    c.say("yes, but send it to my work email")
    assert c.act == "EMAIL_ADDRESS_FIXED" and not c.s.outbox


@pytest.mark.parametrize("text", ["that would be great", "I'd like that", "go for it", "ok thanks", "sure, thanks"])
def test_plain_acceptances_send(agent: Agent, text: str):
    c = at_email_offer(agent)
    c.say(text)
    assert c.act == "EMAIL_SENT" and len(c.s.outbox) == 1


@pytest.mark.parametrize(
    "text", ["what would be in it?", "what does the summary include?", "where does it go?", "What does the email contain?", "what's included?"]
)
def test_any_question_at_the_offer_is_about_the_summary(agent: Agent, text: str):
    c = at_email_offer(agent)
    reply = c.say(text)
    assert c.act == "EMAIL_OFFER_REPEAT" and c.s.phase == POST_PROCESS and c.s.oos_streak == 0
    assert "what we discussed" in reply and not c.s.outbox
    c.say("yes")
    assert c.act == "EMAIL_SENT"


def test_the_summary_lists_only_guidance_that_was_given(convo: Convo):
    convo.say(THREE)
    reply = convo.say("what are the next steps for my auto claim?")
    assert reply.count("no documents") + reply.count("No documents") == 1
    convo.run("that's all", "yes please")
    body = convo.s.outbox[0]["body"]
    assert "The appeal and next steps (claim CL-2102)" in body and "When the documents need to be submitted" not in body


# ---------------------------------------------------------------- finishing


@pytest.mark.parametrize("text", ["I don't have any other questions", "No, I don't have any more questions", "I don't have anything else"])
def test_having_no_more_questions_is_a_closing_not_a_missing_document(verified: Convo, text: str):
    reply = verified.say(text)
    assert verified.act == "EMAIL_OFFER" and "replacement copy" not in reply


@pytest.mark.parametrize("text", ["I can't get the pathology report", "I don't have it anymore"])
def test_a_missing_document_still_gets_the_alternatives(verified: Convo, text: str):
    assert "replacement copy" in verified.say(text)


# ---------------------------------------------------------------- identity details


def test_a_day_first_date_is_read_as_a_date(convo: Convo):
    convo.run("Ava Lopez", "21/08/1990")
    assert convo.s.verification["matched"] == ["full_name", "dob"] and convo.s.oos_streak == 0


@pytest.mark.parametrize(
    "text, note",
    [("3/15", "full date"), ("March 15", "full date"), ("born in 1985", "full date"), ("650-521-283", "looks incomplete"),
     ("1985", "year of birth or the last four digits")],
)
def test_a_detail_given_in_part_is_neither_counted_nor_declined(convo: Convo, text: str, note: str):
    convo.say("Margaret Chen")
    reply = convo.say(text)
    assert convo.act == "VERIFY_PROGRESS" and note in reply
    assert convo.s.failed_attempts == 0 and convo.s.oos_streak == 0 and convo.s.verification["matched"] == ["full_name"]


def test_a_name_given_in_two_messages_is_one_name(convo: Convo):
    convo.say("Margaret")
    assert convo.s.verification["incomplete"] == ["full_name"]
    convo.say("Chen")
    assert convo.s.verification["matched"] == ["full_name"]
    other = Convo(convo.agent)
    other.run("Hi this is Margaret", "my last name is Chen")
    assert other.s.verification["matched"] == ["full_name"]


def test_lower_case_and_unpunctuated_details_are_read(convo: Convo):
    convo.say("hi this is margaret chen")
    assert convo.s.verification["matched"] == ["full_name"]
    other = Convo(convo.agent)
    other.say("margaret chen 1985-03-15 4472")
    assert other.s.verified


def test_a_first_name_reminder_is_not_repeated_on_every_turn(convo: Convo):
    assert "first and last name" in convo.say("I'm Margaret")
    assert "first and last name" not in convo.say("hmm")
    convo.say("never mind, goodbye")
    assert convo.act == "VERIFY_GOODBYE"


@pytest.mark.parametrize("text", ["my husband's birthday is 1980-01-01", "call me back at 650-000-0000", "my doctor's number is 650-111-2222"])
def test_a_value_that_belongs_to_someone_else_is_not_an_identity_attempt(convo: Convo, text: str):
    convo.say("Margaret Chen")
    convo.say(text)
    assert convo.s.failed_attempts == 0 and convo.s.verification["matched"] == ["full_name"]


def test_a_stray_value_keeps_earlier_matches_and_counts_as_wrong(convo: Convo):
    convo.say("date of birth 1985-03-15")
    reply = convo.say("SSN last four 6688")  # another policyholder's
    assert convo.act == "VERIFY_MISMATCH" and convo.s.failed_attempts == 1 and "confirmed your date of birth" in reply
    convo.run("policy POL-0001", "policy POL-0002")
    assert convo.s.phase == ESCALATED  # wrong policy numbers are counted in a tie as well


def test_a_tie_without_history_asks_for_more_and_counts_nothing(agent: Agent):
    result = verify_identity(agent.data, {"dob": "1985-03-15", "id_last4": "6688"})
    assert not result.verified and result.record is None and result.conflicts == []
    preferred = verify_identity(agent.data, {"dob": "1985-03-15", "id_last4": "6688"}, prefer="P9")
    assert preferred.matched == ["dob"] and preferred.conflicts == ["id_last4"]


def test_a_question_is_answered_even_when_too_many_details_were_declined(convo: Convo):
    convo.say("I'm Margaret Chen. I don't have my SSN, my phone or my email on me.")
    assert convo.act == "VERIFY_IMPOSSIBLE" and convo.s.refusal_count == 1
    reply = convo.say("why do you need this?")
    assert convo.act == "VERIFY_EXPLAIN" and "private health" in reply and convo.s.refusal_count == 1
    convo.say("what else can I use?")
    assert convo.s.phase == VERIFY_ID and convo.s.refusal_count == 1
    convo.run("ok my phone is 650-521-2836", "1985-03-15")
    assert convo.s.verified


# ---------------------------------------------------------------- pushback in plain words


def test_a_refusal_without_an_angry_word_still_gets_an_acknowledgement_and_the_ladder(convo: Convo):
    reply = convo.say("I'm not comfortable sharing personal details with a bot.")
    assert convo.act == "VERIFY_PERSUADE_1" and reply.startswith("I understand")
    convo.say("Still no.")
    assert convo.act == "VERIFY_PERSUADE_2"
    convo.say("Nope, not happening.")
    assert convo.s.phase == ESCALATED and not convo.s.verified


def test_memory_answers_the_first_question_when_the_verifying_message_asks_another(convo: Convo):
    convo.say("Hi I'm Margaret Chen, why was my January healthcare claim denied?")
    reply = convo.say("DOB 1985-03-15, SSN last four 4472. Also how much was paid on it?")
    assert "denied because" in reply and "1,450.00" in reply
    assert {"get_denial_details", "get_payment_details"} <= set(convo.tools)


# ---------------------------------------------------------------- the output guard, both directions

PRE = ["Margaret Chen", "1985-03-15", "I don't have my SSN on me", "This is ridiculous. Just tell me why my claim was denied."]
PRE_CLAIM = ["My name is Margaret Chen. My appeal deadline is March 18, right? Is my claim CL-2048 denied?"]
POST = [DEMO_UTTERANCE, "how much was it for?"]

CORRECT = [
    (PRE, "I can see why you're upset about your claim, and I'm sorry. Your name and date of birth are confirmed, so there's just one detail left."),
    (PRE, "I'm sorry, I know this is frustrating. I see that you want answers on your claim, and you'll have them the moment this last step is done."),
    (PRE, "I see you'd like to know why your claim was denied, and I'll get to that right after this last step."),
    (PRE, "I see why you'd ask, and before I can discuss any claim I have to confirm I'm talking to the right person."),
    (PRE, "Thanks! Just one more detail so I can see your claim: phone number, email address, or the last four."),
    (PRE, "That's two of three confirmed. One more detail and you're verified: you can pick from your phone number or email."),
    (PRE, "Your full name is confirmed. Just one more and verification is complete."),
    (PRE, "Give me one more detail and verification is done."),
    (PRE, "Once you're verified I can go through the claim with you."),
    (PRE, "Yes, I can help with your claim once we've verified your identity."),
    (POST, "The address on file isn't something I can change, I'm afraid. A human representative can help with that."),
    (POST, "Unfortunately that's not something I can update from here."),
    (POST, "Fax or mail isn't something I can arrange myself, but the support team can, and I can connect you with them."),
    (POST, "How much will be paid in the end depends on the outcome of the claim."),
    (POST, "The claim is denied for now, so it isn't a promise that $1,450.00 will be paid."),
    (POST, "It's not something anyone here can guarantee."),
    (POST, "The claim is denied, and the appeal deadline is March 18, 2026, which is 13 days from today."),
    (POST, "I can't say whether it will be approved."),
    (POST, "Nothing has been sent yet. Would you like the summary?"),
    (["Margaret Chen, 1985-03-15, 4472"], "You're all set. I see an auto claim in progress, filed February 28, 2026 (CL-2102), and three others. Which one?"),
    (["Margaret Chen, 1985-03-15, 4472", "the dental one", "how much did you pay?"], "The expected reimbursement of $425.00 was paid in full."),
]
WRONG = [
    (PRE_CLAIM, "Thanks Margaret. After checking those details, you're verified."),
    (PRE_CLAIM, "Thank you. When I checked the name, everything matched and you are now verified."),
    (PRE_CLAIM, "Great news, Margaret: after a quick check your identity is confirmed, so we can talk about your claim."),
    (PRE_CLAIM, "If I may say so, verification is complete. What can I do for you?"),
    (PRE_CLAIM, "You’re verified! How can I help?"),
    (PRE_CLAIM, "I’m transferring you to a representative now."),
    (PRE_CLAIM, "I’ve sent you a verification code."),
    (PRE_CLAIM, "Great, that's everything. Verification complete!"),
    (PRE_CLAIM, "Thank you, Margaret, I was able to verify you."),
    (PRE_CLAIM, "You've passed the security check, so go ahead and ask me about your claim."),
    (PRE_CLAIM, "Yes, I do have a denied claim on file for you."),
    (PRE_CLAIM, "Thanks Margaret. Your healthcare claim is still under review."),
    (PRE_CLAIM, "Yes, claim CL-2048 was denied."),
    (PRE_CLAIM, "Yes, your appeal deadline is March 18."),
    (PRE_CLAIM, "I can see your claim was denied in January."),
    (POST, "I've reopened the claim and flagged it for priority review."),
    (POST, "I've set up a fax submission for you."),
    (POST, "I'm going to file the appeal for you right now."),
    (POST, "We'll waive the deadline for you."),
    (POST, "Let me transfer you to a representative now."),
    (POST, "I'll connect you to a representative right now, please hold."),
    (POST, "I'll make sure your appeal gets approved."),
    (POST, "Your claim was approved yesterday and payment is scheduled."),
    (POST, "I've gone ahead and emailed you the summary."),
    (POST, "Your summary has just been emailed to m***@email.com."),
    (POST, "You have until the 28th to appeal."),
    (POST, "The appeal deadline is next Friday at 5 pm."),
    (POST, "The review takes 24 hours."),
    (POST, "You can upload them at www.claims-portal.com/upload."),
    (POST, "Call the claims team on 555-0142 to arrange a fax."),
    (POST, "The plan covers 80% of that."),
    (POST, "Once you send the report, your claim will be approved."),
    (POST, "If you send the documents today, your claim will be approved within 48 hours."),
    (POST, "Once you send the documents you'll definitely get paid."),
    (POST, "Email sent! Take care."),
    (POST, "I went ahead and extended your appeal deadline."),
    (POST, "Let me expedite that for you right away."),
]


def _verdict(agent: Agent, lines: list[str], reply: str):
    c = Convo(agent)
    c.run(*lines)
    directive = Directive(act="X", goal="", fallback=c.last["reply"], facts=c.trace["directive"]["facts"])
    return check_reply(reply, session=c.s, directive=directive, data=agent.data)


@pytest.mark.parametrize("lines, reply", CORRECT, ids=[reply[:48] for _lines, reply in CORRECT])
def test_guard_lets_a_natural_correct_reply_through(agent: Agent, lines: list[str], reply: str):
    verdict = _verdict(agent, lines, reply)
    assert verdict.ok, verdict.violations


@pytest.mark.parametrize("lines, reply", WRONG, ids=[reply[:48] for _lines, reply in WRONG])
def test_guard_stops_a_false_claim_however_it_is_worded(agent: Agent, lines: list[str], reply: str):
    assert not _verdict(agent, lines, reply).ok


def test_guard_accepts_the_consent_turn_of_the_representative_script(agent: Agent):
    c = Convo(agent)
    c.run("Hi, this is David Chen, I'm calling on behalf of my wife Margaret Chen. Her DOB is 1985-03-15 and her SSN ends in 4472.")
    assert c.act == "CONSENT_REQUESTED"
    c.say("OK, she just approved it.")
    assert c.s.verified
    directive = Directive(act="X", goal="", fallback=c.last["reply"], facts=c.trace["directive"]["facts"])
    for reply in ("Thanks, David. Margaret's consent has been approved, so I can help you now.", "Margaret approved the request I sent, so we can go ahead."):
        assert check_reply(reply, session=c.s, directive=directive, data=agent.data).ok, reply


# ---------------------------------------------------------------- server, console, metering


@pytest.fixture
def client() -> TestClient:
    from server.app import app

    return TestClient(app)


def _start(client: TestClient) -> str:
    return client.post("/api/session", json={}).json()["session_id"]


def test_bad_requests_get_one_true_sentence(client: TestClient):
    sid = _start(client)
    blank = client.post("/api/chat", json={"session_id": sid, "message": "   "})
    assert blank.status_code == 422 and blank.json()["detail"] == "The message is empty."
    number = client.post("/api/chat", json={"session_id": sid, "message": 5})
    assert number.json()["detail"] == "The message must be text."
    broken = client.post("/api/chat", content="{not json", headers={"Content-Type": "application/json"})
    assert broken.status_code == 422 and broken.json()["detail"] == "The request body is not valid JSON."
    missing = client.post("/api/chat", json={})
    assert "session_id is required." in missing.json()["detail"] and "message is required." in missing.json()["detail"]


def test_the_notice_says_what_actually_produced_the_reply(client: TestClient, monkeypatch):
    import server.app as server

    def refuse(*_args):
        raise LLMError("refusal", "The model declined this request.")

    monkeypatch.setattr(server, "get_anthropic_llm", lambda *a, **k: ScriptedLLM(on_json=refuse, on_text=lambda *_: "Could you give me your date of birth?"))
    out = client.post("/api/chat", json={"session_id": _start(client), "message": "My name is Margaret Chen"}, headers={"X-API-Key": "k"}).json()
    assert out["trace"]["reply_source"] == "model" and out["engine"] != "offline"
    assert "offline rules read it instead" in out["notice"] and "still written by the model" in out["notice"]
    assert "ran on the offline engine" not in out["notice"]


def test_a_hand_off_keeps_its_phase_when_the_inspector_is_off(client: TestClient, monkeypatch):
    monkeypatch.setenv("SOP_INSPECTOR", "off")
    sid = _start(client)
    state = client.post("/api/chat", json={"session_id": sid, "message": "I want a human", "engine": "offline"}).json()["state"]
    assert state["phase"] == "ESCALATED" and state["handoff"]["phase_at_handoff"] == "VERIFY_ID"
    assert set(state["handoff"]) == {"ticket_id", "phase_at_handoff"}


def test_a_billed_call_that_ends_in_an_error_carries_its_usage():
    truncated = _response("partial", stop_reason="max_tokens")
    endpoint = _Endpoint(truncated)
    llm = AnthropicLLM(api_key="k", model="claude-haiku-5-5", client=_client(endpoint, endpoint))
    with pytest.raises(LLMError) as caught:
        llm.complete_text(system="s", user="u")
    assert caught.value.kind == "truncated" and caught.value.result is not None
    assert caught.value.result.output_tokens == truncated.usage.output_tokens
    endpoint = _Endpoint(_response("not json"))
    llm = AnthropicLLM(api_key="k", model="claude-haiku-5-5", client=_client(endpoint, endpoint))
    with pytest.raises(LLMError) as caught:
        llm.complete_json(system="s", user="u", schema={"type": "object"})
    assert caught.value.result is not None


def test_the_spend_meter_counts_calls_that_failed_after_being_billed():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import live_check

    class Failing:
        name, model = "llm:x", "claude-opus-5-5"

        def complete_text(self, **_kwargs):
            raise LLMError("truncated", "cut off", live_check.LLMResult(text="", model=self.model, input_tokens=3000, output_tokens=16000))

    meter = live_check.Meter(cap=0.50)
    metered = live_check.MeteredLLM(Failing(), meter)
    with pytest.raises(LLMError):
        metered.complete_text(system="s", user="u")
    assert meter.spent > 0.30 and not meter.exhausted
    with pytest.raises(LLMError):
        metered.complete_text(system="s", user="u")
    assert meter.exhausted
    with pytest.raises(LLMError) as caught:
        metered.complete_text(system="s", user="u")
    assert caught.value.kind == "budget"


# ---------------------------------------------------------------- gaps the mutation run found


@pytest.mark.parametrize(
    "text",
    [
        "I don't understand any of this, can I speak to someone?",
        "This is not working, let me talk to a person",
        "I'm not happy with this, connect me to a manager",
        "I never got an answer, get me a supervisor",
        "I'm not happy connect me to a manager",
    ],
)
def test_a_request_for_a_person_after_a_complaint_is_honoured_in_any_phase(agent: Agent, text: str):
    before = Convo(agent)
    before.say(text)
    assert before.s.phase == ESCALATED, "before verification"
    after = Convo(agent)
    after.run(DEMO_UTTERANCE, text)
    assert after.s.phase == ESCALATED, "after verification"


def test_the_policy_number_never_counts_as_one_of_the_three(convo: Convo):
    result = verify_identity(convo.agent.data, {"policy_number": "POL-9921", "full_name": "Margaret Chen", "dob": "1985-03-15"})
    assert not result.verified and result.matched == ["full_name", "dob"]
    reply = convo.say("Margaret Chen, policy POL-9921, date of birth 1985-03-15")
    assert not convo.s.verified and convo.s.phase == VERIFY_ID and "one more detail" in reply
    assert_no_leak(reply, allow=("POL-",))


def test_the_policy_number_breaks_a_tie_and_an_unknown_one_is_wrong(agent: Agent):
    tied = verify_identity(agent.data, {"dob": "1985-03-15", "id_last4": "6688"})
    assert tied.record is None
    pinned = verify_identity(agent.data, {"dob": "1985-03-15", "id_last4": "6688", "policy_number": "POL-8836"})
    assert pinned.record.name == "Ma Tian" and pinned.matched == ["id_last4"] and pinned.conflicts == ["dob"]
    alone = verify_identity(agent.data, {"policy_number": "POL-0000"})
    assert alone.conflicts == ["policy_number"] and not alone.verified


@pytest.mark.parametrize(
    "text",
    ["Send it? No, skip it.", "please skip it, there is no reason to send it", "why would you send it?", "who would you send it to?"],
)
def test_send_and_skip_in_one_message_or_a_question_about_sending_sends_nothing(agent: Agent, text: str):
    c = at_email_offer(agent)
    c.say(text)
    assert not c.s.outbox and c.s.phase == POST_PROCESS and "send_email_summary" not in c.tools


def test_asking_for_the_summary_again_after_it_was_sent(agent: Agent):
    c = at_email_offer(agent)
    c.say("yes please")
    assert len(c.s.outbox) == 1
    for text in ("can you send it again?", "please resend the summary"):
        reply = c.say(text)
        assert c.act == "EMAIL_ALREADY_SENT" and "already gone to" in reply and len(c.s.outbox) == 1
        assert c.s.phase == CLOSED


@pytest.mark.parametrize("text", ["its margaret", "fair enough", "makes sense", "Speak English", "Cell Number", "oh boy"])
def test_a_short_reply_is_not_even_read_as_a_name(convo: Convo, text: str):
    convo.say(text)
    assert "full_name" not in convo.trace["understanding"]["identity"]
    assert convo.act != "VERIFY_NAME_NOT_FOUND" and convo.s.name_misses == 0


def test_guessed_names_are_dropped_a_few_times_and_then_not_tried_at_all(convo: Convo):
    for text in ("jane doe", "john smith", "mary jones"):
        convo.say(text)
        assert convo.act == "VERIFY_ASK"
    assert convo.s.name_misses == 3 and convo.s.failed_attempts == 0
    convo.run("peter pan", "wendy darling", "captain hook", "tiger lily")
    assert convo.s.failed_attempts == 0 and convo.s.phase == VERIFY_ID  # further guesses neither probe nor count
    convo.say("margaret chen")
    assert "full_name" not in convo.s.memory.identity  # the allowance is spent, so even a right guess is not tried
    convo.say("My name is Margaret Chen")
    assert convo.s.verification["matched"] == ["full_name"]


def test_a_reply_read_as_a_name_cannot_replace_a_confirmed_name(convo: Convo):
    convo.say("My name is Margaret Chen")
    convo.say("Blue Cross")  # capitalised, so it is read as a name
    assert convo.trace["understanding"]["identity"] == {"full_name": "Blue Cross"}
    assert convo.s.memory.identity["full_name"] == "Margaret Chen" and convo.s.failed_attempts == 0
    assert convo.s.verification["matched"] == ["full_name"]


@pytest.mark.parametrize(
    "reply, problem",
    [
        ("Thanks. Is your policy number POL-9921?", "ungrounded policy number"),
        ("Thanks. Does your SSN end in 4472?", "identity number from the records"),
        ("I can't go into detail yet, but there is an auto claim in progress.", "claim summary that was not provided"),
        ("Is this about the pathology report and office note?", "requested document"),
        ("", "empty reply"),
        ("word " * 500, "reply too long"),
    ],
)
def test_guard_rules_for_values_held_on_file(convo: Convo, reply: str, problem: str):
    convo.say("My name is Margaret Chen")
    directive = Directive(act="VERIFY_PROGRESS", goal="", fallback="ok", facts={})
    verdict = check_reply(reply, session=convo.s, directive=directive, data=convo.agent.data)
    assert not verdict.ok and any(problem in violation for violation in verdict.violations), verdict.violations


def test_guard_ignores_earlier_facts_until_the_caller_is_verified(convo: Convo):
    convo.say("My name is Margaret Chen")
    convo.s.grounded.append('{"amount": "$1,450.00", "claim": "CL-2048"}')  # must not count before verification
    directive = Directive(act="VERIFY_PROGRESS", goal="", fallback="ok", facts={})
    verdict = check_reply("Your claim CL-2048 was for $1,450.00.", session=convo.s, directive=directive, data=convo.agent.data)
    assert not verdict.ok and any("ungrounded" in violation for violation in verdict.violations)


def test_declining_too_many_details_three_times_ends_with_a_person(convo: Convo):
    convo.say("I'm Margaret Chen. I don't have my SSN, my phone or my email on me.")
    assert convo.act == "VERIFY_IMPOSSIBLE"
    convo.say("I told you, I don't have them")
    assert convo.act == "VERIFY_IMPOSSIBLE" and convo.s.phase == VERIFY_ID
    convo.say("no, I won't give those")
    assert convo.s.phase == ESCALATED and not convo.s.verified


def test_no_at_the_claim_question_is_taken_as_not_on_file(convo: Convo):
    convo.say(LISTED)
    reply = convo.say("no")
    assert convo.act == "NOT_ON_FILE" and "CL-" not in reply and convo.s.pending == "human_offer"


def test_each_token_gets_its_own_model_client():
    from sop_harness.llm import get_anthropic_llm

    first = get_anthropic_llm("token-one", model="claude-haiku-5-5", effort="low")
    again = get_anthropic_llm("token-one", model="claude-haiku-5-5", effort="low")
    other = get_anthropic_llm("token-two", model="claude-haiku-5-5", effort="low")
    assert first is again and first is not other


def test_sessions_do_not_share_state_and_old_ones_expire(client: TestClient, monkeypatch):
    import server.app as server

    one, two = _start(client), _start(client)
    client.post("/api/chat", json={"session_id": one, "message": DEMO_UTTERANCE, "engine": "offline"})
    assert client.get(f"/api/session/{one}").json()["state"]["verified"] is True
    assert client.get(f"/api/session/{two}").json()["state"]["verified"] is False
    monkeypatch.setattr(server, "SESSION_TTL_SECONDS", -1)
    _start(client)  # starting a conversation prunes the expired ones
    assert client.get(f"/api/session/{one}").status_code == 404


def test_nothing_on_file_is_shown_in_the_inspector_before_consent(agent: Agent):
    c = Convo(agent, scenario="timeout")
    c.say("I'm calling for my wife Margaret Chen, her DOB is 1985-03-15, SSN last four 4472. My name is David Chen.")
    state = agent.public_state(c.s)
    assert state["verified"] is False and state["caller"]["policy_number"] == "" and state["caller"]["email_on_file"] == ""
    assert state["case_id"] is None and "get_claim_summary" not in state["allowed_tools"]
