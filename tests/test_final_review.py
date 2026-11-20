"""Regression tests for the last review pass, which checked the second round of fixes.

Its main findings were in the role question added in that round (a "no" read as a yes, and
the question missing on the model path), two regressions (a claim year counted as an ID
number, a bare claim switch answered with "can't do that"), and the output guard's softening
rules in both directions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from apps.insurance_claims import Agent
from apps.insurance_claims.guard import check_reply
from apps.insurance_claims.state import ESCALATED, PROCESS_CASE, RESOLVE_INTENT
from sop_harness.core import Directive
from sop_harness.llm import ScriptedLLM

from .conftest import DEMO_UTTERANCE, Convo, assert_no_leak
from .test_llm_path import reading

ROOT = Path(__file__).resolve().parent.parent
MENTIONS_WIFE = "Hi, my wife is in hospital. Margaret Chen, 1985-03-15, SSN last four 4472. Why was the claim denied?"


# ---------------------------------------------------------------- the role question


@pytest.mark.parametrize(
    "answer", ["No, I am not", "I'm not the policyholder", "no", "not me, I'm calling for her", "No. I am her husband.", "it's not mine"]
)
def test_a_no_to_the_role_question_never_opens_the_gate(convo: Convo, answer: str):
    convo.say(MENTIONS_WIFE)
    assert convo.act == "ROLE_CHECK" and not convo.s.verified
    reply = convo.say(answer)
    assert not convo.s.verified and convo.s.memory.caller_role == "representative" and convo.act == "ASK_REP_NAME"
    assert_no_leak(reply)


@pytest.mark.parametrize("answer", ["yes", "Yes, I am", "I am", "that's me", "I'm the policyholder"])
def test_a_clear_yes_to_the_role_question_goes_on(convo: Convo, answer: str):
    convo.say(MENTIONS_WIFE)
    convo.say(answer)
    assert convo.s.verified and convo.s.memory.caller_role == "policyholder"


def test_an_answer_that_is_neither_yes_nor_no_is_asked_again(convo: Convo):
    convo.say(MENTIONS_WIFE)
    reply = convo.say("yes, why wouldn't I be?")
    assert convo.act == "ROLE_CHECK" and not convo.s.verified and "yes or no" in reply


def test_a_no_wins_even_when_a_model_also_reads_policyholder(agent: Agent):
    c = Convo(agent)
    c.say(MENTIONS_WIFE)
    c.llm = ScriptedLLM(
        on_json=lambda *_: reading(caller_role="policyholder", affirmation="no"),
        on_text=lambda *_: "Thank you. May I have your own full name?",
    )
    reply = c.say("No, I am not")
    assert not c.s.verified and c.s.memory.caller_role == "representative"
    assert_no_leak(reply)


def test_the_role_question_is_asked_on_the_model_path_too(agent: Agent):
    details = reading(identity={"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"})
    c = Convo(agent, llm=ScriptedLLM(on_json=lambda *_: details, on_text=lambda *_: "Thank you. Are you the policyholder yourself?"))
    reply = c.say("My husband told me to call. Margaret Chen, 1985-03-15, 4472")
    assert c.act == "ROLE_CHECK" and not c.s.verified and c.tools == []
    assert_no_leak(reply)
    plain = Convo(agent, llm=ScriptedLLM(on_json=lambda *_: details, on_text=lambda *_: "Thank you, you're verified. Which claim?"))
    plain.say("Margaret Chen, 1985-03-15, 4472")
    assert plain.s.verified  # nobody else was mentioned, so nothing is asked


def test_my_wife_asked_me_to_call_gets_the_role_question_as_the_readme_says(convo: Convo):
    convo.say("Hi, my wife asked me to call about my denied claim")
    assert convo.s.memory.caller_role == "unknown"
    reply = convo.say("I'm Ma Tian, born 1964-09-10, last four 6688")
    assert convo.act == "ROLE_CHECK" and "policyholder yourself" in reply and not convo.s.verified
    convo.say("yes")
    assert convo.s.verified and convo.s.case_id == "CL-3001"


def test_a_caller_taken_for_a_representative_who_gives_their_own_details_is_asked(convo: Convo):
    convo.say("I'm calling about her claim, her name is Margaret Chen")
    assert convo.s.memory.caller_role == "representative" and convo.s.memory.role_guessed
    reply = convo.say("My name is Margaret Chen, my date of birth is 1985-03-15 and my phone number is 650-521-2836")
    assert convo.act == "ROLE_CHECK" and not convo.s.verified and "not listed" not in reply
    convo.say("yes, that's me")
    assert convo.s.verified


def test_a_role_read_from_wording_alone_is_settled_by_the_question(convo: Convo):
    reply = convo.say("I'm calling about her claim. Margaret Chen, 1985-03-15, 4472")
    assert convo.act == "ROLE_CHECK" and not convo.s.verified and "authorized representative" not in reply
    assert_no_leak(reply)
    convo.say("no, I'm her husband")
    assert convo.act == "ASK_REP_NAME" and convo.s.memory.caller_role == "representative" and not convo.s.verified
    yes = Convo(convo.agent)
    yes.run("I'm calling about her claim. Margaret Chen, 1985-03-15, 4472", "yes, it's my own policy")
    assert yes.s.verified and yes.s.memory.caller_role == "policyholder"


def test_a_model_that_takes_a_policyholder_for_a_representative_is_corrected_by_the_question(agent: Agent):
    # What Haiku did live: "my husband told me to call" read as a representative, role kept on the next turn.
    first = reading(caller_role="representative", identity={"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"})
    second = reading(caller_role="representative", affirmation="yes")
    readings = [first, second]
    c = Convo(agent, llm=ScriptedLLM(on_json=lambda *_: readings.pop(0), on_text=lambda *_: "Thanks. Are you the policyholder yourself?"))
    c.say("Hi, my husband told me to call about a claim. I'm Margaret Chen, born 1985-03-15, last four 4472.")
    assert c.act == "ROLE_CHECK" and not c.s.verified
    c.say("Yes, it's my own policy.")
    assert c.s.verified and c.s.memory.caller_role == "policyholder"


def test_a_model_reading_of_representative_stands_when_the_caller_said_so(agent: Agent):
    first = reading(caller_role="representative", identity={"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"})
    c = Convo(agent, llm=ScriptedLLM(on_json=lambda *_: first, on_text=lambda *_: "Thanks. May I have your own full name?"))
    c.say("I'm calling on behalf of my wife Margaret Chen, born 1985-03-15, last four 4472.")
    assert c.act == "ASK_REP_NAME" and not c.s.memory.role_guessed and not c.s.verified


def test_a_caller_refused_as_a_representative_cannot_become_the_policyholder(convo: Convo):
    convo.say("I need to check on my wife's claim. Her name is Margaret Chen, DOB 1985-03-15, SSN last four 4472.")
    assert convo.act == "ROLE_CHECK"
    convo.say("John Smith")  # a name of their own in reply to "are you the policyholder yourself?"
    assert convo.act == "REP_NOT_AUTHORIZED" and convo.s.memory.caller_role == "representative"
    reply = convo.say("Actually I am the policyholder")
    assert not convo.s.verified and convo.s.phase == ESCALATED
    assert_no_leak(reply)


# ---------------------------------------------------------------- numbers that are not an ID


@pytest.mark.parametrize(
    "text",
    ["it's about my healthcare claim from 2025", "I said the claim from 2025", "the one filed in 2025",
     "my phone number ends in 2836", "you owe me 1450", "I've been a customer since 2010"],
)
def test_a_number_inside_a_sentence_is_not_an_id_number(convo: Convo, text: str):
    convo.say("Margaret Chen, born 1985-03-15")
    convo.say(text)
    assert convo.s.failed_attempts == 0 and "id_last4" not in convo.s.memory.identity
    assert convo.trace["understanding"]["partial"] == []


def test_a_claim_year_said_during_verification_is_remembered_not_counted(convo: Convo):
    convo.run("Margaret Chen, born 1985-03-15", "it's about my healthcare claim from 2025")
    convo.say("650-521-2836")
    assert convo.s.verified and convo.s.failed_attempts == 0 and convo.s.case_id == "CL-2011"


def test_repeating_the_agents_wording_for_the_phone_number_is_a_phone_number(convo: Convo):
    convo.say("Ma Tian, 1964-09-10")
    convo.say("the phone number on my policy is 650-208-8799")
    assert convo.s.verified and convo.s.failed_attempts == 0
    other = Convo(convo.agent)
    other.say("the phone number on the policy: 650-521-2836")
    assert other.s.verification["matched"] == ["phone"] and "policy_number" not in other.s.memory.identity
    third = Convo(convo.agent)
    third.say("my policy number is 9921 and my phone is 650-521-2836")
    assert third.s.memory.identity["policy_number"] == "POL-9921" and third.s.verification["matched"] == ["phone"]


# ---------------------------------------------------------------- pointing at another claim


@pytest.mark.parametrize(
    "text, claim",
    [("what about the dental one?", "CL-1899"), ("and CL-2048?", "CL-2048"), ("and the denied one?", "CL-2048"),
     ("what about the one from November?", "CL-1899")],
)
def test_pointing_at_another_claim_gives_its_overview_not_a_refusal(convo: Convo, text: str, claim: str):
    convo.say("My name is Margaret Chen, date of birth 1985-03-15, and the last four of my SSN are 4472. It's about my auto claim.")
    assert convo.s.case_id == "CL-2102"
    reply = convo.say(text)
    assert convo.s.case_id == claim and "not something I can do" not in reply and convo.s.pending == "anything_else"
    convo.say("ok")
    assert convo.s.phase == PROCESS_CASE and convo.s.handoff is None  # "ok" goes on, it does not transfer


def test_a_request_the_assistant_cannot_do_still_gets_the_honest_reply(verified: Convo):
    reply = verified.say("can you approve it?")
    assert "not something I can do" in reply and verified.s.pending == "human_offer"


# ---------------------------------------------------------------- the output guard

NAME_ONLY = ["My name is Margaret Chen"]
ANSWERED = [DEMO_UTTERANCE, "what do I need to send, and how?"]

FALSE_CLAIMS = [
    (NAME_ONLY, "No problem — you're verified now, Margaret."),
    (NAME_ONLY, "No worries at all, I'm pleased to say you're verified."),
    (NAME_ONLY, "I'm happy to say your identity has been verified."),
    (NAME_ONLY, "I just checked those details and you're verified."),
    (NAME_ONLY, "That's the last thing I needed and your identity is confirmed."),
    (NAME_ONLY, "After checking you're verified, so let's look at the claim."),
    (NAME_ONLY, "Is this David Chen? Could you give me three details?"),
    (ANSWERED, "There's no doubt your claim will be approved."),
    (ANSWERED, "It's not a problem and your claim will be approved."),
    (ANSWERED, "You can upload them at portal.claims-help.co.uk/upload."),
    (ANSWERED, "Send them through secure.claims-portal.com today."),
    (ANSWERED, "No problem — I've sent the summary to your address."),
    (ANSWERED, "No question, your appeal will be approved."),
    (ANSWERED, "No worries, the summary has been sent."),
    (NAME_ONLY, "No need to worry, you're verified now."),
    (NAME_ONLY, "No you're verified, go ahead."),
]
CORRECT_REPLIES = [
    (NAME_ONLY, "Yes, I can help with your claim as soon as your identity is verified. Could you give me any two of the details I listed?"),
    (NAME_ONLY, "Correct, the policy number is not one of the three details, it only helps me find the record. I need two more details."),
    (NAME_ONLY, "Exactly, the claim file is private, which is why I check first."),
    (NAME_ONLY, "After the 3rd attempt that doesn't match, a human representative takes over."),
    (NAME_ONLY, "Once you're verified I can go through it with you."),
    (ANSWERED, "You're 100% right to ask."),
    ([DEMO_UTTERANCE, "What do I do now?"], "Please send the pathology report and office note within 1 week."),
    (ANSWERED, "The appeal deadline is Wednesday, March 18, 2026, so you have until Wednesday to appeal."),
    (ANSWERED, "Let me update you on where this stands: the claim is denied."),
    (ANSWERED, "I can arrange for a human representative to look at it with you."),
    (ANSWERED, "Nothing has been sent yet, and I can't promise it will be approved."),
    (ANSWERED, "There's no guarantee it will be approved."),
    (ANSWERED, "No email has been sent."),
    (NAME_ONLY, "I can't confirm you're verified until I have two more details."),
]


def _verdict(agent: Agent, lines: list[str], reply: str):
    c = Convo(agent)
    c.run(*lines)
    directive = Directive(act="X", goal="", fallback=c.last["reply"], facts=c.trace["directive"]["facts"])
    return check_reply(reply, session=c.s, directive=directive, data=agent.data)


@pytest.mark.parametrize("lines, reply", FALSE_CLAIMS, ids=[reply[:44] for _lines, reply in FALSE_CLAIMS])
def test_a_softening_word_or_a_dash_in_front_does_not_excuse_a_false_claim(agent: Agent, lines: list[str], reply: str):
    assert not _verdict(agent, lines, reply).ok


@pytest.mark.parametrize("lines, reply", CORRECT_REPLIES, ids=[reply[:44] for _lines, reply in CORRECT_REPLIES])
def test_ordinary_correct_wording_is_not_rejected(agent: Agent, lines: list[str], reply: str):
    verdict = _verdict(agent, lines, reply)
    assert verdict.ok, verdict.violations


# ---------------------------------------------------------------- the offline mode of the scenario checks


def _live_check(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/live_check.py", "--offline", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=120, env={"SOP_SKIP_DOTENV": "1", "PATH": ""},
    )


def test_offline_checks_say_when_a_scenario_needs_a_model():
    run = _live_check("--scenarios", "messy", "--quiet")
    assert run.returncode == 2 and "can only be followed by a model" in run.stdout and "FAILED" not in run.stdout


def test_offline_checks_honour_turns_and_out(tmp_path):
    out = tmp_path / "run.json"
    run = _live_check("--scenarios", "happy,bonus", "--quiet", "--turns", "2", "--out", str(out))
    assert run.returncode == 0 and "turns 4" in run.stdout
    saved = json.loads(out.read_text())
    assert [len(s["turns"]) for s in saved["scenarios"]] == [2, 2] and saved["failures"] == []


def test_the_claim_list_is_still_reached_after_a_role_question(convo: Convo):
    convo.say("My husband told me to call. Margaret Chen, 1985-03-15, 4472")
    convo.say("yes")
    assert convo.s.phase == RESOLVE_INTENT and convo.act == "ASK_WHICH_CLAIM"


# ---------------------------------------------------------------- found by probing the fixes above

NAME_AND_DOB = "Margaret Chen, born 1985-03-15"


@pytest.mark.parametrize(
    "text",
    ["it is 4472", "they're 4472", "that's 4472", "ends in 4472", "it ends in 4472", "yes 4472", "ok so it's 4472",
     "should be 4472", "I think it's 4472", "the number is 4472", "that would be 4472", "here you go 4472"],
)
def test_a_lead_in_before_the_last_four_is_still_an_answer(convo: Convo, text: str):
    convo.say(NAME_AND_DOB)
    convo.say(text)
    assert convo.s.verified and convo.s.failed_attempts == 0


def test_a_year_after_a_lead_in_gets_the_note_about_the_full_date(convo: Convo):
    convo.say("Margaret Chen")
    convo.say("it was 1985")
    assert convo.trace["understanding"]["partial"] == ["dob_or_id"] and convo.s.failed_attempts == 0


def test_a_phone_number_right_after_a_policy_number_is_still_read(convo: Convo):
    convo.say("Margaret Chen")
    convo.say("policy number 9921 650-521-2836")
    assert convo.s.memory.identity["policy_number"] == "POL-9921" and "phone" in convo.s.verification["matched"]


@pytest.mark.parametrize("answer", ["I'm not sure what you mean", "I'm not comfortable answering that", "who else would I be?", "why do you ask?"])
def test_a_question_about_the_role_question_is_not_read_as_a_no(convo: Convo, answer: str):
    convo.say(MENTIONS_WIFE)
    reply = convo.say(answer)
    assert convo.act == "ROLE_CHECK" and convo.s.memory.caller_role == "unknown" and not convo.s.verified
    assert "yes or no" in reply and "approved" in reply  # asked again, with the reason


@pytest.mark.parametrize("answer", ["it's mine", "the policy is in my name", "I'm Margaret", "Margaret Chen", "obviously", "I'm the one on the policy"])
def test_everyday_ways_of_saying_yes_to_the_role_question(convo: Convo, answer: str):
    convo.say(MENTIONS_WIFE)
    convo.say(answer)
    assert convo.s.verified and convo.s.memory.caller_role == "policyholder"


@pytest.mark.parametrize(
    "answer",
    ["it's her policy", "she's the policyholder", "it's under her name", "it's my wife's policy", "I'm Margaret's husband",
     "I'm Margaret Chen's son"],
)
def test_everyday_ways_of_saying_no_to_the_role_question(convo: Convo, answer: str):
    convo.say(MENTIONS_WIFE)
    reply = convo.say(answer)
    assert not convo.s.verified and convo.s.memory.caller_role == "representative"
    assert_no_leak(reply)


@pytest.mark.parametrize(
    "opening",
    ["Hi, I'm calling for the status of my claim.", "calling for a claim update,", "I'm calling for my claim status.", "I'm calling for the denied claim."],
)
def test_calling_for_a_thing_is_not_calling_for_a_person(convo: Convo, opening: str):
    convo.say(f"{opening} Margaret Chen, 1985-03-15, 4472")
    assert convo.s.verified and convo.s.memory.caller_role == "unknown"


@pytest.mark.parametrize("opening", ["I'm calling for my daughter.", "It's for my mom.", "I'm calling for a friend,", "I'm calling for the policyholder,", "I'm calling for her."])
def test_calling_for_a_person_is_still_a_representative(convo: Convo, opening: str):
    convo.say(f"{opening} Margaret Chen, 1985-03-15, 4472")
    assert not convo.s.verified and convo.s.memory.caller_role == "representative" and convo.act == "ASK_REP_NAME"


@pytest.mark.parametrize(
    "first",
    ["Calling about my son's claim. I'm Ma Tian, born 1964-09-10, last four 6688",
     "I'm calling for my daughter's claim. I'm Ma Tian, born 1964-09-10, last four 6688",
     "This is for my husband's knee surgery claim. I'm Ma Tian, 1964-09-10, 6688"],
)
def test_a_policyholder_calling_about_a_dependents_claim_is_asked_then_verified(convo: Convo, first: str):
    reply = convo.say(first)
    assert convo.act == "ROLE_CHECK" and convo.s.memory.identity["full_name"] == "Ma Tian" and not convo.s.memory.representative_name
    assert "their" not in reply  # spoken to as the person whose details these are
    convo.say("yes I am")
    assert convo.s.verified and convo.s.memory.caller_role == "policyholder" and convo.s.failed_attempts == 0


def test_a_third_party_who_names_both_people_is_not_taken_for_the_policyholder(convo: Convo):
    convo.say("I'm David Chen, calling about my wife's claim. Margaret Chen, born 1985-03-15, last four 4472")
    m = convo.s.memory
    assert m.caller_role == "representative" and m.representative_name == "David Chen" and m.identity["full_name"] == "Margaret Chen"
    assert not convo.s.verified and convo.s.failed_attempts == 0
    other = Convo(convo.agent)
    other.say("I'm David Chen, calling about my wife's claim. Her name is Margaret Chen, born 1985-03-15, last four 4472")
    assert other.s.memory.representative_name == "David Chen" and not other.s.verified and other.s.failed_attempts == 0


def test_a_model_that_gives_the_callers_own_name_as_a_representatives_is_settled_by_the_question(agent: Agent):
    # The caller never says they act for anyone. The model reads "representative" and files their name under it.
    first = reading(caller_role="representative", identity={"dob": "1964-09-10", "id_last4": "6688", "representative_name": "Ma Tian"})
    second = reading(caller_role="representative", identity={"full_name": "Ma Tian"})
    third = reading(caller_role="representative", affirmation="yes")
    readings = [first, second, third]
    c = Convo(agent, llm=ScriptedLLM(on_json=lambda *_: readings.pop(0), on_text=lambda *_: "Thank you."))
    c.say("Calling about my son's claim. I'm Ma Tian, born 1964-09-10, last four 6688")
    assert c.s.memory.role_guessed and c.s.memory.representative_name == "Ma Tian" and not c.s.verified
    c.say("The policyholder is Ma Tian")
    assert c.act == "ROLE_CHECK" and not c.s.verified
    c.say("yes, that's me")
    assert c.s.verified and c.s.memory.caller_role == "policyholder"


def test_saying_i_am_the_policyholder_reopens_a_role_that_was_only_inferred(agent: Agent):
    first = reading(caller_role="representative", identity={"dob": "1964-09-10", "id_last4": "6688", "representative_name": "Ma Tian"})
    second = reading(caller_role="policyholder")
    readings = [first, second]
    c = Convo(agent, llm=ScriptedLLM(on_json=lambda *_: readings.pop(0), on_text=lambda *_: "Thank you."))
    c.say("Calling about my son's claim. I'm Ma Tian, born 1964-09-10, last four 6688")
    c.say("No, I'm the policyholder")
    m = c.s.memory
    assert c.s.verified and m.caller_role == "policyholder" and m.identity["full_name"] == "Ma Tian" and not m.representative_name


def test_a_role_said_outright_later_is_no_longer_treated_as_inferred(convo: Convo):
    convo.say("I'm calling about her claim")
    assert convo.s.memory.role_guessed
    convo.say("I'm her husband. Margaret Chen, 1985-03-15, 4472")
    assert not convo.s.memory.role_guessed and convo.act == "ASK_REP_NAME" and not convo.s.verified



def test_the_policyholders_name_in_the_possessive_is_not_a_yes(convo: Convo):
    convo.say(MENTIONS_WIFE)
    reply = convo.say("It's Margaret Chen's policy")
    assert not convo.s.verified and convo.s.memory.caller_role != "policyholder"
    assert_no_leak(reply)


def test_being_patient_is_not_mentioning_a_patient(convo: Convo):
    convo.say("I've been very patient with this. Margaret Chen, 1985-03-15, 4472")
    assert convo.s.verified and not convo.s.memory.role_unclear
    other = Convo(convo.agent)
    other.say("I'm calling about a patient. Margaret Chen, 1985-03-15, 4472")
    assert other.act == "ROLE_CHECK" and not other.s.verified


# ---------------------------------------------------------------- found by re-reading the brief before submission


def test_a_policy_number_is_never_read_as_a_name(convo: Convo):
    reply = convo.say("Is my policy number enough? It's POL-9921")
    m = convo.s.memory
    assert m.identity.get("policy_number") == "POL-9921" and "full_name" not in m.identity
    assert convo.act == "VERIFY_EXPLAIN" and "doesn't count" in reply


def test_the_car_thing_points_at_the_auto_claim(verified: Convo):
    reply = verified.say("actually what about the car thing?")
    assert verified.s.case_id == "CL-2102" and "not something I can do" not in reply


def test_just_answer_my_question_is_pushback_on_verification(convo: Convo):
    convo.say("I'm not telling you anything about me")
    assert convo.act == "VERIFY_PERSUADE_1"
    reply = convo.say("No. Just answer my question.")
    assert convo.act == "VERIFY_PERSUADE_2" and "human representative" in reply and not convo.s.verified
    other = Convo(convo.agent)
    other.say("Can I just answer two of them?")  # a question about the step, answered as one
    assert other.act == "VERIFY_EXPLAIN" and other.s.refusal_count == 0


def test_try_before_a_number_is_an_answer(convo: Convo):
    convo.run("Margaret Chen, born 1985-03-15", "0000")
    assert convo.s.failed_attempts == 1
    convo.say("try 4472")
    assert convo.s.verified


REP_OPENING = "I'm David Chen, calling for my mother Margaret Chen. Her date of birth is 1985-03-15 and her phone is 650-521-2836."


def test_a_question_about_consent_is_answered_before_the_status_is_checked_again(convo: Convo):
    convo.say(REP_OPENING)
    assert convo.act == "CONSENT_REQUESTED"
    reply = convo.say("Why do you need her consent? I'm her son.")
    assert convo.act == "CONSENT_PENDING" and not convo.s.verified and "check_consent_status" not in convo.tools
    assert "private information" in reply and "still waiting" in reply
    assert_no_leak(reply)
    convo.say("ok she approved it")
    assert convo.s.verified and "check_consent_status" in convo.tools


def test_a_caller_who_says_it_was_approved_gets_the_check_even_with_a_question(convo: Convo):
    convo.say(REP_OPENING)
    convo.say("She approved it, why is this taking so long?")
    assert convo.s.verified
