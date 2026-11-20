"""Regression tests for the third independent check.

Two reviewers who had not seen the code worked in opposite directions. One tried to get claim
details without being the policyholder. The other played honest callers and looked for anyone who
was refused, looped or handed off without cause. Both found faults at the same place, the reading
of who is calling, and lists of phrasings had failed there three times.

So the step changed. Three matching details say whose policy it is, never who is calling, and
that is now always asked unless the caller has said outright that they are the policyholder.
The answer is read three ways. Only a plain yes or an exact statement opens the gate. An
outright "no", another person's name or a description of oneself as somebody's relative sends
the caller through the representative steps for good. Everything else is asked again with a
request for a plain yes or no, and after that a person takes over. Nothing loops.
"""

from __future__ import annotations

import pytest

from apps.insurance_claims import Agent
from apps.insurance_claims.nlu_rules import role_answer, says_policyholder
from apps.insurance_claims.state import CLOSED, ESCALATED
from sop_harness.llm import ScriptedLLM

from .conftest import DEMO_UTTERANCE, Convo, assert_no_leak
from .test_llm_path import reading

D3 = "Margaret Chen, 1985-03-15, 4472"
MENTIONS_MOM = f"My mom asked me to check on something. {D3}"
MENTIONS_HUSBAND = f"My husband told me to call about a claim. {D3}"
FOR_MOTHER = "I'm calling for my mother Margaret Chen. Her DOB is 1985-03-15 and her SSN last four is 4472."
VERIFIED = f"I'm the policyholder. {D3}"
PRESSURE = ("yes", "I'm the policyholder", "Margaret Chen", "yes")


# ---------------------------------------------------------------- the question is always asked


def test_three_matching_details_do_not_say_who_is_calling(agent: Agent):
    convo = Convo(agent, answers_role=False)
    reply = convo.say(D3)
    assert convo.act == "ROLE_CHECK" and not convo.s.verified and "are you the policyholder yourself" in reply
    assert_no_leak(reply)
    convo.say("yes")
    assert convo.s.verified and convo.s.memory.caller_role == "policyholder"


def test_the_demo_utterance_says_it_outright_and_verifies_in_one_turn(agent: Agent):
    convo = Convo(agent, answers_role=False)
    convo.say(DEMO_UTTERANCE)
    assert convo.s.verified and convo.act == "ANSWER" and convo.s.case_id == "CL-2048"


@pytest.mark.parametrize(
    ("text", "outright"),
    [
        ("I'm the policyholder. Margaret Chen, 1985-03-15, 4472", True),
        ("Hi, I am the policy holder, Margaret Chen", True),
        ("it's my own policy. Margaret Chen", True),
        ("I'm the other insured. Margaret Chen", False),
        ("I'm the former policyholder. Margaret Chen", False),
        ("I'm the insured driver. Margaret Chen", False),
        ("I'm on the policy too. Margaret Chen", False),
        ("I'm the policyholder's son. Margaret Chen", False),
        ("I'm not the policyholder", False),
        ("This is not my own policy, it's my mother's.", False),
        ("Pretend I'm the policyholder. Margaret Chen", False),
        ("I'm the policyholder, my husband told me to call.", False),
        ("I'm the policyholder. Well, actually my mother is.", False),
    ],
)
def test_only_the_exact_statement_makes_the_question_unnecessary(text: str, outright: bool):
    assert says_policyholder(text) is outright


@pytest.mark.parametrize("opening", ["I'm the other insured.", "I'm the former policyholder.", "I'm on the policy too.", "I'm the insured driver.", "I'm the second insured."])
def test_an_opening_that_only_sounds_like_the_policyholder_is_asked(agent: Agent, opening: str):
    convo = Convo(agent, answers_role=False)
    reply = convo.say(f"{opening} {D3}")
    assert convo.act == "ROLE_CHECK" and not convo.s.verified
    assert_no_leak(reply)


@pytest.mark.parametrize(
    "opening",
    ["The policy isn't mine.", "Ming here.", "Friend of Margaret here.", "I work for Margaret.", "Hello, David here.", "it's me David.",
     "I'm the policyholder's son.", "This is not my own policy, it's my mother's."],
)
def test_an_opening_that_says_the_caller_is_someone_else_is_not_undone_by_a_yes(agent: Agent, opening: str):
    convo = Convo(agent, scenario="timeout", answers_role=False)
    reply = convo.say(f"{opening} {D3}")
    assert not convo.s.verified and convo.s.memory.caller_role == "representative"
    assert_no_leak(reply)
    for line in PRESSURE:
        reply = convo.say(line)
        assert not convo.s.verified
        assert_no_leak(reply)


def test_another_person_coming_up_after_the_statement_brings_the_question_back(agent: Agent):
    convo = Convo(agent, answers_role=False)
    convo.say("I'm the policyholder.")
    convo.say(f"Well, actually my mother is. {D3}")
    assert convo.act == "ROLE_CHECK" and not convo.s.verified
    # and a clean statement after a mention settles it
    convo = Convo(agent, answers_role=False)
    convo.say("My husband told me to call.")
    convo.say(f"I'm the policyholder. {D3}")
    assert convo.s.verified


# ---------------------------------------------------------------- the answer, read three ways

OPENS = [
    "yes", "Yes, I am", "I am", "that's me", "I'm the policyholder", "the policy is in my name", "Margaret Chen", "Yes!", "YES", "yes yes",
    "Yes ma'am", "Of course", "Yes, of course I am", "I am, yes", "Correct.", "That's correct, I'm the policyholder.", "yes, speaking",
    "Yes, this is Margaret Chen", "Yeah that's me, Margaret", "yes - I am", "Yeah, I am. Why?", "yes. can you tell me the status?",
    "Yes, I'd like to know why my claim was denied", "yes I want to know the status of my claim", "Yes, I'm calling about my denied claim",
    "Yes, please tell me what documents I need", "yes. what's the appeal deadline", "Yes, and I need help with my denied claim", "yes so why was it denied",
]
ASKED_AGAIN = [
    "I'm the other insured", "yes, I'm the second insured", "I'm the former policyholder", "yes, I'm the other member", "yes I'm the other person on the policy",
    "yes, I'm on the policy too", "I'm insured on the policy", "I'm the insured driver", "yes I'm Mr Chen", "yes I'm the other Chen", "yes the policyholder is",
    "Yes I'm the patient", "I'm not sure what you mean", "why do you ask?", "kind of", "ok", "fine", "Yes, si, soy su hijo", "yes, ich bin ihr Sohn",
    "Right now I just need the claim status", "Sure, why was it denied?",
]
FOR_GOOD = [
    "it's me David", "it's me, David", "it's me the daughter", "it's me Ming Chen", "yes Margaret is here", "yes, Margaret is", "she's my mother",
    "I'm calling for Margaret", "Margaret asked me to call", "this isn't Margaret", "I'm not her", "the policy is in my wife's name", "I work for Mrs Chen",
    "yes, I'm her lawyer", "I have power of attorney", "I'm Margaret's son", "Margaret is my mother", "I'm the policyholder. Well, actually my mother is",
    "I'm the policyholder, my mother", "yes, mine. well my wife's", "Friend of Margaret here.", "Ming here.", "The policy isn't mine.", "it isn't my policy",
    "I'm calling on Margaret's behalf", "yes, I work for Margaret", "yes, I'm with Margaret", "Yes, I'm Margaret's hubby", "yes, the policy belongs to Margaret",
]
ONE_MORE_QUESTION = [
    "Yes, I'm the patient and the policyholder", "yes I'm his wife and the policyholder", "Yes, I'm the policyholder, I'm his mother", "Yes I'm the patient",
    "Yes, it's my son's claim on my policy", "Yes, I am. The doctor is Dr. Patel.", "Yes, I'm the policyholder. My husband is covered under my plan.",
    "yes, my husband just dialed for me", "Yes, she's on my policy", "Yes, it's about my son's claim", "Yes, I'm calling about my daughter's surgery claim",
    "I'm the insured", "Yes, the claim is under Margaret Chen", "I'm her mother", "Yes, I'm his mom",
]


@pytest.mark.parametrize("answer", OPENS)
def test_a_plain_yes_opens_the_gate_also_with_talk_about_the_claim_next_to_it(convo: Convo, answer: str):
    convo.say(MENTIONS_HUSBAND)
    assert convo.act == "ROLE_CHECK"
    convo.say(answer)
    assert convo.s.verified and convo.s.memory.caller_role == "policyholder"


@pytest.mark.parametrize("answer", ASKED_AGAIN)
def test_an_answer_that_is_neither_a_plain_yes_nor_a_no_is_asked_again(convo: Convo, answer: str):
    convo.say(MENTIONS_MOM)
    reply = convo.say(answer)
    assert convo.act == "ROLE_CHECK" and not convo.s.verified and "yes or no" in reply and "approved" in reply
    assert_no_leak(reply)


@pytest.mark.parametrize("answer", FOR_GOOD)
def test_saying_outright_that_one_is_someone_else_holds_against_every_later_yes(agent: Agent, answer: str):
    convo = Convo(agent, scenario="timeout")
    convo.say(MENTIONS_MOM)
    reply = convo.say(answer)
    assert not convo.s.verified and convo.s.memory.caller_role == "representative"
    assert_no_leak(reply)
    for line in PRESSURE:
        reply = convo.say(line)
        assert not convo.s.verified
        assert_no_leak(reply)


@pytest.mark.parametrize("answer", ONE_MORE_QUESTION)
def test_a_policyholder_who_says_more_than_yes_gets_one_more_question_and_is_never_locked_out(convo: Convo, answer: str):
    convo.say(MENTIONS_HUSBAND)
    reply = convo.say(answer)
    assert convo.act == "ROLE_CHECK" and convo.s.memory.caller_role == "unknown" and convo.s.phase != ESCALATED
    assert "yes or no" in reply
    convo.say("yes")
    assert convo.s.verified and convo.s.memory.caller_role == "policyholder"


def test_three_unclear_answers_end_with_a_person(convo: Convo):
    convo.say(MENTIONS_MOM)
    for expected in ("ROLE_CHECK", "ROLE_CHECK", "ESCALATED"):
        reply = convo.say("I'm the other insured")
        assert convo.act == expected and not convo.s.verified
        assert_no_leak(reply)
    assert convo.s.phase == ESCALATED


def test_every_name_the_policyholder_goes_by_is_their_own(agent: Agent):
    # verified without a name, so the name on record is what an answer is compared with
    convo = Convo(agent, answers_role=False)
    convo.say("DOB 1985-03-15, phone 650-521-2836, SSN last four 4472")
    assert convo.act == "ROLE_CHECK"
    convo.say("Yes, I'm Margaret Chen")
    assert convo.s.verified and convo.s.memory.caller_role == "policyholder"
    convo = Convo(agent, scenario="timeout", answers_role=False)
    convo.say("DOB 1985-03-15, phone 650-521-2836, SSN last four 4472")
    for line in ("Yes, I'm David Chen", "yes", "yes"):
        convo.say(line)
        assert not convo.s.verified and convo.s.memory.caller_role == "representative"
    assert role_answer("Yes, I'm Margaret Chen", ["", "Margaret Chen"]) == "self"
    assert role_answer("Yes, I'm Margaret Chen", ["Maggie Chen", "Margaret Chen"]) == "self"


@pytest.mark.parametrize("answer", ["I'm the other insured", "it's me David", "yes, I'm Margaret Chen Jr", "Right now I just need the claim status", "I'm her son"])
def test_a_models_yes_never_opens_the_gate_for_words_that_are_not_one(agent: Agent, answer: str):
    details = reading(identity={"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"}, caller_role="policyholder")
    says_yes = reading(caller_role="policyholder", affirmation="yes")
    readings = [details, says_yes]
    convo = Convo(agent, scenario="timeout", llm=ScriptedLLM(on_json=lambda *_: readings.pop(0), on_text=lambda *_: "Thank you."))
    convo.say(MENTIONS_MOM)
    assert convo.act == "ROLE_CHECK" and not convo.s.verified  # the model inferred "policyholder"; the caller never said it
    convo.say(answer)
    assert not convo.s.verified


# ---------------------------------------------------------------- the representative's own name


@pytest.mark.parametrize("answer", ["what?", "huh", "why", "I'd rather not", "no", "Yes, I'm the policyholder"])
def test_the_name_step_ends_with_a_person_and_never_repeats_for_ever(agent: Agent, answer: str):
    convo = Convo(agent, scenario="timeout")
    convo.say("I'm her son.")
    convo.say(D3)
    assert convo.act == "ASK_REP_NAME"
    for _ in range(6):
        if convo.s.phase == ESCALATED:
            break
        reply = convo.say(answer)
        assert not convo.s.verified
        assert_no_leak(reply)
    assert convo.s.phase == ESCALATED


def test_giving_the_policyholders_name_as_ones_own_is_queried_once_then_goes_to_a_person(agent: Agent):
    convo = Convo(agent, scenario="timeout")
    convo.say("I'm her son.")
    convo.say(D3)
    reply = convo.say("Margaret Chen")
    assert convo.act == "ASK_REP_NAME" and "name of the policyholder" in reply and not convo.s.verified
    convo.say("Margaret Chen")
    assert convo.s.phase == ESCALATED and not convo.s.verified
    # the listed representative who typed the wrong name first is not shut out
    convo = Convo(agent, scenario="timeout")
    convo.say("I'm her son.")
    convo.say(D3)
    convo.say("Margaret Chen")
    convo.say("Sorry, David Chen")
    assert convo.act == "CONSENT_REQUESTED"


@pytest.mark.parametrize("name", ["David J. Chen", "David James Chen", "David J Chen", "david chen"])
def test_a_middle_name_or_initial_still_finds_the_listed_representative(agent: Agent, name: str):
    convo = Convo(agent, scenario="timeout")
    convo.say(FOR_MOTHER)
    convo.say(name)
    assert convo.act == "CONSENT_REQUESTED"


def test_a_typo_in_the_name_can_be_corrected_and_another_name_cannot(agent: Agent):
    convo = Convo(agent, scenario="timeout")
    convo.run(FOR_MOTHER, "Davd Chen")
    assert convo.act == "REP_NOT_AUTHORIZED"
    convo.say("sorry, David Chen")
    assert convo.act == "CONSENT_REQUESTED"
    convo = Convo(agent, scenario="timeout")
    convo.run(FOR_MOTHER, "Bob Smith")
    assert convo.act == "REP_NOT_AUTHORIZED"
    convo.say("sorry, David Chen")
    assert convo.s.phase == ESCALATED and not convo.s.verified  # a new name after a refusal is somebody guessing


def test_a_middle_name_does_not_loosen_the_policyholders_own_name(agent: Agent):
    convo = Convo(agent, answers_role=False)
    convo.say("Margaret Rose Chen, 1985-03-15, 4472")
    assert not convo.s.verified and convo.act != "ROLE_CHECK"  # the name is a mismatch, as before


# ---------------------------------------------------------------- nothing is given away on the way


def test_nobody_is_told_which_details_matched_once_another_person_has_come_up(agent: Agent):
    for first in ("I'm calling for my mother, her name is Margaret Chen", "My mom asked me to call. Margaret Chen"):
        convo = Convo(agent, answers_role=False)
        convo.say(first)
        reply = convo.say("1985-03-15").lower()
        assert "confirmed" not in reply and "full name" in reply and "date of birth" in reply  # the whole list, nothing named as matched
    convo = Convo(agent, answers_role=False)
    convo.say("Margaret Chen")
    assert "name" in convo.say("1985-03-15").lower()  # a caller who mentioned nobody is told what is noted


# ---------------------------------------------------------------- the caller's own contact details


@pytest.mark.parametrize(
    "line",
    ["the email on file is margaret@email.com", "my number's 650-521-2836", "you can use margaret@email.com", "phone number on file should be 650-521-2836",
     "you have my email as margaret@email.com", "my email address? margaret@email.com", "my phone number. 650-521-2836"],
)
def test_more_ways_of_giving_ones_own_phone_or_email(agent: Agent, line: str):
    convo = Convo(agent, answers_role=False)
    convo.say("Margaret Chen, born 1985-03-15")
    convo.say(line)
    assert convo.act == "ROLE_CHECK" and len(convo.s.verification["matched"]) == 3 and convo.s.failed_attempts == 0


# ---------------------------------------------------------------- the email step


@pytest.mark.parametrize("line", ["send it to my personal email margaret@email.com", "my work email margaret@email.com please", "margaret@email.com", "yes, margaret@email.com"])
def test_typing_the_address_on_file_at_the_offer_sends_the_summary(convo: Convo, line: str):
    convo.run(f"{VERIFIED}. Why was my January claim denied?", "that's all")
    assert convo.act == "EMAIL_OFFER"
    convo.say(line)
    assert convo.act == "EMAIL_SENT" and convo.s.phase == CLOSED and len(convo.s.outbox) == 1


@pytest.mark.parametrize(
    ("line", "act"),
    [("send it to bob@example.com", "EMAIL_ADDRESS_FIXED"), ("is it margaret@email.com?", "EMAIL_OFFER_REPEAT"), ("don't use margaret@email.com", "EMAIL_OFFER_REPEAT"),
     ("margaret@email.com is my old address", "EMAIL_OFFER_REPEAT")],
)
def test_an_address_that_is_another_one_or_is_questioned_sends_nothing(convo: Convo, line: str, act: str):
    convo.run(f"{VERIFIED}. Why was my January claim denied?", "that's all")
    convo.say(line)
    assert convo.act == act and not convo.s.outbox


# ---------------------------------------------------------------- after verification

HANDS_OFF = [
    "the policy isn't mine", "it isn't my policy", "I'm calling for Margaret", "I'm calling on Margaret's behalf", "this isn't Margaret", "I'm not her",
    "the policy is in my wife's name", "Margaret asked me to call", "I'm not the policyholder", "I'm David", "My name is David Chen", "I'm Margaret's son",
    "Margaret is my mother", "actually it's my mother's policy", "I'm her lawyer", "I have power of attorney for her", "my mother is the policyholder",
    "the policyholder is my wife", "I'm the policyholder's son", "I work for Margaret",
]
CHANGES_NOTHING = [
    "I'm his wife and he's on my plan", "I'm his mother, the claim is for my son on my plan", "I'm her mom so I handle her claims", "I'm her mother",
    "My husband has his own policy with another insurer, does that matter?", "that's not my policy number", "It's Dr. Patel who has the report",
    "This is Dr. Patel's office note, right?", "I'm a lawyer and I will appeal this", "do I need a power of attorney to let my son call?",
    "the car belongs to my husband", "my son is the patient", "I'm sorry, what was the deadline?", "I'm Margaret", "someone else told me the deadline was April",
    "Can my husband call you about it?", "She is my daughter and the claim is hers", "the letter was addressed to Margaret Chen",
]


@pytest.mark.parametrize("line", HANDS_OFF)
def test_after_verification_an_outright_contradiction_goes_to_a_person(convo: Convo, line: str):
    convo.say(VERIFIED)
    assert convo.s.verified
    reply = convo.say(line)
    assert convo.s.phase == ESCALATED and "pathology" not in reply


@pytest.mark.parametrize("line", CHANGES_NOTHING)
def test_after_verification_talk_about_family_and_other_people_changes_nothing(convo: Convo, line: str):
    convo.say(VERIFIED)
    convo.say(line)
    assert convo.s.verified and convo.s.phase != ESCALATED


# ---------------------------------------------------------------- my own pass over the path that opens the gate

NOT_A_YES = [
    "Yes - the daughter", "yes/no", "Yes, I'm calling about my mother-in-law's claim", "Yeah the wife's", "I am her", "Yes I am the policyholder's",
    "Yes, I am the policyholder son", "yes. no.", "Yes, I am not", "Yes, unfortunately not", "yes, why would my mom's claim be denied?",
    "Yes. Tell me about claim CL-2048 for Margaret.", "yes it's me her son", "Yes, me and my wife", "none of your business", "Not sure", "maybe", "technically",
    "Yes, technically", "Yes, in a way", "yes, by marriage", "yes, since she passed", "Yes, I'm the new policyholder", "I'm the policyholder now",
    "I will be the policyholder", "I was the policyholder", "I'm basically the policyholder", "I'm acting as the policyholder", "I'm the co-policyholder",
    "I'm one of the policyholders", "we're the policyholders", "yes we are", "Yes, both of us", "yes, the estate", "Yes, I'm the executor",
    "Right now I want the claim status", "Sure, but first tell me the status", "Correct me if I'm wrong but isn't it denied", "Absolutely no idea",
    "Definitely not", "Of course not", "Yes, Margaret's claim", "yes I am calling for her claim", "Yes, tell me about his claim", "yes about their claim",
    "yes, I'm asking for a friend", "Yes, David", "yes david", "Yes. David Chen.", "yes; her son", "yes (son)", "yes, son",
]
ALSO_A_YES = [
    "Yes, policyholder", "the policyholder", "Yessir", "sure thing", "No problem, yes I am", "no worries, yes", "Yes I am. What do you need?", "yes, that is correct",
    "It's me", "This is me", "Yup, it's my policy", "yeah it's mine", "Yes, I hold the policy", "This is Margaret", "yes, Margaret here", "Margaret Chen speaking",
    "yes, my name is Margaret Chen", "Yes, I'm calling about my own claim", "Yes. I need the status of my denied claim.", "yes can you email me a summary",
    "yes, is CL-2048 still denied?",
]


@pytest.mark.parametrize("answer", NOT_A_YES)
def test_nothing_but_a_plain_yes_or_an_exact_statement_reads_as_one(answer: str):
    assert role_answer(answer, ["Margaret Chen"]) != "self"


@pytest.mark.parametrize("answer", ALSO_A_YES)
def test_the_ways_a_policyholder_says_yes(answer: str):
    assert role_answer(answer, ["Margaret Chen"]) == "self"


@pytest.mark.parametrize(
    ("text", "outright"),
    [
        ("Let's assume, I'm the policyholder. Margaret Chen", False),
        ("Hypothetically, I'm the policyholder.", False),
        ("Is it enough to say I'm the policyholder?", False),
        ("I'm the policy holder for my late wife's account", False),
        ("I'm the policyholder and the estate's lawyer", False),
        ("I'm the policyholder. My agent told me to call.", False),
        ("I'm the policyholder Margaret Chen", False),
        ("So I'm the policyholder, right? Margaret Chen", True),
        ("I'm the policyholder and I want to know why my claim was denied", True),
        ("Hello, I am the policyholder. My name is Margaret Chen.", True),
    ],
)
def test_a_supposition_or_a_statement_with_more_in_it_is_not_the_outright_statement(text: str, outright: bool):
    assert says_policyholder(text) is outright


@pytest.mark.parametrize(
    "lines",
    [
        ["I'm calling for my wife. My name is Margaret Chen, born 1985-03-15, last four 4472, phone 650-521-2836.", "yes"],
        ["I'm calling for my wife. My name is Margaret Chen, born 1985-03-15, last four 4472.", "650-521-2836", "yes"],
        ["I'm calling on behalf of my son, his claim was denied. I'm Margaret Chen, DOB 1985-03-15, SSN last four 4472.", "Yes, I am"],
        ["Hi, I'm his mother. Margaret Chen, 1985-03-15, 4472", "yes"],
        ["Chen here. Margaret Chen, 1985-03-15, 4472", "yes"],
        ["Mom here, calling about my son's claim. Margaret Chen, 1985-03-15, 4472", "yes"],
        ["I'm the policyholder. Margaret Chen, born 1985-03-15", "my SSN ends in 4472, my husband has the card", "yes"],
        [D3, "why do you ask?", "yes"],
    ],
)
def test_a_policyholder_with_family_on_the_plan_is_asked_and_gets_through(agent: Agent, lines: list[str]):
    convo = Convo(agent, scenario="timeout", answers_role=False)
    for line in lines[:-1]:
        reply = convo.say(line)
        assert not convo.s.verified
        assert_no_leak(reply)
    assert convo.act == "ROLE_CHECK"
    convo.say(lines[-1])
    assert convo.s.verified and convo.s.memory.caller_role == "policyholder"


def test_a_model_that_takes_a_parent_for_a_representative_does_not_settle_it(agent: Agent):
    # "On behalf of my son" is what a policyholder says about a child on her plan. Whatever the
    # model concludes, the caller is asked, and the answer decides.
    opening = "I'm calling on behalf of my son, his claim was denied. I'm Margaret Chen, born 1985-03-15, last four 4472."
    details = reading(identity={"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"}, caller_role="representative")
    for answer, verified in (("Yes, I am", True), ("No, the policy is his", False)):
        readings = [details, reading(affirmation="yes" if verified else "no")]
        convo = Convo(agent, scenario="timeout", llm=ScriptedLLM(on_json=lambda *_, r=readings: r.pop(0), on_text=lambda *_: "Thank you."))
        convo.say(opening)
        assert convo.act == "ROLE_CHECK" and not convo.s.verified
        convo.say(answer)
        assert convo.s.verified is verified


@pytest.mark.parametrize("answer", ["The policyholder?", "Margaret Chen?", "Me?", "I am?", "Yes?", "no idea", "no clue", "What do you mean?"])
def test_repeating_the_question_or_not_knowing_is_not_an_answer(convo: Convo, answer: str):
    convo.say(MENTIONS_HUSBAND)
    reply = convo.say(answer)
    assert convo.act == "ROLE_CHECK" and convo.s.memory.caller_role == "unknown" and not convo.s.verified
    assert_no_leak(reply)
    convo.say("Me? Yes.")
    assert convo.s.verified


def test_a_name_on_record_with_three_parts_is_the_policyholders_own_in_every_usual_form():
    names = ["Ya Wen Li", "Yaven Li"]
    for answer in ("Ya Wen Li", "I'm Ya Wen Li", "Yes, this is Ya Wen Li", "Ya Wen", "yes, Li Ya Wen", "Ya-Wen Li", "Yaven Li", "Yes, I'm Ya Wen Li, why?"):
        assert role_answer(answer, names) == "self", answer
    for answer in ("It's David Li", "I'm her husband, Tom Li"):
        assert role_answer(answer, names) == "other", answer
    for answer in ("Li", "Mrs Li", "Ya Wen Li's", "Ya Wen Li Jr"):
        assert role_answer(answer, names) != "self", answer


def test_the_policyholder_with_a_three_part_name_gets_through(agent: Agent):
    convo = Convo(agent, answers_role=False)
    convo.say("Ya Wen Li, 1989-12-03, 5317")
    assert convo.act == "ROLE_CHECK"
    convo.say("Yes, this is Ya Wen Li")
    assert convo.s.verified and convo.s.memory.caller_role == "policyholder"


# ---------------------------------------------------------------- the fourth check


@pytest.mark.parametrize("answer", ["John Smith", "David Chen", "Me, John Smith", "I'm David Chen", "My name is John Smith."])
def test_a_name_typed_at_the_role_question_is_never_the_policyholders_by_assertion(agent: Agent, answer: str):
    # Verified on three details without a name. The name then given is compared with the record.
    convo = Convo(agent, scenario="timeout", answers_role=False)
    convo.say("1985-03-15, 4472, margaret@email.com")
    assert convo.act == "ROLE_CHECK"
    for line in (answer, "yes", "Why was my claim denied? CL-2048"):
        reply = convo.say(line)
        assert not convo.s.verified and convo.s.memory.caller_role == "representative"
        assert_no_leak(reply, allow=("CL-2048", "margaret@email.com"))


def test_a_models_name_at_the_role_question_is_compared_with_the_record_too(agent: Agent):
    details = reading(identity={"dob": "1985-03-15", "id_last4": "4472", "email": "margaret@email.com"})
    named = reading(identity={"full_name": "David Chen"}, caller_role="policyholder", affirmation="yes")
    readings = [details, named]
    convo = Convo(agent, scenario="timeout", llm=ScriptedLLM(on_json=lambda *_: readings.pop(0), on_text=lambda *_: "Thank you."))
    convo.say("1985-03-15, 4472, margaret@email.com")
    convo.say("I'm David Chen")
    assert not convo.s.verified and convo.s.memory.caller_role == "representative"


def test_the_policyholders_own_name_at_the_role_question_still_opens_the_gate(agent: Agent):
    convo = Convo(agent, answers_role=False)
    convo.say("1985-03-15, 4472, margaret@email.com")
    convo.say("Margaret Chen")
    assert convo.s.verified and convo.s.memory.caller_role == "policyholder"


def test_a_dependent_in_either_spelling_is_not_talk_about_the_claim():
    for answer in ("Yes, about the denied claim. Covered as a dependent.", "Yes, about the denied claim. Covered as a dependant.", "I'm only a dependent."):
        assert role_answer(answer, ["Margaret Chen"]) != "self", answer
