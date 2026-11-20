"""Regression tests for the independent check made on the finished version.

Three reviewers who had not seen the code before tried to break the safety gates, the brief's
requirements and the README's claims. The serious finding was at the role question: a yes that
also named a third party ("Yes I'm David, her son") was taken for a plain yes. The others: other
people's contact details counted as failed identity attempts, a representative could work out
which details were right, the output guard let some overclaims through, and a request to send the
summary to somebody else was taken for a yes.
"""

from __future__ import annotations

import pytest

from apps.insurance_claims import Agent
from apps.insurance_claims.guard import check_reply
from apps.insurance_claims.nlu_rules import role_answer
from apps.insurance_claims.state import CLOSED, ESCALATED, VERIFY_ID
from sop_harness.core import Directive
from sop_harness.llm import ScriptedLLM

from .conftest import DEMO_UTTERANCE, Convo, assert_no_leak
from .test_llm_path import reading

MOM_ASKED = "My mom asked me to call about her claim. Margaret Chen, born 1985-03-15, SSN last four 4472."
THREE_DETAILS = "My name is Margaret Chen. DOB is 1985-03-15, SSN last four is 4472."

# ---------------------------------------------------------------- the role question

SAYS_THIRD_PARTY = [
    "No.", "not me, my mother", "I'm her son", "I am her caregiver", "I look after her affairs", "It's my mom's policy",
    "I have power of attorney for her", "well, I'm her husband", "Yes I'm David, her son", "yeah I'm the one who handles her stuff",
    "yes on her behalf", "sure, I'm her guardian", "yes - I'm her legal guardian", "I'm the account holder's son",
    "yes, I'm the son of the policyholder", "Yes, I'm her POA", "yes, for my mother", "it's her policy", "yes it's hers", "I'm David Chen",
]
MENTIONS_SOMEONE = ["she's my mother", "yes, she's my mom", "of course, she is my mother", "yes, my husband just dialed for me", "Yes, I'm David", "yes im david"]
PLAIN_YES = ["yes", "Yes, I am", "it's mine", "Yes, it's my own policy.", "This is she.", "I'm Margaret", "obviously", "Yes. Why was it denied?"]


@pytest.mark.parametrize("answer", SAYS_THIRD_PARTY)
def test_an_answer_that_says_the_caller_is_someone_else_goes_to_the_representative_steps(convo: Convo, answer: str):
    convo.say(MOM_ASKED)
    assert convo.act == "ROLE_CHECK"
    reply = convo.say(answer)
    assert not convo.s.verified and convo.s.memory.caller_role == "representative" and convo.s.phase == VERIFY_ID
    assert_no_leak(reply)


@pytest.mark.parametrize("answer", MENTIONS_SOMEONE)
def test_a_yes_that_mentions_another_person_is_asked_again_never_accepted(convo: Convo, answer: str):
    convo.say(MOM_ASKED)
    reply = convo.say(answer)
    assert convo.act == "ROLE_CHECK" and not convo.s.verified and "yes or no" in reply
    convo.say(answer)
    assert convo.s.phase == ESCALATED and not convo.s.verified  # a second unclear answer goes to a person


@pytest.mark.parametrize("answer", PLAIN_YES)
def test_a_plain_yes_still_opens_the_gate(convo: Convo, answer: str):
    convo.say(MOM_ASKED)
    convo.say(answer)
    assert convo.s.verified and convo.s.memory.caller_role == "policyholder"


@pytest.mark.parametrize(
    "text, model_reading, verified",
    [
        ("yes, for my mother", {"caller_role": "representative", "affirmation": "yes"}, False),
        ("Yes I'm David, her son", {"caller_role": "policyholder", "affirmation": "yes"}, False),
        ("Yes I'm David, her son", {"affirmation": "yes"}, False),
        ("No.", {"affirmation": "yes"}, False),
        ("No.", {"caller_role": "policyholder", "affirmation": "yes"}, False),
        ("I'm the account holder's son", {"caller_role": "policyholder"}, False),
        ("yes, she's my mom", {"caller_role": "policyholder", "affirmation": "yes"}, False),
        ("yes", {"affirmation": "no"}, False),
        ("Yes, it's my own policy.", {"caller_role": "representative", "affirmation": "yes"}, True),
        ("yep that would be me", {"caller_role": "policyholder", "affirmation": "yes"}, True),
        ("si, soy yo", {"caller_role": "policyholder", "affirmation": "yes"}, True),
    ],
)
def test_the_callers_words_decide_at_the_role_question_whatever_the_model_reads(agent: Agent, text: str, model_reading: dict, verified: bool):
    readings = [reading(identity={"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"}), reading(**model_reading)]
    c = Convo(agent, llm=ScriptedLLM(on_json=lambda *_: readings.pop(0), on_text=lambda *_: "Thank you."))
    c.say(MOM_ASKED)
    assert c.act == "ROLE_CHECK"
    reply = c.say(text)
    assert c.s.verified is verified
    if not verified:
        assert_no_leak(reply)


def test_the_role_answer_reading():
    holder = "Margaret Chen"
    assert role_answer("it's her policy", holder) == "other"
    assert role_answer("this is her husband", holder) == "other"
    assert role_answer("It's Margaret Chen's policy", holder) == "third"
    assert role_answer("This is she.", holder) == "self"
    assert role_answer("I'm not sure what you mean", holder) == "none"


@pytest.mark.parametrize(
    "text",
    ["My mom is the policyholder, I'm just helping. Why was the healthcare claim denied?", "it's actually her policy, I'm her daughter",
     "I have power of attorney for her", "I'm just helping my mother with this", "I'm her legal guardian"],
)
def test_after_verification_saying_the_policy_is_someone_elses_goes_to_a_person(convo: Convo, text: str):
    convo.say(THREE_DETAILS)
    assert convo.s.verified
    reply = convo.say(text)
    assert convo.s.phase == ESCALATED and "pathology" not in reply


@pytest.mark.parametrize(
    "text",
    ["my son is covered under my plan, right? what's the status of the dental claim?", "my husband wants to know when the deadline is",
     "can my daughter see this claim too?"],
)
def test_after_verification_mentioning_a_relative_is_not_a_change_of_role(convo: Convo, text: str):
    convo.say(THREE_DETAILS)
    convo.say(text)
    assert convo.s.phase != ESCALATED


def test_a_first_name_alone_is_asked_about_not_refused_as_unlisted(agent: Agent):
    # What a model returns for "Yes I'm David, her son": a representative whose name is only "David".
    readings = [
        reading(identity={"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"}),
        reading(caller_role="representative", affirmation="yes", identity={"representative_name": "David"}),
        reading(caller_role="representative", identity={"representative_name": "David Chen"}),
    ]
    c = Convo(agent, llm=ScriptedLLM(on_json=lambda *_: readings.pop(0), on_text=lambda *_: "Thank you."))
    c.say(MOM_ASKED)
    c.say("Yes I'm David, her son")
    assert c.act == "ASK_REP_NAME" and not c.s.verified and c.s.rep_denials == 0
    c.say("David Chen")
    assert c.act == "CONSENT_REQUESTED" and not c.s.verified  # he is listed, so the next step is the policyholder's consent


def test_typographic_apostrophes_read_like_plain_ones(convo: Convo):
    convo.say("I\u2019m Margaret Chen, born 1985-03-15, last four 4472. My mom asked me to call.")
    assert convo.act == "ROLE_CHECK" and convo.s.memory.identity["full_name"] == "Margaret Chen"
    convo.say("I\u2019m the account holder\u2019s son")
    assert not convo.s.verified and convo.s.memory.caller_role == "representative"
    other = Convo(convo.agent)
    other.run("I\u2019m Margaret Chen, born 1985-03-15, last four 4472. My mom asked me to call.", "yes, it\u2019s mine")
    assert other.s.verified


# ---------------------------------------------------------------- other people's contact details

SOMEONE_ELSES = [
    "I already sent the forms to claims@insurer.com last week.", "My doctor's office is at office@clinic.com if you need it.",
    "The provider's fax is 650 555 1234.", "my doctor's office email is billing@clinic.com", "the hospital's number is 650-555-1234 if that helps",
    "the pharmacy phone is 650-555-9999", "the provider's phone number is 650-555-1234", "My doctor's number is 650-555-0199.",
    "You can reach my doctor at office@clinic.com", "I emailed the adjuster at j.smith@insurer.com twice", "my lawyer's email is counsel@lawfirm.com",
]


@pytest.mark.parametrize("line", SOMEONE_ELSES)
def test_someone_elses_contact_details_are_not_a_failed_identity_attempt(convo: Convo, line: str):
    convo.say("My name is Margaret Chen.")
    convo.say(line)
    assert convo.s.failed_attempts == 0 and "email" not in convo.s.memory.identity and "phone" not in convo.s.memory.identity


def test_three_such_sentences_do_not_lock_the_caller_out(convo: Convo):
    convo.run("My name is Margaret Chen", "my doctor's office email is billing@clinic.com", "the hospital's number is 650-555-1234 if that helps",
              "the pharmacy phone is 650-555-9999")
    assert convo.s.failed_attempts == 0 and convo.s.phase == VERIFY_ID
    convo.say("my phone is 650-521-2836 and my email is margaret@email.com")
    assert convo.s.verified


@pytest.mark.parametrize(
    "line, field",
    [("my email is margaret@email.com", "email"), ("you can reach me at margaret@email.com", "email"), ("the email on my policy is margaret@email.com", "email"),
     ("my phone is 650-521-2836", "phone"), ("the phone number on my policy is 650-521-2836", "phone")],
)
def test_the_callers_own_contact_details_are_still_read(convo: Convo, line: str, field: str):
    convo.say("My name is Margaret Chen.")
    convo.say(line)
    assert field in convo.s.verification["matched"]


# ---------------------------------------------------------------- what a representative is told


def test_a_representative_cannot_work_out_which_details_were_right(convo: Convo):
    convo.say("I'm calling on behalf of my mother. Her name is Margaret Chen, her date of birth is 1985-04-15 and her SSN last four is 4472.")
    assert convo.act == "VERIFY_MISMATCH" and convo.s.failed_attempts == 1
    reply = convo.say("which details do you still need?")
    # the whole list is offered, so nothing shows that the name and the ID were right and the date wrong
    for label in ("full name", "date of birth", "phone number", "email address", "last four"):
        assert label in reply
    assert "noted" not in reply and "confirmed" not in reply
    facts = convo.trace["directive"]["facts"]["verification"]
    assert "details_received_so_far" not in facts and "details_confirmed_so_far" not in facts and len(facts["caller_can_choose_from"]) == 5


# ---------------------------------------------------------------- the output guard

OVERCLAIMS = [
    "I've flagged CL-2048 for priority review.", "I've put in a request to reopen CL-2048.", "Good news: I've requested an extension of your appeal deadline.",
    "Your appeal is going to succeed once you send the pathology report.", "I've added a note so the reviewer prioritises it, and the deadline is now extended.",
    "I've asked the team to expedite this for you.", "I've forwarded the documents request to your doctor.", "I've contacted your provider for you.",
    "Consider it done: the appeal is filed.", "Your appeal has been submitted.", "The claim will be reprocessed and paid within a week.",
    "I'll reach out to your provider directly.", "I can request the records from the hospital for you.", "I can mark this as urgent.",
    "Your claim has now been approved.", "I'll make the change to your phone number now.", "It's approved now, Margaret.", "I've texted you the summary.",
    "You're all set, it's been reversed.", "The denial has been overturned.", "I've raised a complaint on your behalf.", "I'm opening an appeal for you now.",
]
STILL_FINE = [
    "It hasn't been overturned, and I can't say whether it will be.", "I can't mark it as urgent or change its priority.",
    "The appeal deadline is March 18, 2026, and nothing has been submitted on your behalf.", "If it's approved after review, the plan pays up to $1,450.00.",
    "I can connect you with a human representative who can look into that.", "Once the documents are in, the claim goes back into review.",
]


def _verdict(agent: Agent, reply: str):
    c = Convo(agent)
    c.run(THREE_DETAILS, "why was CL-2048 denied, and how much does the plan pay?")
    directive = Directive(act="X", goal="", fallback=c.last["reply"], facts=c.trace["directive"]["facts"])
    return check_reply(reply, session=c.s, directive=directive, data=agent.data)


@pytest.mark.parametrize("reply", OVERCLAIMS, ids=[reply[:40] for reply in OVERCLAIMS])
def test_an_action_or_outcome_that_did_not_happen_is_not_released(agent: Agent, reply: str):
    assert not _verdict(agent, reply).ok


@pytest.mark.parametrize("reply", STILL_FINE, ids=[reply[:40] for reply in STILL_FINE])
def test_saying_what_cannot_be_done_or_promised_is_still_released(agent: Agent, reply: str):
    verdict = _verdict(agent, reply)
    assert verdict.ok, verdict.violations


# ---------------------------------------------------------------- where the summary goes


@pytest.mark.parametrize("request_line", ["send it to my son please", "yes, but send it to my work email", "email it to my lawyer", "to him instead"])
def test_a_request_for_another_recipient_is_not_a_yes(convo: Convo, request_line: str):
    convo.run(DEMO_UTTERANCE, "that's all")
    assert convo.act == "EMAIL_OFFER"
    reply = convo.say(request_line)
    assert convo.act == "EMAIL_ADDRESS_FIXED" and convo.s.outbox == [] and "address on file" in reply
    convo.say("yes")
    assert len(convo.s.outbox) == 1 and convo.s.phase == CLOSED


def test_after_closing_a_request_for_another_address_sends_nothing(convo: Convo):
    convo.run(DEMO_UTTERANCE, "that's all", "no thanks")
    assert convo.s.phase == CLOSED and convo.s.outbox == []
    reply = convo.say("actually send me the summary, but to my work email instead")
    assert convo.act == "EMAIL_ADDRESS_FIXED" and convo.s.outbox == [] and "address on file" in reply
    convo.say("ok send it to the one on file")
    assert len(convo.s.outbox) == 1


def test_after_closing_a_model_reading_with_another_address_sends_nothing(agent: Agent):
    c = Convo(agent)
    c.run(DEMO_UTTERANCE, "that's all", "no thanks")
    c.llm = ScriptedLLM(
        on_json=lambda *_: reading(email_summary_choice="send", identity={"email": "dchen@work.com"}),
        on_text=lambda *_: "Sure, it's on its way to dchen@work.com.",
    )
    reply = c.say("actually send the summary to dchen@work.com")
    assert c.s.outbox == [] and c.act == "EMAIL_ADDRESS_FIXED" and "dchen@work.com" not in reply
