"""POST_PROCESS: the email summary is offered, and sent only on a clear choice."""

from __future__ import annotations

from apps.insurance_claims.state import CLOSED, POST_PROCESS, PROCESS_CASE

from .conftest import Convo


def test_preference_stated_before_verification_is_remembered(convo: Convo):
    convo.say("Hi, it's about my auto claim. And please email me a summary when we're done.")
    assert convo.s.memory.email_pref == "send"
    convo.run("Margaret Chen", "margaret@email.com, and my date of birth is 03/15/1985")
    assert convo.s.case_id == "CL-2102"

    reply = convo.say("thanks, that's all")
    assert convo.act == "EMAIL_OFFER"
    assert "Earlier you asked" in reply
    assert convo.s.outbox == []  # still confirmed before anything is sent
    convo.say("yes")
    assert len(convo.s.outbox) == 1 and convo.s.phase == CLOSED


def test_preference_not_to_be_emailed_is_honoured_without_asking(convo: Convo):
    convo.say("Margaret Chen, 1985-03-15, 4472. It's about my auto claim. Please don't email me anything.")
    assert convo.s.memory.email_pref == "skip"
    reply = convo.say("that's all")
    assert convo.act == "EMAIL_SKIPPED"
    assert "As you asked earlier" in reply
    assert convo.s.outbox == [] and convo.s.phase == CLOSED


def test_unclear_answers_never_send_an_email(verified: Convo):
    verified.say("that's all")
    verified.say("hmm")
    assert verified.act == "EMAIL_OFFER_REPEAT" and verified.s.phase == POST_PROCESS
    verified.say("well")
    verified.say("I don't know")
    assert verified.s.phase == CLOSED
    assert verified.act == "EMAIL_SKIPPED"
    assert verified.s.outbox == []


def test_question_about_the_email_is_answered_then_the_offer_is_repeated(verified: Convo):
    verified.say("that's all")
    reply = verified.say("What will be in the email?")
    assert verified.act == "EMAIL_OFFER_REPEAT"
    assert "what we discussed" in reply
    assert verified.s.outbox == []


def test_a_new_question_during_the_offer_goes_back_to_the_claim(verified: Convo):
    verified.say("that's all")
    assert verified.s.phase == POST_PROCESS
    reply = verified.say("Actually, how much was the allowed amount?")
    assert verified.trace["phase_path"] == [POST_PROCESS, PROCESS_CASE]
    assert "$1,450.00" in reply
    verified.say("ok that's all now")
    assert verified.act == "EMAIL_OFFER"
    verified.say("send it")
    assert len(verified.s.outbox) == 1
    assert "payment amounts" in verified.s.outbox[0]["body"]


def test_changing_your_mind_after_closing_still_sends(verified: Convo):
    verified.run("that's all", "no thanks")
    assert verified.s.phase == CLOSED and verified.s.outbox == []
    verified.say("actually, please send me the summary")
    assert verified.act == "EMAIL_SENT"
    assert len(verified.s.outbox) == 1


def test_a_new_request_after_closing_reopens_without_reverifying(verified: Convo):
    verified.run("that's all", "skip")
    assert verified.s.phase == CLOSED
    reply = verified.say("One more thing, what's the status of my auto claim?")
    assert verified.s.phase == PROCESS_CASE and verified.s.case_id == "CL-2102"
    assert verified.s.verified
    assert "open" in reply


def test_summary_lists_every_claim_discussed(verified: Convo):
    verified.say("What about my auto claim, what's the status?")
    verified.run("that's all", "yes please")
    body = verified.s.outbox[0]["body"]
    assert "CL-2048" in body and "CL-2102" in body
    assert "No action is needed from you right now" in body
