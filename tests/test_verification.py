"""VERIFY_ID: strict gate, natural conversation around it."""

from __future__ import annotations

import pytest

from apps.insurance_claims.data import ClaimsData
from apps.insurance_claims.normalize import find_dates, norm_dob, norm_phone, redact_ssn
from apps.insurance_claims.state import (
    ESCALATED,
    PROCESS_CASE,
    RESOLVE_INTENT,
    VERIFY_ID,
)
from apps.insurance_claims.verification import verify_identity

from .conftest import Convo, assert_no_leak


def test_two_matching_details_are_not_enough(convo: Convo):
    reply = convo.say("My name is Margaret Chen and my date of birth is 1985-03-15")
    assert convo.s.phase == VERIFY_ID and not convo.s.verified
    assert convo.act == "VERIFY_PROGRESS"
    assert "one more detail" in reply
    assert_no_leak(reply)


def test_partial_answers_and_alternate_fields(convo: Convo):
    convo.say("Margaret Chen")
    assert convo.s.verification["matched"] == ["full_name"]
    convo.say("I don't have my SSN on me")
    assert "id_last4" in convo.s.memory.declined
    assert convo.act == "VERIFY_PROGRESS"
    convo.say("my email is margaret@email.com")
    assert not convo.s.verified
    convo.say("and my phone is (650) 521-2836")
    assert convo.s.verified
    assert set(convo.s.verification["matched"]) == {"full_name", "email", "phone"}
    assert convo.s.phase == RESOLVE_INTENT  # several claims, no hint: the agent has to ask which
    assert convo.act == "ASK_WHICH_CLAIM"


def test_clarification_question_is_answered_without_leaving_the_phase(convo: Convo):
    reply = convo.say("Why do you need my date of birth?")
    assert convo.act == "VERIFY_EXPLAIN"
    assert convo.s.phase == VERIFY_ID
    assert "private" in reply
    assert convo.s.refusal_count == 0  # a question is not a refusal


def test_a_first_name_alone_does_not_count(convo: Convo):
    reply = convo.say("my name is Margaret")
    assert convo.s.verification["matched"] == []
    assert convo.s.verification["incomplete"] == ["full_name"]
    assert "first and last name" in reply


def test_mismatch_is_reported_and_can_be_corrected(convo: Convo):
    reply = convo.say("Margaret Chen, date of birth 1985-03-16")
    assert convo.act == "VERIFY_MISMATCH"
    assert convo.s.failed_attempts == 1
    assert "date of birth" in reply
    assert "1985-03-15" not in reply  # what is on file is never revealed
    convo.say("sorry, it's 1985-03-15")
    assert convo.s.verification["matched"] == ["full_name", "dob"]
    convo.say("the last four of my social are 4472")
    assert convo.s.verified


def test_three_failed_attempts_hand_over_to_a_human(convo: Convo):
    convo.say("Margaret Chen, date of birth 1980-01-01")
    convo.say("date of birth 1981-02-02")
    assert convo.s.phase == VERIFY_ID
    reply = convo.say("ok my ssn last four is 0000")
    assert convo.s.phase == ESCALATED
    assert convo.act == "VERIFY_LOCKED"
    assert not convo.s.verified
    assert convo.s.handoff["identity_verified"] is False
    assert_no_leak(reply)


def test_unknown_person_gets_a_generic_answer(convo: Convo):
    reply = convo.say("My name is John Smith, born 1970-05-05, SSN last four 1111")
    assert convo.act == "VERIFY_MISMATCH"
    assert "wasn't able to find a match" in reply
    assert not convo.s.verified


def test_declining_too_many_details_offers_a_human(convo: Convo):
    convo.say("I'd rather not share my phone, my email or my SSN")
    assert set(convo.s.memory.declined) >= {"phone", "email", "id_last4"}
    assert convo.act == "VERIFY_IMPOSSIBLE"
    assert convo.s.phase == VERIFY_ID


def test_full_ssn_is_redacted_before_it_is_stored(convo: Convo):
    convo.say("Margaret Chen, 1985-03-15, my social is 123-45-4472")
    assert convo.s.verified
    stored = " ".join(item["text"] for item in convo.s.history)
    assert "123-45" not in stored
    assert "[SSN ending 4472]" in stored


@pytest.mark.parametrize(
    "lines, party",
    [
        (["Ma Tian, born 1964-09-10, national ID ending 6688"], "P12"),
        (["My name is Yaven Li, date of birth 12/03/1989, email yawen.li@example.com"], "P13"),
        (["Ava Lopez", "08/21/1990", "ava.lopez@email.com"], "P7"),
    ],
)
def test_other_policyholders_and_aliases(convo: Convo, lines, party):
    convo.run(*lines)
    assert convo.s.verified and convo.s.party_id == party


def test_single_claim_on_a_policy_is_picked_without_asking(convo: Convo):
    reply = convo.say("Ma Tian, born 1964-09-10, national ID ending 6688")
    assert convo.s.phase == PROCESS_CASE
    assert convo.s.case_id == "CL-3001"
    assert "diagnosis report" in reply


def test_three_matches_verify_even_with_a_stale_fourth_detail():
    data = ClaimsData()
    result = verify_identity(
        data, {"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472", "phone": "4155550000"}
    )
    assert result.verified and result.conflicts == ["phone"]


def test_policy_number_alone_never_verifies():
    data = ClaimsData()
    result = verify_identity(data, {"policy_number": "POL-9921", "full_name": "Margaret Chen"})
    assert not result.verified and result.matched == ["full_name"]


def test_normalisation():
    assert norm_dob("March 15, 1985") == "1985-03-15"
    assert norm_dob("3/15/85") == "1985-03-15"
    assert norm_dob("15 March 1985") == "1985-03-15"
    assert norm_phone("+1 (650) 521-2836") == "6505212836"
    assert redact_ssn("ssn 123454472 ok") == "ssn [SSN ending 4472] ok"
    assert redact_ssn("call 650-521-2836") == "call 650-521-2836"
    assert [(y, m, d) for y, m, d, _ in find_dates("due March 18, 2026 or Jan 12")] == [(2026, 3, 18), (None, 1, 12)]


def test_guessing_while_two_records_tie_still_counts(convo: Convo):
    convo.say("My name is Margaret Chen and my date of birth is 1964-09-10")  # name of one record, birth date of another
    assert convo.s.failed_attempts == 0 and convo.act == "VERIFY_PROGRESS"
    convo.say("ssn last four 0000")
    assert convo.act == "VERIFY_MISMATCH" and convo.s.failed_attempts == 1
    convo.say("ssn last four 1111")
    convo.say("ssn last four 2222")
    assert convo.s.phase == ESCALATED and not convo.s.verified
