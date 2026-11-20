"""The model path, exercised with scripted stand-ins. No network is used."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from apps.insurance_claims import Agent
from apps.insurance_claims.state import PROCESS_CASE, VERIFY_ID
from apps.insurance_claims.understanding import (
    UNDERSTANDING_SCHEMA,
    understanding_from_json,
)
from sop_harness.llm import FALLBACK_BETA, AnthropicLLM, LLMError, ScriptedLLM

from .conftest import DEMO_UTTERANCE, PROTECTED, Convo


def reading(**overrides) -> dict:
    """A complete understanding object, as the schema requires."""
    base = {
        "scope": "in_scope",
        "caller_role": "unknown",
        "identity": {
            "full_name": "", "dob": "", "phone": "", "email": "", "id_last4": "",
            "policy_number": "", "representative_name": "",
        },
        "declined_fields": [],
        "refuses_verification": False,
        "intent": "none",
        "also_asks": [],
        "case_hints": {"claim_id": "", "claim_type": "none", "status": "none", "month": 0, "year": 0},
        "selected_case_id": "",
        "question": "",
        "followup_topics": [],
        "process_question": False,
        "emotion": "neutral",
        "wants_human": False,
        "affirmation": "none",
        "email_summary_choice": "none",
        "done": False,
        "more": False,
        "notes": [],
    }
    identity = overrides.pop("identity", {})
    hints = overrides.pop("case_hints", {})
    base.update(overrides)
    base["identity"].update(identity)
    base["case_hints"].update(hints)
    return base


DEMO_READING = reading(
    caller_role="policyholder",
    identity={"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472", "policy_number": "POL-9921"},
    intent="general_claim_question",
    case_hints={"claim_type": "healthcare", "status": "denied", "month": 1},
    question="Caller wants to discuss their denied healthcare claim from January.",
)


def scripted(readings: list[dict], replies: list[str]) -> ScriptedLLM:
    readings, replies = list(readings), list(replies)
    return ScriptedLLM(on_json=lambda *_: readings.pop(0), on_text=lambda *_: replies.pop(0))


def test_schema_and_example_reading_agree():
    assert set(UNDERSTANDING_SCHEMA["required"]) == set(DEMO_READING)
    assert UNDERSTANDING_SCHEMA["additionalProperties"] is False


def test_model_reading_drives_the_same_policy(agent: Agent):
    good_reply = (
        "Thanks, Margaret, you're all set. I pulled up the denied healthcare claim from January you mentioned, "
        "CL-2048, filed January 12, 2026. It was denied because the review file didn't include the pathology "
        "report and the treating provider's office note. If you send those in before March 18, 2026, which is "
        "13 days away, it can be reconsidered. Is there anything else you'd like to know about this claim?"
    )
    llm = scripted([DEMO_READING], [good_reply])
    convo = Convo(agent, llm=llm)
    reply = convo.say(DEMO_UTTERANCE)
    assert reply == good_reply
    assert convo.s.verified and convo.s.phase == PROCESS_CASE and convo.s.case_id == "CL-2048"
    assert convo.trace["engine"] == "llm:scripted"
    assert convo.trace["reply_source"] == "model"
    assert convo.trace["guard"]["ok"] is True


def test_no_record_data_reaches_the_model_before_verification(agent: Agent):
    llm = scripted(
        [reading(intent="denial_explanation", case_hints={"status": "denied"}, question="Why was my claim denied?",
                 emotion="frustrated", refuses_verification=True)],
        ["I hear you. I do need to confirm it's you first. Could you give me any three identity details?"],
    )
    convo = Convo(agent, llm=llm)
    convo.say("Just tell me why my claim was denied!")
    assert convo.s.phase == VERIFY_ID
    assert len(llm.calls) == 2
    assert "withheld until the caller is verified" in llm.calls[0]["user"]
    for call in llm.calls:
        for item in PROTECTED:
            assert item.lower() not in call["user"].lower(), (item, call["kind"])


@pytest.mark.parametrize(
    "leak",
    [
        "Your claim CL-2048 was denied because the pathology report was missing.",
        "You're verified! How can I help?",
        "I see a payment of $1,450.00 on your account.",
        "Your appeal deadline is March 18, 2026.",
        "I've sent the summary to your email.",
        "I'm transferring you to an agent now.",
        "Is your date of birth 1985-03-15 and your email margaret@email.com?",
    ],
)
def test_guard_replaces_a_model_reply_that_discloses_or_overclaims(agent: Agent, leak: str):
    llm = scripted([reading(identity={"full_name": "Margaret Chen"})], [leak])
    convo = Convo(agent, llm=llm)
    reply = convo.say("My name is Margaret Chen")
    assert reply != leak
    assert convo.trace["guard"]["ok"] is False
    assert convo.trace["guard"]["rejected_reply"] == leak
    assert convo.trace["reply_source"].startswith("template")
    assert convo.s.phase == VERIFY_ID


@pytest.mark.parametrize(
    "hallucination",
    [
        "It was denied, and you'll receive $2,000.00 once it's reviewed.",
        "Your appeal deadline is April 1, 2026.",
        "That's claim CL-9999, which was denied.",
        "Ma Tian's claim was denied too.",
    ],
)
def test_guard_rejects_ungrounded_values_after_verification(agent: Agent, hallucination: str):
    llm = scripted([DEMO_READING], [hallucination])
    convo = Convo(agent, llm=llm)
    reply = convo.say(DEMO_UTTERANCE)
    assert convo.s.verified
    assert convo.trace["guard"]["ok"] is False
    assert reply != hallucination and "CL-2048" in reply


def test_guard_accepts_natural_progress_wording(agent: Agent):
    natural = (
        "Thanks, Margaret. I've confirmed your full name, so I just need two more details. Your date of birth, "
        "phone number, email address or the last four digits of your SSN would all work."
    )
    llm = scripted([reading(identity={"full_name": "Margaret Chen"})], [natural])
    convo = Convo(agent, llm=llm)
    assert convo.say("My name is Margaret Chen") == natural
    assert convo.trace["guard"]["ok"] is True and convo.trace["reply_source"] == "model"


def test_a_model_that_restates_the_current_claim_does_not_derail_the_email_offer(agent: Agent):
    restating = {"selected_case_id": "CL-2048", "case_hints": {"claim_type": "healthcare", "status": "denied"}}
    llm = scripted(
        [
            DEMO_READING,
            reading(done=True, affirmation="no", **restating),
            reading(**restating),  # an unclear answer to the email offer
            reading(email_summary_choice="send", affirmation="yes", **restating),
        ],
        ["ok", "Would you like a summary by email, or shall we skip it?", "Should I send it or skip it?", "Sent. Take care!"],
    )
    convo = Convo(agent, llm=llm)
    convo.say(DEMO_UTTERANCE)
    convo.say("no that's all")
    assert convo.s.phase == "POST_PROCESS" and convo.act == "EMAIL_OFFER"
    convo.say("hmm")
    assert convo.s.phase == "POST_PROCESS" and convo.act == "EMAIL_OFFER_REPEAT"
    convo.say("yes go ahead")
    assert convo.act == "EMAIL_SENT" and len(convo.s.outbox) == 1


def test_model_cannot_select_another_policyholders_claim(agent: Agent):
    llm = scripted(
        [DEMO_READING, reading(selected_case_id="CL-3001", intent="claim_status", question="Status of CL-3001?")],
        ["ok", "Claim CL-3001 was denied because the diagnosis report is missing."],
    )
    convo = Convo(agent, llm=llm)
    convo.say(DEMO_UTTERANCE)
    reply = convo.say("what about CL-3001")
    assert convo.act == "CLAIM_ID_NOT_FOUND"
    # the other policyholder's data was never given to the model ...
    assert "diagnosis" not in llm.calls[-1]["user"].lower()
    assert convo.trace["directive"]["facts"]["found_on_this_policy"] is False
    # ... and a reply that states it anyway is rejected
    assert convo.trace["guard"]["ok"] is False
    assert "diagnosis" not in reply.lower()


def test_a_failed_model_call_falls_back_to_the_offline_engine(agent: Agent):
    def boom(*_):
        raise LLMError("auth", "The API token was rejected (401). Check the token.")

    llm = ScriptedLLM(on_json=boom, on_text=lambda *_: "should not be used")
    convo = Convo(agent, llm=llm)
    reply = convo.say("My name is Margaret Chen")
    assert convo.trace["engine"] == "offline"
    assert convo.trace["llm_errors"][0]["kind"] == "auth"
    assert len(llm.calls) == 1  # no second call once the token is known to be bad
    assert "confirmed your full name" in reply
    assert convo.s.verification["matched"] == ["full_name"]


def test_malformed_model_output_is_normalised_not_trusted():
    u = understanding_from_json(
        {
            "scope": "everything", "intent": "delete_all_claims", "emotion": 7,
            "identity": {"dob": "15th of March 1985", "phone": "call me", "id_last4": "123-45-4472", "full_name": "  margaret   CHEN "},
            "case_hints": {"month": 99, "year": "soon", "claim_type": "boat"},
            "declined_fields": ["password", "dob"], "followup_topics": ["free_money"], "notes": "x",
        }
    )
    assert (u.scope, u.intent, u.emotion) == ("in_scope", "none", "neutral")
    assert u.identity == {"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"}
    assert u.case_hints == {} and u.declined_fields == ["dob"] and u.followup_topics == []


# ---- AnthropicLLM request shape, against a fake client -------------------------


def _response(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        model="claude-opus-5-5",
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=120, output_tokens=30, cache_read_input_tokens=0),
    )


class _Endpoint:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _client(plain, beta):
    return SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta))


def _status_error(cls, status: int, message: str = "simulated"):
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(message, response=httpx2.Response(status, request=request), body=None)


def test_request_uses_structured_output_effort_and_fallbacks():
    plain, beta = _Endpoint(_response("{}")), _Endpoint(_response('{"scope": "in_scope"}'))
    llm = AnthropicLLM(api_key="test-key", client=_client(plain, beta))
    result = llm.complete_json(system="SYS", user="USER", schema={"type": "object"})
    assert result.data == {"scope": "in_scope"} and result.input_tokens == 120
    assert not plain.calls
    sent = beta.calls[0]
    assert sent["model"] == "claude-opus-5-5"
    assert sent["betas"] == [FALLBACK_BETA] and sent["fallbacks"] == "default"
    assert sent["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["messages"] == [{"role": "user", "content": "USER"}]
    for banned in ("temperature", "top_p", "thinking", "tool_choice"):
        assert banned not in sent


def test_fallback_beta_rejection_degrades_to_a_plain_request():
    import anthropic

    plain = _Endpoint(_response("hello"))
    rejected = "Unexpected value(s) `server-side-fallback-2026-07-01` for the `anthropic-beta` header."
    beta = _Endpoint(_status_error(anthropic.BadRequestError, 400, rejected))
    llm = AnthropicLLM(api_key="test-key", client=_client(plain, beta))
    assert llm.complete_text(system="s", user="u").text == "hello"
    assert "fallbacks" not in plain.calls[0] and "betas" not in plain.calls[0]
    llm.complete_text(system="s", user="u")
    assert len(beta.calls) == 1 and len(plain.calls) == 2  # the beta is not retried


def test_an_unrelated_400_is_not_retried_and_stops_the_turn():
    import anthropic

    plain = _Endpoint(_response("hello"))
    beta = _Endpoint(_status_error(anthropic.BadRequestError, 400, "your organization must have data retention enabled"))
    llm = AnthropicLLM(api_key="test-key", client=_client(plain, beta))
    with pytest.raises(LLMError) as caught:
        llm.complete_text(system="s", user="u")
    assert caught.value.kind == "bad_request" and caught.value.fatal_for_turn
    assert not plain.calls and len(beta.calls) == 1  # one request, not three


def test_models_without_server_fallback_use_the_plain_endpoint():
    plain, beta = _Endpoint(_response("hi")), _Endpoint(_response("hi"))
    AnthropicLLM(api_key="k", model="claude-haiku-5-5", client=_client(plain, beta)).complete_text(system="s", user="u")
    assert len(plain.calls) == 1 and not beta.calls


@pytest.mark.parametrize("stop_reason, kind", [("refusal", "refusal"), ("max_tokens", "truncated")])
def test_stop_reason_is_checked_before_content(stop_reason: str, kind: str):
    endpoint = _Endpoint(_response("partial", stop_reason))
    llm = AnthropicLLM(api_key="k", model="claude-haiku-5-5", client=_client(endpoint, endpoint))
    with pytest.raises(LLMError) as caught:
        llm.complete_text(system="s", user="u")
    assert caught.value.kind == kind


def test_api_errors_are_mapped_to_stable_kinds():
    import anthropic

    cases = [
        (anthropic.AuthenticationError, 401, "auth"),
        (anthropic.PermissionDeniedError, 403, "permission"),
        (anthropic.NotFoundError, 404, "model"),
        (anthropic.RateLimitError, 429, "rate_limit"),
        (anthropic.InternalServerError, 500, "api"),
    ]
    for cls, status, kind in cases:
        endpoint = _Endpoint(_status_error(cls, status))
        llm = AnthropicLLM(api_key="k", model="claude-haiku-5-5", client=_client(endpoint, endpoint))
        with pytest.raises(LLMError) as caught:
            llm.complete_text(system="s", user="u")
        assert caught.value.kind == kind, cls.__name__


def test_invalid_json_from_the_model_is_an_error():
    endpoint = _Endpoint(_response("not json"))
    llm = AnthropicLLM(api_key="k", model="claude-haiku-5-5", client=_client(endpoint, endpoint))
    with pytest.raises(LLMError) as caught:
        llm.complete_json(system="s", user="u", schema={"type": "object"})
    assert caught.value.kind == "bad_output" and not caught.value.fatal_for_turn


def test_a_models_no_to_the_email_offer_is_honoured(agent: Agent):
    c = Convo(agent)
    c.run(DEMO_UTTERANCE, "No, that's all.")
    for answer in (reading(affirmation="no"), reading(email_summary_choice="skip")):
        other = Convo(agent)
        other.run(DEMO_UTTERANCE, "No, that's all.")
        other.llm = scripted([answer], ["No problem, I won't send it. Take care."])
        other.say("I'd rather you didn't, thanks")
        assert other.act == "EMAIL_SKIPPED" and not other.s.outbox and "send_email_summary" not in other.tools
    c.llm = scripted([reading()], ["Would you like me to send it, or skip it?"])
    c.say("hmm, let me think")
    assert c.act == "EMAIL_OFFER_REPEAT" and not c.s.outbox  # no reading of yes means nothing is sent
