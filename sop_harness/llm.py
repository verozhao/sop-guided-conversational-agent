"""Model access.

The harness needs exactly two operations from a language model:

* ``complete_json`` - read a caller message and return a JSON object that
  follows a schema (understanding)
* ``complete_text`` - write the reply for a turn brief (phrasing)

``AnthropicLLM`` implements both with the official Anthropic SDK. Anything
with the same two methods can be plugged in, which is how the tests run
without network access.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Protocol

DEFAULT_MODEL = "claude-opus-5-5"
SELECTABLE_MODELS = [
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "claude-fable-5-2",
]
EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}

# A safety classifier can decline a request on these models. The server-side
# fallback re-runs a declined request on another model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MODELS_WITH_SERVER_FALLBACK = {
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-fable-5-2",
    "claude-fable-5-1",
}


class LLMError(Exception):
    """A model call failed. ``kind`` is stable and safe to show to a tester."""

    def __init__(self, kind: str, message: str, result: "LLMResult | None" = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.result = result  # set when the API answered (and billed) before the call was judged a failure

    @property
    def fatal_for_turn(self) -> bool:
        """True when a second call in the same turn would fail the same way."""
        return self.kind in {"auth", "permission", "model", "billing", "rate_limit", "connection", "bad_request", "budget"}


@dataclass
class LLMResult:
    text: str
    data: dict[str, Any] | None = None
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def usage(self, purpose: str) -> dict[str, Any]:
        return {
            "purpose": purpose,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


class LLM(Protocol):
    name: str

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any]) -> LLMResult: ...

    def complete_text(self, *, system: str, user: str) -> LLMResult: ...


class AnthropicLLM:
    """Claude through the Messages API.

    Notes on the request shape (current Claude models):

    * thinking is adaptive and always on, so ``thinking`` is omitted and depth
      is set with ``output_config.effort``
    * sampling parameters and forced ``tool_choice`` are rejected, so JSON is
      requested with structured outputs (``output_config.format``)
    * ``stop_reason`` is checked before content is read, because a declined
      request is an HTTP 200 with ``stop_reason == "refusal"``
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        effort: str | None = "low",
        max_tokens: int = 16000,
        timeout: float = 90.0,
        use_fallbacks: bool = True,
        client: Any = None,
    ):
        import anthropic

        self._sdk = anthropic
        # The key is always passed explicitly. A bare ``Anthropic()`` would
        # pick up credentials from the environment or a local profile, which
        # this demo must never do on someone's machine.
        self.client = client or anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)
        self.model = model
        self.effort = effort if effort in EFFORT_LEVELS else None
        self.max_tokens = max_tokens
        self.name = f"llm:{model}"
        self._fallbacks_on = use_fallbacks and model in MODELS_WITH_SERVER_FALLBACK
        self.fallback_note = ""

    # -- public API ---------------------------------------------------------

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        result = self._complete(system, user, schema)
        try:
            data = json.loads(result.text)
        except json.JSONDecodeError as exc:
            raise LLMError("bad_output", "The model did not return valid JSON.", result) from exc
        if not isinstance(data, dict):
            raise LLMError("bad_output", "The model returned JSON that is not an object.", result)
        result.data = data
        return result

    def complete_text(self, *, system: str, user: str) -> LLMResult:
        return self._complete(system, user, None)

    # -- internals ----------------------------------------------------------

    def _complete(self, system: str, user: str, schema: dict[str, Any] | None) -> LLMResult:
        output_config: dict[str, Any] = {}
        if self.effort:
            output_config["effort"] = self.effort
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # The system prompt is the stable prefix, so it carries the cache
            # breakpoint. Per-turn content stays in the user message.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
        }
        if output_config:
            kwargs["output_config"] = output_config

        response = self._send(kwargs)

        try:
            stop_reason = response.stop_reason
            text = "".join(block.text for block in response.content if block.type == "text").strip()
            usage = response.usage
        except (AttributeError, TypeError) as exc:
            # A gateway in front of the API answered 200 with something that is not a message.
            raise LLMError("bad_output", "The model endpoint returned a response that is not a message.") from exc
        result = LLMResult(
            text=text,
            model=getattr(response, "model", self.model) or self.model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
        # These calls were answered and billed, so the usage travels with the error.
        if stop_reason == "refusal":
            raise LLMError("refusal", "The model declined this request.", result)
        if stop_reason == "max_tokens":
            raise LLMError("truncated", "The model reply was cut off by max_tokens.", result)
        if not text:
            raise LLMError("empty", "The model returned no text.", result)
        return result

    def _send(self, kwargs: dict[str, Any]) -> Any:
        sdk = self._sdk
        try:
            if self._fallbacks_on:
                try:
                    return self.client.beta.messages.create(
                        betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                    )
                except sdk.BadRequestError as exc:
                    # Only when the fallback option itself was rejected (the beta is not
                    # enabled for this account): send a plain request instead. Any other
                    # 400 would fail the same way again, so it is raised as it is.
                    if not any(word in _short(exc).lower() for word in ("beta", "fallback")):
                        raise
                    self._fallbacks_on = False
                    self.fallback_note = _short(exc)
            return self.client.messages.create(**kwargs)
        except sdk.AuthenticationError as exc:
            raise LLMError("auth", "The API token was rejected (401). Check the token.") from exc
        except sdk.PermissionDeniedError as exc:
            raise LLMError("permission", f"The API token is not allowed to do this: {_short(exc)}") from exc
        except sdk.NotFoundError as exc:
            raise LLMError("model", f"Model '{self.model}' is not available to this token.") from exc
        except sdk.RateLimitError as exc:
            raise LLMError("rate_limit", "Rate limited by the model API. Try again shortly.") from exc
        except sdk.BadRequestError as exc:
            raise LLMError("bad_request", f"The model API rejected the request: {_short(exc)}") from exc
        except sdk.APIStatusError as exc:
            kind = "billing" if exc.status_code == 402 else "api"
            raise LLMError(kind, f"Model API error {exc.status_code}: {_short(exc)}") from exc
        except sdk.APIConnectionError as exc:
            raise LLMError("connection", "Could not reach the model API (network or timeout).") from exc
        except LLMError:
            raise
        except Exception as exc:  # anything else from the client is still only a failed model call
            raise LLMError("api", f"Unexpected error from the model client: {type(exc).__name__}") from exc


def _short(exc: Exception) -> str:
    return (getattr(exc, "message", None) or str(exc))[:240]


class ScriptedLLM:
    """A stand-in model driven by two callables. Used by tests and demos."""

    def __init__(
        self,
        on_json: Callable[[str, str, dict[str, Any]], dict[str, Any]] | None = None,
        on_text: Callable[[str, str], str] | None = None,
        name: str = "llm:scripted",
    ):
        self.on_json = on_json
        self.on_text = on_text
        self.name = name
        self.calls: list[dict[str, str]] = []

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        self.calls.append({"kind": "json", "system": system, "user": user})
        if self.on_json is None:
            raise LLMError("bad_output", "scripted model has no JSON handler")
        data = self.on_json(system, user, schema)
        return LLMResult(text=json.dumps(data), data=data, model=self.name)

    def complete_text(self, *, system: str, user: str) -> LLMResult:
        self.calls.append({"kind": "text", "system": system, "user": user})
        if self.on_text is None:
            raise LLMError("empty", "scripted model has no text handler")
        return LLMResult(text=self.on_text(system, user), model=self.name)


# -- client cache -----------------------------------------------------------

_CACHE: "OrderedDict[str, AnthropicLLM]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 16


def get_anthropic_llm(api_key: str, model: str = DEFAULT_MODEL, effort: str | None = "low") -> AnthropicLLM:
    """Reuse one client per (token, model, effort) so connections are pooled.

    The cache is keyed by a hash of the token. The token itself is held only
    inside the SDK client object, in memory.
    """
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    key = f"{digest}:{model}:{effort}"
    with _CACHE_LOCK:
        llm = _CACHE.get(key)
        if llm is None:
            llm = AnthropicLLM(api_key=api_key, model=model, effort=effort)
            _CACHE[key] = llm
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
        else:
            _CACHE.move_to_end(key)
        return llm
