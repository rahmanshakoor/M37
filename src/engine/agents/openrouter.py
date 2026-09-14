"""OpenRouter as a model provider — the same agent loop as :class:`AnthropicClient`,
spoken over OpenRouter's OpenAI-compatible ``/chat/completions``.

Why a second real client: the case runs on whichever key the operator has. Stages 5
and 6 are written against :class:`~engine.agents.client.ModelClient` only, so this
client takes the same :class:`~engine.agents.client.AgentRequest` and returns the same
:class:`~engine.agents.client.AgentResult` — validated pydantic ``output``, a
transcript of :class:`~engine.agents.client.Turn`/:class:`~engine.agents.client.ToolCall`
entries with the Anthropic stop-reason vocabulary, usage, model, stop reason,
disclosure — so the transcripts and manifests the stages write are shaped identically
whichever provider answered.

Why these request shapes and no others: every field sent here is one the OpenRouter
probe of 2026-09-13 took from OpenRouter's OpenAPI and guides for the Claude
endpoints and the skeptic's re-run confirmed (``tools`` as OpenAI function tools with
``strict``, ``tool_choice``, ``reasoning.effort``, ``response_format`` as a strict
``json_schema``, ``max_tokens``, top-level ``cache_control``, a ``provider`` block);
the probe ran without a key, so the tool round trip itself is documented shape, not
a recorded exchange — the live test in ``tests/test_providers.py`` is where that is
checked once a key exists. Nothing outside the first-party Anthropic endpoint's
parameter list is sent — no ``temperature``, no ``max_completion_tokens``, no
``parallel_tool_calls`` — because the review showed they make a pinned request
unroutable or are silently dropped. The tool loop follows the documented contract:
the ``tools`` array is resent on every request, every tool call of a turn is executed and answered with one
``role: tool`` message per ``tool_call_id`` before the next request, the assistant
message is echoed back verbatim with its ``reasoning_details`` unmodified (Claude's
thinking signatures are checked upstream), and the loop is bounded by ``max_turns``.
This skin has no ``is_error`` flag on a tool result, so an error result carries its
text in ``content`` (it always starts with ``Error:``) and is still recorded as
``is_error`` in the transcript.

Why every request carries ``provider: {"data_collection": "deny"}``: the payload
holds a patient's variants; ``deny`` keeps the request off any upstream that stores
inputs non-transiently or may train on them. The upstream is pinned (``order``,
``allow_fallbacks: false``) so a request cannot silently drift to an endpoint that
lacks structured outputs, and the response's provider/model echo and generation id
are recorded on the result so the manifest can say who actually served each turn.

Why the final answer is streamed and the loop turns are not: a 64k-token answer
with thinking can outlast any sane socket timeout, and OpenRouter's edge cuts a long
unstreamed request (its 524/408). With ``stream: true`` the connection carries
keep-alive comments while the model thinks and bytes as it writes, so the timeout
becomes an idle timeout; the events are assembled once the body is complete —
nothing here needs a token before the answer is whole. A loop turn is not streamed
because its ``reasoning_details`` must go back verbatim and the streamed form of
those blocks is not among the shapes the probe verified; its timeout is sized from
``max_tokens`` at the Anthropic SDK's own planning rate instead
(:func:`turn_timeout`).

Why a model call is never re-sent after a timeout: an identical re-send of a
request the model may already be answering is billed again, up to once per retry,
before the failure is even typed. The transport is built with no retries of its
own; this client retries only an answer that says nothing was generated — a 429 or
a 500/502/503 (:data:`RETRY_STATUSES`), with backoff — and turns a timeout, a dropped
connection or a 504 into one typed failure on the first occurrence.

Transport is :class:`engine.retrieve.http.Http` with ``cache_ok=False`` — a model
call is never served from or written to the evidence cache — and every failure is a
typed :class:`OpenRouterError` (an :class:`~engine.agents.client.AgentError` with the
HTTP status, OpenRouter's ``error_type`` and the generation id when there is one).
The error message carries the status and OpenRouter's short message, never the
request or the ``metadata`` block (a refusal's ``flagged_input`` echoes the prompt).
The key is held on the client and sent in a header; it appears in no log line, no
transcript and no manifest.
"""

from __future__ import annotations

import http.client
import json
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Iterator

from engine.agents import client as ac
from engine.retrieve.http import Http, HttpCache, HttpError, RateLimiter, Response, default_cache_root

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-opus-5"
"""The probe's recommended id: 1M context, 128k output, reasoning on by default,
tools + structured outputs on the first-party Anthropic endpoint."""
DEFAULT_UPSTREAM = "anthropic"
"""Provider slug (the endpoints listing's ``tag``) the request is pinned to. The base
slug does not match the 2x-price ``anthropic/fast`` tier."""
DATA_COLLECTION = "deny"
FINAL_CALL = "chat.completions stream=true response_format=json_schema"
CALL_TIMEOUT = 600.0
"""Seconds a call may go without a byte arriving. The streamed final answer sends
keep-alives while the model thinks, so for it this is an idle timeout; an unstreamed
loop turn must arrive whole, so its timeout is :func:`turn_timeout` — never less."""
SECONDS_PER_TOKEN = 3600.0 / 128_000
"""The Anthropic SDK's planning rate for an unstreamed call (128k tokens an hour): the
SDK refuses one whose ``max_tokens`` would outlast ten minutes at that rate; here the
same rate sizes the loop-turn timeout, since that turn is not streamed."""
REQUEST_RETRIES = 3
"""Re-sends allowed for an answer in :data:`RETRY_STATUSES`; every other failure is
typed on the first occurrence."""
RETRY_STATUSES = frozenset({429, 500, 502, 503})
"""Answers that say nothing was generated — refused, or the upstream failed before
answering — so a re-send is not billed twice. Not 504, 524 or 408: a timeout by any
name means the model may have been generating, and the request is never re-sent."""
STREAM_ACCEPT = "text/event-stream"
STOP_REASONS = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens",
                "content_filter": "refusal", "error": "error"}
"""OpenRouter ``finish_reason`` → the Anthropic stop reason the transcript uses;
``native_finish_reason`` (the raw Anthropic value) wins when present."""
_SCHEMA_NAME = re.compile(r"[^A-Za-z0-9_-]+")


class OpenRouterError(ac.AgentError):
    """OpenRouter (or the network in front of it) did not deliver an answer. ``status``
    is the HTTP status (or the ``error.code`` of an HTTP-200 error body), ``error_type``
    OpenRouter's classification when it gave one; ``request_id`` is the generation id."""

    def __init__(self, message: str, *, status: int | None = None, error_type: str | None = None,
                 request_id: str | None = None, transcript: list[ac.Turn] | None = None):
        super().__init__(message, request_id=request_id, transcript=transcript)
        self.status = status
        self.error_type = error_type


@dataclass
class OpenRouterResult(ac.AgentResult):
    """An :class:`~engine.agents.client.AgentResult` plus what OpenRouter echoed:
    which upstream served the calls, the model id it reports, every generation id
    (the audit handle for ``GET /generation``) and the credits charged."""

    provider_echo: str | None = None
    model_echo: str | None = None
    generation_ids: list[str] = field(default_factory=list)
    cost_usd: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {**super().as_dict(), "openrouter": {
            "provider": self.provider_echo, "model": self.model_echo,
            "generation_ids": list(self.generation_ids), "cost_usd": self.cost_usd,
        }}


def pinned_upstream(model: str, upstream: str | None = DEFAULT_UPSTREAM) -> str | None:
    """``upstream`` when it is the model's own author (``anthropic/claude-…`` →
    ``anthropic``), else ``None``: a pin to Anthropic can serve no DeepSeek or Kimi."""
    if not upstream:
        return None
    return upstream if model.split("/", 1)[0] == upstream else None


def disclosure(model: str, effort: str, *, upstream: str | None = DEFAULT_UPSTREAM,
               data_collection: str = DATA_COLLECTION) -> str:
    """The provider-disclosure line the reports and manifests carry."""
    upstream = pinned_upstream(model, upstream)
    where = (f"pinned to upstream provider {upstream!r} with no fallback" if upstream
             else "routed by OpenRouter across the model's upstream providers that support every "
                  "parameter of the request (require_parameters)")
    return (f"OpenRouter API, model {model}, effort {effort}, {where}; every request sets "
            f"provider.data_collection={data_collection!r}, so no upstream that stores inputs "
            "non-transiently or trains on them may serve it, and OpenRouter itself stores no "
            "prompts or completions unless the account opts in")


# ---------------------------------------------------------------------------- client

class OpenRouterClient:
    """``ModelClient`` over OpenRouter. ``api_key`` is required (the caller resolves it
    from the environment; see :mod:`engine.agents.providers`). ``model`` (an
    ``author/slug`` id) is what is sent when a request carries the Anthropic
    client's bare default (``claude-opus-5``), which OpenRouter does not know; a
    request naming an ``author/slug`` id is sent as it is, like the Anthropic client
    sends ``request.model``, and any other bare id is refused. ``upstream=None``
    lifts the provider pin (OpenRouter then load-balances; ``require_parameters``
    defaults on so structured outputs stay honoured). ``timeout`` is the idle
    timeout of the streamed final answer and the floor of a loop turn's
    (:func:`turn_timeout`). Pass ``http`` to inject a transport (tests) — one built
    with ``retries=0``, since a model call must never be re-sent by the transport;
    the default is built on the first call. ``retries`` and ``sleep`` shape this
    client's own retry of a :data:`RETRY_STATUSES` answer."""

    def __init__(self, api_key: str, *, model: str = DEFAULT_MODEL, base_url: str = BASE_URL,
                 effort: str = ac.DEFAULT_EFFORT, data_collection: str = DATA_COLLECTION,
                 app_title: str = "engine", referer: str | None = None,
                 upstream: str | None = DEFAULT_UPSTREAM, require_parameters: bool | None = None,
                 cache_control: bool = True, http: Http | None = None,
                 timeout: float = CALL_TIMEOUT, retries: int = REQUEST_RETRIES,
                 sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] | None = None):
        if not api_key:
            raise ValueError("OpenRouterClient needs an api_key")
        if "/" not in model:
            raise ValueError(f"OpenRouterClient model must be an OpenRouter id (author/slug), got {model!r}")
        if data_collection not in ("deny", "allow"):
            raise ValueError(f"data_collection must be 'deny' or 'allow', got {data_collection!r}")
        if retries < 0:
            raise ValueError("retries must be at least 0")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.effort = effort
        self.data_collection = data_collection
        self.app_title = app_title
        self.referer = referer
        self.upstream = upstream
        self.require_parameters = require_parameters  # None: decided per model in provider_preferences
        self.cache_control = cache_control
        self.timeout = timeout
        self.retries = retries
        self.sleep = sleep
        self.log = log or (lambda s: None)
        self._http = http

    # -- transport

    @property
    def http(self) -> Http:
        """The transport: no retries of its own (a timed-out model call is never
        re-sent — see the module docstring); :meth:`_chat` retries the statuses
        that are safe to."""
        if self._http is None:
            self._http = Http(HttpCache(default_cache_root()), limiter=RateLimiter(default_per_second=2.0),
                              timeout=self.timeout, retries=0)
        return self._http

    def headers(self, *, stream: bool = False) -> dict[str, str]:
        """Bearer auth, app attribution (kept out of OpenRouter's public rankings) and
        the metadata opt-in that names the serving provider in the body; a streamed
        call accepts SSE."""
        h = {"Authorization": f"Bearer {self._api_key}", "X-OpenRouter-Title": self.app_title,
             "X-OpenRouter-App-Visibility": "hidden", "X-OpenRouter-Metadata": "enabled"}
        if self.referer:
            h["HTTP-Referer"] = self.referer
        if stream:
            h["Accept"] = STREAM_ACCEPT
        return h

    def provider_preferences(self, model: str | None = None) -> dict[str, Any]:
        """The ``provider`` block for ``model``. A model is pinned to ``upstream`` only
        when that is its own author (``anthropic/...`` → Anthropic); any other model is
        routed by OpenRouter across its upstreams, still under ``data_collection`` and,
        unless told otherwise, with ``require_parameters`` so only an endpoint that
        supports every parameter in the request (tools, the JSON-schema answer,
        reasoning) may answer — an endpoint that silently ignores one would break
        the loop."""
        prefs: dict[str, Any] = {"data_collection": self.data_collection}
        pinned = pinned_upstream(model or self.model, self.upstream)
        if pinned:
            prefs["order"] = [pinned]
            prefs["allow_fallbacks"] = False
        if self.require_parameters or (self.require_parameters is None and not pinned):
            prefs["require_parameters"] = True
        return prefs

    def transport_params(self) -> dict[str, Any]:
        """How the calls are made — recorded under ``request`` in the transcript.
        Never the key."""
        return {"api": FINAL_CALL, "base_url": self.base_url, "provider": self.provider_preferences(self.model),
                "cache_control": self.cache_control, "app_title": self.app_title, "referer": self.referer,
                "timeout": self.timeout, "retries": self.retries, "retry_statuses": sorted(RETRY_STATUSES),
                "stream": {"turns": False, "final": True}}

    # -- the loop

    def run(self, request: ac.AgentRequest) -> OpenRouterResult:
        handlers = {t.name: t.handler for t in request.tools}
        model = self._model_for(request)
        base = self._base_body(request, model)
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.user})
        transcript: list[ac.Turn] = []
        usage = ac.Usage()
        echo = _Echo()
        stop_reason = "end_turn"
        turn_seconds = turn_timeout(request.max_tokens, self.timeout)
        n = 0
        while True:
            n += 1
            reply = self._chat({**base, "max_tokens": request.max_tokens, "messages": messages}, transcript,
                               timeout=turn_seconds)
            usage.add(reply.usage)
            echo.record(reply)
            messages.append(reply.assistant_message())
            self.log(f"turn {n}: stop_reason={reply.stop_reason} tool_calls={len(reply.tool_calls)}")
            exhausted = n > request.max_turns
            if reply.stop_reason in ("refusal", "max_tokens"):
                transcript.append(ac.Turn(n, reply.stop_reason, reply.text, request_id=reply.generation_id))
                why = "the model refused" if reply.stop_reason == "refusal" else "the turn was cut off at max_tokens"
                raise OpenRouterError(f"turn {n}: {why}", request_id=reply.generation_id, transcript=transcript)
            if reply.stop_reason != "tool_use" or not reply.tool_calls:
                transcript.append(ac.Turn(n, reply.stop_reason, reply.text, request_id=reply.generation_id))
                stop_reason = reply.stop_reason
                break
            calls: list[ac.ToolCall] = []
            for call in reply.tool_calls:
                inputs, parse_error = _arguments(call)
                if exhausted:
                    text_out, is_error = ac.TOOL_BUDGET_ERROR, True
                elif parse_error:
                    text_out, is_error = parse_error, True
                else:
                    try:
                        text_out, is_error = ac.call_tool(handlers, call.name, inputs)
                    except ac.ToolFailure as e:
                        transcript.append(ac.Turn(n, "tool_use", reply.text, calls, request_id=reply.generation_id))
                        raise ac.ToolFailure(e.tool, e.detail, request_id=reply.generation_id,
                                             transcript=transcript) from e.__cause__
                usage.count_tool_call(is_error)
                calls.append(ac.ToolCall(id=call.id, name=call.name, input=inputs, result=text_out, is_error=is_error))
                # No is_error flag in this skin: the error text is the content, all
                # results go back before the next request, one message per call id.
                messages.append({"role": "tool", "tool_call_id": call.id, "content": text_out})
            transcript.append(ac.Turn(n, "tool_use", reply.text, calls, request_id=reply.generation_id))
            if exhausted:
                stop_reason = "max_turns"
                break

        messages.append({"role": "user", "content": ac.FINAL_INSTRUCTION})
        final = self._chat(self._final_body(request, base, messages), transcript, timeout=self.timeout)
        usage.add(final.usage)
        echo.record(final)
        transcript.append(ac.Turn(n + 1, final.stop_reason, final.text, request_id=final.generation_id))
        output = ac.parse_answer(request.output_model, final.text, final.stop_reason,
                                 request_id=final.generation_id, transcript=transcript)
        return OpenRouterResult(
            output=output, transcript=transcript, usage=usage, model=model, stop_reason=stop_reason,
            effort=request.effort,
            disclosure=disclosure(model, request.effort, upstream=self.upstream, data_collection=self.data_collection),
            final_text=final.text, request={**request.params(), "model": model, "reasoning": {"effort": request.effort},
                                            **self.transport_params(), "turn_timeout": turn_seconds},
            provider_echo=echo.provider, model_echo=echo.model, generation_ids=echo.generation_ids, cost_usd=echo.cost,
        )

    # -- request bodies

    def _model_for(self, request: ac.AgentRequest) -> str:
        """The id sent: the request's own ``author/slug``, or this client's for the
        Anthropic client's bare default. Any other bare id is refused before a request
        is built — sending it would be a 400 after the bundle had left, and silently
        substituting ``self.model`` would answer with a model nobody asked for."""
        if "/" in request.model:
            return request.model
        if request.model == ac.DEFAULT_MODEL:
            return self.model
        raise OpenRouterError(f"OpenRouter model ids are author/slug (e.g. {self.model}), got {request.model!r}")

    def _base_body(self, request: ac.AgentRequest, model: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "reasoning": {"effort": request.effort},
            "provider": self.provider_preferences(model),
        }
        if self.cache_control:
            # Automatic prompt caching (top-level, documented for the Anthropic
            # upstream): the system prompt, bundle and tools repeat on every turn.
            body["cache_control"] = {"type": "ephemeral"}
        if request.tools:
            body["tools"] = [tool_param(t) for t in request.tools]
            body["tool_choice"] = "auto"
        return body

    @staticmethod
    def _final_body(request: ac.AgentRequest, base: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
        """The structured answer's request: ``response_format`` carries the request's
        answer schema (strict); tools stay declared so the transcript's tool messages
        remain valid, but the model may not call them; ``stream`` so the long answer
        arrives over a connection that stays alive (:func:`parse_stream`)."""
        body = {**base, "max_tokens": request.final_max_tokens, "messages": messages, "stream": True,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": _SCHEMA_NAME.sub("_", request.output_model.__name__)[:64] or "answer",
                    "strict": True, "schema": request.output_schema}}}
        if request.tools:
            body["tool_choice"] = "none"
        return body

    # -- one call

    def _chat(self, body: dict[str, Any], transcript: list[ac.Turn], *, timeout: float) -> _Reply:
        """One completion. A :data:`RETRY_STATUSES` answer is re-sent after a backoff,
        up to ``retries`` times; a timeout, a refused connection, a stream that
        dropped mid-answer or any other status is one typed failure — the request
        is never sent twice for it."""
        url = f"{self.base_url}/chat/completions"
        host = urllib.parse.urlparse(url).netloc
        streamed = bool(body.get("stream"))
        attempt = 0
        while True:
            try:
                resp = self.http.request("POST", url, json_body=body, headers=self.headers(stream=streamed),
                                         cache_ok=False, timeout=timeout)
            except HttpError as e:
                if e.status in RETRY_STATUSES and attempt < self.retries:
                    attempt += 1
                    wait = retry_wait(attempt)
                    self.log(f"OpenRouter answered {e.status}: retry {attempt}/{self.retries} in {wait:.0f}s")
                    self.sleep(wait)
                    continue
                raise _http_error(e, transcript) from e
            except OSError as e:  # URLError, sockets, timeouts — the model may be answering: not re-sent
                raise OpenRouterError(f"could not reach OpenRouter at {host}: {type(e).__name__}",
                                      transcript=transcript) from e
            except http.client.HTTPException as e:  # IncompleteRead: the stream dropped mid-answer
                raise OpenRouterError(f"the connection to OpenRouter at {host} dropped mid-answer: "
                                      f"{type(e).__name__}", transcript=transcript) from e
            except RuntimeError as e:  # Http: "request failed after retries" / "offline: ..."
                raise OpenRouterError(f"OpenRouter request failed: {e}", transcript=transcript) from e
            return parse_stream(resp, transcript) if streamed else parse_reply(resp, transcript)


def turn_timeout(max_tokens: int, floor: float = CALL_TIMEOUT) -> float:
    """Seconds an unstreamed loop turn of ``max_tokens`` is given to arrive whole:
    :data:`SECONDS_PER_TOKEN` per token, never under ``floor``."""
    return max(float(floor), max_tokens * SECONDS_PER_TOKEN)


def retry_wait(attempt: int) -> float:
    """Backoff before re-send ``attempt`` (1-based): 2, 4, 8 … seconds, capped at 60 —
    the transport's own curve, since OpenRouter's ``Retry-After`` does not survive
    :class:`~engine.retrieve.http.HttpError`."""
    return min(2.0 ** attempt, 60.0)


# --------------------------------------------------------------------------- shapes

def tool_param(tool: ac.ToolSpec) -> dict[str, Any]:
    """The API's function-tool definition: the same strict, closed schema the
    Anthropic client sends, under OpenAI's ``parameters`` key."""
    p = tool.param()
    return {"type": "function", "function": {"name": p["name"], "description": p["description"],
                                             "strict": True, "parameters": p["input_schema"]}}


@dataclass(frozen=True)
class _Call:
    id: str
    name: str
    arguments: str
    raw: dict[str, Any]


@dataclass
class _Reply:
    """One parsed completion: what the loop needs, plus the assistant message to echo."""

    generation_id: str | None
    model: str | None
    provider: str | None
    stop_reason: str
    text: str
    tool_calls: list[_Call]
    usage: Any
    cost: float | None
    message: dict[str, Any]

    def assistant_message(self) -> dict[str, Any]:
        """The assistant turn as the API must see it next time: content and tool
        calls as returned, ``reasoning_details`` verbatim (signed thinking blocks)."""
        out: dict[str, Any] = {"role": "assistant", "content": self.message.get("content")}
        if self.tool_calls:
            out["tool_calls"] = [c.raw for c in self.tool_calls]
        if self.message.get("reasoning_details"):
            out["reasoning_details"] = self.message["reasoning_details"]
        return out


class _Echo:
    def __init__(self) -> None:
        self.provider: str | None = None
        self.model: str | None = None
        self.generation_ids: list[str] = []
        self.cost: float | None = None

    def record(self, reply: _Reply) -> None:
        self.provider = reply.provider or self.provider
        self.model = reply.model or self.model
        if reply.generation_id:
            self.generation_ids.append(reply.generation_id)
        if reply.cost is not None:
            self.cost = (self.cost or 0.0) + reply.cost


def parse_reply(resp: Response, transcript: list[ac.Turn] | None = None) -> _Reply:
    """A ``/chat/completions`` body as a :class:`_Reply`, or :class:`OpenRouterError`.
    An HTTP 200 whose body carries ``error`` (a late upstream failure) is an error;
    so is a body without a choice."""
    try:
        body = resp.json()
    except ValueError as e:
        raise OpenRouterError(f"OpenRouter answered HTTP {resp.status} with a non-JSON body",
                              status=resp.status, transcript=transcript) from e
    if not isinstance(body, dict):
        raise OpenRouterError(f"OpenRouter answered HTTP {resp.status} with a non-object body",
                              status=resp.status, transcript=transcript)
    gid = _str_or_none(body.get("id"))
    if body.get("error"):
        code, error_type, message = _error_fields(body["error"])
        raise OpenRouterError(f"OpenRouter error {code or resp.status}{_typed(error_type)}: {message}",
                              status=code if isinstance(code, int) else resp.status, error_type=error_type,
                              request_id=gid, transcript=transcript)
    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise OpenRouterError(f"OpenRouter answered HTTP {resp.status} without a choice", status=resp.status,
                              request_id=gid, transcript=transcript)
    choice = choices[0]
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return _Reply(
        generation_id=gid, model=_str_or_none(body.get("model")), provider=_provider_of(body),
        stop_reason=stop_reason(choice), text=_content_text(message.get("content")),
        tool_calls=_tool_calls(message), usage=_usage(usage), cost=_float_or_none(usage.get("cost")), message=message,
    )


def parse_stream(resp: Response, transcript: list[ac.Turn] | None = None) -> _Reply:
    """A ``stream: true`` body — SSE events whose ``data`` is a ``chat.completion.chunk``
    — assembled into one :class:`_Reply`: ``delta.content`` concatenated, the last
    ``finish_reason``/``native_finish_reason`` seen, the ``usage`` of the chunk that
    carries it, the first generation id and the provider/model echo. Comment lines
    (``: OPENROUTER PROCESSING``) are keep-alives and skipped; ``[DONE]`` ends the
    stream. A chunk carrying ``error`` (a failure mid-stream) is an
    :class:`OpenRouterError`, and so is a stream that ends before any finish reason —
    a cut-off answer is never parsed as complete. A body that is one JSON object (an
    error envelope, or an answer sent unstreamed) is read by :func:`parse_reply`.
    Only the text is reassembled: a streamed reply is the final answer, with tools
    off and nothing echoed back, so tool-call and reasoning deltas are not."""
    text = resp.text.lstrip()
    if text.startswith("{"):
        return parse_reply(resp, transcript)
    gid: str | None = None
    model: str | None = None
    provider: str | None = None
    finish: str | None = None
    native: str | None = None
    usage: dict[str, Any] = {}
    parts: list[str] = []
    events = 0
    for payload in _sse_data(text):
        try:
            chunk = json.loads(payload)
        except ValueError as e:
            raise OpenRouterError(f"OpenRouter streamed a non-JSON event (HTTP {resp.status})", status=resp.status,
                                  request_id=gid, transcript=transcript) from e
        if not isinstance(chunk, dict):
            raise OpenRouterError(f"OpenRouter streamed a non-object event (HTTP {resp.status})", status=resp.status,
                                  request_id=gid, transcript=transcript)
        events += 1
        gid = gid or _str_or_none(chunk.get("id"))
        model = _str_or_none(chunk.get("model")) or model
        provider = _provider_of(chunk) or provider
        if chunk.get("error"):
            code, error_type, message = _error_fields(chunk["error"])
            raise OpenRouterError(f"OpenRouter error {code or resp.status}{_typed(error_type)} mid-stream: {message}",
                                  status=code if isinstance(code, int) else resp.status, error_type=error_type,
                                  request_id=gid, transcript=transcript)
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        parts.append(_content_text(delta.get("content")))
        if choice.get("finish_reason") is not None:
            finish = str(choice["finish_reason"])
        if isinstance(choice.get("native_finish_reason"), str) and choice["native_finish_reason"]:
            native = choice["native_finish_reason"]
    if not events:
        raise OpenRouterError(f"OpenRouter answered HTTP {resp.status} with an empty stream", status=resp.status,
                              request_id=gid, transcript=transcript)
    if finish is None and native is None:
        raise OpenRouterError("the OpenRouter stream ended before a finish reason", status=resp.status,
                              request_id=gid, transcript=transcript)
    message: dict[str, Any] = {"role": "assistant", "content": "".join(parts)}
    return _Reply(
        generation_id=gid, model=model, provider=provider,
        stop_reason=stop_reason({"finish_reason": finish, "native_finish_reason": native}),
        text=message["content"], tool_calls=[], usage=_usage(usage), cost=_float_or_none(usage.get("cost")),
        message=message,
    )


def _sse_data(text: str) -> Iterator[str]:
    """The ``data`` payload of every event in an SSE body (a multi-line ``data`` is
    joined with newlines; comments and other fields are skipped), up to the
    ``[DONE]`` sentinel or the end of the body."""
    data: list[str] = []
    for line in text.splitlines() + [""]:
        if line == "":
            if data:
                payload, data = "\n".join(data), []
                if payload.strip() == "[DONE]":
                    return
                yield payload
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        if name == "data":
            data.append(value[1:] if value.startswith(" ") else value)


ANTHROPIC_STOP_REASONS = {"end_turn", "tool_use", "max_tokens", "stop_sequence", "refusal", "pause_turn"}


def stop_reason(choice: dict[str, Any]) -> str:
    """Anthropic's vocabulary, so the transcript reads the same whichever client
    wrote it. ``native_finish_reason`` is the upstream's own word: for Anthropic it
    already is that vocabulary and wins; for any other upstream (DeepSeek, Moonshot,
    …) it is OpenAI's — ``tool_calls``/``stop``/``length`` — and is mapped like
    ``finish_reason``, otherwise the loop would read ``tool_calls`` as a final answer
    and never run the tools."""
    native = choice.get("native_finish_reason")
    if isinstance(native, str) and native:
        if native in ANTHROPIC_STOP_REASONS:
            return native
        if native in STOP_REASONS:
            return STOP_REASONS[native]
    finish = choice.get("finish_reason")
    if finish is not None:
        return STOP_REASONS.get(str(finish), str(finish))
    return native if isinstance(native, str) and native else "end_turn"


def _tool_calls(message: dict[str, Any]) -> list[_Call]:
    out: list[_Call] = []
    for raw in message.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        fn = raw.get("function") if isinstance(raw.get("function"), dict) else {}
        args = fn.get("arguments")
        out.append(_Call(id=str(raw.get("id") or f"call_{len(out) + 1}"), name=str(fn.get("name") or ""),
                         arguments=args if isinstance(args, str) else json.dumps(args or {}), raw=raw))
    return out


def _arguments(call: _Call) -> tuple[dict[str, Any], str | None]:
    """``function.arguments`` is a JSON string — always parsed, never string-matched.
    Text that is not a JSON object becomes an error result the model can repair."""
    try:
        parsed = json.loads(call.arguments) if call.arguments.strip() else {}
    except ValueError:
        return {}, "Error: tool arguments were not valid JSON"
    if not isinstance(parsed, dict):
        return {}, "Error: tool arguments must be a JSON object"
    return parsed, None


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts: keep the text ones
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content)


def _usage(usage: dict[str, Any]) -> SimpleNamespace:
    """OpenRouter's usage under the attribute names :meth:`Usage.add` reads."""
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    return SimpleNamespace(
        input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
        cache_read_input_tokens=details.get("cached_tokens"), cache_creation_input_tokens=details.get("cache_write_tokens"),
    )


def _provider_of(body: dict[str, Any]) -> str | None:
    """The upstream that served the call: the metadata's successful attempt (opted in
    by header), else the top-level echo when OpenRouter includes one."""
    meta = body.get("openrouter_metadata")
    if isinstance(meta, dict):
        attempts = [a for a in meta.get("attempts") or [] if isinstance(a, dict)]
        served = [a for a in attempts if a.get("status") == 200] or attempts
        if served and served[-1].get("provider"):
            return str(served[-1]["provider"])
    return _str_or_none(body.get("provider"))


# --------------------------------------------------------------------------- errors

def _http_error(e: HttpError, transcript: list[ac.Turn]) -> OpenRouterError:
    code, error_type, message = _error_fields(_error_body(e.body))
    if e.status == 429:
        head = "OpenRouter rate limit (429)"
    elif e.status in (401, 403):
        head = f"OpenRouter refused the key or the request ({e.status})"
    else:
        head = f"OpenRouter API error {e.status}"
    return OpenRouterError(f"{head}{_typed(error_type)}: {message}", status=e.status, error_type=error_type,
                           transcript=transcript)


def _error_body(text: str) -> Any:
    try:
        doc = json.loads(text) if text else {}
    except ValueError:
        return {}
    return doc.get("error") if isinstance(doc, dict) else {}


def _error_fields(err: Any) -> tuple[int | str | None, str | None, str]:
    """``(code, error_type, message)`` from an ``error`` object; the message is
    short and never the ``metadata`` block (a refusal echoes the prompt there)."""
    if not isinstance(err, dict):
        return None, None, "no error detail"
    code = err.get("code")
    meta = err.get("metadata") if isinstance(err.get("metadata"), dict) else {}
    error_type = _str_or_none(meta.get("error_type"))
    message = str(err.get("message") or "no error detail")[:200]
    return (code if isinstance(code, (int, str)) else None), error_type, message


def _typed(error_type: str | None) -> str:
    return f" [{error_type}]" if error_type else ""


def _str_or_none(v: Any) -> str | None:
    return str(v) if isinstance(v, (str, int)) and str(v) else None


def _float_or_none(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
