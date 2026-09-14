"""Model access for the two agents — one protocol, a real client and a scripted one.

Why a protocol: stages 5 and 6 are written against :class:`ModelClient` only, so the
same stage code runs against the Anthropic API in production and against
:class:`FakeClient` in tests, where every turn is scripted and no network exists.

Why a manual tool loop rather than the SDK's tool runner: the transcript is an
artefact. Every tool call, its input, its result and whether it failed is recorded in
:class:`AgentResult.transcript` and written into the run directory, so a judge can see
what the model asked for and what it was told. The loop follows the SDK's documented
manual pattern: every ``tool_use`` block of a turn is executed, all results go back in
one ``user`` message, and the loop is bounded by ``max_turns``.

Two kinds of tool failure are kept apart. A handler that rejects its *argument*
(``KeyError`` for an unknown record id, ``ValueError`` for a malformed query) becomes
an ``is_error`` tool result the model can route around. A failure of the *service*
behind the tool — the network, an API that answered badly, a cache miss when offline —
raises :class:`ToolFailure` out of the loop, because an answer written around a
silently missing paper is exactly the kind of absence this engine refuses to produce.

Why a separate final call: the loop is free-form (prose, thinking, tool calls); the
answer is a JSON document. Once the model stops asking for tools, one call with
``output_config.format`` constrains the reply to a JSON schema derived from the
stage's pydantic model (:func:`answer_schema`), and the text is validated against that
model here. The schema never asks for what the engine computes (``classification``)
and pins ``code`` to the ACMG codes, so the API itself refuses an invented one.
Anything short of a schema-valid ``end_turn`` answer — a refusal, a cut-off at
``max_tokens``, a document the model rejects — raises :class:`AgentError`. The error
message names the stop reason and a fingerprint of the text, never the text itself
(it may hold a patient variant); the raw text travels on ``AgentError.final_text`` for
the stage to write into the run directory.

Every call is streamed (``messages.stream(...).get_final_message()``): thinking tokens
count against ``max_tokens`` under adaptive thinking, so a turn can be long, and the
SDK refuses an unstreamed request whose ``max_tokens`` could outlast its ten-minute
timeout. The request id of a streamed call lives on the stream, not the message; it
is captured there and recorded on every turn — it is what a failure report needs.

Every request pins the model, adaptive thinking and an effort level; every result
carries the provider disclosure line the reports print and the request parameters a
manifest must record, including a hash of the exact answer schema and tool set.
"""

from __future__ import annotations

import copy
import hashlib
import json
import urllib.parse
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, get_args, runtime_checkable

import anthropic
import pydantic
from pydantic import BaseModel

from engine.agents.schema import ALL_CODES, Criterion
from engine.retrieve.http import HttpError

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOKENS = 32000
"""Per loop turn (streamed). Thinking counts against it; a turn that hits it is a
hard failure, so the budget is generous."""
DEFAULT_FINAL_MAX_TOKENS = 16000
"""For the streamed final answer: a full evidence chain plus its thinking is long."""
EFFORTS = ("low", "medium", "high", "xhigh", "max")
THINKING = {"type": "adaptive"}
"""Adaptive thinking: the documented mode for claude-opus-5 (``budget_tokens`` is
rejected there)."""
ENGINE_FIELDS = ("classification",)
"""Fields the engine fills in after validation; never asked of the model, so the
default answer schema drops them wherever they occur."""
FINAL_CALL = "messages.stream"

FINAL_INSTRUCTION = (
    "Write your final answer now as a single JSON document in the required schema. "
    "Cite only record ids that appear in the bundle or were returned by a tool in this "
    "conversation, spelled exactly as given; never invent a record id, a PMID or a trial "
    "id. Leave a list empty rather than guess."
)
TOOL_BUDGET_ERROR = "Error: the tool-call budget for this task is exhausted; answer with the evidence already gathered."

ToolHandler = Callable[[dict[str, Any]], Any]


class AgentError(RuntimeError):
    """The model API failed, the loop could not finish, a tool's service failed, or
    the final answer was not a schema-valid document. Never raised for a handler that
    rejects its argument — that is returned to the model.

    The message never quotes the model's text. What there is of a final answer is on
    ``final_text``; the turns completed before the failure are on ``transcript``."""

    def __init__(self, message: str, *, request_id: str | None = None,
                 final_text: str | None = None, transcript: list[Turn] | None = None):
        super().__init__(f"{message} (request id {request_id})" if request_id else message)
        self.request_id = request_id
        self.final_text = final_text
        self.transcript = list(transcript or [])


class ToolFailure(AgentError):
    """The service behind a tool failed (network, API, offline cache miss). Carries the
    tool's name; the underlying exception is chained."""

    def __init__(self, tool: str, detail: str, *, request_id: str | None = None,
                 transcript: list[Turn] | None = None):
        super().__init__(f"tool {tool!r} failed: {detail}", request_id=request_id, transcript=transcript)
        self.tool = tool
        self.detail = detail


def disclosure(model: str, effort: str) -> str:
    return (f"Anthropic API, model {model}, effort {effort}; API inputs are not used for "
            "training under Anthropic's commercial terms")


# ---------------------------------------------------------------------------- schemas

def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``schema`` closed for strict use: every object gets
    ``additionalProperties: false`` and *every* property listed in ``required`` (an
    optional argument must be spelled as nullable, not omitted — that is what the
    API's strict mode guarantees against). Applied recursively through
    ``properties``, ``items``, ``anyOf``/``oneOf``/``allOf`` and ``$defs``."""
    out = copy.deepcopy(schema)
    _close(out)
    return out


def _close(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _close(item)
        return
    if not isinstance(node, dict):
        return
    if node.get("type") == "object" or "properties" in node:
        props = node.setdefault("properties", {})
        node["additionalProperties"] = False
        node["required"] = sorted(props)
        for p in props.values():
            _close(p)
    for key in ("items", "anyOf", "oneOf", "allOf"):
        if key in node:
            _close(node[key])
    if "$defs" in node:
        for d in node["$defs"].values():
            _close(d)


def answer_schema(model: type[BaseModel], *, drop: Iterable[str] = (),
                  enum: Mapping[str, Sequence[Any]] | None = None) -> dict[str, Any]:
    """The JSON schema the final answer is constrained to (``output_config.format``):
    the model's own schema, rewritten by the SDK's ``transform_schema`` into the subset
    the API accepts, then closed by :func:`strict_schema` so every field must be
    written — a list the model has nothing for is ``[]``, never absent.

    ``drop`` removes named properties from every object: a field the engine fills in
    (``classification``) is not asked of the model. ``enum`` pins named properties to
    a value set, so the API rejects a spelling the model would otherwise invent."""
    schema = copy.deepcopy(model.model_json_schema())
    _prune(schema, set(drop), dict(enum or {}))
    return strict_schema(anthropic.transform_schema(schema))


def default_answer_schema(model: type[BaseModel]) -> dict[str, Any]:
    """:func:`answer_schema` with the engine's own pins: :data:`ENGINE_FIELDS` dropped,
    and ``code`` pinned to the ACMG codes wherever ``model`` embeds a
    :class:`~engine.agents.schema.Criterion` (only there — a ``code`` field of some
    other model means something else)."""
    enum = {"code": sorted(ALL_CODES)} if Criterion in models_used(model) else {}
    return answer_schema(model, drop=ENGINE_FIELDS, enum=enum)


def models_used(model: type[BaseModel]) -> set[type[BaseModel]]:
    """``model`` and every pydantic model reachable from its fields (``list[X]``,
    ``X | None``, nested)."""
    seen: set[type[BaseModel]] = set()

    def walk(m: type[BaseModel]) -> None:
        if m in seen:
            return
        seen.add(m)
        for info in m.model_fields.values():
            for t in _model_types(info.annotation):
                walk(t)

    walk(model)
    return seen


def _model_types(annotation: Any) -> list[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return [t for arg in get_args(annotation) for t in _model_types(arg)]


def _prune(node: Any, drop: set[str], enum: dict[str, Sequence[Any]]) -> None:
    if isinstance(node, list):
        for item in node:
            _prune(item, drop, enum)
        return
    if not isinstance(node, dict):
        return
    props = node.get("properties")
    if isinstance(props, dict):
        for name in list(props):
            if name in drop:
                del props[name]
            elif name in enum:
                props[name] = {**props[name], "enum": list(enum[name])}
        if isinstance(node.get("required"), list):
            node["required"] = [r for r in node["required"] if r not in drop]
        for p in props.values():
            _prune(p, drop, enum)
    for key in ("items", "anyOf", "oneOf", "allOf"):
        if key in node:
            _prune(node[key], drop, enum)
    if "$defs" in node:
        for d in node["$defs"].values():
            _prune(d, drop, enum)


# ---------------------------------------------------------------------------- request

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def param(self) -> dict[str, Any]:
        """The API's tool definition (``strict: true`` with a closed input schema)."""
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": strict_schema(self.input_schema),
        }


@dataclass
class AgentRequest:
    system: str
    user: str
    output_model: type[BaseModel]
    """The pydantic model the final answer must validate against."""
    tools: list[ToolSpec] = field(default_factory=list)
    max_turns: int = 10
    """Tool-calling turns allowed. The turn after the last one has its tool calls
    refused with :data:`TOOL_BUDGET_ERROR` and the final answer is demanded."""
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    max_tokens: int = DEFAULT_MAX_TOKENS
    """Per loop turn."""
    final_max_tokens: int = DEFAULT_FINAL_MAX_TOKENS
    """For the streamed final answer."""
    output_schema: dict[str, Any] | None = None
    """What ``output_config.format`` constrains the final answer to. Default:
    :func:`default_answer_schema` of ``output_model``; pass one built with
    :func:`answer_schema`'s ``drop``/``enum`` to ask the model for less."""

    def __post_init__(self) -> None:
        if self.effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}, got {self.effort!r}")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        names = [t.name for t in self.tools]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tool names: {names}")
        if self.output_schema is None:
            self.output_schema = default_answer_schema(self.output_model)

    def params(self) -> dict[str, Any]:
        """Every parameter that shapes the answer — what a stage manifest records. The
        answer schema and the tool definitions are represented by their hashes, so two
        runs with different pins (a dropped field, an enum) are told apart."""
        return {
            "model": self.model,
            "effort": self.effort,
            "thinking": dict(THINKING),
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
            "final_max_tokens": self.final_max_tokens,
            "tools": [t.name for t in self.tools],
            "tool_definitions_sha256": sha256_json([t.param() for t in self.tools]),
            "output_model": self.output_model.__name__,
            "output_schema_sha256": sha256_json(self.output_schema),
        }


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ----------------------------------------------------------------------------- result

@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]
    result: str
    is_error: bool = False


@dataclass
class Turn:
    n: int
    stop_reason: str
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    request_id: str | None = None
    """The API's ``request-id`` for this turn — what Anthropic asks for in a failure report."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    api_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    """Tool calls answered with ``is_error`` (bad argument, unknown tool, budget)."""

    def add(self, usage: Any) -> None:
        """Accumulate one response's ``usage`` object (attributes may be ``None``)."""
        self.api_calls += 1
        for name in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            v = getattr(usage, name, None)
            if v:
                setattr(self, name, getattr(self, name) + int(v))

    def count_tool_call(self, is_error: bool) -> None:
        self.tool_calls += 1
        self.tool_errors += int(is_error)


@dataclass
class AgentResult:
    output: Any
    """The final answer, validated against ``AgentRequest.output_model``."""
    transcript: list[Turn]
    usage: Usage
    model: str
    stop_reason: str
    effort: str
    disclosure: str
    final_text: str = ""
    """The final answer exactly as the model wrote it (JSON), before any validation
    downstream — what a judge compares the rendered report against."""
    request: dict[str, Any] = field(default_factory=dict)
    """:meth:`AgentRequest.params` plus how the client made the calls."""

    def as_dict(self) -> dict[str, Any]:
        """Everything but the parsed output, for manifests and transcript files."""
        return {
            "model": self.model,
            "effort": self.effort,
            "stop_reason": self.stop_reason,
            "disclosure": self.disclosure,
            "request": dict(self.request),
            "usage": asdict(self.usage),
            "transcript": [asdict(t) for t in self.transcript],
            "final_text": self.final_text,
        }


@runtime_checkable
class ModelClient(Protocol):
    def run(self, request: AgentRequest) -> AgentResult: ...


# ------------------------------------------------------------------------ tool calls

def call_tool(handlers: dict[str, ToolHandler], name: str, inputs: dict[str, Any]) -> tuple[str, bool]:
    """Run one tool. Returns ``(text, is_error)``: an unknown tool or a handler that
    rejects its argument is reported to the model as an error result. A handler whose
    *service* failed (:func:`is_infrastructure_failure`) raises :class:`ToolFailure`."""
    handler = handlers.get(name)
    if handler is None:
        return f"Error: unknown tool {name!r}", True
    try:
        return _as_text(handler(dict(inputs))), False
    except Exception as e:  # every failure is caught, then sorted into "the model's problem" and "ours"
        if is_infrastructure_failure(e):
            raise ToolFailure(name, _describe(e)) from e
        return f"Error: {type(e).__name__}: {e}", True


def is_infrastructure_failure(e: BaseException) -> bool:
    """Whether ``e`` is a failure of the world rather than of the model's argument:
    an HTTP error after retries, anything ``OSError`` (``URLError``, sockets,
    timeouts), a retriever's own "the API answered, but not usably" error (every one
    is a ``RuntimeError`` defined under ``engine.retrieve`` or ``engine.medicine``), or
    :class:`~engine.retrieve.http.Http`'s offline cache miss."""
    if isinstance(e, (HttpError, OSError)):
        return True
    if isinstance(e, RuntimeError):
        module = type(e).__module__ or ""
        return module.startswith(("engine.retrieve.", "engine.medicine.")) or str(e).startswith("offline:")
    return False


def _describe(e: BaseException) -> str:
    """The failure without its payload: an HTTP error names status and host only, an
    offline miss says so, anything else is its class — the chained exception has the rest."""
    if isinstance(e, HttpError):
        return f"HTTP {e.status} from {urllib.parse.urlparse(e.url).netloc}"
    if isinstance(e, RuntimeError) and str(e).startswith("offline:"):
        return "offline: no cached response for the request"
    return type(e).__name__


def _as_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    return json.dumps(result, sort_keys=True, ensure_ascii=False, default=str)


def _text_of(content: list[Any]) -> str:
    return "".join(getattr(b, "text", "") for b in content if getattr(b, "type", "") == "text")


def _request_id(response: Any) -> str | None:
    rid = getattr(response, "_request_id", None)
    return str(rid) if rid else None


def _stop_details(response: Any) -> str:
    """`` (category cyber)`` when a refusal carries structured details, else ''."""
    details = getattr(response, "stop_details", None)
    category = getattr(details, "category", None) if details is not None else None
    return f" (category {category})" if category else ""


# ------------------------------------------------------------------ final answer

def fingerprint(text: str) -> str:
    """``1234 chars, sha256 0f3a…`` — enough to match a text file, none of its content."""
    return f"{len(text)} chars, sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}"


def parse_answer(model: type[BaseModel], text: str, stop_reason: str, *, request_id: str | None = None,
                 stop_details: str = "", transcript: list[Turn] | None = None) -> BaseModel:
    """The final text as ``model``, or :class:`AgentError`. Only an ``end_turn`` answer
    is parsed: ``max_tokens`` means a truncated document, ``refusal`` means none. A
    validation failure is reported by field path and error type — never by value."""
    if stop_reason != "end_turn":
        why = {"max_tokens": "the answer was cut off at max_tokens",
               "refusal": "the model refused" + stop_details}.get(stop_reason, f"stop_reason {stop_reason!r}")
        raise AgentError(f"final answer not usable: {why}; {fingerprint(text)}",
                         request_id=request_id, final_text=text, transcript=transcript)
    try:
        return model.model_validate_json(text)
    except pydantic.ValidationError as e:
        where = "; ".join(f"{'.'.join(str(x) for x in err['loc']) or '<root>'}: {err['type']}" for err in e.errors()[:6])
        raise AgentError(f"final answer does not fit {model.__name__}: {e.error_count()} error(s) — {where}; "
                         f"{fingerprint(text)}", request_id=request_id, final_text=text, transcript=transcript) from e


# ----------------------------------------------------------------- Anthropic client

class AnthropicClient:
    """The real thing. ``anthropic.Anthropic()`` resolves credentials from the
    environment; pass ``client`` to inject a configured or fake SDK client. Every
    call goes through :meth:`call` — ``messages.stream``, whose final message and
    request id are returned together."""

    def __init__(self, client: Any | None = None, *, log: Callable[[str], None] | None = None):
        self.client = client if client is not None else anthropic.Anthropic()
        self.log = log or (lambda s: None)

    def run(self, request: AgentRequest) -> AgentResult:
        try:
            return self._run(request)
        except anthropic.RateLimitError as e:
            retry_after = e.response.headers.get("retry-after", "?") if getattr(e, "response", None) is not None else "?"
            raise AgentError(f"Anthropic API rate limit (retry-after {retry_after}s): {e.message}") from e
        except anthropic.APIStatusError as e:
            raise AgentError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise AgentError(f"could not reach the Anthropic API: {e}") from e

    def call(self, **kwargs: Any) -> tuple[Any, str | None]:
        """One streamed request: ``(final message, request id)``. The SDK raises a
        plain ``TypeError``/``ValueError`` for a request it will not send (no
        credentials resolved, an argument it rejects) — those become
        :class:`AgentError` too, so a stage sees one error type for "no answer"."""
        try:
            with self.client.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
                return message, getattr(stream, "request_id", None) or _request_id(message)
        except (TypeError, ValueError) as e:
            raise AgentError(f"the Anthropic SDK refused the request: {type(e).__name__}: {e}") from e

    def _params(self, request: AgentRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "thinking": THINKING,
            "output_config": {"effort": request.effort},
            # Automatic prompt caching: the bundle and tool definitions are the same on
            # every turn of the loop, so the prefix is served from cache after the first.
            "cache_control": {"type": "ephemeral"},
        }
        if request.system:
            params["system"] = request.system
        if request.tools:
            params["tools"] = [t.param() for t in request.tools]
        return params

    def _run(self, request: AgentRequest) -> AgentResult:
        handlers = {t.name: t.handler for t in request.tools}
        params = self._params(request)
        messages: list[dict[str, Any]] = [{"role": "user", "content": request.user}]
        transcript: list[Turn] = []
        usage = Usage()
        stop_reason = "end_turn"
        n = 0
        while True:
            n += 1
            response, rid = self.call(messages=messages, **params)
            usage.add(response.usage)
            text = _text_of(response.content)
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            messages.append({"role": "assistant", "content": response.content})
            self.log(f"turn {n}: stop_reason={response.stop_reason} tool_calls={len(tool_uses)}")
            exhausted = n > request.max_turns
            if response.stop_reason == "pause_turn" and not exhausted:
                # Server paused mid-turn: re-send as is, no extra user message.
                transcript.append(Turn(n, "pause_turn", text, request_id=rid))
                continue
            if response.stop_reason in ("refusal", "max_tokens"):
                transcript.append(Turn(n, str(response.stop_reason), text, request_id=rid))
                why = "the model refused" + _stop_details(response) if response.stop_reason == "refusal" \
                    else "the turn was cut off at max_tokens"
                raise AgentError(f"turn {n}: {why}", request_id=rid, transcript=transcript)
            if response.stop_reason != "tool_use" or not tool_uses:
                transcript.append(Turn(n, str(response.stop_reason), text, request_id=rid))
                stop_reason = "max_turns" if exhausted and response.stop_reason == "pause_turn" else str(response.stop_reason)
                break
            calls: list[ToolCall] = []
            results: list[dict[str, Any]] = []
            for b in tool_uses:
                inputs = dict(b.input) if isinstance(b.input, dict) else {}
                if exhausted:
                    text_out, is_error = TOOL_BUDGET_ERROR, True
                else:
                    try:
                        text_out, is_error = call_tool(handlers, b.name, inputs)
                    except ToolFailure as e:
                        transcript.append(Turn(n, "tool_use", text, calls, request_id=rid))
                        raise ToolFailure(e.tool, e.detail, request_id=rid, transcript=transcript) from e.__cause__
                usage.count_tool_call(is_error)
                calls.append(ToolCall(id=b.id, name=b.name, input=inputs, result=text_out, is_error=is_error))
                result: dict[str, Any] = {"type": "tool_result", "tool_use_id": b.id, "content": text_out}
                if is_error:
                    result["is_error"] = True
                results.append(result)
            transcript.append(Turn(n, "tool_use", text, calls, request_id=rid))
            messages.append({"role": "user", "content": results})  # all results, one message
            if exhausted:
                stop_reason = "max_turns"
                break

        messages.append({"role": "user", "content": FINAL_INSTRUCTION})
        final, rid = self.call(**self._final_params(request, params, messages))
        usage.add(final.usage)
        final_text = _text_of(final.content)
        transcript.append(Turn(n + 1, str(final.stop_reason), final_text, request_id=rid))
        output = parse_answer(request.output_model, final_text, str(final.stop_reason), request_id=rid,
                              stop_details=_stop_details(final), transcript=transcript)
        return AgentResult(
            output=output, transcript=transcript, usage=usage, model=request.model,
            stop_reason=stop_reason, effort=request.effort,
            disclosure=disclosure(request.model, request.effort), final_text=final_text,
            request={**request.params(), "api": FINAL_CALL},
        )

    @staticmethod
    def _final_params(request: AgentRequest, params: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
        """The structured answer's request: ``output_config.format`` carries the
        request's answer schema; tools stay declared so the transcript's tool blocks
        remain valid, but the model may not call them."""
        kwargs = dict(
            params, messages=messages, max_tokens=request.final_max_tokens,
            output_config={"effort": request.effort,
                           "format": {"type": "json_schema", "schema": request.output_schema}},
        )
        if request.tools:
            kwargs["tool_choice"] = {"type": "none"}
        return kwargs


# ---------------------------------------------------------------------- fake client

@dataclass
class FakeTurn:
    """One scripted assistant turn: the tools it 'asks for' (name, input) and any prose."""

    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    text: str = ""


class FakeClient:
    """Scripted model for tests. Runs the request's real tool handlers for every
    scripted call (so tool wiring is exercised, and a service failure raises here as
    it would in production), then returns the scripted output — which may say
    anything at all, including a fabricated evidence id or a PMID that no tool ever
    fetched. That is the point: the validator, not the client, is the line of
    defence, and tests need a model that lies.

    ``output`` is a pydantic object, a dict (validated against the request's
    ``output_model`` unless ``validate_output=False``), a callable of the request, or a
    list of those consumed one per ``run``.
    """

    def __init__(self, output: Any, turns: list[FakeTurn] | None = None, *,
                 model: str = "fake-model", effort: str = "low", validate_output: bool = True):
        self.turns = list(turns or [])
        self.outputs = list(output) if isinstance(output, list) else [output]
        self.model = model
        self.effort = effort
        self.validate_output = validate_output
        self.requests: list[AgentRequest] = []
        self.calls: list[ToolCall] = []

    def run(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        handlers = {t.name: t.handler for t in request.tools}
        transcript: list[Turn] = []
        usage = Usage()
        for i, turn in enumerate(self.turns, 1):
            calls = []
            for j, (name, inputs) in enumerate(turn.tool_calls, 1):
                try:
                    text, is_error = call_tool(handlers, name, inputs)
                except ToolFailure as e:
                    transcript.append(Turn(i, "tool_use", turn.text, calls))
                    raise ToolFailure(e.tool, e.detail, request_id=f"fake_{i}", transcript=transcript) from e.__cause__
                usage.count_tool_call(is_error)
                calls.append(ToolCall(id=f"toolu_fake_{i}_{j}", name=name, input=dict(inputs), result=text, is_error=is_error))
            self.calls.extend(calls)
            transcript.append(Turn(i, "tool_use" if calls else "end_turn", turn.text, calls))
        if not self.outputs:
            raise AgentError("FakeClient has no scripted output left")
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if callable(out) and not isinstance(out, BaseModel):
            out = out(request)
        if isinstance(out, dict) and self.validate_output:
            out = request.output_model.model_validate(out)
        final_text = out.model_dump_json() if isinstance(out, BaseModel) else json.dumps(out, sort_keys=True)
        transcript.append(Turn(len(self.turns) + 1, "end_turn", final_text))
        return AgentResult(
            output=out, transcript=transcript, usage=usage, model=self.model,
            stop_reason="end_turn", effort=self.effort,
            disclosure=f"FakeClient (scripted, no API call), model {self.model}, effort {self.effort}",
            final_text=final_text, request={**request.params(), "api": None},
        )
