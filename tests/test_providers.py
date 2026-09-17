"""Model providers: the OpenRouter client over a stub transport, and the provider
selection, dotenv and disclosure logic the stage CLIs use.

Network-free. The OpenRouter client is driven by ``FakeHttp`` serving the synthetic
public responses in ``tests/fixtures/openrouter/`` (a toy arithmetic conversation —
see the README there); every request body it receives is checked for the privacy
settings the contract demands. The stage CLIs run over the same public run directory
``tests/test_reason.py`` builds (fixture evidence, no patient data). The one test that
opens a socket talks to a loopback server it starts itself, to prove a timed-out call
is sent exactly once. One live test sends a toy question through the real transport
when ``ENGINE_LIVE_TESTS`` and ``OPENROUTER_API_KEY`` are both set.

A pytest session never reads the dotenv beside the checkout (``providers.load_env_file``
sees ``PYTEST_CURRENT_TEST``); the tests that exercise that default location lift the
guard through the ``operator_shell`` fixture — always with the checkout redirected to
a temporary directory, so the operator's real dotenv is never read here.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import urllib.error
from dataclasses import asdict, dataclass
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from pydantic import BaseModel

from engine.agents import client as ac
from engine.agents import openrouter as orc
from engine.agents import providers
from engine.agents.schema import EvidenceChain, MedicineReport
from engine.cli import main
from engine.reason import run as rr
from engine.retrieve.http import Http, HttpCache, HttpError, Response
from engine.retrieve.store import EvidenceRecord, EvidenceStore

FIXTURES = Path(__file__).parent / "fixtures"
OPENROUTER = FIXTURES / "openrouter"
EVIDENCE = FIXTURES / "agents" / "evidence"
REASON = FIXTURES / "reason"
HPO = ["HP:0002205", "HP:0006528", "HP:0012236"]
KEY = "sk-or-v1-" + "0" * 64
"""A syntactically valid, unknown key (the probe's shape); never sent anywhere here."""


# ------------------------------------------------------------------------ fixtures

class Sums(BaseModel):
    results: list[int]
    note: str


def add_tool() -> ac.ToolSpec:
    def handler(inputs: dict[str, Any]) -> dict[str, int]:
        return {"sum": int(inputs["a"]) + int(inputs["b"])}
    return ac.ToolSpec("add", "Add two integers.",
                       {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}, handler)


def request(**kw: Any) -> ac.AgentRequest:
    kw.setdefault("tools", [add_tool()])
    kw.setdefault("max_turns", 3)
    return ac.AgentRequest(system="You add numbers with the add tool.", user="Compute 2+3 and 10+20.",
                           output_model=Sums, model="anthropic/claude-opus-5", effort="low", **kw)


def fixture(name: str) -> dict[str, Any]:
    """A scripted response: ``<name>.json`` (status, headers, JSON body) or
    ``<name>.sse`` (the raw body of a streamed 200)."""
    sse = OPENROUTER / f"{name}.sse"
    if sse.exists():
        return {"status": 200, "headers": {"content-type": "text/event-stream"}, "body": sse.read_text()}
    return json.loads((OPENROUTER / f"{name}.json").read_text())


@dataclass
class Call:
    method: str
    url: str
    body: dict[str, Any]
    headers: dict[str, str]
    cache_ok: bool
    timeout: float | None


class FakeHttp:
    """Answers scripted fixtures in order; a status >= 400 is raised as ``HttpError``
    exactly as :class:`engine.retrieve.http.Http` does (built with ``retries=0``, it
    raises on the first one). A name in ``failures`` raises that exception instead."""

    retries = 0

    def __init__(self, *names: str, failures: dict[int, BaseException] | None = None):
        self.queue = [fixture(n) for n in names]
        self.calls: list[Call] = []
        self.failures = dict(failures or {})

    def request(self, method: str, url: str, *, params: Any = None, json_body: Any = None, headers: Any = None,
                cache_404: bool = False, cache_ok: bool = True, timeout: float | None = None) -> Response:
        self.calls.append(Call(method, url, json.loads(json.dumps(json_body)), dict(headers or {}), cache_ok, timeout))
        if len(self.calls) in self.failures:
            raise self.failures[len(self.calls)]
        if not self.queue:
            raise AssertionError(f"no scripted response left for call {len(self.calls)}")
        fx = self.queue.pop(0)
        text = fx["body"] if isinstance(fx["body"], str) else json.dumps(fx["body"])
        if fx["status"] >= 400:
            raise HttpError(fx["status"], url, text)
        return Response(fx["status"], text, "2026-09-13T00:00:00+00:00", False, "fake", fx.get("headers", {}))

    def post(self, url: str, json_body: Any, **kw: Any) -> Response:
        return self.request("POST", url, json_body=json_body, **kw)


def client(*names: str, failures: dict[int, BaseException] | None = None, **kw: Any) -> tuple[orc.OpenRouterClient, FakeHttp]:
    """A client over ``FakeHttp``; its backoff sleeps are recorded on ``c.slept``,
    never slept."""
    http = FakeHttp(*names, failures=failures)
    log: list[str] = []
    slept: list[float] = []
    c = orc.OpenRouterClient(KEY, http=http, log=log.append, sleep=slept.append, **kw)
    c.lines = log  # type: ignore[attr-defined]
    c.slept = slept  # type: ignore[attr-defined]
    return c, http


def check_privacy(http: FakeHttp) -> None:
    """Every request: data_collection=deny, pinned upstream, Bearer auth, never cached,
    only the verified parameters, the key nowhere but the header; a loop turn is
    unstreamed with a timeout sized to its max_tokens, the final answer is streamed
    with the idle timeout."""
    assert http.calls, "no request was made"
    for call in http.calls:
        assert call.url == "https://openrouter.ai/api/v1/chat/completions" and call.method == "POST"
        assert call.body["provider"]["data_collection"] == "deny"
        assert call.body["provider"]["order"] == ["anthropic"] and call.body["provider"]["allow_fallbacks"] is False
        assert call.cache_ok is False
        assert call.headers["Authorization"] == f"Bearer {KEY}"
        assert call.headers["X-OpenRouter-App-Visibility"] == "hidden" and call.headers["X-OpenRouter-Metadata"] == "enabled"
        assert set(call.body) <= {"model", "max_tokens", "reasoning", "provider", "cache_control", "tools", "tool_choice",
                                  "messages", "response_format", "stream"}
        for absent in ("temperature", "max_completion_tokens", "parallel_tool_calls", "top_p", "top_k", "verbosity",
                       "stream_options", "usage"):
            assert absent not in call.body
        if "response_format" in call.body:  # the final answer
            assert call.body["stream"] is True and call.headers["Accept"] == orc.STREAM_ACCEPT
            assert call.timeout == orc.CALL_TIMEOUT
        else:  # a loop turn
            assert "stream" not in call.body and "Accept" not in call.headers
            assert call.timeout == orc.turn_timeout(call.body["max_tokens"]) >= orc.CALL_TIMEOUT
        assert KEY not in json.dumps(call.body)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """The public stage-5 run directory ``tests/test_reason.py`` builds."""
    run = tmp_path / "run"
    shutil.copytree(EVIDENCE, run / "02_retrieve" / "evidence")
    (run / "03_filter").mkdir()
    shutil.copy(REASON / "candidates.json", run / "03_filter" / "candidates.json")
    (run / "04_rank").mkdir()
    shutil.copy(REASON / "joined.json", run / "04_rank" / "joined.json")
    rank_store = EvidenceStore(run / "04_rank" / "evidence")
    rank_store.put(EvidenceRecord(
        record_id="exomiser:CFTR", source="exomiser", source_version="exomiser-cli 14.0.0 · data 2406",
        query={"hpoIds": HPO, "genomeAssembly": "hg38", "analysis_yaml_sha256": "0" * 64},
        url="https://www.ncbi.nlm.nih.gov/gene/1080", retrieved_at="2026-09-12T00:00:00+00:00",
        payload={"ranking": {"rank": 1, "gene_symbol": "CFTR", "moi": "AR", "exomiser_score": 0.97}, "citation": "Exomiser"}))
    rank_store.write_index()
    # the case terms' hpo: records, so stage 5 serves them and never asks the JAX API
    shutil.copytree(FIXTURES / "hpo" / "records", run / "05_reason" / "evidence", dirs_exist_ok=True)
    return run


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """No model key, no dotenv anywhere the code looks, a throw-away cache. The keys
    are set *empty* rather than deleted so that whatever a dotenv loads under a test is
    restored at teardown (monkeypatch records nothing for a variable that was absent)."""
    for var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.setenv(var, "")
    monkeypatch.delenv("ENGINE_ENV", raising=False)
    monkeypatch.setenv("ENGINE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(providers, "repo_root", lambda: tmp_path / "engine")  # so <repo>/../.env is tmp_path/.env
    (tmp_path / "engine").mkdir(exist_ok=True)
    return tmp_path


@pytest.fixture
def operator_shell(no_keys: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``no_keys`` plus "this is an operator's shell, not a test session": the
    ``PYTEST_CURRENT_TEST`` guard is lifted (pytest re-exports the variable for the
    call phase after fixtures run, so the check is patched, not the variable) and the
    default dotenv location — ``tmp_path/.env`` under ``no_keys`` — is read as in
    production."""
    monkeypatch.setattr(providers, "test_session", lambda: False)
    return no_keys


def read(path: Path) -> Any:
    return json.loads(path.read_text())


# ---------------------------------------------------------------- the tool loop

def test_two_turn_tool_loop_executes_every_call_and_answers_with_the_schema():
    c, http = client("tools_round1", "tools_round2", "tools_done", "final_stream")
    result = c.run(request())

    # the answer: validated against the request's model, the streamed text assembled verbatim
    # (byte for byte what the unstreamed envelope would carry)
    assert isinstance(result, orc.OpenRouterResult) and isinstance(result, ac.AgentResult)
    assert result.output == Sums(results=[5, 30, 35], note="sums via the add tool")
    assert result.final_text == fixture("final_answer")["body"]["choices"][0]["message"]["content"]
    assert result.stop_reason == "end_turn" and result.model == "anthropic/claude-opus-5" and result.effort == "low"

    # the transcript: Anthropic's stop-reason vocabulary, every call with its result
    assert [(t.n, t.stop_reason, len(t.tool_calls)) for t in result.transcript] == \
        [(1, "tool_use", 2), (2, "tool_use", 3), (3, "end_turn", 0), (4, "end_turn", 0)]
    assert [t.request_id for t in result.transcript] == ["gen-round1", "gen-round2", "gen-done", "gen-final"]
    r1 = result.transcript[0].tool_calls
    assert [(x.id, x.name, x.input, x.result, x.is_error) for x in r1] == [
        ("toolu_01", "add", {"a": 2, "b": 3}, '{"sum": 5}', False),
        ("toolu_02", "add", {"a": 10, "b": 20}, '{"sum": 30}', False)]
    r2 = result.transcript[1].tool_calls
    assert (r2[0].result, r2[0].is_error) == ('{"sum": 35}', False)
    assert r2[1].is_error and r2[1].result == "Error: unknown tool 'nope'"
    assert r2[2].is_error and r2[2].result == "Error: tool arguments were not valid JSON" and r2[2].input == {}
    assert result.transcript[1].text.startswith("Adding the two sums")

    # usage and the echo
    assert (result.usage.api_calls, result.usage.tool_calls, result.usage.tool_errors) == (4, 5, 2)
    assert result.usage.input_tokens == 180 + 260 + 320 + 400 and result.usage.output_tokens == 60 + 70 + 25 + 30
    assert result.usage.cache_read_input_tokens == 180 + 320
    assert result.provider_echo == "Anthropic" and result.model_echo == "anthropic/claude-opus-5"
    assert result.generation_ids == ["gen-round1", "gen-round2", "gen-done", "gen-final"]
    assert result.cost_usd == pytest.approx(0.0024 + 0.0031 + 0.0018 + 0.0021)
    assert result.request["api"] == orc.FINAL_CALL and result.request["provider"]["data_collection"] == "deny"
    assert result.request["reasoning"] == {"effort": "low"} and result.request["model"] == "anthropic/claude-opus-5"
    assert result.request["retries"] == 3 and result.request["retry_statuses"] == [429, 500, 502, 503]
    assert result.request["stream"] == {"turns": False, "final": True}
    assert result.request["timeout"] == 600.0 and result.request["turn_timeout"] == orc.turn_timeout(ac.DEFAULT_MAX_TOKENS) == 900.0
    assert c.slept == []
    assert "OpenRouter API, model anthropic/claude-opus-5, effort low, pinned to upstream provider 'anthropic'" in result.disclosure
    assert "data_collection='deny'" in result.disclosure
    assert c.lines == ["turn 1: stop_reason=tool_use tool_calls=2", "turn 2: stop_reason=tool_use tool_calls=3",
                       "turn 3: stop_reason=end_turn tool_calls=0"]
    check_privacy(http)


def test_the_requests_carry_the_verified_shapes_and_every_result_before_the_next_turn():
    c, http = client("tools_round1", "tools_round2", "tools_done", "final_stream")
    c.run(request())
    b1, b2, b3, b4 = (call.body for call in http.calls)

    # turn 1: system + user, tools as strict function tools, reasoning effort, caching
    assert b1["messages"] == [{"role": "system", "content": "You add numbers with the add tool."},
                              {"role": "user", "content": "Compute 2+3 and 10+20."}]
    assert b1["model"] == "anthropic/claude-opus-5" and b1["max_tokens"] == ac.DEFAULT_MAX_TOKENS
    assert b1["reasoning"] == {"effort": "low"} and b1["cache_control"] == {"type": "ephemeral"}
    assert b1["tool_choice"] == "auto" and "response_format" not in b1
    [tool] = b1["tools"]
    assert tool["type"] == "function" and tool["function"]["name"] == "add" and tool["function"]["strict"] is True
    assert tool["function"]["parameters"] == ac.strict_schema(add_tool().input_schema)
    assert tool["function"]["parameters"]["additionalProperties"] is False

    # turn 2: the assistant message echoed with tool_calls and reasoning_details unmodified,
    # then one tool message per call id — both parallel results, in order — and the tools again
    round1 = fixture("tools_round1")["body"]["choices"][0]["message"]
    assert b2["messages"][2] == {"role": "assistant", "content": None, "tool_calls": round1["tool_calls"],
                                 "reasoning_details": round1["reasoning_details"]}
    assert b2["messages"][3:] == [{"role": "tool", "tool_call_id": "toolu_01", "content": '{"sum": 5}'},
                                  {"role": "tool", "tool_call_id": "toolu_02", "content": '{"sum": 30}'}]
    assert b2["tools"] == b1["tools"] and b2["tool_choice"] == "auto"

    # turn 3: the error results travel as content (this skin has no is_error flag)
    assert b3["messages"][6:] == [{"role": "tool", "tool_call_id": "toolu_03", "content": '{"sum": 35}'},
                                  {"role": "tool", "tool_call_id": "toolu_04", "content": "Error: unknown tool 'nope'"},
                                  {"role": "tool", "tool_call_id": "toolu_05", "content": "Error: tool arguments were not valid JSON"}]
    assert "is_error" not in json.dumps(b3["messages"])

    # the final call: the instruction, the strict schema, tools declared but not callable
    assert b4["messages"][-1] == {"role": "user", "content": ac.FINAL_INSTRUCTION}
    assert b4["messages"][-2] == {"role": "assistant", "content": "2+3 = 5 and 10+20 = 30; together 35."}
    assert b4["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "Sums", "strict": True, "schema": ac.default_answer_schema(Sums)}}
    assert b4["tool_choice"] == "none" and b4["tools"] == b1["tools"] and b4["max_tokens"] == ac.DEFAULT_FINAL_MAX_TOKENS
    assert b4["stream"] is True and all("stream" not in b for b in (b1, b2, b3))
    check_privacy(http)


def test_parallel_tool_calls_are_all_executed_and_all_answered_in_one_turn():
    c, http = client("tools_round1", "tools_done", "final_stream")
    result = c.run(request())
    [turn] = [t for t in result.transcript if t.tool_calls]
    assert [x.id for x in turn.tool_calls] == ["toolu_01", "toolu_02"] and not any(x.is_error for x in turn.tool_calls)
    tool_msgs = [m for m in http.calls[1].body["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["toolu_01", "toolu_02"]
    assert result.usage.tool_calls == 2 and result.usage.tool_errors == 0
    check_privacy(http)


def test_final_json_schema_answer_without_any_tool_call():
    c, http = client("tools_done", "final_stream")
    result = c.run(request())
    assert result.output.results == [5, 30, 35] and result.stop_reason == "end_turn"
    assert [t.stop_reason for t in result.transcript] == ["end_turn", "end_turn"]
    assert http.calls[1].body["response_format"]["json_schema"]["strict"] is True
    check_privacy(http)


def test_a_streamed_request_answered_with_one_json_object_is_still_read():
    """A server that ignores ``stream`` (or answers an error envelope unstreamed)
    sends one JSON object: the same reader applies."""
    c, http = client("tools_done", "final_answer")
    result = c.run(request())
    assert result.output.results == [5, 30, 35] and result.generation_ids == ["gen-done", "gen-final"]
    assert http.calls[1].body["stream"] is True
    c, _ = client("tools_done", "final_truncated")
    with pytest.raises(ac.AgentError, match="final answer not usable: the answer was cut off at max_tokens"):
        c.run(request())
    check_privacy(http)


def test_max_turns_exhaustion_refuses_the_calls_and_still_asks_for_the_answer():
    c, http = client("tools_round1", "tools_round1", "final_stream")
    result = c.run(request(max_turns=1))
    assert result.stop_reason == "max_turns"
    t1, t2, final = result.transcript
    assert not any(x.is_error for x in t1.tool_calls)
    assert all(x.is_error and x.result == ac.TOOL_BUDGET_ERROR for x in t2.tool_calls)
    assert final.stop_reason == "end_turn" and result.output.note == "sums via the add tool"
    assert result.usage.tool_errors == 2
    budget = [m for m in http.calls[2].body["messages"] if m["role"] == "tool"][-2:]
    assert [m["content"] for m in budget] == [ac.TOOL_BUDGET_ERROR, ac.TOOL_BUDGET_ERROR]
    check_privacy(http)


def test_a_401_and_a_429_are_typed_agent_errors_that_name_no_key():
    c, http = client("error_401")
    with pytest.raises(orc.OpenRouterError) as e:
        c.run(request())
    assert isinstance(e.value, ac.AgentError)
    assert e.value.status == 401 and e.value.error_type is None
    assert str(e.value) == "OpenRouter refused the key or the request (401): User not found."
    assert KEY not in str(e.value) and e.value.transcript == [] and len(http.calls) == 1 and c.slept == []
    check_privacy(http)

    c, http = client("tools_round1", "error_429", "error_429", "error_429", "error_429")
    with pytest.raises(orc.OpenRouterError) as e:
        c.run(request())
    assert e.value.status == 429 and e.value.error_type == "rate_limit_exceeded"
    assert str(e.value) == "OpenRouter rate limit (429) [rate_limit_exceeded]: Rate limit exceeded"
    assert len(e.value.transcript) == 1 and e.value.transcript[0].tool_calls[0].name == "add"  # turns kept for .failed.json
    assert len(http.calls) == 5 and c.slept == [2.0, 4.0, 8.0]  # the turn, then three re-sends with backoff
    assert c.lines[1:] == [f"OpenRouter answered 429: retry {i}/3 in {w}s" for i, w in ((1, 2), (2, 4), (3, 8))]
    check_privacy(http)


def test_a_refused_or_failed_upstream_is_retried_and_a_timeout_status_is_not():
    """429/500/502/503 say nothing was generated, so a re-send costs nothing twice;
    a 504 is a timeout by another name — the model may have been generating — and is
    one typed failure."""
    c, http = client("error_500", "tools_done", "final_stream")
    result = c.run(request())
    assert result.output.results == [5, 30, 35] and len(http.calls) == 3 and c.slept == [2.0]
    assert [call.body["messages"] for call in http.calls[:2]] == [http.calls[0].body["messages"]] * 2  # the identical request
    assert result.usage.api_calls == 2  # only answers count
    check_privacy(http)

    c, http = client("tools_done", "error_504")
    with pytest.raises(orc.OpenRouterError, match=r"OpenRouter API error 504: Gateway Timeout") as e:
        c.run(request())
    assert e.value.status == 504 and len(http.calls) == 2 and c.slept == []
    assert "Anthropic" not in str(e.value)  # the metadata block never travels

    c, http = client("error_429", retries=0)  # no retries at all when asked
    with pytest.raises(orc.OpenRouterError, match="rate limit"):
        c.run(request())
    assert len(http.calls) == 1 and c.slept == []
    assert orc.retry_wait(1) == 2.0 and orc.retry_wait(3) == 8.0 and orc.retry_wait(10) == 60.0


def test_a_200_with_an_error_body_a_refusal_and_a_truncated_answer_are_agent_errors():
    c, _ = client("error_late_200")
    with pytest.raises(orc.OpenRouterError) as e:
        c.run(request())
    assert e.value.status == 502 and e.value.error_type == "provider_unavailable" and e.value.request_id == "gen-late"
    assert str(e.value) == "OpenRouter error 502 [provider_unavailable]: Provider returned error (request id gen-late)"

    c, _ = client("refusal")
    with pytest.raises(ac.AgentError, match=r"turn 1: the model refused \(request id gen-refused\)") as e:
        c.run(request())
    assert e.value.transcript[0].stop_reason == "refusal"

    c, _ = client("tools_done", "final_stream_truncated")
    with pytest.raises(ac.AgentError, match="final answer not usable: the answer was cut off at max_tokens") as e:
        c.run(request())
    assert e.value.final_text == '{"results": [5, 30' and "5, 30" not in str(e.value)
    assert e.value.request_id == "gen-cut" and e.value.transcript[-1].stop_reason == "max_tokens"


def test_a_tool_whose_service_fails_stops_the_loop_as_a_tool_failure():
    def broken(inputs: dict[str, Any]) -> Any:
        raise HttpError(503, "https://www.ebi.ac.uk/europepmc/x", "down")
    tool = ac.ToolSpec("add", "Add two integers.", {"type": "object", "properties": {"a": {"type": "integer"},
                                                                                       "b": {"type": "integer"}}}, broken)
    c, http = client("tools_round1")
    with pytest.raises(ac.ToolFailure, match="tool 'add' failed: HTTP 503 from www.ebi.ac.uk") as e:
        c.run(request(tools=[tool]))
    assert e.value.request_id == "gen-round1" and len(http.calls) == 1


def test_a_timeout_or_a_dropped_connection_is_one_typed_failure_never_a_resend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The finding this guards against: a client-side timeout on a long answer used to
    be re-sent identically three more times by the transport — four generations billed
    for one failure. Now the transport has no retries and the client re-sends only a
    :data:`RETRY_STATUSES` answer."""
    for exc, why in ((TimeoutError("timed out"), "TimeoutError"),
                     (urllib.error.URLError(TimeoutError("timed out")), "URLError"),
                     (ConnectionResetError(54, "reset"), "ConnectionResetError")):
        c, http = client("tools_done", failures={2: exc})
        with pytest.raises(orc.OpenRouterError, match=f"could not reach OpenRouter at openrouter.ai: {why}") as e:
            c.run(request())
        assert len(http.calls) == 2 and c.slept == [] and e.value.status is None
        assert len(e.value.transcript) == 1 and e.value.transcript[0].stop_reason == "end_turn"  # kept for .failed.json

    c, http = client("tools_done", failures={2: IncompleteRead(b"data: {}")})
    with pytest.raises(orc.OpenRouterError, match="dropped mid-answer: IncompleteRead"):
        c.run(request())
    assert len(http.calls) == 2 and c.slept == []

    # the default transport is built without retries of its own
    monkeypatch.setenv("ENGINE_CACHE", str(tmp_path / "cache"))
    transport = orc.OpenRouterClient(KEY).http
    assert isinstance(transport, Http) and transport.retries == 0 and transport.timeout == orc.CALL_TIMEOUT


class _SlowHandler(BaseHTTPRequestHandler):
    """Counts POSTs and answers only after the client has given up."""

    hits = 0
    stop = threading.Event()

    def do_POST(self) -> None:  # noqa: N802 — http.server's name
        type(self).hits += 1
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).stop.wait(5.0)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"id": "gen-late", "choices": []}')
        except OSError:
            pass  # the client is long gone

    def log_message(self, *a: Any) -> None:
        pass


def test_loopback_proof_a_timed_out_call_is_posted_exactly_once(tmp_path: Path):
    """The reviewer's probe, kept: a server that answers after the client's timeout
    must see ONE request per ``run``. Loopback only; nothing leaves the machine."""
    _SlowHandler.hits, _SlowHandler.stop = 0, threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        transport = Http(HttpCache(tmp_path / "cache"), retries=0)
        c = orc.OpenRouterClient(KEY, base_url=f"http://127.0.0.1:{server.server_port}/api/v1", http=transport, timeout=0.3)
        started = time.monotonic()
        with pytest.raises(orc.OpenRouterError, match=r"could not reach OpenRouter at 127\.0\.0\.1:\d+: (TimeoutError|URLError)"):
            c.run(request(tools=[], max_tokens=10))  # a turn timeout of max(0.3, 10 tokens' worth) = 0.3 s
        assert time.monotonic() - started < 3.0
        _SlowHandler.stop.set()
        time.sleep(0.05)
        assert _SlowHandler.hits == 1
        assert not any((tmp_path / "cache").rglob("*.json"))
    finally:
        _SlowHandler.stop.set()
        server.shutdown()
        server.server_close()


def test_network_failures_and_unusable_bodies_are_typed_errors():
    class Down:
        def request(self, *a: Any, **kw: Any) -> Response:
            raise OSError("connection refused")
    with pytest.raises(orc.OpenRouterError, match="could not reach OpenRouter at openrouter.ai: OSError"):
        orc.OpenRouterClient(KEY, http=Down()).run(request())

    class Junk:
        def request(self, *a: Any, **kw: Any) -> Response:
            return Response(200, "<html>", "t", False, "k", {})
    with pytest.raises(orc.OpenRouterError, match="non-JSON body") as e:
        orc.OpenRouterClient(KEY, http=Junk()).run(request())
    assert e.value.status == 200

    class NoChoice:
        def request(self, *a: Any, **kw: Any) -> Response:
            return Response(200, '{"id": "gen-x", "choices": []}', "t", False, "k", {})
    with pytest.raises(orc.OpenRouterError, match="without a choice") as e:
        orc.OpenRouterClient(KEY, http=NoChoice()).run(request())
    assert e.value.request_id == "gen-x"


def test_the_transcript_has_the_same_shape_as_the_other_clients_write():
    c, _ = client("tools_round1", "tools_done", "final_stream")
    ours = c.run(request()).as_dict()
    theirs = ac.FakeClient({"results": [1], "note": "n"}, turns=[ac.FakeTurn([("add", {"a": 1, "b": 2})])]).run(request()).as_dict()
    assert set(ours) == set(theirs) | {"openrouter"}
    assert {k for t in ours["transcript"] for k in t} == {k for t in theirs["transcript"] for k in t}
    assert {k for t in ours["transcript"] for x in t["tool_calls"] for k in x} == \
        {k for t in theirs["transcript"] for x in t["tool_calls"] for k in x}
    assert set(ours["usage"]) == set(theirs["usage"])
    assert ours["openrouter"] == {"provider": "Anthropic", "model": "anthropic/claude-opus-5",
                                  "generation_ids": ["gen-round1", "gen-done", "gen-final"],
                                  "cost_usd": pytest.approx(0.0024 + 0.0018 + 0.0021)}


def test_the_bare_anthropic_default_model_is_sent_as_the_clients_openrouter_id():
    c, http = client("tools_done", "final_stream", model="anthropic/claude-sonnet-5")
    result = c.run(ac.AgentRequest(system="", user="Sum.", output_model=Sums, tools=[], effort="medium"))  # model claude-opus-5
    assert http.calls[0].body["model"] == "anthropic/claude-sonnet-5" and result.model == "anthropic/claude-sonnet-5"
    assert http.calls[0].body["messages"][0]["role"] == "user" and "tools" not in http.calls[0].body
    assert "tool_choice" not in http.calls[1].body and http.calls[1].body["reasoning"] == {"effort": "medium"}
    assert result.request["model"] == "anthropic/claude-sonnet-5"

    # any other bare id is refused before a request is built — never a 400 after the bundle left,
    # never a silent switch to the client's model
    c, http = client("tools_done", "final_stream")
    with pytest.raises(orc.OpenRouterError, match=r"OpenRouter model ids are author/slug \(e.g. anthropic/claude-opus-5\), got 'claude-sonnet-5'"):
        c.run(ac.AgentRequest(system="", user="Sum.", output_model=Sums, tools=[], model="claude-sonnet-5"))
    assert http.calls == []
    with pytest.raises(ValueError, match="OpenRouterClient model must be an OpenRouter id"):
        orc.OpenRouterClient(KEY, model="claude-opus-5")


def test_client_options_shape_the_provider_block_and_never_the_key():
    c = orc.OpenRouterClient(KEY, upstream=None, referer="https://example.invalid/engine", app_title="engine test")
    assert c.provider_preferences() == {"data_collection": "deny", "require_parameters": True}
    assert c.headers()["HTTP-Referer"] == "https://example.invalid/engine" and c.headers()["X-OpenRouter-Title"] == "engine test"
    assert KEY not in json.dumps(c.transport_params())
    assert orc.OpenRouterClient(KEY, data_collection="allow", require_parameters=True).provider_preferences() == \
        {"data_collection": "allow", "order": ["anthropic"], "allow_fallbacks": False, "require_parameters": True}
    with pytest.raises(ValueError, match="api_key"):
        orc.OpenRouterClient("")
    with pytest.raises(ValueError, match="data_collection"):
        orc.OpenRouterClient(KEY, data_collection="maybe")
    with pytest.raises(ValueError, match="retries"):
        orc.OpenRouterClient(KEY, retries=-1)
    assert orc.turn_timeout(ac.DEFAULT_MAX_TOKENS) == 900.0 and orc.turn_timeout(2000) == orc.CALL_TIMEOUT
    assert orc.turn_timeout(2000, floor=30.0) == 56.25 and orc.OpenRouterClient(KEY, timeout=30.0).timeout == 30.0
    assert "routed by OpenRouter across" in orc.disclosure("m", "low", upstream=None)
    assert orc.stop_reason({"finish_reason": "tool_calls"}) == "tool_use"
    assert orc.stop_reason({"finish_reason": "stop", "native_finish_reason": "end_turn"}) == "end_turn"
    assert orc.stop_reason({"finish_reason": "content_filter"}) == "refusal" and orc.stop_reason({}) == "end_turn"


# ------------------------------------------------------------- the streamed answer

def test_the_final_answer_is_streamed_and_assembled_from_the_events():
    c, http = client("tools_done", "final_stream")
    result = c.run(request())
    final = http.calls[1]
    assert final.body["stream"] is True and final.headers["Accept"] == "text/event-stream" and final.timeout == 600.0
    assert result.final_text == '{"results": [5, 30, 35], "note": "sums via the add tool"}'
    assert result.transcript[-1].stop_reason == "end_turn" and result.transcript[-1].request_id == "gen-final"
    assert result.usage.api_calls == 2 and result.usage.input_tokens == 320 + 400 and result.usage.output_tokens == 25 + 30
    assert result.usage.cache_read_input_tokens == 180 + 320  # the usage chunk of the stream counted once
    assert result.cost_usd == pytest.approx(0.0018 + 0.0021)
    assert result.provider_echo == "Anthropic" and result.model_echo == "anthropic/claude-opus-5"
    assert result.generation_ids == ["gen-done", "gen-final"]
    check_privacy(http)


def test_a_stream_that_fails_or_is_cut_is_a_typed_error_never_a_partial_answer():
    c, _ = client("tools_done", "final_stream_error")
    with pytest.raises(orc.OpenRouterError) as e:
        c.run(request())
    assert str(e.value) == "OpenRouter error 502 mid-stream: Provider disconnected mid-stream (request id gen-final-err)"
    assert e.value.status == 502 and e.value.request_id == "gen-final-err" and e.value.error_type is None
    assert "upstream connection reset" not in str(e.value) and "[5" not in str(e.value)  # no metadata, no answer text
    assert len(e.value.transcript) == 1  # the loop turn survives for .failed.json

    c, _ = client("tools_done", "final_stream_dropped")
    with pytest.raises(orc.OpenRouterError, match="the OpenRouter stream ended before a finish reason") as e:
        c.run(request())
    assert e.value.request_id == "gen-dropped" and e.value.status == 200

    def streamed(text: str) -> Response:
        return Response(200, text, "t", False, "k", {"content-type": "text/event-stream"})
    with pytest.raises(orc.OpenRouterError, match="empty stream"):
        orc.parse_stream(streamed(": OPENROUTER PROCESSING\n\ndata: [DONE]\n\n"))
    with pytest.raises(orc.OpenRouterError, match="non-JSON event"):
        orc.parse_stream(streamed("data: {not json\n\n"))
    with pytest.raises(orc.OpenRouterError, match="non-object event"):
        orc.parse_stream(streamed("data: [1, 2]\n\n"))
    # a stream whose only event is the error envelope OpenRouter sends unstreamed
    with pytest.raises(orc.OpenRouterError, match=r"OpenRouter error 502 \[provider_unavailable\]"):
        orc.parse_stream(streamed(json.dumps(fixture("error_late_200")["body"])))


def test_sse_events_are_read_by_the_spec_comments_multiline_data_and_done():
    body = (": keep-alive\r\n\r\n"
            "event: message\r\ndata: {\"a\":\r\ndata: 1}\r\n\r\n"
            "data:{\"b\":2}\n\n"
            "id: 7\ndata: {\"c\":3}\n"          # no blank line before EOF: still an event
            )
    assert list(orc._sse_data(body)) == ['{"a":\n1}', '{"b":2}', '{"c":3}']
    assert list(orc._sse_data("data: x\n\ndata: [DONE]\n\ndata: after\n\n")) == ["x"]
    assert list(orc._sse_data("")) == []

    reply = orc.parse_stream(Response(200, fixture("final_stream")["body"], "t", False, "k", {}))
    assert reply.stop_reason == "end_turn" and reply.generation_id == "gen-final" and reply.provider == "Anthropic"
    assert reply.message == {"role": "assistant", "content": reply.text} and reply.tool_calls == []
    assert reply.usage.input_tokens == 400 and reply.usage.cache_read_input_tokens == 320 and reply.cost == 0.0021
    # finish_reason alone (no native) maps through the same table; usage may be absent
    reply = orc.parse_stream(Response(200, 'data: {"id":"g","choices":[{"index":0,"delta":{"content":"{}"},'
                                           '"finish_reason":"length"}]}\n\ndata: [DONE]\n\n', "t", False, "k", {}))
    assert reply.stop_reason == "max_tokens" and reply.text == "{}" and reply.cost is None and reply.usage.input_tokens is None


# ---------------------------------------------------------------------- providers

def test_dotenv_never_overrides_the_environment_and_lives_outside_the_repo(operator_shell: Path, tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch):
    repo = tmp_path / "checkout"
    repo.mkdir()
    (tmp_path / ".env").write_text('# keys\nexport OPENROUTER_API_KEY="sk-or-v1-file"\nFOO=from-file # comment\nBAD LINE\n')
    monkeypatch.setenv("FOO", "real")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")   # empty is absent — and restored at teardown
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("ENGINE_ENV", raising=False)

    assert providers.env_path(repo=repo) == (tmp_path / ".env").resolve()
    loaded = providers.load_env_file(repo=repo)
    assert loaded == {"OPENROUTER_API_KEY": "sk-or-v1-file", "FOO": "from-file"}
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-v1-file" and os.environ["FOO"] == "real"  # precedence
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    # inside the checkout: refused, whichever way it is named
    (repo / ".env").write_text("OPENROUTER_API_KEY=inside\n")
    with pytest.raises(ValueError, match="inside the repository"):
        providers.load_env_file(repo / ".env", repo=repo)
    monkeypatch.setenv("ENGINE_ENV", str(repo / "sub" / ".env"))
    with pytest.raises(ValueError, match="inside the repository"):
        providers.load_env_file(repo=repo)
    assert not os.environ["OPENROUTER_API_KEY"]

    # $ENGINE_ENV elsewhere wins over the default; an explicit path must exist
    other = tmp_path / "elsewhere" / "engine.env"
    other.parent.mkdir()
    other.write_text("ANTHROPIC_API_KEY=sk-ant-file\n")
    monkeypatch.setenv("ENGINE_ENV", str(other))
    assert providers.load_env_file(repo=repo) == {"ANTHROPIC_API_KEY": "sk-ant-file"}
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-file"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("ENGINE_ENV", str(tmp_path / "missing.env"))
    with pytest.raises(FileNotFoundError):
        providers.load_env_file(repo=repo)
    monkeypatch.delenv("ENGINE_ENV")
    (tmp_path / ".env").unlink()
    assert providers.load_env_file(repo=repo) == {}  # the default may be absent

    # ENGINE_ENV="" reads nothing, even with a file in the default place (an explicit path still does)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-or-v1-file\n")
    monkeypatch.setenv("ENGINE_ENV", "")
    assert providers.dotenv_disabled() and providers.load_env_file(repo=repo) == {} and not os.environ["OPENROUTER_API_KEY"]
    assert providers.load_env_file(other, repo=repo) == {"ANTHROPIC_API_KEY": "sk-ant-file"}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("ENGINE_ENV")
    assert not providers.dotenv_disabled()
    assert providers.parse_dotenv("A='x y'\nB=\nC\n=d\n") == {"A": "x y", "B": ""}

    # under pytest the default location is not read at all, whatever it holds; ENGINE_ENV
    # naming a file is explicit and still is
    monkeypatch.setattr(providers, "test_session", lambda: True)
    assert providers.dotenv_disabled()
    assert providers.load_env_file(repo=repo) == {} and not os.environ["OPENROUTER_API_KEY"]
    assert providers.load_env_file(other, repo=repo) == {"ANTHROPIC_API_KEY": "sk-ant-file"}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("ENGINE_ENV", str(other))
    assert not providers.dotenv_disabled() and providers.load_env_file(repo=repo) == {"ANTHROPIC_API_KEY": "sk-ant-file"}
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-file"


def test_default_provider_prefers_openrouter_then_anthropic_then_names_both(no_keys: Path, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(providers.ProviderError, match="set OPENROUTER_API_KEY or ANTHROPIC_API_KEY") as e:
        providers.default_provider()
    assert isinstance(e.value, ac.AgentError) and str(no_keys / ".env") in str(e.value)
    assert providers.default_provider(strict=False) == "anthropic"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    assert providers.default_provider() == "anthropic"
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    assert providers.default_provider() == "openrouter"
    assert providers.default_model("openrouter") == "anthropic/claude-opus-5"
    assert providers.default_model("anthropic") == "claude-opus-5" and providers.default_model("fake") == "fake-model"
    with pytest.raises(ValueError, match="unknown provider"):
        providers.default_model("gemini")


def test_provider_model_settles_the_id_the_provider_will_send():
    assert providers.provider_model("openrouter") == "anthropic/claude-opus-5"
    assert providers.provider_model("openrouter", "") == "anthropic/claude-opus-5"
    assert providers.provider_model("openrouter", "claude-opus-5") == "anthropic/claude-opus-5"  # the stages' bare default
    assert providers.provider_model("openrouter", "anthropic/claude-sonnet-5") == "anthropic/claude-sonnet-5"
    assert providers.provider_model("anthropic") == "claude-opus-5" and providers.provider_model("anthropic", "x") == "x"
    assert providers.provider_model("fake") == "fake-model" and providers.provider_model("fake", "m") == "m"
    with pytest.raises(providers.ProviderError, match=r"openrouter model ids are author/slug \(e.g. anthropic/claude-opus-5\), got 'claude-sonnet-5'") as e:
        providers.provider_model("openrouter", "claude-sonnet-5")
    assert isinstance(e.value, ac.AgentError)
    with pytest.raises(ValueError, match="unknown provider"):
        providers.provider_model("gemini")


def test_select_client_builds_all_three_and_a_missing_key_is_named(no_keys: Path, monkeypatch: pytest.MonkeyPatch):
    fake = providers.select_client("fake", effort="low")
    assert isinstance(fake, ac.FakeClient) and fake.model == "fake-model"
    result = fake.run(ac.AgentRequest(system="", user="u", output_model=EvidenceChain, tools=[]))
    assert result.output == EvidenceChain(candidate_id="", variants=[], phase_statement="", mechanism_hypothesis="")
    assert result.disclosure == providers.disclosure("fake", "fake-model", "low")
    report = providers.select_client("fake").run(ac.AgentRequest(system="", user="u", output_model=MedicineReport, tools=[]))
    assert report.output == MedicineReport(candidate_id="", gene_symbol="", mechanism=[], candidates=[])

    with pytest.raises(providers.ProviderError, match="provider openrouter needs OPENROUTER_API_KEY"):
        providers.select_client("openrouter")
    with pytest.raises(providers.ProviderError, match="no model key"):
        providers.select_client(None)
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    lines: list[str] = []
    c = providers.select_client(None, log=lines.append)
    assert isinstance(c, orc.OpenRouterClient) and c.model == "anthropic/claude-opus-5" and c.effort == "high"
    assert c.headers()["Authorization"] == f"Bearer {KEY}" and c.app_title == "engine"
    assert isinstance(providers.select_client("openrouter", "anthropic/claude-sonnet-5", "low"), orc.OpenRouterClient)
    assert providers.select_client("openrouter", ac.DEFAULT_MODEL).model == orc.DEFAULT_MODEL  # never sent bare
    with pytest.raises(providers.ProviderError, match="author/slug"):
        providers.select_client("openrouter", "claude-sonnet-5")
    assert not lines

    built: list[dict[str, Any]] = []
    monkeypatch.setattr(ac, "AnthropicClient", lambda **kw: built.append(kw) or ac.FakeClient({}, validate_output=False))
    assert isinstance(providers.select_client("anthropic", log=lines.append), ac.FakeClient)
    assert built == [{"log": lines.append}]

    def refuse(**kw: Any) -> None:
        raise TypeError("Could not resolve authentication method.")
    monkeypatch.setattr(ac, "AnthropicClient", refuse)
    with pytest.raises(ac.AgentError, match="could not build the Anthropic client: Could not resolve authentication"):
        providers.select_client("anthropic")
    with pytest.raises(ValueError, match="unknown provider"):
        providers.select_client("gemini")


def test_disclosure_and_status_name_the_provider_never_a_key(no_keys: Path, monkeypatch: pytest.MonkeyPatch):
    assert providers.disclosure("anthropic", "claude-opus-5", "high") == ac.disclosure("claude-opus-5", "high")
    line = providers.disclosure("openrouter", "anthropic/claude-opus-5", "high")
    assert line.startswith("OpenRouter API, model anthropic/claude-opus-5, effort high, pinned to upstream provider 'anthropic' "
                           "with no fallback; every request sets provider.data_collection='deny'")
    assert line == orc.disclosure("anthropic/claude-opus-5", "high")
    assert providers.disclosure("fake", "fake-model", "low") == "FakeClient (scripted, no API call), model fake-model, effort low"

    st = providers.status()
    assert {name: (v["key_present"], v["default_model"]) for name, v in st.items()} == {
        "anthropic": (False, "claude-opus-5"),
        "openrouter": (False, "anthropic/claude-opus-5"),
        "fake": (True, "fake-model"),
    }
    # the model choices carry ids and labels only — never a key
    for v in st.values():
        assert v["models"] and all(set(m) == {"id", "label"} for m in v["models"])
    assert st["openrouter"]["models"][0]["id"] == "anthropic/claude-opus-5"
    assert {m["id"] for m in st["openrouter"]["models"]} >= {"deepseek/deepseek-v4-pro", "moonshotai/kimi-k3"}
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    s = providers.status()
    assert s["openrouter"]["key_present"] is True and KEY not in json.dumps(s)


# ---------------------------------------------------------------------- the stages

def test_the_stage_cli_takes_a_provider_and_defaults_the_model(run_dir: Path, no_keys: Path, monkeypatch: pytest.MonkeyPatch):
    cache = str(no_keys / "cache")
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--top", "1", "--provider", "fake", "--effort", "low",
                                       "--cache", cache, "--offline"])
    assert result.exit_code == 0, result.output
    assert "reason · " in result.output and " · fake · fake-model · effort low" in result.output
    assert "candidate 1/1: agent (fake-model, effort low)" in result.output and "CFTR" not in result.output
    m = read(run_dir / "05_reason" / "manifest.json")
    assert m["params"]["provider"] == "fake" and m["params"]["model"] == "fake-model"
    assert m["params"]["disclosure"] == providers.disclosure("fake", "fake-model", "low")
    assert m["params"]["agent"]["CFTR:hom"]["disclosure"] == m["params"]["disclosure"]
    chain = read(run_dir / "05_reason" / "chains" / "CFTR:hom.json")
    # the empty answer: the stage pins the identity and fills the bundle's variant in with no criteria
    assert chain["candidate_id"] == "CFTR:hom"
    assert [(v["key"], v["criteria"], v["classification"]) for v in chain["variants"]] == [("7:117559590:ATCT:A", [], "vus")]

    # a dry run needs no key: the provider is recorded with its default model and disclosure
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--top", "1", "--provider", "openrouter", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert " · openrouter · anthropic/claude-opus-5 · effort high" in result.output
    m = read(run_dir / "05_reason" / "manifest.json")
    assert m["params"]["provider"] == "openrouter" and m["params"]["model"] == "anthropic/claude-opus-5"
    assert m["params"]["disclosure"] == providers.disclosure("openrouter", "anthropic/claude-opus-5", "high")
    assert read(run_dir / "05_reason" / "prompts" / "CFTR:hom.json")["model"] == "anthropic/claude-opus-5"

    # the stages' bare default id under openrouter is the OpenRouter default — what is sent is what is recorded
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--top", "1", "--provider", "openrouter",
                                       "--model", "claude-opus-5", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert " · openrouter · anthropic/claude-opus-5 · effort high" in result.output
    assert read(run_dir / "05_reason" / "manifest.json")["params"]["model"] == "anthropic/claude-opus-5"
    assert read(run_dir / "05_reason" / "prompts" / "CFTR:hom.json")["model"] == "anthropic/claude-opus-5"
    # any other bare id: one FAILED line before anything is written
    (run_dir / "05_reason" / "prompts" / "CFTR:hom.json").unlink()
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--top", "1", "--provider", "openrouter",
                                       "--model", "claude-sonnet-5", "--dry-run"])
    assert result.exit_code == 1 and result.output.count("FAILED: ") == 1
    assert "FAILED: openrouter model ids are author/slug (e.g. anthropic/claude-opus-5), got 'claude-sonnet-5'" in result.output
    assert not (run_dir / "05_reason" / "prompts" / "CFTR:hom.json").exists()

    # the same run without a key: one FAILED line naming the variable (before any model call), no chain
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--top", "1", "--provider", "openrouter",
                                       "--cache", cache, "--offline"])
    assert result.exit_code == 1 and result.output.count("FAILED: ") == 1
    assert "provider openrouter needs OPENROUTER_API_KEY" in result.output and "CFTR" not in result.output
    assert not (run_dir / "05_reason" / "chains").exists() and not (run_dir / "05_reason" / "manifest.json").exists()
    assert (run_dir / "05_reason" / "prompts" / "CFTR:hom.json").exists()  # the prompt is on disk before the client is built

    # no --provider and no key: anthropic is the default and its default model; the SDK decides about credentials
    monkeypatch.setattr(ac, "AnthropicClient", lambda **kw: providers.select_client("fake", "claude-opus-5", "high"))
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--top", "1", "--cache", cache, "--offline"])
    assert result.exit_code == 0, result.output
    assert " · anthropic · claude-opus-5 · effort high" in result.output
    assert read(run_dir / "05_reason" / "manifest.json")["params"]["provider"] == "anthropic"

    # ... and when the SDK then finds no credential either, the failure names both keys once
    def refuse(**kw: Any) -> None:
        raise TypeError("Could not resolve authentication method.")
    monkeypatch.setattr(ac, "AnthropicClient", refuse)
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--top", "1", "--cache", cache, "--offline"])
    assert result.exit_code == 1 and result.output.count("FAILED: ") == 1
    assert "FAILED: could not build the Anthropic client: Could not resolve authentication method." in result.output
    assert f"no model key: set OPENROUTER_API_KEY or ANTHROPIC_API_KEY in the environment or in {no_keys / '.env'}, " \
           "or pass --provider fake" in result.output
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--top", "1", "--provider", "anthropic",
                                       "--cache", cache, "--offline"])
    assert result.exit_code == 1 and "no model key" not in result.output  # an explicit provider gets no hint

    # a dotenv inside the checkout is refused before anything runs
    monkeypatch.setenv("ENGINE_ENV", str(no_keys / "engine" / ".env"))
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--top", "1", "--dry-run"])
    assert result.exit_code == 1 and "FAILED: refusing a dotenv inside the repository" in result.output
    monkeypatch.setenv("ENGINE_ENV", str(no_keys / "nowhere.env"))
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--top", "1", "--dry-run"])
    assert result.exit_code == 1 and "FAILED: dotenv not found" in result.output


def test_the_cli_defaults_to_openrouter_when_its_key_comes_from_the_dotenv(run_dir: Path, operator_shell: Path, monkeypatch: pytest.MonkeyPatch):
    no_keys = operator_shell
    (no_keys / ".env").write_text(f"OPENROUTER_API_KEY={KEY}\nENGINE_TEST_MARKER=from-dotenv\n")
    monkeypatch.setenv("ENGINE_TEST_MARKER", "real")
    stub = FakeHttp("tools_done", "final_answer")
    real = orc.OpenRouterClient
    monkeypatch.setattr(orc, "OpenRouterClient", lambda key, **kw: real(key, **{**kw, "http": stub}))
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--top", "1", "--cache", str(no_keys / "cache"), "--offline"])
    assert result.exit_code == 1, result.output  # the toy answer does not fit an EvidenceChain — the client validated it
    assert " · openrouter · anthropic/claude-opus-5 · effort high" in result.output
    assert "FAILED: final answer does not fit EvidenceChain" in result.output
    assert os.environ["OPENROUTER_API_KEY"] == KEY and os.environ["ENGINE_TEST_MARKER"] == "real"  # loaded, never overriding
    assert KEY not in result.output
    assert stub.calls and stub.calls[0].headers["Authorization"] == f"Bearer {KEY}"
    check_privacy(stub)
    failed = read(run_dir / "05_reason" / "transcripts" / "CFTR:hom.failed.json")
    assert failed["request_id"] == "gen-final" and [t["stop_reason"] for t in failed["transcript"]] == ["end_turn", "end_turn"]
    assert KEY not in json.dumps(failed)


def test_a_pytest_session_never_reads_the_dotenv_beside_the_checkout(run_dir: Path, no_keys: Path, monkeypatch: pytest.MonkeyPatch):
    """The finding this guards against: a plain ``pytest`` beside the operator's dotenv
    routed the stage tests that patch ``ac.AnthropicClient`` to a live OpenRouter call.
    Under pytest the default location is not read, so ``--provider`` defaults to
    anthropic and the patched client answers; a dotenv named by ENGINE_ENV still is."""
    (no_keys / ".env").write_text(f"OPENROUTER_API_KEY={KEY}\n")
    assert os.environ[providers.TEST_SESSION_VAR].endswith("(call)")  # what pytest exports while a test runs
    assert providers.test_session() and providers.dotenv_disabled()
    assert providers.resolve() == ("anthropic", "claude-opus-5") and not os.environ["OPENROUTER_API_KEY"]
    with monkeypatch.context() as m:
        m.delenv(providers.TEST_SESSION_VAR)
        assert not providers.test_session() and not providers.dotenv_disabled()
    monkeypatch.setattr(ac, "AnthropicClient", lambda **kw: providers.select_client("fake", "claude-opus-5", "high"))
    result = CliRunner().invoke(main, ["reason", "--run", str(run_dir), "--top", "1", "--cache", str(no_keys / "cache"), "--offline"])
    assert result.exit_code == 0, result.output
    assert " · anthropic · claude-opus-5 · effort high" in result.output and "api calls: 0" in result.output
    assert not os.environ["OPENROUTER_API_KEY"] and KEY not in result.output
    assert read(run_dir / "05_reason" / "manifest.json")["params"]["provider"] == "anthropic"

    monkeypatch.setenv("ENGINE_ENV", str(no_keys / ".env"))  # explicit: honoured even under pytest
    assert providers.resolve() == ("openrouter", "anthropic/claude-opus-5") and os.environ["OPENROUTER_API_KEY"] == KEY


def test_the_medicine_cli_shares_the_provider_choice(run_dir: Path, no_keys: Path):
    chains = run_dir / "05_reason" / "chains"
    chains.mkdir(parents=True)
    shutil.copy(FIXTURES / "medicine" / "chain_CFTR_hom.json", chains / "CFTR:hom.json")
    result = CliRunner().invoke(main, ["medicine", "--run", str(run_dir), "--provider", "openrouter", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "medicine · " in result.output and " · openrouter · anthropic/claude-opus-5 · effort high" in result.output
    m = read(run_dir / "06_medicine" / "manifest.json")
    assert m["params"]["provider"] == "openrouter" and m["params"]["model"] == "anthropic/claude-opus-5"
    assert m["params"]["disclosure"] == providers.disclosure("openrouter", "anthropic/claude-opus-5", "high")
    result = CliRunner().invoke(main, ["medicine", "--run", str(run_dir), "--provider", "openrouter", "--model", "claude-opus-5",
                                       "--dry-run"])
    assert result.exit_code == 0 and " · openrouter · anthropic/claude-opus-5 · " in result.output
    assert read(run_dir / "06_medicine" / "prompts" / "CFTR:hom.json")["model"] == "anthropic/claude-opus-5"
    result = CliRunner().invoke(main, ["medicine", "--run", str(run_dir), "--provider", "openrouter", "--model", "claude-sonnet-5",
                                       "--dry-run"])
    assert result.exit_code == 1 and "FAILED: openrouter model ids are author/slug" in result.output
    result = CliRunner().invoke(main, ["medicine", "--run", str(run_dir), "--provider", "openrouter",
                                       "--cache", str(no_keys / "cache"), "--offline"])
    assert result.exit_code == 1 and "FAILED: provider openrouter needs OPENROUTER_API_KEY" in result.output
    help_text = CliRunner().invoke(main, ["medicine", "--help"]).output
    assert "--provider [anthropic|openrouter|fake]" in help_text and "anthropic/claude-opus-5 for openrouter" in help_text


def test_resolve_loads_the_dotenv_then_settles_provider_and_model(operator_shell: Path, monkeypatch: pytest.MonkeyPatch):
    no_keys = operator_shell
    assert providers.resolve() == ("anthropic", "claude-opus-5")
    assert providers.resolve("fake") == ("fake", "fake-model")
    assert providers.resolve("openrouter", "anthropic/claude-sonnet-5") == ("openrouter", "anthropic/claude-sonnet-5")
    assert providers.resolve("openrouter", "claude-opus-5") == ("openrouter", "anthropic/claude-opus-5")
    with pytest.raises(providers.ProviderError, match="author/slug"):
        providers.resolve("openrouter", "claude-sonnet-5")
    (no_keys / ".env").write_text(f"OPENROUTER_API_KEY={KEY}\n")
    assert providers.resolve() == ("openrouter", "anthropic/claude-opus-5")
    assert providers.resolve("anthropic") == ("anthropic", "claude-opus-5")
    assert providers.default_provider() == "openrouter" and providers.status()["openrouter"]["key_present"] is True


def test_run_reason_records_the_provider_it_was_given(run_dir: Path, no_keys: Path):
    m = read(rr.run_reason(run_dir, 1, dry_run=True, provider="fake"))
    assert m["params"]["provider"] == "fake" and m["params"]["model"] == "fake-model"
    assert m["params"]["disclosure"].startswith("FakeClient")
    m = read(rr.run_reason(run_dir, 1, dry_run=True))
    assert m["params"]["provider"] is None and m["params"]["model"] == "claude-opus-5"
    assert m["params"]["disclosure"].startswith("Anthropic API, model claude-opus-5")
    # programmatic openrouter with the default model: the OpenRouter id everywhere, never the bare one
    m = read(rr.run_reason(run_dir, 1, dry_run=True, provider="openrouter"))
    assert m["params"]["model"] == "anthropic/claude-opus-5"
    assert m["params"]["disclosure"].startswith("OpenRouter API, model anthropic/claude-opus-5")
    assert read(run_dir / "05_reason" / "prompts" / "CFTR:hom.json")["model"] == "anthropic/claude-opus-5"
    m = read(rr.run_reason(run_dir, 1, dry_run=True, provider="openrouter", model="claude-opus-5"))
    assert m["params"]["model"] == "anthropic/claude-opus-5"
    with pytest.raises(providers.ProviderError, match="author/slug"):
        rr.run_reason(run_dir, 1, dry_run=True, provider="openrouter", model="claude-sonnet-5")


# --------------------------------------------------------------------------- live

@pytest.mark.live
@pytest.mark.skipif(not (os.environ.get("ENGINE_LIVE_TESTS") and os.environ.get("OPENROUTER_API_KEY")),
                    reason="set ENGINE_LIVE_TESTS=1 and export OPENROUTER_API_KEY for the OpenRouter smoke test "
                           "(a test session never reads the dotenv beside the checkout)")
def test_live_toy_question_through_openrouter(tmp_path: Path):
    """One toy arithmetic question, tiny budgets, the real transport (no retries of its
    own, as the client requires) over a throw-away cache that stays empty (model calls
    are never cached). Nothing from a case."""
    http = Http(HttpCache(tmp_path / "cache"), timeout=120.0, retries=0)
    c = orc.OpenRouterClient(os.environ["OPENROUTER_API_KEY"], http=http)
    req = ac.AgentRequest(system="You add integers with the add tool and answer briefly.",
                          user="Compute 2+3 and 10+20 with the add tool, then answer.", output_model=Sums,
                          tools=[add_tool()], max_turns=3, model="anthropic/claude-opus-5", effort="low",
                          max_tokens=2000, final_max_tokens=2000)
    result = c.run(req)
    assert isinstance(result.output, Sums) and {5, 30} <= set(result.output.results)
    assert result.provider_echo and result.generation_ids and result.usage.api_calls >= 2
    assert not any((tmp_path / "cache").rglob("*.json"))
    assert asdict(result.usage)["input_tokens"] > 0
