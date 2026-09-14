"""Jobs — a stage launched from the page as a subprocess, its output kept as events.

Why a subprocess and not a thread: a stage is ``uv run engine <stage> …`` exactly as
an operator types it, so what the page launches is what the shell would run, with
the same manifest, the same log lines and the same exit code — and a stage that
hangs or is cancelled cannot take the server with it. The command is assembled
here from a per-stage **whitelist** (:data:`WHITELIST`): a request names its
arguments by key (``{"top": 2, "dry_run": false}``), each key is checked against
the stage's table, its value against the table's type, and anything else is
refused before a process exists. Fixed arguments — the run directory, the case
file, the caches and reference bundles the operator configured — are added by the
runner, never taken from a request, so a request cannot point a stage at another
directory.

Output is captured line by line from both pipes into a bounded in-memory ring
(:class:`EventLog`, what the SSE stream reads) and into a log file beside the
stage's outputs (``<run>/<stage-dir>/ui-job-<id>.log``, what remains after the
server is gone). For stages 5 and 6 a :class:`TranscriptWatcher` polls the stage's
``transcripts/`` directory while the process runs and turns every tool call of a
transcript the stage writes into an event, so the page shows what the agent asked
for as each candidate finishes. Cancelling terminates the process group — ``uv``
and the ``engine`` it started — and records the job as cancelled.

Nothing here reads a key: the subprocess inherits the server's environment, and the
stage CLIs find their provider keys the way they always do.

Three kinds of job share the machinery. A **stage** job is one ``engine <stage>``.
A **setup** job is one of the three reference commands (:data:`SETUP_STAGES`) with a
fixed argv — ``engine clinvar-download --ref-dir <ref>`` and its siblings — that
takes nothing from a request. A **pipeline** job (:meth:`JobRunner.start_pipeline`)
runs the six stages in order, one subprocess after another, inside one job: each
stage gets the defaults the page would carry (:func:`pipeline_args`), a stage whose
manifest is already newer than every earlier stage's is skipped unless a rerun was
asked for (:func:`freshness`), the first failure stops the run — except ``rank``
without a usable Exomiser bundle, or killed for memory (exit 137), which is marked
skipped and the pipeline goes on, since stages 5 and 6 do not need it — and a
cancel ends the current stage and the pipeline. The event stream carries one
``stage`` event per transition (``started``/``done``/``skipped``/``failed``) with the
stage's wall seconds, in addition to the lines and tool calls.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

log = logging.getLogger("engine.ui")

STAGE_DIRS: dict[str, str] = {
    "ingest": "01_ingest",
    "retrieve": "02_retrieve",
    "filter": "03_filter",
    "rank": "04_rank",
    "reason": "05_reason",
    "medicine": "06_medicine",
}
"""Stage name → the directory it writes under the run, in pipeline order."""

AGENT_STAGES = ("reason", "medicine")
"""The stages whose ``transcripts/`` are watched for tool calls."""

PIPELINE = tuple(STAGE_DIRS)
"""The six stages in the order the pipeline runs them."""
PIPELINE_STAGE = "pipeline"
SETUP_STAGES = ("clinvar-download", "funnel", "exomiser-download")
"""The reference commands the Setup panel may start; each has a fixed argv and takes no argument from a request."""
SETUP_RUN = ".setup"
"""The pseudo-run setup jobs belong to (not a valid run name, so it never collides with a run)."""
OOM_EXIT = 137
"""What a container killed by the daemon's memory limit exits with."""
DEFAULT_VEP_WORKERS = 6
"""``retrieve --vep-workers`` in a pipeline."""
_OOM_TEXT = ("exited with code 137", "OutOfMemoryError", "out of memory")

DEFAULT_COMMAND: tuple[str, ...] = ("uv", "run", "engine")
RUN_RECORD = "ui-run.json"
"""What ``POST /api/runs`` writes into a new run directory: the case path as given."""
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
"""A run name is one path segment: no slash, no leading dot, nothing a shell needs quoted."""

PROVIDERS = ("anthropic", "openrouter", "fake")
"""``engine.agents.providers.PROVIDERS``, likewise copied and checked by the tests."""
EFFORTS = ("low", "medium", "high", "xhigh", "max")
"""The stage CLIs' choices (``engine.agents.client.EFFORTS``), copied so a request is
checked without importing the SDK; ``tests/test_ui.py`` asserts the two agree."""
SOURCES = ("vep", "clinvar", "gnomad")
_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")
_HEAP = re.compile(r"^[1-9][0-9]{0,4}[gGmM]$")
_CANDIDATE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")


class ArgumentError(ValueError):
    """A request argument the whitelist refuses (the API answers 400)."""


class JobConflict(RuntimeError):
    """A job is already running for the run (the API answers 409)."""


class NotConfigured(RuntimeError):
    """The stage needs a fixed input the operator did not configure (the API answers 409)."""


# ---------------------------------------------------------------------- whitelist

@dataclass(frozen=True)
class Arg:
    """One argument a stage accepts from a request and how it becomes a CLI option."""

    name: str
    kind: str
    """``bool`` (a flag), ``int``, ``choice``, ``path``, ``text``, ``candidate``, ``heap``, ``sources``."""
    option: str
    help: str
    choices: tuple[str, ...] = ()
    minimum: int | None = None
    maximum: int | None = None
    default: Any = None

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "option": self.option, "help": self.help,
                "choices": list(self.choices), "min": self.minimum, "max": self.maximum, "default": self.default}


_PROVIDER = Arg("provider", "choice", "--provider", "Who answers: anthropic, openrouter or fake.", choices=PROVIDERS)
_MODEL = Arg("model", "text", "--model", "Model id for the provider (default: the provider's).")
_EFFORT = Arg("effort", "choice", "--effort", "Effort for every call.", choices=EFFORTS, default="high")
_DRY_RUN = Arg("dry_run", "bool", "--dry-run", "Write the bundles and prompts, call no model.", default=True)

WHITELIST: dict[str, tuple[Arg, ...]] = {
    "ingest": (
        Arg("regions", "path", "--regions", "BED of regions to restrict to (overrides the case file)."),
    ),
    "retrieve": (
        Arg("funnel", "path", "--funnel", "BED restricting which variants are sent to the sources."),
        Arg("sources", "sources", "--sources", "Comma-separated subset of vep,clinvar,gnomad."),
        Arg("vep_workers", "int", "--vep-workers", "Concurrent VEP batches.", minimum=1, maximum=12),
    ),
    "filter": (),
    "rank": (
        Arg("heap", "heap", "--heap", "JVM -Xmx for Exomiser, e.g. 6g."),
    ),
    "reason": (
        _PROVIDER, _MODEL, _EFFORT,
        Arg("top", "int", "--top", "How many stage-3 candidates to reason about.", minimum=1, maximum=1000),
        _DRY_RUN,
    ),
    "medicine": (
        _PROVIDER, _MODEL, _EFFORT, _DRY_RUN,
        Arg("candidate", "candidate", "--candidate", "Candidate id to report on (default: the best stage-5 chain)."),
    ),
}
"""Every argument a request may pass, per stage — the CLI's own options and nothing
else. ``reason`` has no ``candidate`` and ``medicine`` no ``top`` because the commands
do not; a key the stage does not list is refused, never dropped or passed through."""


def describe_stages() -> list[dict[str, Any]]:
    """The whitelist as the page reads it: stage, directory, arguments, whether it is an agent stage."""
    return [{"stage": name, "dir": STAGE_DIRS[name], "agent": name in AGENT_STAGES,
             "args": [a.describe() for a in WHITELIST[name]]} for name in STAGE_DIRS]


def check_args(stage: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """The request's arguments, validated against the stage's whitelist; a copy with
    each value normalised (``"3"`` → ``3`` for an int, ``"true"`` → ``True`` for a flag).
    Raises :class:`ArgumentError` naming every unknown key and the first bad value."""
    if stage not in WHITELIST:
        raise ArgumentError(f"unknown stage {stage!r}; one of {', '.join(STAGE_DIRS)}")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ArgumentError("args must be an object")
    table = {a.name: a for a in WHITELIST[stage]}
    unknown = sorted(str(k) for k in args if k not in table)
    if unknown:
        allowed = ", ".join(table) or "none"
        raise ArgumentError(f"stage {stage} does not take {', '.join(unknown)} (allowed: {allowed})")
    out: dict[str, Any] = {}
    for name, value in args.items():
        if value is None:
            continue
        out[name] = _check_value(table[name], value)
    return out


def _check_value(arg: Arg, value: Any) -> Any:
    kind = arg.kind
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false", "1", "0", "yes", "no"):
            return value.lower() in ("true", "1", "yes")
        raise ArgumentError(f"{arg.name} must be true or false")
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ArgumentError(f"{arg.name} must be an integer")
        try:
            n = int(value)
        except ValueError:
            raise ArgumentError(f"{arg.name} must be an integer") from None
        if arg.minimum is not None and n < arg.minimum or arg.maximum is not None and n > arg.maximum:
            raise ArgumentError(f"{arg.name} must be between {arg.minimum} and {arg.maximum}")
        return n
    if not isinstance(value, str):
        raise ArgumentError(f"{arg.name} must be a string")
    text = value.strip()
    if not text:
        raise ArgumentError(f"{arg.name} must not be empty")
    if kind == "choice":
        if text not in arg.choices:
            raise ArgumentError(f"{arg.name} must be one of {', '.join(arg.choices)}")
        return text
    if kind == "sources":
        parts = tuple(p.strip() for p in text.split(",") if p.strip())
        bad = [p for p in parts if p not in SOURCES]
        if not parts or bad:
            raise ArgumentError(f"sources must be a comma-separated subset of {','.join(SOURCES)}")
        return ",".join(parts)
    if kind == "heap":
        if not _HEAP.match(text):
            raise ArgumentError("heap must look like 6g or 4096m")
        return text
    if kind == "candidate":
        if not _CANDIDATE.match(text):
            raise ArgumentError("candidate must be a candidate id such as CFTR:comphet")
        return text
    if kind == "path":
        if text.startswith("-") or "\x00" in text or "\n" in text:
            raise ArgumentError(f"{arg.name} must be a file path")
        return text
    if kind == "text":
        if not _TEXT.match(text):
            raise ArgumentError(f"{arg.name} holds a character the command line cannot take")
        return text
    raise ArgumentError(f"{arg.name}: unknown argument kind {kind!r}")  # pragma: no cover


# ------------------------------------------------------------------- run record

def read_run_record(run_dir: Path) -> dict[str, Any]:
    """``ui-run.json`` if the page created the run, else what the manifests say: the
    case file a stage recorded as an input (stages 4–6 do) and the VCF stage 1 read."""
    run_dir = Path(run_dir)
    doc = _read_json(run_dir / RUN_RECORD)
    out: dict[str, Any] = {"name": run_dir.name, "case_path": None, "regions": None, "created_at": None, "source": None}
    if isinstance(doc, dict):
        out.update({k: doc.get(k) for k in ("case_path", "regions", "created_at")})
        out["source"] = RUN_RECORD
    if not out["case_path"]:
        for stage_dir in ("04_rank", "05_reason", "06_medicine"):
            m = _read_json(run_dir / stage_dir / "manifest.json")
            case = ((m or {}).get("inputs") or {}).get("case") if isinstance(m, dict) else None
            if isinstance(case, dict) and case.get("path"):
                out["case_path"], out["source"] = str(case["path"]), f"{stage_dir}/manifest.json"
                break
    ingest = _read_json(run_dir / "01_ingest" / "manifest.json")
    vcf = ((ingest or {}).get("inputs") or {}).get("vcf") if isinstance(ingest, dict) else None
    out["vcf_path"] = vcf.get("path") if isinstance(vcf, dict) else None
    return out


def write_run_record(run_dir: Path, case_path: str, regions: str | None = None) -> Path:
    path = Path(run_dir) / RUN_RECORD
    path.write_text(json.dumps({"name": Path(run_dir).name, "case_path": case_path, "regions": regions,
                                "created_at": _now()}, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------- command

@dataclass(frozen=True)
class RunnerConfig:
    """The fixed inputs the operator configured for the stages that need one — passed
    on the command line by the runner, never by a request. ``None`` means the stage's
    own default applies (``retrieve``/``reason``/``medicine`` default their cache to
    ``$ENGINE_CACHE``); a stage that has no default for a missing input is refused
    with :class:`NotConfigured` before a process starts."""

    cache: Path | None = None
    clinvar_vcf: Path | None = None
    exomiser_data: Path | None = None
    funnel_bed: Path | None = None
    """The funnel BED: ``retrieve --funnel`` and ``rank --regions`` unless a request names another."""

    def describe(self) -> dict[str, Any]:
        return {"cache": _s(self.cache), "clinvar_vcf": _s(self.clinvar_vcf), "exomiser_data": _s(self.exomiser_data),
                "funnel_bed": _s(self.funnel_bed)}


def build_argv(stage: str, args: dict[str, Any], run_dir: Path, *, record: dict[str, Any],
               config: RunnerConfig = RunnerConfig(), command: tuple[str, ...] = DEFAULT_COMMAND) -> list[str]:
    """The full command line: the prefix, the stage, the fixed arguments the stage needs
    (run directory, case file, configured caches and bundles), then the whitelisted
    arguments in whitelist order. ``args`` must already have passed :func:`check_args`."""
    argv = [*command, stage]
    case = record.get("case_path")
    if stage == "ingest":
        if not case:
            raise NotConfigured("ingest needs the run's case file: create the run through the page (POST /api/runs) "
                                "so the case path is recorded")
        argv += ["--case", str(case), "--out", str(run_dir)]
        if record.get("regions") and "regions" not in args:
            argv += ["--regions", str(record["regions"])]
    else:
        argv += ["--run", str(run_dir)]
    if stage == "retrieve":
        sources = str(args.get("sources") or "vep,clinvar,gnomad")
        if "clinvar" in sources:
            if config.clinvar_vcf is None:
                raise NotConfigured("retrieve with the clinvar source needs the pinned ClinVar VCF: run clinvar-download "
                                    "from the Setup panel, start the UI with --clinvar-vcf (or CLINVAR_VCF), or pass "
                                    "sources without clinvar")
            argv += ["--clinvar-vcf", str(config.clinvar_vcf)]
        if config.funnel_bed is not None and "funnel" not in args:
            argv += ["--funnel", str(config.funnel_bed)]
    if stage == "rank":
        if not case:
            raise NotConfigured("rank needs the run's case file (HPO terms): no ui-run.json and no manifest records it")
        if config.exomiser_data is None:
            raise NotConfigured("rank needs a verified Exomiser data bundle: run exomiser-download from the Setup panel "
                                "or start the UI with --exomiser-data (or EXOMISER_DATA)")
        argv += ["--case", str(case), "--exomiser-data", str(config.exomiser_data)]
        if config.funnel_bed is not None:
            argv += ["--regions", str(config.funnel_bed)]  # the funnel's regions only: never the whole genome into Exomiser
    if stage in AGENT_STAGES and case:
        argv += ["--case", str(case)]
    if stage in ("retrieve", *AGENT_STAGES) and config.cache is not None:
        argv += ["--cache", str(config.cache)]
    for arg in WHITELIST[stage]:
        if arg.name not in args:
            continue
        value = args[arg.name]
        if arg.kind == "bool":
            if value:
                argv.append(arg.option)
        else:
            argv += [arg.option, str(value)]
    return argv


# ---------------------------------------------------------------------- events

@dataclass(frozen=True)
class Event:
    seq: int
    type: str
    data: dict[str, Any]
    time: str

    def sse(self) -> str:
        body = json.dumps({"seq": self.seq, "time": self.time, **self.data}, ensure_ascii=False, default=str)
        return f"event: {self.type}\nid: {self.seq}\ndata: {body}\n\n"


class EventLog:
    """A bounded, thread-safe ring of :class:`Event` with a sequence number per event
    and a condition a reader can wait on. Lines a client missed because the ring
    turned over are in the job's log file."""

    def __init__(self, maxlen: int = 5000):
        self._events: deque[Event] = deque(maxlen=maxlen)
        self._seq = 0
        self._cond = threading.Condition()
        self.closed = False

    def append(self, type: str, data: dict[str, Any]) -> Event:
        with self._cond:
            self._seq += 1
            e = Event(self._seq, type, data, _now())
            self._events.append(e)
            if type == "done":
                self.closed = True
            self._cond.notify_all()
            return e

    @property
    def last_seq(self) -> int:
        with self._cond:
            return self._seq

    def since(self, seq: int) -> list[Event]:
        with self._cond:
            return [e for e in self._events if e.seq > seq]

    def wait(self, seq: int, timeout: float) -> list[Event]:
        """Events after ``seq``, waiting up to ``timeout`` seconds for one to arrive;
        empty when none did (a keepalive is due) or the log is closed."""
        with self._cond:
            if self._seq <= seq and not self.closed:
                self._cond.wait(timeout)
            return [e for e in self._events if e.seq > seq]

    def snapshot(self) -> list[Event]:
        with self._cond:
            return list(self._events)


# ------------------------------------------------------------------ transcripts

class TranscriptWatcher:
    """Tails ``<stage-dir>/transcripts/`` for the files stages 5 and 6 write when a
    candidate's agent finishes (``<candidate>.json``, or ``<candidate>.failed.json``
    when the model or a tool failed) and emits one ``tool_call`` event per tool call
    plus one ``transcript`` event per file. A file is taken up when it is new since
    :meth:`snapshot` or has changed since it was last read; one that is not yet
    complete JSON (the stage is still writing it) is retried on the next poll."""

    def __init__(self, stage_dir: Path, emit: Callable[[str, dict[str, Any]], Any], *, preview_chars: int = 400):
        self.dir = Path(stage_dir) / "transcripts"
        self.emit = emit
        self.preview_chars = preview_chars
        self._seen: dict[str, tuple[int, int]] = {}
        self._emitted: dict[str, int] = {}
        self.files = 0
        self.tool_calls = 0

    def snapshot(self) -> None:
        """Remember what is there before the stage starts, so an earlier run's
        transcripts are not replayed (the stage removes them, but not before the
        first poll could see them)."""
        self._seen = self._stat_all()

    def poll(self) -> None:
        current = self._stat_all()
        for name, stamp in sorted(current.items()):
            if self._seen.get(name) == stamp:
                continue
            if self._read(name):
                self._seen[name] = stamp

    def _stat_all(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        try:
            for p in self.dir.iterdir():
                if p.suffix == ".json" and p.is_file():
                    st = p.stat()
                    out[p.name] = (st.st_mtime_ns, st.st_size)
        except OSError:
            pass
        return out

    def _read(self, name: str) -> bool:
        path = self.dir / name
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False  # still being written, or gone: try again next poll
        if not isinstance(doc, dict):
            return True
        failed = name.endswith(".failed.json")
        stem = name[:-len(".failed.json")] if failed else name[:-len(".json")]
        candidate = str(doc.get("candidate_id") or stem)
        calls: list[dict[str, Any]] = []
        for turn in doc.get("transcript") or []:
            if not isinstance(turn, dict):
                continue
            for call in turn.get("tool_calls") or []:
                if isinstance(call, dict):
                    calls.append({**call, "turn": turn.get("n")})
        already = self._emitted.get(name, 0)
        for i, call in enumerate(calls[already:], already + 1):
            result = str(call.get("result", ""))
            self.emit("tool_call", {
                "candidate_id": candidate, "file": name, "n": i, "turn": call.get("turn"),
                "id": call.get("id"), "name": call.get("name"), "input": call.get("input"),
                "is_error": bool(call.get("is_error")),
                "result_chars": len(result), "result_preview": result[:self.preview_chars],
            })
            self.tool_calls += 1
        self._emitted[name] = len(calls)
        self.files += 1
        usage = doc.get("usage") if isinstance(doc.get("usage"), dict) else {}
        self.emit("transcript", {
            "candidate_id": candidate, "file": name, "failed": failed,
            "error": doc.get("error") if failed else None, "request_id": doc.get("request_id") if failed else None,
            "turns": len(doc.get("transcript") or []), "tool_calls": len(calls),
            "stop_reason": doc.get("stop_reason"), "model": doc.get("model"),
            "usage": {k: usage.get(k) for k in ("input_tokens", "output_tokens", "api_calls", "tool_calls", "tool_errors")},
        })
        return True


# ------------------------------------------------------------------------ jobs

class Process(Protocol):
    """What the runner needs from a process — :class:`subprocess.Popen` has it all; a
    test's fake needs only this."""

    pid: int
    returncode: int | None
    stdout: Any
    stderr: Any

    def wait(self, timeout: float | None = None) -> int: ...
    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


Spawn = Callable[[list[str], Path, dict[str, str]], Process]


class _Popen(subprocess.Popen):
    """A process in its own session, so terminating it reaches the ``engine`` that
    ``uv run`` started and not only ``uv``."""

    def terminate(self) -> None:
        try:
            os.killpg(self.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, AttributeError, OSError):
            super().terminate()

    def kill(self) -> None:
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, AttributeError, OSError):
            super().kill()


def popen(argv: list[str], cwd: Path, env: dict[str, str]) -> Process:
    """The default :data:`Spawn`: pipes for both outputs, no stdin, a new session."""
    return _Popen(argv, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                  stdin=subprocess.DEVNULL, start_new_session=True)


@dataclass
class StageState:
    """One stage's place in a pipeline job."""

    stage: str
    dir: str
    status: str = "pending"
    """``pending`` → ``running`` → ``done`` | ``skipped`` | ``failed`` | ``cancelled``."""
    reason: str | None = None
    seconds: float | None = None
    exit_code: int | None = None
    argv: list[str] | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "dir": self.dir, "status": self.status, "reason": self.reason, "seconds": self.seconds,
                "exit_code": self.exit_code, "argv": self.argv, "started_at": self.started_at, "finished_at": self.finished_at}


@dataclass
class PipelineState:
    """What a pipeline job was asked for and where it is."""

    live: bool
    rerun: bool
    top: int
    provider: str
    model: str
    effort: str
    heap: str
    stages: list[StageState]
    current: str | None = None
    heap_note: str | None = None

    def stage(self, name: str) -> StageState:
        return next(s for s in self.stages if s.stage == name)

    def to_dict(self) -> dict[str, Any]:
        return {"live": self.live, "rerun": self.rerun, "top": self.top, "provider": self.provider, "model": self.model,
                "effort": self.effort, "heap": self.heap, "heap_note": self.heap_note, "current": self.current,
                "current_dir": STAGE_DIRS.get(self.current) if self.current else None,
                "stages": [s.to_dict() for s in self.stages]}


@dataclass
class Job:
    id: str
    run: str
    run_dir: Path
    stage: str
    args: dict[str, Any]
    argv: list[str]
    log_path: Path
    events: EventLog = field(default_factory=EventLog)
    status: str = "starting"
    """``starting`` → ``running`` → ``done`` (exit 0) | ``failed`` (any other exit) | ``cancelled``."""
    pid: int | None = None
    exit_code: int | None = None
    started_at: str = field(default_factory=lambda: _now())
    finished_at: str | None = None
    cancel_requested: bool = False
    tool_calls: int = 0
    transcripts: int = 0
    kind: str = "stage"
    """``stage`` (one ``engine <stage>``), ``pipeline`` (the six in order), ``setup`` (a reference command)."""
    pipeline: PipelineState | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def stage_dir(self) -> Path:
        if self.kind == "stage":
            return self.run_dir / STAGE_DIRS[self.stage]
        return self.run_dir

    @property
    def manifest_path(self) -> Path | None:
        if self.kind == "stage":
            return self.stage_dir / "manifest.json"
        if self.kind == "pipeline":
            return self.run_dir / STAGE_DIRS[PIPELINE[-1]] / "manifest.json"
        return None

    @property
    def live(self) -> bool:
        """An agent stage that will call a model (``dry_run`` off), or a live pipeline."""
        if self.kind == "pipeline":
            return bool(self.pipeline and self.pipeline.live)
        return self.kind == "stage" and self.stage in AGENT_STAGES and not self.args.get("dry_run", False)

    @property
    def agent(self) -> bool:
        return self.kind == "pipeline" or (self.kind == "stage" and self.stage in AGENT_STAGES)

    def finished(self) -> bool:
        return self.status in ("done", "failed", "cancelled")

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            manifest = self.manifest_path
            current = self.pipeline.current if self.pipeline else None
            return {
                "id": self.id, "run": self.run, "run_dir": str(self.run_dir), "stage": self.stage, "kind": self.kind,
                "stage_dir": STAGE_DIRS.get(self.stage), "args": dict(self.args), "argv": list(self.argv),
                "status": self.status, "pid": self.pid, "exit_code": self.exit_code,
                "started_at": self.started_at, "finished_at": self.finished_at,
                "log": str(self.log_path), "manifest": None if manifest is None else str(manifest),
                "manifest_present": manifest is not None and manifest.exists(), "live": self.live,
                "agent": self.agent, "tool_calls": self.tool_calls, "transcripts": self.transcripts,
                "events": self.events.last_seq, "cancel_requested": self.cancel_requested,
                "current_stage": current, "current_dir": STAGE_DIRS.get(current) if current else None,
                "pipeline": self.pipeline.to_dict() if self.pipeline else None,
            }


# ------------------------------------------------------------- pipeline defaults

def pipeline_args(stage: str, *, live: bool, top: int, provider: str, model: str | None, effort: str, heap: str,
                  candidate: str | None = None) -> dict[str, Any]:
    """The whitelisted arguments a pipeline carries into ``stage`` — what the page would
    fill in by hand: 6 VEP workers for ``retrieve`` (the funnel BED and the ClinVar VCF
    are fixed inputs :func:`build_argv` adds), the sized heap for ``rank``, and for the
    agent stages the provider whose key is present with its default model, effort
    ``high``, the top-N and — for ``medicine`` — the first stage-3 candidate. Live off
    means the scripted ``fake`` provider: the stages run and write, no model is called."""
    if stage == "retrieve":
        return {"vep_workers": DEFAULT_VEP_WORKERS}
    if stage == "rank":
        return {"heap": heap}
    if stage in AGENT_STAGES:
        chosen = provider if live else "fake"
        args: dict[str, Any] = {"provider": chosen, "effort": effort, "dry_run": False}
        if model and chosen == provider:
            args["model"] = model
        if stage == "reason":
            args["top"] = int(top)
        elif candidate:
            args["candidate"] = candidate
        return check_args(stage, args)
    return {}


def first_candidate(run_dir: Path) -> str | None:
    """The first candidate id in ``03_filter/candidates.json`` (stage 3 sorts by priority),
    or ``None`` when there is none or the id would not pass the whitelist."""
    doc = _read_json(Path(run_dir) / STAGE_DIRS["filter"] / "candidates.json")
    cands = doc.get("candidates") if isinstance(doc, dict) else None
    if not isinstance(cands, list) or not cands:
        return None
    first = cands[0].get("candidate_id") if isinstance(cands[0], dict) else None
    return first if isinstance(first, str) and _CANDIDATE.match(first) else None


def freshness(run_dir: Path, stage: str, *, live: bool = False) -> tuple[bool, str]:
    """Whether ``stage`` may be skipped: its manifest exists and is not older than any
    earlier stage's manifest (an earlier stage rerun since makes it stale). An agent
    stage's dry-run manifest is stale for a live pipeline: the chain or report it did
    not write is what is wanted. Returns ``(fresh, why)``."""
    run_dir = Path(run_dir)
    own = run_dir / STAGE_DIRS[stage] / "manifest.json"
    if not own.exists():
        return False, "no manifest yet"
    own_time = _mtime(own)
    newer = []
    for earlier in PIPELINE[:PIPELINE.index(stage)]:
        m = run_dir / STAGE_DIRS[earlier] / "manifest.json"
        if m.exists() and _mtime(m) > own_time:
            newer.append(earlier)
    if newer:
        return False, f"older than {', '.join(newer)}"
    if stage in AGENT_STAGES and live:
        doc = _read_json(own)
        if isinstance(doc, dict) and (doc.get("params") or {}).get("dry_run"):
            return False, "the last run was a dry run"
    return True, "outputs exist and are newer than every earlier stage's"


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# ---------------------------------------------------------------------- runner

class JobRunner:
    """Starts, lists and cancels jobs; one running job per run directory.

    ``spawn`` is how a process is made (:func:`popen`; a test passes a fake), ``command``
    the prefix before the stage name, ``repo`` the checkout ``uv run`` executes in.
    ``env`` overrides the environment the subprocess inherits (``VIRTUAL_ENV`` is
    always dropped, so ``uv`` uses the project's own; ``PYTHONUNBUFFERED`` is set so
    lines arrive as they are printed)."""

    def __init__(self, work: Path, *, repo: Path | None = None, spawn: Spawn = popen,
                 command: tuple[str, ...] = DEFAULT_COMMAND, config: RunnerConfig = RunnerConfig(),
                 env: dict[str, str] | None = None, poll_interval: float = 0.5, ring_size: int = 5000,
                 kill_after: float = 10.0):
        self.work = Path(work)
        self.repo = Path(repo) if repo else repo_root()
        self.spawn = spawn
        self.command = tuple(command)
        self.config = config
        self.env = env
        self.poll_interval = poll_interval
        self.ring_size = ring_size
        self.kill_after = kill_after
        self._jobs: dict[str, Job] = {}
        self._procs: dict[str, Process] = {}
        self._lock = threading.Lock()
        self._n = 0

    # -- queries

    def list(self, run: str | None = None) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [j for j in jobs if run is None or j.run == run]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def running(self, run: str) -> Job | None:
        for j in self.list(run):
            if not j.finished():
                return j
        return None

    # -- lifecycle

    def start(self, run: str, run_dir: Path, stage: str, args: dict[str, Any] | None) -> Job:
        """Validate, assemble, spawn. Raises :class:`ArgumentError` (400),
        :class:`JobConflict` (409), :class:`NotConfigured` (409)."""
        checked = check_args(stage, args)
        record = read_run_record(run_dir)
        argv = build_argv(stage, checked, run_dir, record=record, config=self.config, command=self.command)
        stage_dir = run_dir / STAGE_DIRS[stage]
        job = self._register(run, run_dir, stage, checked, argv, stage_dir, "ui-job")
        log_file, log_lock = self._open_log(job, argv)
        watcher = TranscriptWatcher(stage_dir, lambda t, d: self._emit_transcript(job, t, d)) if stage in AGENT_STAGES else None
        if watcher is not None:
            watcher.snapshot()
        proc = self._spawn(job, argv, log_file)
        if proc is None:
            return job
        job.events.append("status", {"status": "running", "pid": proc.pid, "argv": argv, "stage": stage, "run": run,
                                     "live": job.live, "log": str(job.log_path)})
        threading.Thread(target=self._run_single, args=(job, proc, stage, watcher, log_file, log_lock), daemon=True).start()
        log.debug("job %s started: %s %s (pid %s)", job.id, stage, run, proc.pid)
        return job

    def start_setup(self, stage: str, argv_tail: list[str]) -> Job:
        """One of :data:`SETUP_STAGES` with the fixed argv the setup gave (``argv_tail``
        is ``[<cmd>, --ref-dir, <ref>, …]``); one setup job at a time."""
        if stage not in SETUP_STAGES or not argv_tail or argv_tail[0] != stage:
            raise ArgumentError(f"setup stage must be one of {', '.join(SETUP_STAGES)}")
        argv = [*self.command, *argv_tail]
        setup_dir = self.work / SETUP_RUN
        job = self._register(SETUP_RUN, setup_dir, stage, {}, argv, setup_dir, "ui-setup", kind="setup")
        log_file, log_lock = self._open_log(job, argv)
        proc = self._spawn(job, argv, log_file)
        if proc is None:
            return job
        job.events.append("status", {"status": "running", "pid": proc.pid, "argv": argv, "stage": stage, "run": SETUP_RUN,
                                     "live": False, "log": str(job.log_path)})
        threading.Thread(target=self._run_single, args=(job, proc, stage, None, log_file, log_lock), daemon=True).start()
        return job

    def start_pipeline(self, run: str, run_dir: Path, *, live: bool, rerun: bool, top: int, provider: str,
                       model: str | None, effort: str, heap: str, heap_note: str | None = None) -> Job:
        """The six stages in order as one job. Every stage's argument set is built and
        checked here, before anything runs, so a missing fixed input (the ClinVar VCF
        for ``retrieve``) is a refusal now, not a failure four stages in — except
        ``rank``, whose missing bundle becomes a skipped stage."""
        record = read_run_record(run_dir)
        if not record.get("case_path"):
            raise NotConfigured("the pipeline needs the run's case file: create the run through the page (POST /api/runs)")
        plan: list[StageState] = []
        per_stage: dict[str, dict[str, Any]] = {}
        for stage in PIPELINE:
            args = pipeline_args(stage, live=live, top=top, provider=provider, model=model, effort=effort, heap=heap)
            per_stage[stage] = args
            state = StageState(stage, STAGE_DIRS[stage])
            try:
                build_argv(stage, args, run_dir, record=record, config=self.config, command=self.command)
            except NotConfigured as e:
                if stage != "rank":
                    raise
                state.status, state.reason = "skipped", str(e)
            plan.append(state)
        pipeline = PipelineState(live=live, rerun=rerun, top=int(top), provider=provider if live else "fake",
                                 model=model or "" if live else "fake-model", effort=effort, heap=heap, stages=plan,
                                 heap_note=heap_note)
        argv = [*self.command, PIPELINE_STAGE]  # what the job shows as its command: the stages carry their own
        job = self._register(run, run_dir, PIPELINE_STAGE, {"live": live, "rerun": rerun, "top": int(top)}, argv,
                             run_dir, "ui-pipeline", kind="pipeline", pipeline=pipeline)
        log_file, log_lock = self._open_log(job, argv)
        with job._lock:
            job.status = "running"
        job.events.append("status", {"status": "running", "argv": argv, "stage": PIPELINE_STAGE, "run": run,
                                     "live": live, "log": str(job.log_path), "pipeline": pipeline.to_dict()})
        threading.Thread(target=self._run_pipeline, args=(job, record, per_stage, log_file, log_lock), daemon=True).start()
        log.debug("pipeline %s started for %s (live=%s rerun=%s)", job.id, run, live, rerun)
        return job

    def cancel(self, job_id: str) -> Job | None:
        """Terminate a running job — for a pipeline, the current stage and the rest;
        ``None`` for an unknown id, the job unchanged when it has already finished."""
        job = self.get(job_id)
        if job is None or job.finished():
            return job
        with job._lock:
            job.cancel_requested = True
        job.events.append("status", {"status": "cancelling"})
        with self._lock:
            proc = self._procs.get(job_id)
        if proc is not None:
            try:
                proc.terminate()
            except OSError as e:  # already gone
                log.debug("terminate %s: %s", job_id, e)
        return job

    # -- pieces

    def _register(self, run: str, run_dir: Path, stage: str, args: dict[str, Any], argv: list[str], log_dir: Path,
                  log_prefix: str, *, kind: str = "stage", pipeline: PipelineState | None = None) -> Job:
        with self._lock:
            for j in self._jobs.values():
                if j.run == run and not j.finished():
                    raise JobConflict(f"job {j.id} ({j.stage}) is still {j.status} for run {run}")
            self._n += 1
            job_id = f"{self._n:03d}-{int(time.time()) % 100000:05d}"
            log_dir.mkdir(parents=True, exist_ok=True)
            job = Job(job_id, run, run_dir, stage, args, argv, log_dir / f"{log_prefix}-{job_id}.log",
                      events=EventLog(self.ring_size), kind=kind, pipeline=pipeline)
            self._jobs[job_id] = job
        return job

    def _open_log(self, job: Job, argv: list[str]) -> tuple[Any, threading.Lock]:
        log_file = open(job.log_path, "a", encoding="utf-8")
        log_file.write(f"# engine ui {job.kind} {job.id} · {job.started_at}\n# $ {' '.join(argv)}\n")
        log_file.flush()
        return log_file, threading.Lock()

    def _spawn(self, job: Job, argv: list[str], log_file: Any) -> Process | None:
        """Spawn ``argv`` for ``job``; on failure the job is finished as ``failed`` here
        (exit -1) and ``None`` is returned — the caller has nothing to supervise."""
        try:
            proc = self.spawn(argv, self.repo, self._env())
        except OSError as e:
            with job._lock:
                job.status, job.exit_code, job.finished_at = "failed", -1, _now()
            log_file.write(f"# could not start: {e}\n")
            log_file.close()
            job.events.append("status", {"status": "failed", "error": str(e), "argv": argv})
            job.events.append("done", {"exit_code": -1, "status": "failed", "error": str(e),
                                       "manifest": _s(job.manifest_path), "manifest_present": False})
            return None
        with job._lock:
            job.pid, job.status = proc.pid, "running"
        with self._lock:
            self._procs[job.id] = proc
        return proc

    def _supervise(self, job: Job, proc: Process, stage: str, watcher: TranscriptWatcher | None, log_file: Any,
                   log_lock: threading.Lock, tail: deque[str] | None = None) -> int:
        """Read both pipes until the process exits, polling the transcript watcher and
        honouring a cancel (terminate, then kill after ``kill_after``); the exit code."""
        readers = [threading.Thread(target=self._read, args=(job, proc.stdout, "stdout", stage, log_file, log_lock, tail), daemon=True),
                   threading.Thread(target=self._read, args=(job, proc.stderr, "stderr", stage, log_file, log_lock, tail), daemon=True)]
        for t in readers:
            t.start()
        killed_at: float | None = None
        while True:
            try:
                proc.wait(timeout=self.poll_interval)
                break
            except subprocess.TimeoutExpired:
                pass
            if watcher is not None:
                watcher.poll()
            if job.cancel_requested:
                if killed_at is None:
                    killed_at = time.monotonic()
                    try:
                        proc.terminate()
                    except OSError:
                        pass
                elif time.monotonic() - killed_at > self.kill_after:
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    killed_at = float("inf")
        for t in readers:
            t.join(timeout=5)
        if watcher is not None:
            watcher.poll()
        with self._lock:
            self._procs.pop(job.id, None)
        return proc.returncode if proc.returncode is not None else -1

    def _run_single(self, job: Job, proc: Process, stage: str, watcher: TranscriptWatcher | None, log_file: Any,
                    log_lock: threading.Lock) -> None:
        code = self._supervise(job, proc, stage, watcher, log_file, log_lock)
        status = "cancelled" if job.cancel_requested else "done" if code == 0 else "failed"
        with job._lock:
            job.exit_code, job.status, job.finished_at = code, status, _now()
        manifest = job.manifest_path
        present = manifest is not None and manifest.exists()
        with log_lock:
            log_file.write(f"# exit {code} · {status} · {job.finished_at}\n")
            log_file.close()
        job.events.append("done", {"exit_code": code, "status": status, "manifest": _s(manifest),
                                   "manifest_present": present, "tool_calls": job.tool_calls,
                                   "transcripts": job.transcripts})
        log.debug("job %s finished: %s exit %s", job.id, status, code)

    def _run_pipeline(self, job: Job, record: dict[str, Any], per_stage: dict[str, dict[str, Any]], log_file: Any,
                      log_lock: threading.Lock) -> None:
        pipeline = job.pipeline
        assert pipeline is not None
        outcome, exit_code = "done", 0
        for state in pipeline.stages:
            stage = state.stage
            if job.cancel_requested:
                state.status, state.reason = "cancelled", "cancelled before it started"
                outcome = "cancelled"
                continue
            if outcome != "done":
                state.status, state.reason = "skipped", "an earlier stage failed"
                self._stage_event(job, state, log_file, log_lock)
                continue
            if state.status == "skipped":  # decided at planning time: no Exomiser bundle
                self._stage_event(job, state, log_file, log_lock)
                continue
            if not pipeline.rerun:
                fresh, why = freshness(job.run_dir, stage, live=pipeline.live)
                if fresh:
                    state.status, state.reason, state.seconds = "skipped", why, 0.0
                    self._stage_event(job, state, log_file, log_lock)
                    continue
            args = dict(per_stage[stage])
            if stage == "medicine" and "candidate" not in args:
                cid = first_candidate(job.run_dir)
                if cid:
                    args["candidate"] = cid
            try:
                argv = build_argv(stage, args, job.run_dir, record=record, config=self.config, command=self.command)
            except NotConfigured as e:  # cannot happen after planning, except for rank whose bundle vanished
                state.status, state.reason = ("skipped" if stage == "rank" else "failed"), str(e)
                self._stage_event(job, state, log_file, log_lock)
                if stage != "rank":
                    outcome, exit_code = "failed", -1
                continue
            state.argv, state.status, state.started_at = argv, "running", _now()
            with job._lock:
                pipeline.current = stage
            stage_dir = job.run_dir / STAGE_DIRS[stage]
            stage_dir.mkdir(parents=True, exist_ok=True)
            watcher = TranscriptWatcher(stage_dir, lambda t, d, st=stage: self._emit_transcript(job, t, d, st)) \
                if stage in AGENT_STAGES else None
            if watcher is not None:
                watcher.snapshot()
            with log_lock:
                log_file.write(f"## {STAGE_DIRS[stage]} · $ {' '.join(argv)}\n")
                log_file.flush()
            self._stage_event(job, state, log_file, log_lock, "started")
            t0 = time.monotonic()
            try:
                proc = self.spawn(argv, self.repo, self._env())
            except OSError as e:
                state.status, state.reason, state.exit_code = "failed", f"could not start: {e}", -1
                state.seconds, state.finished_at = round(time.monotonic() - t0, 1), _now()
                self._stage_event(job, state, log_file, log_lock)
                outcome, exit_code = "failed", -1
                continue
            with job._lock:
                job.pid = proc.pid
            with self._lock:
                self._procs[job.id] = proc
            tail: deque[str] = deque(maxlen=200)
            code = self._supervise(job, proc, stage, watcher, log_file, log_lock, tail)
            state.exit_code, state.seconds, state.finished_at = code, round(time.monotonic() - t0, 1), _now()
            manifest_present = (stage_dir / "manifest.json").exists()
            if job.cancel_requested:
                state.status, state.reason = "cancelled", f"exit {code}"
                outcome, exit_code = "cancelled", code
            elif code == 0:
                state.status = "done"
                if not manifest_present:
                    state.reason = "exit 0 but no manifest was written"
            elif stage == "rank" and (code == OOM_EXIT or any(t in line for line in tail for t in _OOM_TEXT)):
                state.status, state.reason = "skipped", "Docker memory"
            else:
                state.status, state.reason = "failed", f"exit {code}"
                outcome, exit_code = "failed", code
            with job._lock:
                pipeline.current = None
            self._stage_event(job, state, log_file, log_lock)
        if job.cancel_requested:
            outcome = "cancelled"
        with job._lock:
            job.exit_code, job.status, job.finished_at, pipeline.current = exit_code, outcome, _now(), None
        manifest = job.manifest_path
        with log_lock:
            log_file.write(f"# exit {exit_code} · {outcome} · {job.finished_at}\n")
            log_file.close()
        job.events.append("done", {"exit_code": exit_code, "status": outcome, "manifest": _s(manifest),
                                   "manifest_present": manifest is not None and manifest.exists(),
                                   "tool_calls": job.tool_calls, "transcripts": job.transcripts,
                                   "stages": [s.to_dict() for s in pipeline.stages]})
        log.debug("pipeline %s finished: %s", job.id, outcome)

    def _stage_event(self, job: Job, state: StageState, log_file: Any, log_lock: threading.Lock,
                     status: str | None = None) -> None:
        shown = status or state.status
        with log_lock:
            log_file.write(f"# stage {state.stage} · {shown}{' · ' + state.reason if state.reason and shown != 'started' else ''}"
                           f"{' · ' + str(state.seconds) + ' s' if state.seconds is not None and shown != 'started' else ''}\n")
            log_file.flush()
        job.events.append("stage", {"stage": state.stage, "dir": state.dir, "status": shown, "seconds": state.seconds,
                                    "reason": state.reason if shown != "started" else None, "exit_code": state.exit_code,
                                    "argv": state.argv if shown == "started" else None})

    # -- threads

    def _read(self, job: Job, pipe: Any, stream: str, stage: str, log_file: Any, log_lock: threading.Lock,
              tail: deque[str] | None = None) -> None:
        if pipe is None:
            return
        try:
            for raw in iter(pipe.readline, b""):
                text = raw.decode("utf-8", "replace").rstrip("\r\n")
                if tail is not None:
                    tail.append(text)
                job.events.append("line", {"stream": stream, "text": text, "stage": stage})
                with log_lock:
                    log_file.write(("! " if stream == "stderr" else "  ") + text + "\n")
                    log_file.flush()
        except (OSError, ValueError) as e:
            log.debug("job %s %s reader stopped: %s", job.id, stream, e)
        finally:
            try:
                pipe.close()
            except OSError:
                pass

    def _emit_transcript(self, job: Job, type: str, data: dict[str, Any], stage: str | None = None) -> None:
        with job._lock:
            if type == "tool_call":
                job.tool_calls += 1
            elif type == "transcript":
                job.transcripts += 1
        job.events.append(type, {**data, "stage": stage or job.stage})

    def _env(self) -> dict[str, str]:
        env = dict(os.environ if self.env is None else self.env)
        env.pop("VIRTUAL_ENV", None)
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env


# --------------------------------------------------------------------- helpers

def repo_root() -> Path:
    """The checkout ``uv run`` executes in: the directory holding ``pyproject.toml``
    (``src/engine/ui/`` is three levels below it)."""
    return Path(__file__).resolve().parents[3]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _s(p: Path | None) -> str | None:
    return None if p is None else str(p)

