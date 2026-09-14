"""The UI — ``engine.ui`` (CONTRACTS.md, "UI").

Network-free and subprocess-free by default: the server runs on a free loopback port
in a thread, over a run directory assembled from the public demo fixtures exactly as
``tests/test_report.py`` does (stages 2–6 written by the real stage code, the agents
scripted), and jobs are started through a fake spawner that pretends to be the
process — writes lines to real pipes, exits with a chosen code, hangs until
terminated when asked — so the runner's threads, the event ring, the log file, the
transcript watcher and the SSE stream are the real ones over a fake ``engine``.

One test starts a real subprocess: ``engine filter`` through the server against a
copy of the public run, its stdout read back from the event stream with urllib. It is
skipped without ``uv`` on the PATH. Nothing here derives from a person.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import engine.cli
from engine.agents import client as ac
from engine.agents import providers
from engine.filter.run import run_filter
from engine.medicine.run import run_medicine
from engine.rank.exomiser import DEFAULT_CONFIG, ExomiserConfig
from engine.rank.run import run_rank
from engine.reason.run import run_reason
from engine.report import views
from engine.report.render import render_run
from engine.ui import jobs as jobmod
from engine.ui import setup as setupmod
from engine.ui.cli import commands, ui, ui_command
from engine.ui.jobs import JobRunner, RunnerConfig, TranscriptWatcher, check_args, build_argv
from engine.ui.server import HOST, STATIC_DIR, App, make_server, serve_in_thread
from tests.test_medicine import retrievers, stub_http
from tests.test_public_case import CFTR_PAIR_IDS, FIX, make_run_from_recorded_stage2
from tests.test_rank import FakeDocker, make_data_dir
from tests.test_report import scripted_chain, scripted_report

CASE = FIX / "case.yaml"
PUBLIC_RUN = Path(os.environ.get("PUBLIC_RUN_DIR") or Path(os.environ.get("TMPDIR", "/tmp")) / "engine-public-case" / "run")


# ------------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Stages 2–6 of the public demo, every one written by the real stage code with
    scripted agents (the same recipe as ``tests/test_report.py``)."""
    tmp = tmp_path_factory.mktemp("ui")
    run = make_run_from_recorded_stage2(tmp)
    run_filter(run)
    cfg = ExomiserConfig.load(DEFAULT_CONFIG)
    run_rank(run, CASE, make_data_dir(tmp, cfg), run=FakeDocker(cfg.image_digest))
    reader = ac.FakeClient(scripted_chain(), turns=[ac.FakeTurn([("get_record", {"record_id": rid}) for rid in CFTR_PAIR_IDS], text="reading")])
    run_reason(run, 1, reader, "fake-model", "low", False, http=stub_http(), case_path=CASE)
    writer = ac.FakeClient(scripted_report(), turns=[ac.FakeTurn([("drugs_for_gene", {"gene": "CFTR"})], text="drugs"),
                                                     ac.FakeTurn([("search_trials", {"condition": "cystic fibrosis", "intervention": "ivacaftor",
                                                                                     "term": None, "max_results": 3})], text="trials")])
    run_medicine(run, None, writer, "fake-model", "low", False, retrievers=retrievers(stub_http()), case_path=CASE)
    return run


class FakeProcess:
    """A process that is not one: real pipes fed from a thread, a chosen exit code,
    and — with ``hang`` — no exit until terminated. ``on_start`` runs before the first
    line (a stage writing a transcript, say)."""

    def __init__(self, argv: list[str], *, stdout: list[str] = (), stderr: list[str] = (), exit_code: int = 0,
                 hang: bool = False, on_start: Any = None, pid: int = 4242):
        self.argv, self.pid, self.returncode = argv, pid, None
        r_out, self._w_out = os.pipe()
        r_err, self._w_err = os.pipe()
        self.stdout, self.stderr = os.fdopen(r_out, "rb"), os.fdopen(r_err, "rb")
        self._lines = (list(stdout), list(stderr))
        self._exit_code, self._hang, self._on_start = exit_code, hang, on_start
        self._done, self._term, self.terminated = threading.Event(), threading.Event(), False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        if self._on_start:
            self._on_start()
        for line in self._lines[0]:
            os.write(self._w_out, (line + "\n").encode())
            time.sleep(0.005)
        for line in self._lines[1]:
            os.write(self._w_err, (line + "\n").encode())
        if self._hang:
            self._term.wait(timeout=30)
        os.close(self._w_out)
        os.close(self._w_err)
        self.returncode = -15 if self.terminated else self._exit_code
        self._done.set()

    def wait(self, timeout: float | None = None) -> int:
        if not self._done.wait(timeout):
            raise subprocess.TimeoutExpired(self.argv, timeout or 0)
        return self.returncode  # type: ignore[return-value]

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self._term.set()

    kill = terminate


class FakeSpawn:
    """Records every launch and hands out the :class:`FakeProcess` configured for it."""

    def __init__(self, **kw: Any):
        self.kw, self.calls, self.procs = kw, [], []

    def __call__(self, argv: list[str], cwd: Path, env: dict[str, str]) -> FakeProcess:
        self.calls.append({"argv": list(argv), "cwd": Path(cwd), "env": dict(env)})
        proc = FakeProcess(argv, **self.kw)
        self.procs.append(proc)
        return proc


class Client:
    """urllib over the test server: JSON in, JSON out, errors as (status, body)."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def request(self, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None) -> tuple[int, str, bytes]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers or {})
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def get(self, path: str) -> tuple[int, Any]:
        status, ct, body = self.request("GET", path)
        return status, json.loads(body) if "json" in ct else body

    def post(self, path: str, body: Any = None) -> tuple[int, Any]:
        status, ct, out = self.request("POST", path, body if body is not None else {})
        return status, json.loads(out) if "json" in ct else out

    def sse(self, path: str, *, headers: dict[str, str] | None = None, until: str = "done", timeout: float = 60) -> list[dict[str, Any]]:
        """Read the event stream frame by frame until an event of type ``until``."""
        req = urllib.request.Request(self.url + path, headers=headers or {})
        events: list[dict[str, Any]] = []
        with urllib.request.urlopen(req, timeout=timeout) as r:
            assert r.headers.get("Content-Type", "").startswith("text/event-stream")
            frame: dict[str, Any] = {}
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                raw = r.readline()
                if not raw:
                    break
                line = raw.decode("utf-8").rstrip("\r\n")
                if line == "":
                    if frame.get("event"):
                        frame["data"] = json.loads(frame.get("data", "{}"))
                        events.append(frame)
                        if frame["event"] == until:
                            return events
                    frame = {}
                elif line.startswith(":"):
                    continue
                else:
                    key, _, value = line.partition(":")
                    frame[key] = value[1:] if value.startswith(" ") else value
        return events


@pytest.fixture
def work(demo_run: Path, tmp_path: Path) -> Path:
    """A work directory holding a private copy of the demo run as ``public``."""
    work = tmp_path / "work"
    work.mkdir()
    shutil.copytree(demo_run, work / "public")
    return work


@pytest.fixture
def served(work: Path):
    """The server on a free loopback port over a stub runner; yields (client, app, spawn)."""
    spawn = FakeSpawn(stdout=["filter · run · config=configs/filter.yaml", "  rows in: 12 · kept: 3", "  manifest: m"], stderr=["! a warning"])
    runner = JobRunner(work, spawn=spawn, command=("engine",), poll_interval=0.05, config=RunnerConfig(cache=Path("/cache")))
    app = App(work, runner=runner)
    server = make_server(work, 0, app=app)
    serve_in_thread(server)
    try:
        yield Client(server.url), app, spawn
    finally:
        server.shutdown()
        server.server_close()


def start_and_finish(client: Client, run: str, stage: str, args: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    status, job = client.post(f"/api/runs/{run}/jobs", {"stage": stage, "args": args or {}})
    assert status == 201, job
    events = client.sse(f"/api/jobs/{job['id']}/events")
    return job, events


# ------------------------------------------------------------------------ whitelist

def test_whitelist_matches_the_stage_clis_and_the_shared_choices():
    assert jobmod.PROVIDERS == providers.PROVIDERS
    assert jobmod.EFFORTS == ac.EFFORTS
    assert list(jobmod.STAGE_DIRS) == [name for _, name in views.STAGES] and list(jobmod.STAGE_DIRS.values()) == [d for d, _ in views.STAGES]
    for stage, args in jobmod.WHITELIST.items():
        cmd = engine.cli.main.commands[stage]
        options = {o for p in cmd.params for o in getattr(p, "opts", [])}
        for a in args:
            assert a.option in options, f"{stage}: {a.option} is not an option of `engine {stage}`"
    assert [a.name for a in jobmod.WHITELIST["reason"]] == ["provider", "model", "effort", "top", "dry_run"]
    assert [a.name for a in jobmod.WHITELIST["medicine"]] == ["provider", "model", "effort", "dry_run", "candidate"]
    assert [a.name for a in jobmod.WHITELIST["rank"]] == ["heap"]
    assert [a.name for a in jobmod.WHITELIST["retrieve"]] == ["funnel", "sources", "vep_workers"]
    assert [a.name for a in jobmod.WHITELIST["ingest"]] == ["regions"]
    assert jobmod.WHITELIST["filter"] == ()


def test_check_args_normalises_and_refuses():
    assert check_args("reason", {"top": "2", "dry_run": "false", "effort": "low", "provider": "fake", "model": "fake-model"}) == \
        {"top": 2, "dry_run": False, "effort": "low", "provider": "fake", "model": "fake-model"}
    assert check_args("filter", None) == {} and check_args("retrieve", {"sources": " vep, gnomad ", "vep_workers": 3}) == {"sources": "vep,gnomad", "vep_workers": 3}
    assert check_args("rank", {"heap": "6g"}) == {"heap": "6g"} and check_args("medicine", {"candidate": "CFTR:comphet"}) == {"candidate": "CFTR:comphet"}
    for stage, args, why in [
        ("reason", {"case": "/etc/passwd"}, "does not take case"),
        ("reason", {"candidate": "X"}, "does not take candidate"),
        ("medicine", {"top": 1}, "does not take top"),
        ("filter", {"config": "x"}, "does not take config"),
        ("reason", {"provider": "openai"}, "must be one of"),
        ("reason", {"effort": "extreme"}, "must be one of"),
        ("reason", {"top": "many"}, "must be an integer"),
        ("reason", {"top": 0}, "between 1 and"),
        ("reason", {"model": "--offline"}, "character"),
        ("reason", {"model": "a b"}, "character"),
        ("reason", {"dry_run": "maybe"}, "true or false"),
        ("retrieve", {"sources": "vep,omim"}, "subset"),
        ("retrieve", {"vep_workers": 99}, "between 1 and 12"),
        ("retrieve", {"funnel": "--clinvar-vcf"}, "file path"),
        ("rank", {"heap": "lots"}, "heap must"),
        ("medicine", {"candidate": "../x"}, "candidate must"),
        ("bench", {}, "unknown stage"),
        ("reason", ["top"], "must be an object"),
    ]:
        with pytest.raises(jobmod.ArgumentError, match=why):
            check_args(stage, args)  # type: ignore[arg-type]


def test_build_argv_adds_only_fixed_inputs(tmp_path: Path):
    run = tmp_path / "r"
    rec = {"case_path": "/cases/c.yaml", "regions": "/cases/r.bed"}
    cfg = RunnerConfig(cache=Path("/c"), clinvar_vcf=Path("/ref/clinvar.vcf.gz"), exomiser_data=Path("/ref/exomiser"))
    assert build_argv("ingest", {}, run, record=rec, config=cfg) == ["uv", "run", "engine", "ingest", "--case", "/cases/c.yaml", "--out", str(run), "--regions", "/cases/r.bed"]
    assert build_argv("ingest", {"regions": "/other.bed"}, run, record=rec, config=cfg)[-2:] == ["--regions", "/other.bed"]
    assert build_argv("retrieve", {"sources": "vep,clinvar", "vep_workers": 2}, run, record=rec, config=cfg) == \
        ["uv", "run", "engine", "retrieve", "--run", str(run), "--clinvar-vcf", "/ref/clinvar.vcf.gz", "--cache", "/c", "--sources", "vep,clinvar", "--vep-workers", "2"]
    assert build_argv("retrieve", {"sources": "vep"}, run, record=rec, config=RunnerConfig()) == ["uv", "run", "engine", "retrieve", "--run", str(run), "--sources", "vep"]
    assert build_argv("filter", {}, run, record=rec, config=cfg) == ["uv", "run", "engine", "filter", "--run", str(run)]
    assert build_argv("rank", {"heap": "4g"}, run, record=rec, config=cfg) == \
        ["uv", "run", "engine", "rank", "--run", str(run), "--case", "/cases/c.yaml", "--exomiser-data", "/ref/exomiser", "--heap", "4g"]
    assert build_argv("reason", {"provider": "fake", "model": "m", "effort": "low", "top": 1, "dry_run": True}, run, record=rec, config=cfg) == \
        ["uv", "run", "engine", "reason", "--run", str(run), "--case", "/cases/c.yaml", "--cache", "/c", "--provider", "fake", "--model", "m", "--effort", "low", "--top", "1", "--dry-run"]
    live = build_argv("medicine", {"dry_run": False, "candidate": "CFTR:comphet"}, run, record={"case_path": None}, config=RunnerConfig())
    assert live == ["uv", "run", "engine", "medicine", "--run", str(run), "--candidate", "CFTR:comphet"] and "--dry-run" not in live
    with pytest.raises(jobmod.NotConfigured, match="case file"):
        build_argv("ingest", {}, run, record={"case_path": None})
    with pytest.raises(jobmod.NotConfigured, match="ClinVar"):
        build_argv("retrieve", {}, run, record=rec, config=RunnerConfig())
    with pytest.raises(jobmod.NotConfigured, match="Exomiser"):
        build_argv("rank", {}, run, record=rec, config=RunnerConfig())


def test_run_record_from_the_page_or_the_manifests(demo_run: Path, tmp_path: Path):
    rec = jobmod.read_run_record(demo_run)
    assert rec["case_path"] == str(CASE) and rec["source"] == "04_rank/manifest.json" and rec["vcf_path"] is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert jobmod.read_run_record(empty)["case_path"] is None
    jobmod.write_run_record(empty, "/x/case.yaml", "/x/r.bed")
    rec = jobmod.read_run_record(empty)
    assert rec["case_path"] == "/x/case.yaml" and rec["regions"] == "/x/r.bed" and rec["source"] == "ui-run.json" and rec["created_at"]


# ------------------------------------------------------------------------ read API

def test_binds_loopback_only(served):
    client, app, _ = served
    host, port = urllib.parse.urlsplit(client.url).hostname, urllib.parse.urlsplit(client.url).port
    assert host == HOST == "127.0.0.1" and port and port > 0
    status, doc = client.get("/api/runs")
    assert status == 200 and doc["work"] == str(app.work)


def test_server_socket_is_bound_to_127_0_0_1(work: Path):
    server = make_server(work, 0)
    try:
        assert server.socket.getsockname()[0] == "127.0.0.1" and server.server_address[0] == "127.0.0.1"
        assert server.url.startswith("http://127.0.0.1:")
    finally:
        server.server_close()


def test_run_listing(served, work: Path):
    client, _, _ = served
    (work / ".hidden").mkdir()
    (work / "not a run").mkdir()
    (work / "stray.txt").write_text("x")
    status, doc = client.get("/api/runs")
    assert status == 200 and [r["name"] for r in doc["runs"]] == ["public"]
    run = doc["runs"][0]
    assert run["stages_present"] == ["02_retrieve", "03_filter", "04_rank", "05_reason", "06_medicine"]
    by_dir = {s["dir"]: s for s in run["stages"]}
    assert by_dir["01_ingest"]["present"] is False and by_dir["03_filter"]["counts"]["candidates"] == 2
    assert by_dir["03_filter"]["finished_at"] and by_dir["05_reason"]["dry_run"] is False
    assert run["case_path"] == str(CASE) and run["job"] is None


def test_every_view_endpoint_is_the_report_view(served, work: Path):
    client, _, _ = served
    run = work / "public"
    same = lambda doc: json.loads(json.dumps(doc, default=str))  # noqa: E731 - the JSON round trip the API does
    status, summary = client.get("/api/runs/public/summary")
    assert status == 200 and {k: v for k, v in summary.items() if k != "record"} == same(views.run_summary(run))
    assert summary["record"]["case_path"] == str(CASE)
    assert client.get("/api/runs/public/candidates") == (200, same(views.candidates_view(run)))
    assert client.get("/api/runs/public/ranking") == (200, same(views.ranking_view(run)))
    assert client.get("/api/runs/public/ranking?top=1") == (200, same(views.ranking_view(run, top_n=1)))
    assert client.get("/api/runs/public/chain") == (200, same(views.chain_view(run)))
    assert client.get("/api/runs/public/chain/CFTR:comphet") == (200, same(views.chain_view(run, "CFTR:comphet")))
    assert client.get("/api/runs/public/chain/CFTR%3Acomphet")[1]["chains"][0]["candidate_id"] == "CFTR:comphet"
    assert client.get("/api/runs/public/chain/NOPE:x")[1]["chains"] == []
    assert client.get("/api/runs/public/medicine") == (200, same(views.medicine_view(run)))
    assert client.get("/api/runs/public/provenance") == (200, same(views.provenance_view(run)))
    status, doc = client.get("/api/runs/public/ranking?top=zero")
    assert status == 400 and "top" in doc["error"]
    candidates = client.get("/api/runs/public/candidates")[1]
    assert [c["candidate_id"] for c in candidates["candidates"]] == ["CFTR:comphet", "TP53:het_single"]
    assert all(e["url"].startswith("https://") for v in candidates["candidates"][0]["variants"] for e in v["evidence"])


def test_report_html_is_the_rendered_report(served, work: Path):
    client, _, _ = served
    status, ct, body = client.request("GET", "/api/runs/public/report.html")
    assert status == 200 and ct == "text/html; charset=utf-8"
    assert body.decode("utf-8") == render_run(work / "public")
    assert client.request("GET", "/api/runs/public/report.html?top=1")[2].decode() == render_run(work / "public", top_n=1)


def test_static_files_served_from_the_package(served):
    client, _, _ = served
    status, ct, index = client.request("GET", "/")
    assert status == 200 and ct == "text/html; charset=utf-8" and index == (STATIC_DIR / "index.html").read_bytes()
    assert client.request("GET", "/static/index.html")[2] == index
    for name, ct_expected in [("app.js", "text/javascript; charset=utf-8"), ("app.css", "text/css; charset=utf-8")]:
        status, ct, body = client.request("GET", f"/static/{name}")
        assert status == 200 and ct == ct_expected and body == (STATIC_DIR / name).read_bytes()
    text = index.decode()
    assert '/static/app.js' in text and '/static/app.css' in text and "fonts.googleapis.com" in text
    assert "<script" in text and "EventSource" in (STATIC_DIR / "app.js").read_text()
    css = (STATIC_DIR / "app.css").read_text()
    assert ':root:not([data-theme="light"])' in css and ':root[data-theme="dark"]' in css and "tabular-nums" in css
    assert '"Spectral"' in css and '"IBM Plex Sans"' in css and '"IBM Plex Mono"' in css and "#2A5D7C" in css
    assert "innerHTML" not in (STATIC_DIR / "app.js").read_text().split("*/", 1)[1]
    for bad in ["/static/server.py", "/static/__init__.py", "/static/../server.py", "/static/..%2Fserver.py", "/static/nope.js"]:
        status, _, body = client.request("GET", bad)
        assert status == 404, (bad, body)
    assert client.request("GET", "/nope")[0] == 404 and client.request("POST", "/api/providers", {})[0] == 405


def test_providers_reports_presence_never_a_value(served, monkeypatch: pytest.MonkeyPatch):
    client, _, _ = served
    secret = "sk-ant-test-value-that-must-never-leave-the-process"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    status, ct, body = client.request("GET", "/api/providers")
    doc = json.loads(body)
    assert status == 200 and secret not in body.decode() and "sk-ant" not in body.decode()
    assert doc["providers"]["anthropic"]["key_present"] is True and doc["providers"]["anthropic"]["default_model"] == ac.DEFAULT_MODEL
    assert doc["providers"]["anthropic"]["models"][0]["id"] == ac.DEFAULT_MODEL
    assert doc["providers"]["openrouter"]["key_present"] is False and doc["providers"]["fake"]["default_model"] == "fake-model"
    assert doc["default_provider"] == "anthropic" and doc["any_key"] is True
    assert doc["efforts"] == list(ac.EFFORTS) and doc["default_effort"] == ac.DEFAULT_EFFORT
    assert set(doc["disclosure"]) == {"anthropic", "openrouter", "fake"} and secret not in json.dumps(doc)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    doc = client.get("/api/providers")[1]
    assert doc["providers"]["anthropic"]["key_present"] is False and doc["any_key"] is False


def test_stages_endpoint_describes_the_whitelist(served):
    client, _, _ = served
    status, doc = client.get("/api/stages")
    assert status == 200 and [s["stage"] for s in doc["stages"]] == list(jobmod.STAGE_DIRS)
    reason = next(s for s in doc["stages"] if s["stage"] == "reason")
    assert reason["agent"] is True and reason["dir"] == "05_reason"
    assert [a["name"] for a in reason["args"]] == ["provider", "model", "effort", "top", "dry_run"]
    assert next(a for a in reason["args"] if a["name"] == "dry_run")["default"] is True
    assert doc["command"] == ["engine"] and doc["config"]["cache"] == "/cache"


# ------------------------------------------------------------------------ traversal

def test_run_name_traversal_is_rejected(served, work: Path, tmp_path: Path):
    client, _, spawn = served
    outside = tmp_path / "outside"
    shutil.copytree(work / "public", outside)
    (work / "link").symlink_to(outside)
    for name, status in [("..", 400), ("%2e%2e", 400), (".hidden", 400), ("-x", 400), ("a%2Fb", 404), ("public%00", 400), ("link", 400), ("nope", 404)]:
        for path in [f"/api/runs/{name}/summary", f"/api/runs/{name}/candidates", f"/api/runs/{name}/chain", f"/api/runs/{name}/report.html"]:
            got, _, body = client.request("GET", path)
            assert got == status, (path, body)
    assert client.request("GET", "/api/runs/../summary")[0] == 400
    assert client.post("/api/runs/../jobs", {"stage": "filter"})[0] == 400
    assert client.post("/api/runs/%2e%2e/jobs", {"stage": "filter"})[0] == 400
    assert client.post("/api/runs/link/jobs", {"stage": "filter"})[0] == 400
    assert [r["name"] for r in client.get("/api/runs")[1]["runs"]] == ["public"]  # the symlink out is not listed
    for name in ["../x", "a/b", "..", ".x", "x y"]:
        status, doc = client.post("/api/runs", {"name": name, "case_path": str(CASE)})
        assert status == 400, (name, doc)
    assert spawn.calls == []


# ------------------------------------------------------------------------ jobs

def test_job_start_streams_events_then_done(served, work: Path):
    client, app, spawn = served
    job, events = start_and_finish(client, "public", "filter")
    assert job["stage"] == "filter" and job["run"] == "public" and job["status"] == "running" and job["pid"] == 4242
    assert job["argv"] == ["engine", "filter", "--run", str(work / "public")] and job["live"] is False and job["agent"] is False
    assert spawn.calls[0]["argv"] == job["argv"] and spawn.calls[0]["cwd"] == app.runner.repo
    assert "VIRTUAL_ENV" not in spawn.calls[0]["env"] and spawn.calls[0]["env"]["PYTHONUNBUFFERED"] == "1"
    types = [e["event"] for e in events]
    assert types[0] == "status" and types[-1] == "done" and types.count("line") == 4
    assert events[0]["data"]["status"] == "running" and events[0]["data"]["argv"] == job["argv"]
    lines = [(e["data"]["stream"], e["data"]["text"]) for e in events if e["event"] == "line"]
    assert lines[:3] == [("stdout", "filter · run · config=configs/filter.yaml"), ("stdout", "  rows in: 12 · kept: 3"), ("stdout", "  manifest: m")]
    assert ("stderr", "! a warning") in lines
    assert [int(e["id"]) for e in events] == sorted(int(e["id"]) for e in events) and int(events[0]["id"]) == 1
    done = events[-1]["data"]
    assert done["exit_code"] == 0 and done["status"] == "done" and done["manifest"] == str(work / "public" / "03_filter" / "manifest.json")
    assert done["manifest_present"] is True
    log_path = Path(job["log"])
    assert log_path == work / "public" / "03_filter" / f"ui-job-{job['id']}.log"
    text = log_path.read_text()
    assert "# $ engine filter --run" in text and "  filter · run · config=configs/filter.yaml\n" in text and "! ! a warning\n" in text and "# exit 0 · done" in text
    status, doc = client.get(f"/api/jobs/{job['id']}")
    assert status == 200 and doc["status"] == "done" and doc["exit_code"] == 0 and doc["finished_at"] and doc["events"] == len(events)
    assert client.get("/api/jobs")[1]["jobs"][0]["id"] == job["id"] and client.get("/api/jobs?run=other")[1]["jobs"] == []
    assert client.request("GET", f"/api/jobs/{job['id']}/log")[2].decode() == text
    assert client.get("/api/jobs/nope")[0] == 404 and client.get("/api/jobs/nope/events")[0] == 404
    # a late reader gets the whole ring; Last-Event-ID resumes after what it saw
    again = client.sse(f"/api/jobs/{job['id']}/events")
    assert [e["event"] for e in again] == types
    resumed = client.sse(f"/api/jobs/{job['id']}/events", headers={"Last-Event-ID": events[2]["id"]})
    assert [e["id"] for e in resumed] == [e["id"] for e in events[3:]]
    assert client.sse(f"/api/jobs/{job['id']}/events?after={events[-1]['id']}") == []


def test_job_failure_exit_code(work: Path):
    spawn = FakeSpawn(stdout=["reason · x"], stderr=["  FAILED: no model key"], exit_code=1)
    runner = JobRunner(work, spawn=spawn, command=("engine",), poll_interval=0.05)
    app = App(work, runner=runner)
    server = make_server(work, 0, app=app)
    serve_in_thread(server)
    try:
        client = Client(server.url)
        job, events = start_and_finish(client, "public", "reason", {"dry_run": True})
        assert events[-1]["data"] == {**events[-1]["data"], "exit_code": 1, "status": "failed"}
        assert client.get(f"/api/jobs/{job['id']}")[1]["status"] == "failed"
        assert "--dry-run" in job["argv"] and job["live"] is False
    finally:
        server.shutdown()
        server.server_close()


def test_whitelist_rejection_through_the_api(served, work: Path):
    client, _, spawn = served
    for stage, args in [("reason", {"case": "/etc/passwd"}), ("reason", {"run": "/elsewhere"}), ("filter", {"config": "/x"}),
                        ("reason", {"provider": "openai"}), ("reason", {"top": "lots"}), ("medicine", {"top": 1}),
                        ("reason", {"candidate": "CFTR:comphet"}), ("retrieve", {"clinvar_vcf": "/x"}), ("rank", {"image": "evil"})]:
        status, doc = client.post(f"/api/runs/public/jobs", {"stage": stage, "args": args})
        assert status == 400 and "error" in doc, (stage, args, doc)
        assert doc["stage"] == stage and set(doc["allowed"]) == {a.name for a in jobmod.WHITELIST[stage]}
    assert client.post("/api/runs/public/jobs", {"stage": "bench"})[0] == 400
    assert client.post("/api/runs/public/jobs", {"args": {}})[0] == 400
    assert client.post("/api/runs/public/jobs", {"stage": "filter", "args": ["x"]})[0] == 400
    assert client.post("/api/runs/nope/jobs", {"stage": "filter"})[0] == 404
    status, _, body = client.request("POST", "/api/runs/public/jobs", headers={"Content-Type": "application/json"})
    assert status == 400  # an empty body names no stage
    req = urllib.request.Request(client.url + "/api/runs/public/jobs", data=b"not json", method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400
    assert spawn.calls == [], "a refused request must never reach the spawner"
    assert not list((work / "public").glob("*/ui-job-*.log"))


def test_agent_job_argv_live_and_dry(served, work: Path):
    client, _, spawn = served
    job, events = start_and_finish(client, "public", "reason", {"provider": "fake", "model": "fake-model", "effort": "low", "top": 1, "dry_run": False})
    assert job["argv"] == ["engine", "reason", "--run", str(work / "public"), "--case", str(CASE), "--cache", "/cache",
                           "--provider", "fake", "--model", "fake-model", "--effort", "low", "--top", "1"]
    assert job["live"] is True and job["agent"] is True and events[0]["data"]["live"] is True
    job, _ = start_and_finish(client, "public", "medicine", {"dry_run": True, "candidate": "CFTR:comphet"})
    assert job["argv"] == ["engine", "medicine", "--run", str(work / "public"), "--case", str(CASE), "--cache", "/cache", "--dry-run", "--candidate", "CFTR:comphet"]
    assert job["live"] is False
    job, _ = start_and_finish(client, "public", "medicine", {})
    assert job["argv"][-2:] == ["--case", str(CASE)] or "--cache" in job["argv"]
    assert "--dry-run" not in job["argv"] and job["live"] is True  # dry_run omitted means the CLI's default: live
    for call in spawn.calls:
        assert "ANTHROPIC_API_KEY" not in json.dumps(call["argv"]) and all(not a.startswith("sk-") for a in call["argv"])


def test_transcript_watcher_emits_tool_calls_as_files_appear(served, work: Path, demo_run: Path):
    client, app, spawn = served
    run = work / "public"
    transcripts = run / "05_reason" / "transcripts"
    recorded = json.loads((demo_run / "05_reason" / "transcripts" / "CFTR:comphet.json").read_text())
    n_calls = sum(len(t["tool_calls"]) for t in recorded["transcript"])
    assert n_calls == len(CFTR_PAIR_IDS) > 0

    def stage_writes_a_transcript() -> None:
        # what a real stage 5 does: replaces transcripts/, then writes the file when the candidate is done
        shutil.rmtree(transcripts, ignore_errors=True)
        transcripts.mkdir(parents=True)
        (transcripts / "CFTR:comphet.json").write_text("{not yet")  # half-written: the watcher must wait
        time.sleep(0.15)
        (transcripts / "CFTR:comphet.json").write_text(json.dumps(recorded))
        (transcripts / "TP53:het_single.failed.json").write_text(json.dumps({
            "candidate_id": "TP53:het_single", "error": "model overloaded", "request_id": "req_1",
            "transcript": [{"n": 1, "stop_reason": "tool_use", "text": "", "tool_calls": [
                {"id": "t1", "name": "search_literature", "input": {"query": "TP53"}, "result": "x" * 1000, "is_error": True}]}],
            "final_text": "", "tools": {}}))
        time.sleep(0.2)

    spawn.kw = dict(stdout=["reason · live"], on_start=stage_writes_a_transcript)
    job, events = start_and_finish(client, "public", "reason", {"provider": "fake", "dry_run": False, "top": 2})
    calls = [e["data"] for e in events if e["event"] == "tool_call"]
    assert [c["name"] for c in calls] == ["get_record"] * n_calls + ["search_literature"]
    assert [c["input"]["record_id"] for c in calls[:n_calls]] == CFTR_PAIR_IDS
    assert calls[0]["candidate_id"] == "CFTR:comphet" and calls[0]["file"] == "CFTR:comphet.json" and calls[0]["turn"] == 1 and calls[0]["n"] == 1
    assert calls[0]["result_chars"] > 0 and calls[0]["result_preview"] and calls[0]["is_error"] is False
    assert calls[-1] == {**calls[-1], "candidate_id": "TP53:het_single", "is_error": True, "result_chars": 1000, "result_preview": "x" * 400}
    ts = [e["data"] for e in events if e["event"] == "transcript"]
    assert [(t["file"], t["failed"], t["turns"], t["tool_calls"]) for t in ts] == [("CFTR:comphet.json", False, len(recorded["transcript"]), n_calls), ("TP53:het_single.failed.json", True, 1, 1)]
    assert ts[1]["error"] == "model overloaded" and ts[1]["request_id"] == "req_1" and ts[0]["usage"]["api_calls"] == recorded["usage"]["api_calls"]
    done = events[-1]["data"]
    assert done["tool_calls"] == n_calls + 1 and done["transcripts"] == 2
    assert client.get(f"/api/jobs/{job['id']}")[1]["tool_calls"] == n_calls + 1
    # the ring holds the same events a late reader sees
    assert [e["event"] for e in client.sse(f"/api/jobs/{job['id']}/events")] == [e["event"] for e in events]


def test_transcript_watcher_ignores_what_was_there_before(tmp_path: Path, demo_run: Path):
    stage = tmp_path / "05_reason"
    shutil.copytree(demo_run / "05_reason" / "transcripts", stage / "transcripts")
    seen: list[tuple[str, dict[str, Any]]] = []
    w = TranscriptWatcher(stage, lambda t, d: seen.append((t, d)))
    w.snapshot()
    w.poll()
    assert seen == []
    doc = json.loads((stage / "transcripts" / "CFTR:comphet.json").read_text())
    (stage / "transcripts" / "NEW:one.json").write_text(json.dumps(doc))
    w.poll()
    assert [t for t, _ in seen] == ["tool_call"] * len(CFTR_PAIR_IDS) + ["transcript"]
    assert seen[-1][1]["candidate_id"] == "NEW:one"  # a stage transcript carries no candidate_id: the file name is the candidate
    w.poll()
    assert len(seen) == len(CFTR_PAIR_IDS) + 1  # unchanged files are not replayed
    # a rewritten file with one more call emits only the new call
    doc["transcript"].append({"n": 9, "stop_reason": "end_turn", "text": "", "tool_calls": [{"id": "z", "name": "get_paper", "input": {"pmid": "pmid:1"}, "result": "", "is_error": False}]})
    time.sleep(0.01)
    (stage / "transcripts" / "NEW:one.json").write_text(json.dumps(doc))
    w.poll()
    assert [t for t, _ in seen[len(CFTR_PAIR_IDS) + 1:]] == ["tool_call", "transcript"] and seen[-2][1]["name"] == "get_paper"
    assert w.files == 2 and w.tool_calls == len(CFTR_PAIR_IDS) + 1


def test_cancel_terminates_the_process(served, work: Path):
    client, _, spawn = served
    spawn.kw = dict(stdout=["retrieve · working"], hang=True)
    status, job = client.post("/api/runs/public/jobs", {"stage": "retrieve", "args": {"sources": "vep"}})
    assert status == 201
    assert client.post("/api/runs/public/jobs", {"stage": "filter"})[0] == 409, "one job per run at a time"
    deadline = time.monotonic() + 5
    while client.get(f"/api/jobs/{job['id']}")[1]["events"] < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    status, doc = client.post(f"/api/jobs/{job['id']}/cancel")
    assert status == 200 and doc["cancel_requested"] is True and doc["id"] == job["id"]
    events = client.sse(f"/api/jobs/{job['id']}/events")
    assert spawn.procs[0].terminated
    assert [e["event"] for e in events][-2:] == ["status", "done"] and events[-2]["data"]["status"] == "cancelling"
    assert events[-1]["data"]["status"] == "cancelled" and events[-1]["data"]["exit_code"] == -15
    doc = client.get(f"/api/jobs/{job['id']}")[1]
    assert doc["status"] == "cancelled" and doc["exit_code"] == -15
    assert client.post(f"/api/jobs/{job['id']}/cancel")[0] == 409 and client.post("/api/jobs/nope/cancel")[0] == 404
    assert "# exit -15 · cancelled" in Path(job["log"]).read_text()
    status, job2 = client.post("/api/runs/public/jobs", {"stage": "filter"})  # the run is free again
    assert status == 201
    client.post(f"/api/jobs/{job2['id']}/cancel")
    client.sse(f"/api/jobs/{job2['id']}/events")


def test_run_listing_shows_the_running_job(served):
    client, _, spawn = served
    spawn.kw = dict(hang=True)
    status, job = client.post("/api/runs/public/jobs", {"stage": "filter"})
    assert status == 201
    listed = client.get("/api/runs")[1]["runs"][0]["job"]
    assert listed and listed["id"] == job["id"] and listed["status"] == "running"
    client.post(f"/api/jobs/{job['id']}/cancel")
    client.sse(f"/api/jobs/{job['id']}/events")
    assert client.get("/api/runs")[1]["runs"][0]["job"] is None


def test_spawn_failure_is_a_failed_job(work: Path):
    def broken(argv: list[str], cwd: Path, env: dict[str, str]) -> Any:
        raise FileNotFoundError("uv: command not found")

    runner = JobRunner(work, spawn=broken, poll_interval=0.05)
    job = runner.start("public", work / "public", "filter", {})
    assert job.status == "failed" and job.exit_code == -1 and job.events.closed
    assert [e.type for e in job.events.snapshot()] == ["status", "done"] and "uv: command not found" in job.events.snapshot()[0].data["error"]
    assert "# could not start: uv: command not found" in job.log_path.read_text()
    assert runner.running("public") is None


def test_create_run_records_the_case_path_and_ingest_uses_it(served, work: Path):
    client, _, spawn = served
    status, doc = client.post("/api/runs", {"name": "case-2", "case_path": str(CASE)})
    assert status == 201 and doc["name"] == "case-2" and doc["case_path"] == str(CASE) and doc["source"] == "ui-run.json"
    record = json.loads((work / "case-2" / "ui-run.json").read_text())
    assert record == {"name": "case-2", "case_path": str(CASE), "regions": None, "created_at": record["created_at"]}
    assert not list((work / "case-2").glob("0*")) and not (Path(engine.cli.__file__).parent / "case-2").exists()
    assert client.post("/api/runs", {"name": "case-2", "case_path": str(CASE)})[0] == 409
    assert client.post("/api/runs", {"name": "case-3", "case_path": str(work / "missing.yaml")})[0] == 400
    assert client.post("/api/runs", {"name": "case-3"})[0] == 400
    status, doc = client.post("/api/runs", {"case_path": str(CASE)})  # no name: <proband_id>-<yyyymmdd-hhmm>
    assert status == 201 and re.fullmatch(r"PUBLIC01-\d{8}-\d{4}", doc["name"]), doc
    assert doc["name"] == setupmod.default_run_name("PUBLIC01", datetime.strptime(doc["name"][9:], "%Y%m%d-%H%M"))
    shutil.rmtree(work / doc["name"])
    assert client.post("/api/runs", {"name": "case-3", "case_path": str(CASE), "regions": "/nope.bed"})[0] == 400
    listed = {r["name"]: r for r in client.get("/api/runs")[1]["runs"]}
    assert listed["case-2"]["stages_present"] == [] and listed["case-2"]["case_path"] == str(CASE)
    summary = client.get("/api/runs/case-2/summary")[1]
    assert summary["stages_present"] == [] and summary["record"]["case_path"] == str(CASE)
    job, events = start_and_finish(client, "case-2", "ingest", {})
    assert job["argv"] == ["engine", "ingest", "--case", str(CASE), "--out", str(work / "case-2")]
    assert events[-1]["data"]["manifest_present"] is False and Path(job["log"]).parent == work / "case-2" / "01_ingest"
    status, doc = client.post("/api/runs/case-2/jobs", {"stage": "rank", "args": {"heap": "2g"}})
    assert status == 409 and "Exomiser" in doc["error"]
    status, doc = client.post("/api/runs/case-2/jobs", {"stage": "retrieve", "args": {}})
    assert status == 409 and "ClinVar" in doc["error"]
    assert len(spawn.calls) == 1


# ------------------------------------------------------------------------ CLI

def test_cli_command_is_exposed_for_registration():
    assert commands == [ui] and ui_command is ui and ui.name == "ui"
    result = CliRunner().invoke(ui, ["--help"])
    assert result.exit_code == 0 and "--work" in result.output and "--port" in result.output and "127.0.0.1" in result.output
    assert "--host" not in result.output
    registered = engine.cli.main.commands.get("ui")
    if registered is None:
        pytest.skip("engine.cli.STAGE_PACKAGES has no `ui` entry yet: `engine ui` is not a sub-command until the orchestrator adds it")
    assert registered is ui


def test_cli_serves_and_stops(work: Path, monkeypatch: pytest.MonkeyPatch):
    """The command binds, prints the URL, and stops on KeyboardInterrupt."""
    import engine.ui.server as server_mod

    seen: dict[str, Any] = {}

    def fake_serve(work_dir: Path, port: int, *, on_ready: Any = None, **kw: Any) -> None:
        seen.update({"work": work_dir, "port": port, "kw": kw})
        if on_ready:
            on_ready("http://127.0.0.1:0/")
        raise KeyboardInterrupt

    monkeypatch.setattr(server_mod, "serve", fake_serve)
    monkeypatch.setenv("CLINVAR_VCF", "/ref/clinvar.vcf.gz")
    monkeypatch.delenv("EXOMISER_DATA", raising=False)
    monkeypatch.delenv("ENGINE_CACHE", raising=False)
    project = work.parent  # an empty project directory: nothing to discover, every override honoured as given
    result = CliRunner().invoke(ui, ["--project", str(project), "--work", str(work), "--port", "0", "--exomiser-data", str(work)])
    assert result.exit_code in (0, 1, 130), result.output  # click turns the interrupt into an exit, not a traceback
    assert seen["work"] == work and seen["port"] == 0
    cfg = seen["kw"]["config"]
    assert cfg.clinvar_vcf == Path("/ref/clinvar.vcf.gz") and cfg.exomiser_data == work and cfg.funnel_bed is None
    assert cfg.cache == project / "cache"  # discovered by convention, whether or not it exists yet
    assert "serving on http://127.0.0.1:0/" in result.output and "clinvar_vcf: /ref/clinvar.vcf.gz · missing (env)" in result.output
    assert f"project={project}" in result.output and "funnel_bed: not configured · missing — engine funnel --ref-dir" in result.output
    assert seen["kw"]["project"] == project and callable(seen["kw"]["discovery"])


# ------------------------------------------------------------------------ real smoke

@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv on the PATH to run `uv run engine filter`")
def test_real_filter_through_the_server(work: Path):
    """A real subprocess: `uv run engine filter` on a copy of the public run, its
    lines read from the event stream with urllib, the manifest rewritten."""
    run = work / "public"
    manifest = run / "03_filter" / "manifest.json"
    before = json.loads(manifest.read_text())
    env = {k: v for k, v in os.environ.items() if k not in ("PYTEST_CURRENT_TEST",)}
    env["ENGINE_ENV"] = ""  # no dotenv for the child either
    app = App(work, runner=JobRunner(work, env=env, poll_interval=0.2))
    server = make_server(work, 0, app=app)
    serve_in_thread(server)
    try:
        client = Client(server.url)
        status, job = client.post("/api/runs/public/jobs", {"stage": "filter", "args": {}})
        assert status == 201 and job["argv"] == ["uv", "run", "engine", "filter", "--run", str(run)]
        events = client.sse(f"/api/jobs/{job['id']}/events", timeout=180)
        lines = [e["data"]["text"] for e in events if e["event"] == "line"]
        done = events[-1]["data"]
        assert done["exit_code"] == 0 and done["status"] == "done" and done["manifest_present"] is True, lines
        assert any(l.startswith("filter · ") for l in lines) and any("candidates: 2" in l for l in lines), lines
        assert any(l.strip().startswith("manifest: ") and l.strip().endswith("03_filter/manifest.json") for l in lines)
        after = json.loads(manifest.read_text())
        assert after["counts"] == before["counts"] and after["started_at"] >= before["started_at"]
        assert after["finished_at"] != before["finished_at"] or after["started_at"] != before["started_at"]
        log = Path(job["log"]).read_text()
        assert "# $ uv run engine filter --run" in log and "# exit 0 · done" in log and "filter · " in log
        assert client.get(f"/api/jobs/{job['id']}")[1]["status"] == "done"
        assert client.get("/api/runs/public/candidates")[1]["counts"]["candidates"] == 2
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(not (PUBLIC_RUN / "03_filter" / "candidates.json").exists(), reason="the public run has not been made (scripts/run_public_case.sh)")
def test_public_run_is_served_when_present(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    shutil.copytree(PUBLIC_RUN, work / "public", ignore=shutil.ignore_patterns("report.html"))
    server = make_server(work, 0)
    serve_in_thread(server)
    try:
        client = Client(server.url)
        doc = client.get("/api/runs")[1]
        assert [r["name"] for r in doc["runs"]] == ["public"] and "03_filter" in doc["runs"][0]["stages_present"]
        c = client.get("/api/runs/public/candidates")[1]
        assert c["present"] and c["candidates"][0]["candidate_id"] == "CFTR:comphet"
        chain = client.get("/api/runs/public/chain")[1]
        assert chain["present"] and all(ch["candidate_id"] for ch in chain["chains"])
        status, ct, html = client.request("GET", "/api/runs/public/report.html")
        assert status == 200 and b"CFTR" in html
    finally:
        server.shutdown()
        server.server_close()
