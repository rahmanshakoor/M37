"""The HTTP server — routes, the JSON API, the event stream and the static page.

Why the standard library: the UI is a local, single-user tool that must run wherever
the engine runs, with nothing to install and nothing listening beyond the loopback
interface. :class:`http.server.ThreadingHTTPServer` gives one thread per connection,
which is what a Server-Sent Events stream needs (a connection held open while a
job runs, without blocking the page's other requests), and binding to ``127.0.0.1``
is the whole access control: no auth, no CORS, nothing reachable from another host.

Every read endpoint is a view from :mod:`engine.report.views` serialised as JSON and
``/api/runs/<run>/report.html`` is :func:`engine.report.render.render_run`, so the
page and the report can never disagree about a run. The server adds no fact of its
own except what it knows about jobs.

Two things a request may name are checked before a path is touched: a run name
(one segment, :data:`engine.ui.jobs.RUN_NAME`, resolved and required to lie under
``--work``) and a static file name (one of the files the package ships). A job
request's arguments are checked by :func:`engine.ui.jobs.check_args`; the errors it
raises become 400/409 answers here. ``/api/providers`` says which keys are present
and never what they are.

Discovery (:mod:`engine.ui.setup`) is re-run on every ``GET /api/setup`` and before a
pipeline starts, and what it finds becomes the runner's fixed inputs — so a ClinVar
release downloaded from the Setup panel a minute ago is what the next ``retrieve``
gets, with no restart. ``GET /api/cases`` lists the case files of the project
directory (proband id, VCF name, HPO count; an unusable one with its error) and
``POST /api/runs/<run>/pipeline`` starts the six stages as one job.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlsplit

from engine import __version__
from engine.ui import jobs as jobmod
from engine.ui import setup as setupmod
from engine.ui.jobs import ArgumentError, JobConflict, JobRunner, NotConfigured, RUN_NAME

log = logging.getLogger("engine.ui")

HOST = "127.0.0.1"
"""The only address the server ever binds. Not an option."""
DEFAULT_PORT = 8765
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8",
                "app.css": "text/css; charset=utf-8"}
"""The files the package ships, by name; nothing else under ``static/`` is served."""
KEEPALIVE_S = 15.0
"""How often an idle event stream sends a comment so a proxy or browser keeps it open."""
HEAP_TTL_S = 30.0
"""How long a ``docker info`` reading is reused by ``GET /api/setup`` before it is taken again."""


class HttpError(Exception):
    def __init__(self, status: int, message: str, **extra: Any):
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra


# ------------------------------------------------------------------------- app

class App:
    """The API over one ``--work`` directory and one :class:`JobRunner`. Every method
    returns plain data (a dict/list for JSON, a ``(content_type, bytes)`` pair for a
    document) or raises :class:`HttpError`; the handler does the HTTP."""

    def __init__(self, work: Path, *, runner: JobRunner | None = None, static_dir: Path | None = None,
                 repo: Path | None = None, config: jobmod.RunnerConfig | None = None,
                 discovery: Callable[[], setupmod.Setup] | None = None, project: Path | None = None,
                 heap_probe: setupmod.HeapProbe | None = None):
        self.work = Path(work).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.runner = runner or JobRunner(self.work, repo=repo, config=config or jobmod.RunnerConfig())
        self.static_dir = Path(static_dir or STATIC_DIR)
        self.project = Path(project).resolve() if project else self.work.parent
        self.discovery = discovery or self._default_discovery
        self.heap_probe = heap_probe
        self._heap: tuple[float, dict[str, Any]] | None = None
        self._setup_lock = threading.Lock()

    def _default_discovery(self) -> setupmod.Setup:
        """Discovery over the work directory's parent with the runner's configured
        inputs as overrides — what a server built without the CLI gets."""
        c = self.runner.config
        given = {k: "option" for k, v in (("cache", c.cache), ("clinvar_vcf", c.clinvar_vcf),
                                          ("exomiser_data", c.exomiser_data), ("funnel_bed", c.funnel_bed)) if v is not None}
        return setupmod.discover(self.project, self.runner.repo, work=self.work, cache=c.cache, clinvar_vcf=c.clinvar_vcf,
                                 exomiser_data=c.exomiser_data, funnel_bed=c.funnel_bed, sources=given)

    # -- runs

    def run_dir(self, name: str, *, must_exist: bool = True) -> Path:
        """The run directory for ``name``: one path segment, under ``--work`` after
        resolution — ``..``, a slash or a symlink out of the tree is refused."""
        if not RUN_NAME.match(name) or name in (".", ".."):
            raise HttpError(400, f"invalid run name {name!r}: one path segment of letters, digits, '.', '_' or '-'")
        path = (self.work / name).resolve()
        if path.parent != self.work:
            raise HttpError(400, f"run name {name!r} does not stay under the work directory")
        if must_exist and not path.is_dir():
            raise HttpError(404, f"no run named {name!r} under {self.work}")
        return path

    def runs(self) -> dict[str, Any]:
        from engine.report.views import run_summary

        out = []
        for path in sorted(self.work.iterdir(), key=_mtime, reverse=True):
            if not path.is_dir() or path.name.startswith(".") or not RUN_NAME.match(path.name):
                continue
            if path.resolve().parent != self.work:
                continue  # a symlink out of the tree: not a run of this work directory
            s = run_summary(path)
            record = jobmod.read_run_record(path)
            job = self.runner.running(path.name)
            out.append({
                "name": path.name, "run_dir": str(path), "sample": s.get("sample"),
                "case_path": record.get("case_path"), "created_at": record.get("created_at"),
                "stages": [{"dir": st["dir"], "stage": st["stage"], "present": st["present"], "manifest": st["manifest"],
                            "finished_at": st.get("finished_at"), "duration_s": st.get("duration_s"),
                            "dry_run": st.get("dry_run"), "counts": st.get("counts") or {},
                            "headline": st.get("headline") or [], "failed": st.get("failed") or []}
                           for st in s["stages"]],
                "stages_present": s["stages_present"],
                "job": job.to_dict() if job else None,
            })
        return {"work": str(self.work), "runs": out}

    def create_run(self, body: dict[str, Any]) -> dict[str, Any]:
        name = str(body.get("name") or "").strip()
        case_path = str(body.get("case_path") or "").strip()
        regions = body.get("regions")
        if not case_path:
            raise HttpError(400, "case_path is required")
        if case_path.startswith("-") or not Path(case_path).expanduser().is_file():
            raise HttpError(400, f"case file not found: {case_path}")
        case = setupmod.describe_case(Path(case_path).expanduser())
        if case.get("error"):  # stage 1 would refuse it the same way; better now than after the run directory exists
            raise HttpError(400, f"case file refused: {case['error']}", case=case)
        if not name:  # the default: <proband_id>-<yyyymmdd-hhmm>, read from the case file
            name = case["default_run_name"]
        regions = str(regions).strip() if regions else None
        if regions and (regions.startswith("-") or not Path(regions).expanduser().is_file()):
            raise HttpError(400, f"regions file not found: {regions}")
        path = self.run_dir(name, must_exist=False)
        if path.exists():
            raise HttpError(409, f"a run named {name!r} already exists")
        path.mkdir(parents=True)
        record = jobmod.write_run_record(path, str(Path(case_path).expanduser()), regions and str(Path(regions).expanduser()))
        return {"name": name, "run_dir": str(path), "record": str(record), **jobmod.read_run_record(path)}

    def view(self, name: str, view: str, query: dict[str, list[str]]) -> Any:
        from engine.report import views

        run_dir = self.run_dir(name)
        if view == "summary":
            return {**views.run_summary(run_dir), "record": jobmod.read_run_record(run_dir)}
        if view == "candidates":
            return views.candidates_view(run_dir)
        if view == "ranking":
            return views.ranking_view(run_dir, top_n=_int(query, "top", views.DEFAULT_TOP_N, 1, 10000))
        if view == "medicine":
            return views.medicine_view(run_dir)
        if view == "provenance":
            return views.provenance_view(run_dir)
        raise HttpError(404, f"no view {view!r}")  # pragma: no cover - the route regex limits the names

    def chain(self, name: str, candidate_id: str | None) -> Any:
        from engine.report.views import chain_view

        return chain_view(self.run_dir(name), candidate_id)

    def report(self, name: str, query: dict[str, list[str]]) -> tuple[str, bytes]:
        from engine.report.render import render_run
        from engine.report.views import DEFAULT_TOP_N

        html = render_run(self.run_dir(name), top_n=_int(query, "top", DEFAULT_TOP_N, 1, 10000))
        return "text/html; charset=utf-8", html.encode("utf-8")

    # -- setup, cases, pipeline

    def refresh_setup(self) -> setupmod.Setup:
        """Re-run discovery and make what it found the runner's fixed inputs."""
        with self._setup_lock:
            found = self.discovery()
            self.runner.config = found.runner_config()
            return found

    def heap(self, *, fresh: bool = False) -> dict[str, Any]:
        """``--heap`` for rank from ``docker info`` (cached :data:`HEAP_TTL_S` seconds)."""
        with self._setup_lock:
            if not fresh and self._heap is not None and time.monotonic() - self._heap[0] < HEAP_TTL_S:
                return self._heap[1]
            value = setupmod.docker_heap(self.heap_probe)
            self._heap = (time.monotonic(), value)
            return value

    def pipeline_defaults(self) -> dict[str, Any]:
        """What a pipeline carries unless the request says otherwise: live on when a key
        is present, that provider and its default model, effort high, top 3, and the
        cost band that follows."""
        from engine.agents import providers

        d = setupmod.model_defaults(providers.status())
        d["cost"] = setupmod.cost_band(live=d["live"], provider=d["provider"], top=d["top"])
        d["vep_workers"] = jobmod.DEFAULT_VEP_WORKERS
        return d

    def setup(self) -> dict[str, Any]:
        found = self.refresh_setup()
        setup_job = self.runner.running(jobmod.SETUP_RUN)
        return {**found.to_dict(), "pipeline": self.pipeline_defaults(), "docker": self.heap(),
                "setup_job": setup_job.to_dict() if setup_job else None,
                "setup_jobs": [j.to_dict() for j in self.runner.list(jobmod.SETUP_RUN)]}

    def cases(self) -> dict[str, Any]:
        found = self.refresh_setup()
        return {"project": str(found.project), "cases": setupmod.list_cases(found.project)}

    def start_setup_job(self, body: dict[str, Any]) -> dict[str, Any]:
        stage = body.get("stage")
        if not isinstance(stage, str) or stage not in jobmod.SETUP_STAGES:
            raise HttpError(400, f"stage must be one of {', '.join(jobmod.SETUP_STAGES)}")
        if body.get("args"):
            raise HttpError(400, f"setup stage {stage} takes no argument: its command line is fixed", stage=stage, allowed=[])
        found = self.refresh_setup()
        item = next(name for name, cmd in setupmod.SETUP_COMMANDS.items() if cmd == stage)
        try:
            job = self.runner.start_setup(stage, found.command_for(item))
        except ArgumentError as e:
            raise HttpError(400, str(e)) from e
        except JobConflict as e:
            raise HttpError(409, str(e)) from e
        return job.to_dict()

    def start_pipeline(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        run_dir = self.run_dir(name)
        found = self.refresh_setup()
        defaults = self.pipeline_defaults()
        live = _flag(body, "live", defaults["live"])
        rerun = _flag(body, "rerun", False)
        top = body.get("top", defaults["top"])
        from engine.agents.client import EFFORTS
        model = str(body.get("model") or defaults["model"]).strip()
        effort = str(body.get("effort") or defaults["effort"]).strip()
        if effort not in EFFORTS:
            raise HttpError(400, f"effort must be one of {', '.join(EFFORTS)}")
        if defaults["provider"] == "openrouter" and "/" not in model:
            raise HttpError(400, "an OpenRouter model id is author/slug, e.g. deepseek/deepseek-v4-pro")
        if any(ch in model for ch in " ;&|`$\n\"'"):
            raise HttpError(400, "model holds a character the command line cannot take")
        if isinstance(top, bool) or not isinstance(top, (int, str)) or not str(top).isdigit() or not 1 <= int(top) <= 1000:
            raise HttpError(400, "top must be an integer between 1 and 1000")
        if live and not defaults["any_key"]:
            from engine.agents import providers

            raise HttpError(409, "live mode needs a model key in the server environment: " + providers.no_key_message()
                            + " — or start the pipeline with live off (the scripted fake provider)")
        heap = self.heap(fresh=True)
        try:
            job = self.runner.start_pipeline(name, run_dir, live=live, rerun=rerun, top=int(top),
                                             provider=defaults["provider"], model=model,
                                             effort=effort, heap=heap["heap"],
                                             heap_note="docker info unreadable: the pin's fallback" if heap["fallback"]
                                             else f"docker memory {heap['docker_memory_bytes']:,} bytes minus the 1 GiB reserve")
        except ArgumentError as e:
            raise HttpError(400, str(e)) from e
        except (JobConflict, NotConfigured) as e:
            raise HttpError(409, str(e), setup=found.to_dict()["items"]) from e
        return job.to_dict()

    # -- providers and stages

    def providers(self) -> dict[str, Any]:
        """Which providers have a key (presence only, never a value), each provider's
        default model, the effort choices and where a dotenv would be read from."""
        from engine.agents import providers
        from engine.agents.client import DEFAULT_EFFORT, EFFORTS

        dotenv: dict[str, Any] = {"path": None, "loaded": False, "note": None}
        try:
            dotenv["path"] = str(providers.env_path())
            loaded = providers.load_env_file()
            dotenv["loaded"] = bool(loaded)
        except (FileNotFoundError, ValueError) as e:
            dotenv["note"] = str(e)
        status = providers.status()
        default = providers.default_provider(strict=False)
        return {
            "providers": status,
            "default_provider": default,
            "any_key": any(v["key_present"] for k, v in status.items() if k != "fake"),
            "efforts": list(EFFORTS),
            "default_effort": DEFAULT_EFFORT,
            "dotenv": dotenv,
            "disclosure": {name: providers.disclosure(name, status[name]["default_model"], DEFAULT_EFFORT) for name in status},
        }

    def stages(self) -> dict[str, Any]:
        return {"stages": jobmod.describe_stages(), "command": list(self.runner.command), "repo": str(self.runner.repo),
                "config": self.runner.config.describe()}

    # -- jobs

    def start_job(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        run_dir = self.run_dir(name)
        stage = body.get("stage")
        if not isinstance(stage, str) or stage not in jobmod.STAGE_DIRS:
            raise HttpError(400, f"stage must be one of {', '.join(jobmod.STAGE_DIRS)}")
        args = body.get("args", {})
        try:
            job = self.runner.start(name, run_dir, stage, args)
        except ArgumentError as e:
            raise HttpError(400, str(e), stage=stage, allowed=[a.name for a in jobmod.WHITELIST[stage]]) from e
        except (JobConflict, NotConfigured) as e:
            raise HttpError(409, str(e)) from e
        return job.to_dict()

    def jobs(self, query: dict[str, list[str]]) -> dict[str, Any]:
        run = (query.get("run") or [None])[0]
        return {"jobs": [j.to_dict() for j in self.runner.list(run)]}

    def job(self, job_id: str) -> jobmod.Job:
        job = self.runner.get(job_id)
        if job is None:
            raise HttpError(404, f"no job {job_id!r}")
        return job

    def job_log(self, job_id: str) -> tuple[str, bytes]:
        job = self.job(job_id)
        try:
            return "text/plain; charset=utf-8", job.log_path.read_bytes()
        except OSError:
            return "text/plain; charset=utf-8", b""

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.job(job_id)
        if job.finished():
            raise HttpError(409, f"job {job_id} has already {job.status}")
        self.runner.cancel(job_id)
        return job.to_dict()

    # -- static

    def static(self, name: str) -> tuple[str, bytes]:
        if name not in STATIC_FILES:
            raise HttpError(404, f"no such file {name!r}")
        path = (self.static_dir / name).resolve()
        if path.parent != self.static_dir.resolve():
            raise HttpError(404, f"no such file {name!r}")
        try:
            return STATIC_FILES[name], path.read_bytes()
        except OSError as e:
            raise HttpError(404, f"{name}: {e.strerror or e}") from e


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _flag(body: dict[str, Any], key: str, default: bool) -> bool:
    value = body.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false", "1", "0", "yes", "no"):
        return value.lower() in ("true", "1", "yes")
    raise HttpError(400, f"{key} must be true or false")


def _int(query: dict[str, list[str]], key: str, default: int, lo: int, hi: int) -> int:
    raw = (query.get(key) or [None])[0]
    if raw is None or raw == "":
        return default
    try:
        n = int(raw)
    except ValueError:
        raise HttpError(400, f"{key} must be an integer") from None
    if not lo <= n <= hi:
        raise HttpError(400, f"{key} must be between {lo} and {hi}")
    return n


# --------------------------------------------------------------------- routing

Route = tuple[str, re.Pattern[str], str]
ROUTES: list[Route] = [
    ("GET", re.compile(r"^/$"), "index"),
    ("GET", re.compile(r"^/static/(?P<name>[A-Za-z0-9_.-]+)$"), "static"),
    ("GET", re.compile(r"^/api/runs$"), "runs"),
    ("POST", re.compile(r"^/api/runs$"), "create_run"),
    ("GET", re.compile(r"^/api/runs/(?P<run>[^/]+)/(?P<view>summary|candidates|ranking|medicine|provenance)$"), "view"),
    ("GET", re.compile(r"^/api/runs/(?P<run>[^/]+)/chain(?:/(?P<candidate>[^/]+))?$"), "chain"),
    ("GET", re.compile(r"^/api/runs/(?P<run>[^/]+)/report\.html$"), "report"),
    ("POST", re.compile(r"^/api/runs/(?P<run>[^/]+)/jobs$"), "start_job"),
    ("POST", re.compile(r"^/api/runs/(?P<run>[^/]+)/pipeline$"), "start_pipeline"),
    ("GET", re.compile(r"^/api/providers$"), "providers"),
    ("GET", re.compile(r"^/api/stages$"), "stages"),
    ("GET", re.compile(r"^/api/setup$"), "setup"),
    ("POST", re.compile(r"^/api/setup/jobs$"), "start_setup_job"),
    ("GET", re.compile(r"^/api/cases$"), "cases"),
    ("GET", re.compile(r"^/api/jobs$"), "jobs"),
    ("GET", re.compile(r"^/api/jobs/(?P<job>[A-Za-z0-9_-]+)$"), "job"),
    ("GET", re.compile(r"^/api/jobs/(?P<job>[A-Za-z0-9_-]+)/events$"), "events"),
    ("GET", re.compile(r"^/api/jobs/(?P<job>[A-Za-z0-9_-]+)/log$"), "job_log"),
    ("POST", re.compile(r"^/api/jobs/(?P<job>[A-Za-z0-9_-]+)/cancel$"), "cancel"),
]
MAX_BODY = 1 << 20


class Handler(BaseHTTPRequestHandler):
    """One request. ``app`` is set by :func:`make_server` on a per-server subclass."""

    app: App
    server_version = f"engine-ui/{__version__}"
    sys_version = ""

    # -- entry points

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the base class's name
        log.debug("%s %s", self.address_string(), format % args)

    # -- dispatch

    def _handle(self, method: str) -> None:
        url = urlsplit(self.path)
        path = unquote(url.path)
        query = parse_qs(url.query, keep_blank_values=True)
        try:
            for verb, pattern, name in ROUTES:
                m = pattern.match(path)
                if m is None:
                    continue
                if verb != method:
                    continue
                self._route(name, {k: unquote(v) for k, v in m.groupdict().items() if v is not None}, query)
                return
            if any(p.match(path) for _, p, _ in ROUTES):
                raise HttpError(405, f"{method} is not allowed on {path}")
            raise HttpError(404, f"no route for {path}")
        except HttpError as e:
            self._json({"error": e.message, **e.extra}, status=e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # a bug, reported as such rather than as a dropped connection
            log.exception("ui: %s %s failed", method, path)
            self._json({"error": f"{type(e).__name__}: {e}"}, status=500)

    def _route(self, name: str, params: dict[str, str], query: dict[str, list[str]]) -> None:
        app = self.app
        if name == "index":
            self._document(*app.static("index.html"))
        elif name == "static":
            self._document(*app.static(params["name"]))
        elif name == "runs":
            self._json(app.runs())
        elif name == "create_run":
            self._json(app.create_run(self._body()), status=201)
        elif name == "view":
            self._json(app.view(params["run"], params["view"], query))
        elif name == "chain":
            self._json(app.chain(params["run"], params.get("candidate")))
        elif name == "report":
            self._document(*app.report(params["run"], query))
        elif name == "start_job":
            self._json(app.start_job(params["run"], self._body()), status=201)
        elif name == "start_pipeline":
            self._json(app.start_pipeline(params["run"], self._body()), status=201)
        elif name == "setup":
            self._json(app.setup())
        elif name == "start_setup_job":
            self._json(app.start_setup_job(self._body()), status=201)
        elif name == "cases":
            self._json(app.cases())
        elif name == "providers":
            self._json(app.providers())
        elif name == "stages":
            self._json(app.stages())
        elif name == "jobs":
            self._json(app.jobs(query))
        elif name == "job":
            self._json(app.job(params["job"]).to_dict())
        elif name == "job_log":
            self._document(*app.job_log(params["job"]))
        elif name == "cancel":
            self._json(app.cancel(params["job"]))
        elif name == "events":
            self._events(app.job(params["job"]), query)
        else:  # pragma: no cover
            raise HttpError(404, f"no route {name}")

    # -- request / response

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise HttpError(413, "request body too large")
        raw = self.rfile.read(length) if length else b""
        if not raw.strip():
            return {}
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise HttpError(400, "request body must be JSON") from None
        if not isinstance(doc, dict):
            raise HttpError(400, "request body must be a JSON object")
        return doc

    def _json(self, doc: Any, status: int = 200) -> None:
        body = json.dumps(doc, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def _document(self, content_type: str, body: bytes) -> None:
        self._send(200, content_type, body)

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _events(self, job: jobmod.Job, query: dict[str, list[str]]) -> None:
        """The Server-Sent Events stream: every event after ``Last-Event-ID`` (or
        ``?after=``), then each new one as it arrives, a keepalive comment while
        nothing does, and the stream ends after the ``done`` event."""
        after = _int({"after": [self.headers.get("Last-Event-ID") or (query.get("after") or ["0"])[0]]},
                     "after", 0, 0, 1 << 62)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.wfile.write(f": job {job.id} · {job.stage} · {job.status}\n\n".encode("utf-8"))
        self.wfile.flush()
        while True:
            events = job.events.wait(after, KEEPALIVE_S)
            if not events:
                if job.events.closed:
                    return
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
                continue
            for e in events:
                self.wfile.write(e.sse().encode("utf-8"))
                after = e.seq
            self.wfile.flush()
            if any(e.type == "done" for e in events):
                return


# ---------------------------------------------------------------------- server

class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, app: App, port: int = DEFAULT_PORT):
        handler = type("BoundHandler", (Handler,), {"app": app})
        super().__init__((HOST, port), handler)
        self.app = app

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/"


def make_server(work: Path, port: int = DEFAULT_PORT, *, app: App | None = None, **app_kwargs: Any) -> Server:
    """A server bound to ``127.0.0.1:port`` (``0`` picks a free port) and not yet
    serving; call :meth:`Server.serve_forever` (a thread in tests, the CLI's
    foreground) and :meth:`Server.shutdown` to stop."""
    return Server(app or App(work, **app_kwargs), port)


def serve_in_thread(server: Server) -> threading.Thread:
    t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
    t.start()
    return t


def serve(work: Path, port: int = DEFAULT_PORT, *, on_ready: Callable[[str], None] | None = None,
          **app_kwargs: Any) -> None:
    """Serve until interrupted (the CLI). ``on_ready`` is told the URL once bound."""
    server = make_server(work, port, **app_kwargs)
    if on_ready is not None:
        on_ready(server.url)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
