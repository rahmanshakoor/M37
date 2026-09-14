"""Setup — what the page finds without being told: the project's directories, the
reference files a stage needs, the case files it may run, and the defaults a
pipeline carries into each stage.

Why a discovery step: ``engine ui`` must run a case end to end with no path typed
and no setting known. The layout is the one the whole project already uses — the
checkout sits inside a *project directory* that holds the case file(s), ``ref/``
(public reference releases), ``cache/`` (the HTTP cache) and ``work/`` (the runs) —
so every input has one conventional place, and :func:`discover` looks there. An
option or an environment variable still overrides any item; what was discovered
and what was overridden are both reported, never silently merged.

What "present" means is the same test each stage applies, read here without
touching the bulk of the data: the ClinVar VCF must be indexed (``.tbi``/``.csi``
beside it), the funnel BED must exist, and the Exomiser data must be *verified* —
every required file present and the ``.verified.json`` sidecars the downloader wrote
carrying the pinned hashes (:mod:`engine.rank.download`; the same check
``scripts/run_public_case.sh`` makes before stage 4). A directory that is there but
fails that test is ``partial`` and is not used as the default, so a half-extracted
bundle never reaches Exomiser by accident.

Case files are parsed with :class:`engine.config.CaseConfig` — the engine's own
reading — and a case whose VCF is missing is listed *with* its error rather than
hidden, so the page can say why a run cannot start. Nothing from a case beyond its
proband id, the VCF's file name, its HPO count and the sample name leaves this
module; the HPO terms themselves are not listed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from engine.agents import providers
from engine.ui.jobs import RUN_NAME, RunnerConfig

CASE_GLOBS = ("case.yaml", "case.*.yaml")
"""What "New run" offers: the case file and its siblings in the project directory."""
CLINVAR_GLOB = "clinvar_*.vcf.gz"
FUNNEL_GLOB = "funnel*.bed"
EXOMISER_SUBDIR = "exomiser"
"""``<ref>/exomiser`` — what ``engine exomiser-download --ref-dir`` fills and ``engine rank --exomiser-data`` reads."""
DEFAULT_TOP = 3
DEFAULT_EFFORT = "high"
DEFAULT_VEP_WORKERS = 6
HEAP_FALLBACK = "2g"
"""``--heap`` when ``docker info`` cannot be read (the pin's own fallback)."""

STATES = ("ready", "partial", "missing")
"""An item's state: usable as it is, there but not usable (the detail says why), not there."""

SETUP_COMMANDS: dict[str, str] = {
    "clinvar_vcf": "clinvar-download",
    "funnel_bed": "funnel",
    "exomiser_data": "exomiser-download",
}
"""Setup item → the ``engine`` sub-command that creates it (a job the page can start)."""

_CLINVAR_DATE = re.compile(r"^clinvar_(\d{8})\.vcf\.gz$")


# ------------------------------------------------------------------------ items

@dataclass(frozen=True)
class Item:
    """One discovered input: where it is (or would be), whether it is usable, and the
    command that would create it."""

    name: str
    label: str
    path: Path | None
    state: str
    detail: str
    source: str
    """``discovered`` · ``option`` · ``env`` · ``default`` — where the path came from."""
    command: str | None = None
    setup_stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "label": self.label, "path": None if self.path is None else str(self.path),
                "state": self.state, "ready": self.state == "ready", "detail": self.detail, "source": self.source,
                "command": self.command, "setup_stage": self.setup_stage}


@dataclass(frozen=True)
class Setup:
    """Everything :func:`discover` found, plus the :class:`RunnerConfig` it implies."""

    project: Path
    repo: Path
    ref: Path
    work: Path
    cache: Path | None
    clinvar_vcf: Path | None
    funnel_bed: Path | None
    exomiser_data: Path | None
    items: tuple[Item, ...] = ()
    overrides: dict[str, str] = field(default_factory=dict)

    def runner_config(self) -> RunnerConfig:
        return RunnerConfig(cache=self.cache, clinvar_vcf=self.clinvar_vcf, exomiser_data=self.exomiser_data,
                            funnel_bed=self.funnel_bed)

    def item(self, name: str) -> Item:
        return next(i for i in self.items if i.name == name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": str(self.project), "repo": str(self.repo), "ref": str(self.ref), "work": str(self.work),
            "items": [i.to_dict() for i in self.items],
            "ready": all(i.state == "ready" for i in self.items if i.name != "cache"),
            "overrides": dict(self.overrides),
            "commands": {name: self.command_for(name) for name in SETUP_COMMANDS},
        }

    def command_for(self, item: str) -> list[str]:
        """The fixed argv of the setup job that creates ``item`` (without the ``uv run`` prefix)."""
        return setup_argv(SETUP_COMMANDS[item], ref=self.ref, cache=self.cache)


def setup_argv(command: str, *, ref: Path, cache: Path | None = None) -> list[str]:
    """``engine <cmd> --ref-dir <ref>`` — nothing from a request goes into it. The
    Exomiser bundle lives in ``<ref>/exomiser`` (the directory ``rank`` reads); the
    funnel keeps its CHECKSUMS observation in the same HTTP cache as the stages."""
    if command == "clinvar-download":
        return ["clinvar-download", "--ref-dir", str(ref)]
    if command == "funnel":
        argv = ["funnel", "--ref-dir", str(ref)]
        if cache is not None:
            argv += ["--cache", str(cache)]
        return argv
    if command == "exomiser-download":
        return ["exomiser-download", "--ref-dir", str(ref / EXOMISER_SUBDIR)]
    raise ValueError(f"not a setup command: {command!r}")


# -------------------------------------------------------------------- discovery

def project_root(repo: Path) -> Path:
    """The directory above the checkout — where ``case.yaml``, ``ref/``, ``cache/`` and ``work/`` live."""
    return Path(repo).resolve().parent


def discover(project: Path, repo: Path, *, work: Path | None = None, ref: Path | None = None, cache: Path | None = None,
             clinvar_vcf: Path | None = None, funnel_bed: Path | None = None, exomiser_data: Path | None = None,
             sources: dict[str, str] | None = None) -> Setup:
    """Look in the conventional places, honour every override, report each item.
    ``sources`` names where an override came from (``option``/``env``) for the report."""
    project = Path(project).resolve()
    repo = Path(repo).resolve()
    sources = dict(sources or {})
    ref = Path(ref).expanduser() if ref else project / "ref"
    work = Path(work).expanduser() if work else project / "work"
    overrides: dict[str, str] = {k: v for k, v in sources.items()}
    items: list[Item] = []

    # work
    n_runs = _count_runs(work)
    items.append(Item("work", "Work directory", work, "ready" if work.is_dir() else "missing",
                      f"{n_runs} run{'s' if n_runs != 1 else ''}" if work.is_dir() else "created when the first run is made",
                      sources.get("work", "default")))

    # cache
    cache_path = Path(cache).expanduser() if cache else project / "cache"
    items.append(Item("cache", "HTTP cache", cache_path, "ready" if cache_path.is_dir() else "missing",
                      "every source answer is kept here" if cache_path.is_dir() else "created by the first stage that fetches",
                      sources.get("cache", "default")))

    # ClinVar
    if clinvar_vcf is not None:
        cv_path, cv_source = Path(clinvar_vcf).expanduser(), sources.get("clinvar_vcf", "option")
    else:
        cv_path, cv_source = find_clinvar(ref), "discovered"
    cv_state, cv_detail = _clinvar_state(cv_path, ref)
    items.append(Item("clinvar_vcf", "ClinVar VCF", cv_path, cv_state, cv_detail, cv_source,
                      _cmd(setup_argv("clinvar-download", ref=ref)), "clinvar-download"))

    # funnel
    if funnel_bed is not None:
        fb_path, fb_source = Path(funnel_bed).expanduser(), sources.get("funnel_bed", "option")
    else:
        fb_path, fb_source = find_funnel(ref), "discovered"
    fb_state, fb_detail = _funnel_state(fb_path, ref)
    items.append(Item("funnel_bed", "Funnel BED", fb_path, fb_state, fb_detail, fb_source,
                      _cmd(setup_argv("funnel", ref=ref, cache=cache_path)), "funnel"))

    # Exomiser
    if exomiser_data is not None:
        ex_path, ex_source = Path(exomiser_data).expanduser(), sources.get("exomiser_data", "option")
    else:
        ex_path, ex_source = ref / EXOMISER_SUBDIR, "discovered"
    ex_state, ex_detail = exomiser_state(ex_path)
    items.append(Item("exomiser_data", "Exomiser data", ex_path, ex_state, ex_detail, ex_source,
                      _cmd(setup_argv("exomiser-download", ref=ref)), "exomiser-download"))

    def usable(item: Item, forced: bool) -> Path | None:
        # an override is passed on as given (the operator said so); a discovery only when it is ready
        if item.path is None:
            return None
        return item.path if forced or item.state == "ready" else None

    by_name = {i.name: i for i in items}
    return Setup(
        project=project, repo=repo, ref=ref, work=work, cache=cache_path,
        clinvar_vcf=usable(by_name["clinvar_vcf"], clinvar_vcf is not None),
        funnel_bed=usable(by_name["funnel_bed"], funnel_bed is not None),
        exomiser_data=usable(by_name["exomiser_data"], exomiser_data is not None),
        items=tuple(items), overrides=overrides,
    )


def find_clinvar(ref: Path) -> Path | None:
    """The newest ``clinvar_YYYYMMDD.vcf.gz`` under ``ref`` — by the date in the name,
    which is the release, not the download time."""
    if not ref.is_dir():
        return None
    found = [p for p in ref.glob(CLINVAR_GLOB) if p.is_file() and _CLINVAR_DATE.match(p.name)]
    if not found:
        return None
    return max(found, key=lambda p: _CLINVAR_DATE.match(p.name).group(1))  # type: ignore[union-attr]


def find_funnel(ref: Path) -> Path | None:
    """The newest ``funnel*.bed`` under ``ref`` (by modification time: the name carries
    the release and padding, not a date)."""
    if not ref.is_dir():
        return None
    found = [p for p in ref.glob(FUNNEL_GLOB) if p.is_file()]
    if not found:
        return None
    return max(found, key=lambda p: (p.stat().st_mtime, p.name))


def _clinvar_state(path: Path | None, ref: Path) -> tuple[str, str]:
    if path is None:
        return "missing", f"no {CLINVAR_GLOB} under {ref}"
    if not path.is_file():
        return "missing", f"not found: {path}"
    indexed = any(path.with_name(path.name + ext).exists() for ext in (".tbi", ".csi"))
    m = _CLINVAR_DATE.match(path.name)
    release = f"release {m.group(1)}" if m else path.name
    size = f"{path.stat().st_size / 1e6:,.0f} MB"
    if not indexed:
        return "partial", f"{release} · {size} · no .tbi/.csi beside it (bcftools index -t)"
    return "ready", f"{release} · {size} · indexed"


def _funnel_state(path: Path | None, ref: Path) -> tuple[str, str]:
    if path is None:
        return "missing", f"no {FUNNEL_GLOB} under {ref} — without it every ingested variant goes to the sources and Exomiser sees the whole VCF"
    if not path.is_file():
        return "missing", f"not found: {path}"
    detail = f"{path.stat().st_size / 1e6:,.1f} MB"
    counts = _read_json(Path(str(path) + ".manifest.json"))
    counts = (counts or {}).get("counts") if isinstance(counts, dict) else None
    if isinstance(counts, dict) and counts.get("n_intervals"):
        detail += f" · {int(counts['n_intervals']):,} intervals"
        if counts.get("genome_fraction"):
            detail += f" · {float(counts['genome_fraction']) * 100:.1f}% of the genome"
    return "ready", detail


def exomiser_state(data_dir: Path | None) -> tuple[str, str]:
    """``ready`` when every required file of both bundles is present and the
    ``.verified.json`` sidecars carry the pinned hashes; ``partial`` with the reasons
    otherwise; ``missing`` when the directory is not there. Reads sidecars, not data."""
    from engine.rank.download import members_status, read_verified
    from engine.rank.exomiser import DEFAULT_CONFIG, ExomiserConfig

    if data_dir is None or not data_dir.is_dir():
        return "missing", "not downloaded (about 70 GB of zips; needs Docker to run)"
    try:
        cfg = ExomiserConfig.load(DEFAULT_CONFIG)
    except (OSError, KeyError, ValueError) as e:
        return "partial", f"cannot read the Exomiser pin: {e}"
    problems: list[str] = []
    for name, files in cfg.missing_data(data_dir).items():
        problems.append(f"{name}: missing {', '.join(files[:3])}{'…' if len(files) > 3 else ''}")
    for name, b in cfg.bundles.items():
        try:
            meta = read_verified(data_dir / b.filename)
        except (OSError, ValueError):
            meta = None
        if meta is None:
            problems.append(f"{name}: no {b.filename}.verified.json sidecar")
            continue
        if b.sha256 and meta.get("sha256") != b.sha256:
            problems.append(f"{name}: sidecar sha256 differs from the pin")
        if b.member_sha256 and not members_status(b, data_dir)["all_pinned_verified"]:
            problems.append(f"{name}: pinned members not all verified (engine exomiser-download --verify-extracted)")
    if problems:
        return "partial", "; ".join(problems)
    return "ready", f"release {cfg.data_version} · {', '.join(cfg.bundles)} verified"


def _count_runs(work: Path) -> int:
    try:
        return sum(1 for p in work.iterdir() if p.is_dir() and not p.name.startswith(".") and RUN_NAME.match(p.name))
    except OSError:
        return 0


def _cmd(argv: list[str]) -> str:
    return "engine " + " ".join(argv)


# ------------------------------------------------------------------------ cases

def list_cases(project: Path) -> list[dict[str, Any]]:
    """Every ``case.yaml`` / ``case.*.yaml`` in the project directory, read the way the
    engine reads it. A file that does not parse, or whose VCF is missing or not
    indexed, is listed with ``error`` set — visible, never hidden."""
    project = Path(project)
    if not project.is_dir():
        return []
    seen: dict[str, Path] = {}
    for pattern in CASE_GLOBS:
        for p in project.glob(pattern):
            if p.is_file():
                seen[p.name] = p
    return [describe_case(seen[name]) for name in sorted(seen)]


def describe_case(path: Path) -> dict[str, Any]:
    """Proband id, VCF file name, HPO count and sample of one case file; ``error`` when
    the engine would refuse it (the fields it could read are still filled)."""
    from engine.config import CaseConfig

    path = Path(path).resolve()
    out: dict[str, Any] = {"path": str(path), "name": path.name, "proband_id": None, "vcf": None, "vcf_present": None,
                           "hpo_count": None, "regions": None, "sample": None, "error": None, "default_run_name": None}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("the file is not a mapping")
    except (OSError, ValueError, yaml.YAMLError) as e:
        out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        return out
    # what the file says, before validation — so a refused file still shows who it is for
    out["proband_id"] = str(raw["proband_id"]) if raw.get("proband_id") else None
    out["vcf"] = Path(str(raw["vcf"])).name if raw.get("vcf") else None
    out["hpo_count"] = len(raw["hpo"]) if isinstance(raw.get("hpo"), list) else None
    out["sample"] = str(raw["sample"]) if raw.get("sample") else None
    if out["proband_id"]:
        out["default_run_name"] = default_run_name(out["proband_id"])
    base = path.parent
    for key in ("vcf", "reference_fasta", "regions"):
        if raw.get(key):
            raw[key] = (base / str(raw[key])).resolve()
    try:
        case = CaseConfig(**raw)
    except Exception as e:  # pydantic's ValidationError, or a value of the wrong shape
        out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        return out
    out.update({"proband_id": case.proband_id, "vcf": case.vcf.name, "vcf_present": case.vcf.is_file(),
                "hpo_count": len(case.hpo), "regions": case.regions.name if case.regions else None, "sample": case.sample,
                "default_run_name": default_run_name(case.proband_id)})
    try:
        case.check_files()
    except FileNotFoundError as e:
        out["error"] = str(e).splitlines()[0]
    return out


def default_run_name(proband_id: str, now: datetime | None = None) -> str:
    """``<proband_id>-<yyyymmdd-hhmm>`` in local time, made a valid run name."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M")
    head = re.sub(r"[^A-Za-z0-9._-]+", "-", str(proband_id or "").strip()).strip(".-") or "case"
    name = f"{head}-{stamp}"
    if not RUN_NAME.match(name):
        name = f"case-{stamp}"
    return name


# ---------------------------------------------------------------- pipeline defaults

HeapProbe = Callable[[], int | None]


def docker_heap(probe: HeapProbe | None = None) -> dict[str, Any]:
    """``--heap`` for ``rank``: the daemon's memory (``docker info``) minus the pin's
    reserve (1 GiB), floored and rounded as the stage does; the pin's fallback when
    the daemon cannot be read. ``probe`` is injected by the tests."""
    from engine.rank.exomiser import DEFAULT_CONFIG, ExomiserConfig, docker_info_memory_bytes, heap_for

    total = (probe or docker_info_memory_bytes)()
    try:
        cfg = ExomiserConfig.load(DEFAULT_CONFIG)
        heap = heap_for(cfg, total)
    except (OSError, KeyError, ValueError):
        heap = HEAP_FALLBACK
    return {"docker_memory_bytes": total, "heap": heap, "fallback": total is None}


def model_defaults(status: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Provider, model, effort and the live default from what keys are present
    (:func:`engine.agents.providers.status`): openrouter first, then anthropic; no
    key → the scripted ``fake`` provider and live off."""
    status = providers.status() if status is None else status
    provider = next((name for name in ("openrouter", "anthropic") if (status.get(name) or {}).get("key_present")), None)
    any_key = provider is not None
    chosen = provider or "fake"
    return {
        "any_key": any_key,
        "live": any_key,
        "provider": chosen,
        "model": (status.get(chosen) or {}).get("default_model") or providers.default_model(chosen),
        "models": (status.get(chosen) or {}).get("models") or [],
        "efforts": ["low", "medium", "high", "xhigh", "max"],
        "effort": DEFAULT_EFFORT,
        "top": DEFAULT_TOP,
        "providers": {name: bool((status.get(name) or {}).get("key_present")) for name in ("openrouter", "anthropic")},
    }


def cost_band(*, live: bool, provider: str, top: int) -> dict[str, Any]:
    """What a live pipeline will spend, in the only units known before it runs: the
    number of agent conversations (one per stage-5 candidate, one for stage 6) and
    the turns each may take. No price is guessed."""
    from engine.reason.run import DEFAULT_MAX_TURNS

    calls = int(top) + 1
    if not live or provider == "fake":
        return {"band": "none", "calls": 0, "max_turns": 0,
                "note": "the scripted fake provider answers; no model is called and nothing is paid for"}
    return {"band": "paid", "calls": calls, "max_turns": DEFAULT_MAX_TURNS,
            "note": f"up to {calls} agent conversations ({top} candidate{'s' if top != 1 else ''} + the medicine report), "
                    f"each up to {DEFAULT_MAX_TURNS} tool-calling turns at effort {DEFAULT_EFFORT}, billed by {provider}"}


def _read_json(path: Path) -> Any:
    import json

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
