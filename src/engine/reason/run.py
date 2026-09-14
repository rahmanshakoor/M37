"""Stage 5 orchestrator — ``engine reason``.

For each of the top-N stage-3 candidates: build the bundle (everything the model may
argue from, written to disk so a judge can open it), give it to the reasoning agent
with the three tools, validate the answer against exactly what that candidate was
shown, compute the classification from the surviving criteria, render. The model
never sees anything that is not in the run directory, and nothing the model says
reaches the report without passing the validator.

The citation scope is per candidate: a chain may cite the bundle's record ids and the
records the tools returned in *this* conversation — not another candidate's variant
records, not a paper an earlier run left in the stage store, not a later stage's
evidence. The validator is given an index of just those records, so a mis-citation
of any other real record is rejected like an invented one, and the manifest notes
which rejected ids exist elsewhere in the run.

What is written, and why in this form:

* ``bundles/<candidate_id>.json`` — what the model was given (``engine.agents.bundle``).
* ``prompts/<candidate_id>.json`` — the request as the stage built it: system prompt,
  user prompt, tool definitions as sent, the output schema, model, effort, turn
  budget, and the two fixed texts the client adds on its own (the final-answer
  instruction, the tool-budget refusal). How the client transmits it — prompt
  caching, streaming of the final call, ``tool_choice`` on that call — is the
  client's and is recorded under ``request`` in the transcript. ``.md`` beside it is
  the same prompt for reading. Written in every run; a ``--dry-run`` stops here, so a
  prompt can be reviewed before a model is paid for.
* ``transcripts/<candidate_id>.json`` — every turn, every tool call and result, token
  usage and the final text exactly as the model wrote it. A failed run leaves
  ``<candidate_id>.failed.json`` with the turns completed before the failure.
* ``evidence/`` — the ``pmid:`` (and ``pmid-search:``) records the tools fetched, in the
  stage-2 format, so a paper cited in a chain is a record like any other.
* ``chains/<candidate_id>.json`` — the validated chain, classification filled by the
  engine; ``validation/<candidate_id>.json`` — every rejection and dispute.
* ``evidence_chain.md`` — the chains rendered with a References section per candidate.
* ``manifest.json`` — inputs (sha256), model, effort, the prompt's version and hashes,
  every threshold the validator applied, token usage, the provider disclosure line,
  and the validator's rejections as notes.

A run replaces the stage's outputs: the per-candidate directories, the report and
the manifest are removed before the first candidate, so nothing a manifest does not
claim survives beside it. ``evidence/`` is a store, like stage 2's, and is kept — a
paper fetched once is served from there on a rerun.

Determinism: bundles, prompts, transcripts and chains sort their keys and carry no
clock; with a scripted client and a warm cache a rerun — in place or in a fresh copy
of the run directory — is byte-identical. The manifest is the only file that carries
a time, in its own fields.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import anthropic
import yaml

from engine.agents import client as ac
from engine.agents import providers
from engine.agents.bundle import Bundle, build_bundle, write_bundle
from engine.agents.render import cited_ids, render_evidence_chain
from engine.agents.schema import ALL_CODES, EvidenceChain, VariantChain, combine_acmg
from engine.agents.validator import (
    AF_FIELD, AF_FIELDS, THRESHOLDS, EvidenceIndex, Rejection, ValidationReport, canonical_key, frequency_rules, validate,
)
from engine.config import HPO_ID
from engine.manifest import Manifest
from engine.reason.checks import AccessionResolver, stage_checks
from engine.reason.prompts import PROMPT_VERSION, SYSTEM_PROMPT, instructions_sha256, prompt_sha256, user_prompt
from engine.reason.tools import ABSTRACT_CHARS, COORDINATE_RULE, DEFAULT_MAX_RESULTS, MAX_RESULTS_CAP, ReasonTools
from engine.retrieve.http import Http, HttpCache, RateLimiter, default_cache_root
from engine.retrieve.literature import LiteratureRetriever
from engine.retrieve.run import BACKOFF_FLOOR, RATE_LIMITS, RETRIES
from engine.retrieve.store import EvidenceRecord, EvidenceStore

STAGE_DIR = "05_reason"
FILTER_DIR = "03_filter"
RANK_DIR = "04_rank"
RETRIEVE_DIR = "02_retrieve"
EVIDENCE_STAGES = (RETRIEVE_DIR, RANK_DIR, STAGE_DIR)
"""Where a record this stage may serve or cite can live: the stages before it, and
its own store. A later stage's evidence is never an input here."""
OUTPUT_DIRS = ("bundles", "prompts", "transcripts", "chains", "validation")
DEFAULT_TOP_N = 3
DEFAULT_MAX_TURNS = 10
CITATION_SCOPE = "the bundle's record ids plus the records returned by this candidate's own tool calls"

Progress = Callable[[str], None]

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._:-]+")


def run_reason(
    run_dir: Path,
    top_n: int = DEFAULT_TOP_N,
    client: ac.ModelClient | None = None,
    model: str | None = None,
    effort: str = ac.DEFAULT_EFFORT,
    dry_run: bool = False,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    cache_root: Path | None = None,
    offline: bool = False,
    http: Http | None = None,
    case_hpo: list[str] | None = None,
    case_path: Path | None = None,
    af_field: str = AF_FIELD,
    thresholds: dict[str, float] | None = None,
    literature_max_results: int = DEFAULT_MAX_RESULTS,
    provider: str | None = None,
    progress: Progress = lambda s: None,
) -> Path:
    """Run stage 5 into ``run_dir/05_reason``; returns the manifest path.

    ``client`` defaults to the ``provider``'s client (``engine.agents.providers``;
    ``None`` is the Anthropic SDK client), built when the first candidate needs it
    (never in a dry run); ``model`` defaults to the provider's and is settled by
    ``providers.provider_model`` so the requests, the manifest and the disclosure
    carry the id the provider actually sends. ``http`` (tests) or
    ``cache_root``/``offline`` shape the literature retriever. HPO terms come from
    ``case_hpo``, else ``case_path``, else stage 4's ``joined.json``, else none. Raises
    :class:`~engine.agents.client.AgentError` when the model or a tool's service
    fails — a chain is never written around a failure.
    """
    run_dir = Path(run_dir)
    out_dir = run_dir / STAGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = run_dir / FILTER_DIR / "candidates.json"
    if not candidates_path.exists():
        raise FileNotFoundError(f"stage 3 has not run in {run_dir} ({candidates_path} missing)")
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1, got {top_n}")
    if af_field not in AF_FIELDS:
        raise ValueError(f"af_field must be one of {AF_FIELDS}, got {af_field!r}")
    model = providers.provider_model(provider or "anthropic", model)
    th = {**THRESHOLDS, **(thresholds or {})}

    m = Manifest(stage="reason")
    m.add_input("candidates", candidates_path)
    for name, path in _optional_inputs(run_dir).items():
        m.add_input(name, path)
    m.tools["anthropic-sdk"] = anthropic.__version__

    hpo, hpo_source = resolve_hpo(case_hpo, case_path, run_dir / RANK_DIR / "joined.json")
    if case_path is not None:
        m.add_input("case", case_path, checksum=False)
    candidates = sorted(json.loads(candidates_path.read_text()).get("candidates", []),
                        key=lambda c: (c.get("priority", 1 << 30), str(c.get("candidate_id"))))
    chosen = candidates[:top_n]

    _clear_outputs(out_dir)
    store = EvidenceStore(out_dir / "evidence")  # exists before the index is built, so it is in it
    index = EvidenceIndex.from_run(run_dir, stages=EVIDENCE_STAGES)
    literature = None if dry_run else _literature(http, cache_root, offline)
    disclosure = providers.disclosure(provider or "anthropic", model, effort) if dry_run else None

    m.params.update({
        "top_n": top_n,
        "candidates": [c["candidate_id"] for c in chosen],
        "dry_run": dry_run,
        "provider": provider,
        "model": model,
        "effort": effort,
        "max_turns": max_turns,
        "max_tokens": ac.DEFAULT_MAX_TOKENS,
        "final_max_tokens": ac.DEFAULT_FINAL_MAX_TOKENS,
        "thinking": dict(ac.THINKING),
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": prompt_sha256(),
        "final_instruction_sha256": prompt_sha256(ac.FINAL_INSTRUCTION),
        "tool_budget_error_sha256": prompt_sha256(ac.TOOL_BUDGET_ERROR),
        "instructions_sha256": instructions_sha256(),
        "tools": ["get_record", "search_literature", "get_paper"],
        "output_schema": "EvidenceChain without classification; code pinned to the ACMG codes, key to the bundle's variants",
        "citation_scope": CITATION_SCOPE,
        "evidence_stages": list(EVIDENCE_STAGES),
        "validator": {"thresholds": th, "af_field": af_field, "rules": frequency_rules(th, af_field)},
        "stage_checks": {"bare_accessions_in_prose": "redacted unless carried by a citable record",
                         "paper_backed_codes": ["PS3", "BS3"]},
        "literature": {"default_max_results": literature_max_results, "max_results_cap": MAX_RESULTS_CAP,
                       "abstract_chars": ABSTRACT_CHARS, "peer_reviewed_only": True,
                       "coordinates_in_queries": COORDINATE_RULE},
        "hpo": list(hpo),
        "hpo_source": hpo_source,
        "offline": offline,
        "cache_root": str(_cache_root(cache_root)) if http is None and not dry_run else None,
    })
    m.counts.update({"candidates_total": len(candidates), "candidates_selected": len(chosen),
                     "candidates_reasoned": 0, "chains_written": 0})

    results: list[tuple[Bundle, EvidenceChain, EvidenceIndex]] = []
    usage_total = ac.Usage()
    validation: dict[str, Any] = {}
    written: list[str] = []
    for i, cand in enumerate(chosen, 1):
        cid = str(cand["candidate_id"])
        progress(f"candidate {i}/{len(chosen)}: bundle")
        bundle = build_bundle(cand, run_dir, hpo)
        write_bundle(bundle, out_dir)
        tools = ReasonTools(index, store, literature, citable=bundle.record_ids, default_max_results=literature_max_results)
        request = build_request(bundle, tools, model=model, effort=effort, max_turns=max_turns)
        write_prompt(request, out_dir, cid)
        if dry_run:
            continue
        if client is None:
            client = _client(provider, model, effort, progress)
        progress(f"candidate {i}/{len(chosen)}: agent ({model}, effort {effort})")
        try:
            result = client.run(request)
        except ac.AgentError as e:
            _write_json(out_dir / "transcripts" / f"{_name(cid)}.failed.json", {
                "candidate_id": cid, "error": str(e), "request_id": e.request_id,
                "transcript": [asdict(t) for t in e.transcript], "final_text": e.final_text,
                "tools": tools.log.as_dict(),
            })
            store.write_index()
            raise
        _write_json(out_dir / "transcripts" / f"{_name(cid)}.json", {**result.as_dict(), "tools": tools.log.as_dict()})
        usage_total = _add_usage(usage_total, result.usage)
        written.extend(tools.log.written)

        records = citable_records(bundle, tools, index)
        scoped = EvidenceIndex.from_records(records)
        cleaned, report = check_chain(result.output, bundle, records, scoped, index, af_field=af_field, thresholds=th)
        _write_json(out_dir / "chains" / f"{_name(cid)}.json", cleaned.model_dump())
        _write_json(out_dir / "validation" / f"{_name(cid)}.json", {"candidate_id": cid, **report.as_dict()})
        validation[cid] = _validation_summary(cleaned, report, result, tools)
        for r in report.rejections:
            m.note(f"{cid}: rejected {r.path}: {r.reason}")
        for d in report.disputes:
            m.note(f"{cid}: disputed {d.path}: {d.reason}")
        for n in report.notes:
            m.note(f"{cid}: {n}")
        results.append((bundle, cleaned, scoped))
        disclosure = disclosure or result.disclosure
        progress(f"candidate {i}/{len(chosen)}: {_classification_line(cleaned)} · "
                 f"{report.counts['items_dropped']} rejected · {report.counts['frequency_disputed']} disputed")

    index_path = store.write_index()
    m.add_output("evidence_index", index_path)
    m.counts["evidence_records"] = store.count()
    m.counts["evidence_records_added"] = len(dict.fromkeys(written))
    if dry_run:
        m.note("dry run: bundles and prompts written; no model was called and no chain was produced.")
    else:
        if not chosen:
            m.note("no candidate selected: stage 3 listed none; no model was called.")
        md_path = _write_text(out_dir / "evidence_chain.md",
                              render_document(results, disclosure=disclosure, model=model, effort=effort))
        m.add_output("evidence_chain", md_path)
        m.counts["candidates_reasoned"] = len(results)
        m.counts["chains_written"] = len(results)
        m.counts["usage"] = asdict(usage_total)
        m.counts["classifications"] = {cid: v["classifications"] for cid, v in validation.items()}
        m.counts["validation"] = {cid: v["counts"] for cid, v in validation.items()}
        m.counts["rejections"] = sum(v["counts"]["items_dropped"] + v["counts"]["literature_removed"] + v["counts"]["redactions"]
                                     for v in validation.values())
        m.params["validation"] = {cid: {k: v[k] for k in ("rejections", "disputes", "notes")} for cid, v in validation.items()}
        m.params["tools_used"] = {cid: v["tools"] for cid, v in validation.items()}
        m.params["agent"] = {cid: v["agent"] for cid, v in validation.items()}
    m.params["disclosure"] = disclosure or providers.disclosure(provider or "anthropic", model, effort)
    manifest_path = out_dir / "manifest.json"
    m.write(manifest_path)
    return manifest_path


# ------------------------------------------------------------------------- request

def build_request(bundle: Bundle, tools: ReasonTools, *, model: str, effort: str, max_turns: int) -> ac.AgentRequest:
    """The agent request for one candidate: the frozen system prompt, the bundle as the
    user turn, the three tools, and an answer schema that omits ``classification`` (the
    engine's) and pins ``code`` to the ACMG codes and ``key`` to the bundle's variants,
    so the API itself refuses a respelled or foreign variant."""
    return ac.AgentRequest(
        system=SYSTEM_PROMPT,
        user=user_prompt(bundle, max_turns=max_turns),
        output_model=EvidenceChain,
        tools=tools.specs(),
        max_turns=max_turns,
        model=model,
        effort=effort,
        output_schema=ac.answer_schema(EvidenceChain, drop=ac.ENGINE_FIELDS, enum=_enums(bundle)),
    )


def _enums(bundle: Bundle) -> dict[str, list[str]]:
    enums = {"code": sorted(ALL_CODES)}
    if bundle.variants:  # an empty enum is not a schema
        enums["key"] = [v.key for v in bundle.variants]
    return enums


def write_prompt(request: ac.AgentRequest, stage_dir: Path, candidate_id: str) -> Path:
    """``prompts/<candidate_id>.json`` (the request as built, every text the model
    receives) and ``.md`` (for reading)."""
    doc = {
        "candidate_id": candidate_id,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": prompt_sha256(request.system),
        "instructions_sha256": instructions_sha256(request.system),
        **request.params(),
        "system": request.system,
        "user": request.user,
        "final_instruction": ac.FINAL_INSTRUCTION,
        "tool_budget_error": ac.TOOL_BUDGET_ERROR,
        "tool_definitions": [t.param() for t in request.tools],
        "output_schema": request.output_schema,
    }
    path = _write_json(stage_dir / "prompts" / f"{_name(candidate_id)}.json", doc)
    md = (f"# Prompt — {candidate_id}\n\nmodel: {request.model} · effort: {request.effort} · max_turns: {request.max_turns} · "
          f"tools: {', '.join(t.name for t in request.tools)} · prompt version {PROMPT_VERSION}\n\n"
          f"## System\n\n{request.system}\n\n## User\n\n{request.user}\n\n## Final instruction (added by the client)\n\n"
          f"{ac.FINAL_INSTRUCTION}\n\n## Tool-budget refusal (added by the client)\n\n{ac.TOOL_BUDGET_ERROR}\n")
    _write_text(stage_dir / "prompts" / f"{_name(candidate_id)}.md", md)
    return path


# ------------------------------------------------------------------- after the model

def citable_records(bundle: Bundle, tools: ReasonTools, index: EvidenceIndex) -> list[EvidenceRecord]:
    """The records this candidate's chain may cite: the bundle's and every record a
    tool returned in this conversation. Each one is read from the run's stores — the
    bundle read them, the tools wrote or served them — so a missing one is a bug."""
    records: list[EvidenceRecord] = []
    for rid in tools.citable_ids():
        rec = index.get(rid)
        if rec is None:
            raise RuntimeError(f"record {rid!r} is on the candidate's citable list but not in any store of the run")
        records.append(rec)
    return records


def check_chain(chain: EvidenceChain, bundle: Bundle, records: list[EvidenceRecord], scoped: EvidenceIndex,
                run_index: EvidenceIndex, *, af_field: str, thresholds: dict[str, float]) -> tuple[EvidenceChain, ValidationReport]:
    """Everything between the model's answer and the chain on disk, in order: pin the
    identity to the bundle (:func:`align_chain`), the stage's own checks
    (:func:`~engine.reason.checks.stage_checks`), the shared validator over the
    candidate's scoped index — told the candidate's keys, so its frequency checks look
    up the right variant and it would drop a foreign entry even if the alignment had
    let one through — a note for every rejected id that exists elsewhere in the run,
    and the classification. One report carries all of it."""
    aligned, notes, dropped = align_chain(chain, bundle)
    checks = stage_checks(aligned, AccessionResolver(records))
    cleaned, report = validate(checks.chain, scoped, af_field=af_field, thresholds=thresholds,
                               keys=[v.key for v in bundle.variants])
    report.rejections[:0] = dropped + checks.dropped + checks.redacted
    report.counts["items_dropped"] += len(dropped) + len(checks.dropped)
    for name in ("redactions", "ids_checked", "ids_unknown"):
        report.counts[name] += len(checks.redacted)
    out_of_scope = [rid for rid in cited_ids(aligned, run_index) if rid not in scoped and rid in run_index]
    report.counts["citations_out_of_scope"] = len(out_of_scope)
    report.notes = notes + [f"out of scope: {rid} exists in the run's evidence but was neither in the bundle nor "
                           "returned by a tool in this conversation; rejected" for rid in out_of_scope] + report.notes
    return classify(cleaned), report


def align_chain(chain: EvidenceChain, bundle: Bundle) -> tuple[EvidenceChain, list[str], list[Rejection]]:
    """The model's chain with its identity pinned to the bundle: the candidate id is the
    bundle's; a variant key the model respelled is matched back to the bundle's key by
    the validator's own spelling rule (:func:`~engine.agents.validator.canonical_key`:
    case, ``chr`` prefix, separators); an entry whose key is not a variant of this
    candidate is dropped — with its criteria — and returned as a rejection, so a chain
    never reports a foreign variant as the candidate's and the stage checks never see
    one; a bundle variant the model left out gets an empty entry, so every variant of
    the candidate is classified (as VUS, with nothing to support it) and the omission
    is on record."""
    notes: list[str] = []
    rejections: list[Rejection] = []
    data = chain.model_dump()
    if data["candidate_id"] != bundle.candidate_id:
        notes.append(f"candidate_id: the model wrote {data['candidate_id']!r}; replaced by the bundle's {bundle.candidate_id!r}")
        data["candidate_id"] = bundle.candidate_id
    keys = [v.key for v in bundle.variants]
    by_canonical = {canonical_key(k) or k: k for k in keys}
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for i, v in enumerate(data["variants"]):
        raw = str(v["key"])
        match = by_canonical.get(canonical_key(raw) or raw)
        if match is None:
            rejections.append(Rejection(f"variants[{i}]", f"key {v['key']!r} is not a variant of this candidate "
                                        f"({bundle.candidate_id}: {', '.join(keys)}); the entry and its "
                                        f"{len(v['criteria'])} criteria were dropped"))
            continue
        if match != v["key"]:
            notes.append(f"variants: key {v['key']!r} respelled by the model; matched to the bundle's {match!r}")
            v["key"] = match
        if match in seen:
            rejections.append(Rejection(f"variants[{i}]", f"a second entry for {match!r}; the first is kept"))
            continue
        seen.add(match)
        kept.append(v)
    for k in keys:
        if k not in seen:
            notes.append(f"variants: the model returned no entry for {k!r}; an empty entry was added (no criteria → vus)")
            kept.append(VariantChain(key=k, criteria=[], summary="The model returned no criteria for this variant.").model_dump())
    data["variants"] = sorted(kept, key=lambda v: keys.index(v["key"]))
    return EvidenceChain.model_validate(data), notes, rejections


def classify(chain: EvidenceChain) -> EvidenceChain:
    """Every variant's classification from its surviving criteria (ACMG/AMP 2015
    combining rules). The validator already did this; doing it here too keeps the
    stage's own promise visible and idempotent."""
    for v in chain.variants:
        v.classification = combine_acmg(v.criteria)
    return chain


# --------------------------------------------------------------------------- render

def render_document(chains: list[tuple[Bundle, EvidenceChain, EvidenceIndex]], *,
                    disclosure: str | None, model: str, effort: str) -> str:
    """``evidence_chain.md``: a summary table, then every chain with its own References,
    resolved against the index the chain was validated with."""
    out = ["# Evidence chains", ""]
    out.append(f"{len(chains)} candidate(s) · model {model} · effort {effort} · classification computed by the "
               "engine from the validated criteria (ACMG/AMP 2015). Every claim cites a record id; the References "
               "under each chain resolve them.")
    out.append("")
    out.extend(["| # | candidate | gene | model | variant | classification |", "|---|---|---|---|---|---|"])
    for i, (bundle, chain, _) in enumerate(chains, 1):
        c = bundle.candidate
        for v in chain.variants:
            label = (v.classification or "not computed").replace("_", " ")
            out.append(f"| {i} | {chain.candidate_id} | {c.get('gene_symbol') or '-'} | {c.get('model') or '-'} | {v.key} | {label} |")
    out.append("")
    for _, chain, index in chains:
        out.append("---")
        out.append("")
        out.append(render_evidence_chain(chain, index, disclosure=disclosure).rstrip("\n"))
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


# --------------------------------------------------------------------------- pieces

def resolve_hpo(case_hpo: list[str] | None, case_path: Path | None, joined_path: Path) -> tuple[list[str], str]:
    """HPO terms and where they came from: an explicit list, the case file, stage 4's
    ``joined.json`` (which recorded what Exomiser was given), or nothing."""
    if case_hpo is not None:
        return _hpo(case_hpo), "argument"
    if case_path is not None:
        raw = yaml.safe_load(Path(case_path).read_text()) or {}
        return _hpo(raw.get("hpo") or []), f"case file {case_path}"
    if joined_path.exists():
        doc = json.loads(joined_path.read_text())
        if isinstance(doc, dict) and isinstance(doc.get("hpo"), list):
            return _hpo(doc["hpo"]), "04_rank/joined.json"
    return [], "none"


def _hpo(terms: list[Any]) -> list[str]:
    out = [str(t) for t in terms]
    bad = [t for t in out if not HPO_ID.match(t)]
    if bad:
        raise ValueError(f"not HPO ids (expected HP:0000000): {bad}")
    return out


def _optional_inputs(run_dir: Path) -> dict[str, Path]:
    """Every earlier-stage file the bundle or the index reads when it exists — each one
    shapes what the model sees, so each one is a checksummed input."""
    paths = {
        "retrieve_evidence_index": run_dir / RETRIEVE_DIR / "evidence" / "index.json",
        "filter_shortlist": run_dir / FILTER_DIR / "shortlist.tsv.gz",
        "rank_joined": run_dir / RANK_DIR / "joined.json",
        "rank_evidence_index": run_dir / RANK_DIR / "evidence" / "index.json",
    }
    return {name: p for name, p in paths.items() if p.exists()}


def _clear_outputs(out_dir: Path) -> None:
    """A rerun replaces the stage's outputs — every per-candidate file, the report and
    the manifest — so nothing a manifest does not claim survives beside it. The
    ``evidence/`` store stays: a paper fetched once is a record, whichever run fetched it."""
    for d in OUTPUT_DIRS:
        shutil.rmtree(out_dir / d, ignore_errors=True)
    for name in ("evidence_chain.md", "manifest.json"):
        (out_dir / name).unlink(missing_ok=True)


def _literature(http: Http | None, cache_root: Path | None, offline: bool) -> LiteratureRetriever:
    if http is None:
        http = Http(HttpCache(_cache_root(cache_root)), limiter=RateLimiter(RATE_LIMITS), offline=offline,
                    backoff_floor=BACKOFF_FLOOR, retries=RETRIES)
    return LiteratureRetriever(http)


def _client(provider: str | None, model: str, effort: str, progress: Progress) -> ac.ModelClient:
    """The real client for ``provider`` (``None``: the Anthropic SDK client, which
    resolves credentials on the first request, not here — so a missing key surfaces
    from the first ``client.run``, after the first candidate's bundle and prompt are
    on disk, as an :class:`~engine.agents.client.AgentError`). A missing OpenRouter
    key or a constructor the SDK refuses is an ``AgentError`` from the selector."""
    return providers.select_client(provider or "anthropic", model, effort, log=progress)


def _cache_root(cache_root: Path | None) -> Path:
    return Path(cache_root) if cache_root is not None else default_cache_root()


def _validation_summary(chain: EvidenceChain, report: ValidationReport, result: ac.AgentResult, tools: ReasonTools) -> dict[str, Any]:
    return {
        "classifications": {v.key: v.classification for v in chain.variants},
        "counts": {**report.counts,
                   "criteria_kept": sum(len(v.criteria) for v in chain.variants),
                   "criteria_met": sum(1 for v in chain.variants for c in v.criteria if c.met)},
        "rejections": [asdict(r) for r in report.rejections],
        "disputes": [asdict(d) for d in report.disputes],
        "notes": list(report.notes),
        "tools": tools.log.as_dict(),
        "agent": {"model": result.model, "effort": result.effort, "stop_reason": result.stop_reason,
                  "disclosure": result.disclosure, "usage": asdict(result.usage), "request": dict(result.request)},
    }


def _classification_line(chain: EvidenceChain) -> str:
    return ", ".join(v.classification or "not computed" for v in chain.variants) or "no variants"


def _add_usage(total: ac.Usage, usage: ac.Usage) -> ac.Usage:
    out = ac.Usage(**asdict(total))
    for name, value in asdict(usage).items():
        setattr(out, name, getattr(out, name) + value)
    return out


def _name(candidate_id: str) -> str:
    """File stem for a candidate id (``CFTR:hom`` is a valid file name; a stray slash is not)."""
    return _SAFE_NAME.sub("_", candidate_id) or "candidate"


def _write_json(path: Path, doc: Any) -> Path:
    return _write_text(path, json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False, default=str) + "\n")


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.exists() and path.read_text() == text):
        path.write_text(text)
    return path
