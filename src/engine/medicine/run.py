"""Stage 6 orchestrator — ``engine medicine``.

One candidate per run: the stage-5 chain names the gene, the mechanism hypothesis and
the classification the engine computed; this stage asks the second agent what, if
anything, could act on that mechanism — and holds the answer to the same rule as
stage 5: nothing reaches the report unless a record in the run directory carries it.

The citation scope is per candidate, as in stage 5: the stage-5 bundle's record ids,
every id the validated chain cites (papers in ``05_reason/evidence``, the variant
records, the Exomiser record), and every record the tools return in *this*
conversation — every Open Targets, DGIdb and ChEMBL record ``drugs_for_gene`` wrote,
every trial ``search_trials`` returned, every paper the literature tools fetched. A
drug candidate stands only on a record it cites from a drug source that carries the
drug's name — a drug named from memory, or cited to a record about another drug, is
deleted; a ChEMBL id no record the candidate cites carries is cleared; a bare NCT id,
DOI or PMID in prose that no citable record carries is redacted.

What is written, and why in this form (the layout mirrors stage 5 so a judge reads
both the same way):

* ``bundles/<candidate_id>.json`` — what the model was given: the stage-5 variant
  bundle (records with payloads), the chain as validated, the ids it cites, the case
  HPO terms, and the prompt text built from them.
* ``prompts/<candidate_id>.json`` / ``.md`` — the request as built: system prompt, user
  prompt, tool definitions as sent, the output schema (identity pinned by enum), model,
  effort, turn budget, and the two fixed texts the client adds. Written in every run;
  a ``--dry-run`` stops here.
* ``transcripts/<candidate_id>.json`` — every turn, tool call and result, token usage
  and the final text as the model wrote it; ``.failed.json`` on a failure.
* ``evidence/`` — the drug, trial and paper records the tools fetched, in the stage-2
  format, so anything the report cites is a record like any other.
* ``report.json`` — the validated :class:`~engine.agents.schema.MedicineReport`;
  ``validation/<candidate_id>.json`` — every rejection and redaction;
  ``report.md`` — the report rendered in the rubric's order (mechanism → candidates
  with counter-arguments → follow-up → limits) with References; ``manifest.json``.

A run replaces the stage's outputs (the per-candidate directories, the report, the
manifest) before it starts; ``evidence/`` is a store and is kept — a record fetched
once is served from there on a rerun. Bundles, prompts, transcripts and the report
sort their keys and carry no clock; with a scripted client and a warm cache a rerun is
byte-identical except for the manifest's own timestamps.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import anthropic
import pydantic

from engine.agents import client as ac
from engine.agents import providers
from engine.agents.bundle import Bundle, VariantBundle, build_bundle, find_candidate
from engine.agents.render import cited_ids, render_medicine_report
from engine.agents.schema import EvidenceChain, MedicineReport, combine_acmg
from engine.agents.validator import EvidenceIndex, Rejection, ValidationReport, canonical_id, validate
from engine.manifest import Manifest
from engine.medicine.chembl import ChemblRetriever
from engine.medicine.ctgov import TrialsRetriever
from engine.medicine.dgidb import DgidbRetriever
from engine.medicine.opentargets import OpenTargetsRetriever
from engine.medicine.prompts import PROMPT_VERSION, SYSTEM_PROMPT, instructions_sha256, prompt_sha256, user_prompt
from engine.medicine.tools import (DEFAULT_TRIALS, DISEASES_PER_GENE, FIELD_CHARS, MAX_GENES, MAX_TRIALS, MedicineRetrievers,
                                   MedicineTools)
from engine.reason.run import resolve_hpo
from engine.reason.tools import ABSTRACT_CHARS, COORDINATE_RULE, DEFAULT_MAX_RESULTS, MAX_RESULTS_CAP
from engine.retrieve.http import Http, HttpCache, RateLimiter, default_cache_root
from engine.retrieve.literature import LiteratureRetriever
from engine.retrieve.run import BACKOFF_FLOOR, RATE_LIMITS, RETRIES
from engine.retrieve.store import EvidenceRecord, EvidenceStore

STAGE_DIR = "06_medicine"
REASON_DIR = "05_reason"
FILTER_DIR = "03_filter"
RANK_DIR = "04_rank"
RETRIEVE_DIR = "02_retrieve"
EVIDENCE_STAGES = (RETRIEVE_DIR, RANK_DIR, REASON_DIR, STAGE_DIR)
"""Where a record this stage may serve or cite can live: the stages before it and its
own store."""
OUTPUT_DIRS = ("bundles", "prompts", "transcripts", "validation")
DEFAULT_MAX_TURNS = 10
CITATION_SCOPE = ("the stage-5 bundle's record ids, the ids the validated chain cites, and the records returned by "
                  "this candidate's own tool calls")
DRUG_RECORD_SOURCES = ("opentargets:drug:", "dgidb:", "chembl:", "nct:", "pmid:")
"""Where a record *about a drug* can come from: a drug row, an interaction, a ChEMBL
object, a trial or a paper. A drug candidate must cite one of these that carries the
drug's name (see :meth:`AccessionResolver.carries_name`) — a record about another drug,
a paper about the gene alone or the variant records cannot hold a candidate up."""
NAME_RULE = ("the candidate's name, lower-cased — or each of its '/'-separated components, a parenthesised alias "
             "counting for its component — must occur in the id, URL or payload of a record it cites from a drug source")
"""The rule as the manifest records it (``params.stage_checks.drug_candidate``)."""
REDACTED = "[citation removed: no such record]"
MIN_NAME_CHARS = 3
"""A name (or alias) shorter than this matches any corpus by accident and counts for nothing."""

Progress = Callable[[str], None]

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._:-]+")
_CHEMBL = re.compile(r"^CHEMBL\d+$")
_BARE_NCT = re.compile(r"^(?:nct:)?(NCT\d{8})$", re.IGNORECASE)
"""The validator's spelling rule for ``trial_ids``: a bare ``NCT01807923`` is ``nct:NCT01807923``."""
_PAREN = re.compile(r"\(([^()]*)\)")
_SPACE = re.compile(r"\s+")
_ACCESSION = re.compile(
    r"(?<![\w:/.-])(?P<aux>(?:nct-search|pmid-search|dgidb-gene):[A-Za-z0-9_-]+)"
    r"|(?P<url>(?:https?://)?(?:www\.)?(?:"
    r"europepmc\.org/(?:article|abstract)/MED/(?P<url_pmid>\d+)"
    r"|pubmed\.ncbi\.nlm\.nih\.gov/(?P<url_pmid2>\d+)"
    r"|ncbi\.nlm\.nih\.gov/pubmed/(?P<url_pmid3>\d+)"
    r"|ncbi\.nlm\.nih\.gov/clinvar/variation/(?P<url_vcv>\d+)"
    r"|clinicaltrials\.gov/(?:study|ct2/show)/(?P<url_nct>NCT\d{8})"
    r"|ebi\.ac\.uk/chembl/(?:explore/)?(?:compound|target)(?:_report_card)?/(?P<url_chembl>CHEMBL\d+)"
    r"|platform\.opentargets\.org/(?:drug|target|evidence)/(?P<url_ot>[A-Za-z0-9_]+)"
    r"|dgidb\.org/(?:genes|drugs)/(?P<url_dgidb>[A-Za-z0-9:._-]+)"
    r")[^\s\]\)>\"']*)"
    r"|(?<![\w:/.])(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(?P<doi>10\.\d{4,9}/[^\s\[\]()<>\"']+)"
    r"|(?<![\w:/.])pubmed\s*(?:id)?\s*:?\s*(?P<pubmed>\d{4,9})\b"
    r"|(?<![\w:/.])(?P<cv_prefix>VCV|SCV|RCV)0*(?P<cv_number>\d{1,9})\b"
    r"|(?<![\w:/.])rs(?P<rs>\d{3,})\b"
    r"|(?<![\w:/.])NCT(?P<nct>\d{8})\b"
    r"|(?<![\w:/.])CHEMBL(?P<chembl>\d+)\b"
    r"|(?<![\w:/.])PMC(?P<pmc>\d{4,})\b",
    re.IGNORECASE,
)
"""Bare accessions a medicine report might carry without a record: stage 5's set plus
ChEMBL ids, the drug databases' URLs, and the ids of the engine's own auxiliary records
(``nct-search:``, ``pmid-search:``, ``dgidb-gene:``), which the validator treats as
citations only while a record of that source is in scope."""
_TRAIL = ".,;:"


# ------------------------------------------------------------------------------ run

def run_medicine(
    run_dir: Path,
    candidate_id: str | None = None,
    client: ac.ModelClient | None = None,
    model: str | None = None,
    effort: str = ac.DEFAULT_EFFORT,
    dry_run: bool = False,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    cache_root: Path | None = None,
    offline: bool = False,
    http: Http | None = None,
    retrievers: MedicineRetrievers | None = None,
    case_hpo: list[str] | None = None,
    case_path: Path | None = None,
    literature_max_results: int = DEFAULT_MAX_RESULTS,
    max_genes: int = MAX_GENES,
    trials_max_results: int = DEFAULT_TRIALS,
    provider: str | None = None,
    progress: Progress = lambda s: None,
) -> Path:
    """Run stage 6 into ``run_dir/06_medicine`` for one candidate; returns the manifest
    path. ``candidate_id`` defaults to the stage-5 chain of the best stage-3 priority.
    ``client`` defaults to the ``provider``'s client (``engine.agents.providers``;
    ``None`` is the Anthropic SDK client), built when needed (never in a dry run);
    ``model`` defaults to the provider's and is settled by ``providers.provider_model``
    (see stage 5); ``retrievers`` (tests) or ``http``/``cache_root``/
    ``offline`` shape the drug, trial and literature services. Raises
    :class:`~engine.agents.client.AgentError` when the model or a tool's service fails
    — a report is never written around a failure."""
    run_dir = Path(run_dir)
    out_dir = run_dir / STAGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    chains_dir = run_dir / REASON_DIR / "chains"
    if not chains_dir.is_dir():
        raise FileNotFoundError(f"stage 5 has not run in {run_dir} ({chains_dir} missing)")
    if max_turns < 1:
        raise ValueError(f"max_turns must be at least 1, got {max_turns}")
    model = providers.provider_model(provider or "anthropic", model)
    cid, chain_path = select_candidate(run_dir, candidate_id)
    chain = load_chain(chain_path)
    candidate = find_candidate(run_dir, cid)

    m = Manifest(stage="medicine")
    m.add_input("chain", chain_path)
    m.add_input("candidates", run_dir / FILTER_DIR / "candidates.json")
    for name, path in _optional_inputs(run_dir).items():
        m.add_input(name, path)
    m.tools["anthropic-sdk"] = anthropic.__version__
    hpo, hpo_source = resolve_hpo(case_hpo, case_path, run_dir / RANK_DIR / "joined.json")
    if case_path is not None:
        m.add_input("case", case_path, checksum=False)

    _clear_outputs(out_dir)
    store = EvidenceStore(out_dir / "evidence")  # exists before the index is built, so it is in it
    index = EvidenceIndex.from_run(run_dir, stages=EVIDENCE_STAGES)
    http_used: Http | None = None  # the manifest describes the client the sources actually ran on
    if not dry_run and retrievers is None:
        http_used = _http(http, cache_root, offline)
        retrievers = default_retrievers(http_used)
    disclosure = providers.disclosure(provider or "anthropic", model, effort) if dry_run else None

    m.params.update({
        "candidate": cid,
        "candidate_requested": candidate_id,
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
        "tools": ["get_record", "search_literature", "get_paper", "drugs_for_gene", "search_trials"],
        "output_schema": "MedicineReport; candidate_id and gene_symbol pinned to the candidate's",
        "citation_scope": CITATION_SCOPE,
        "evidence_stages": list(EVIDENCE_STAGES),
        "stage_checks": {"drug_candidate": f"dropped unless {NAME_RULE}", "drug_record_sources": list(DRUG_RECORD_SOURCES),
                         "min_name_chars": MIN_NAME_CHARS,
                         "chembl_id": "cleared unless a record the candidate itself cites carries it",
                         "bare_accessions_in_prose": "redacted unless carried by a citable record",
                         "id_spelling": "as the validator: source prefix case-folded, whitespace stripped, a bare NCT id read as nct:NCT…"},
        "drugs": {"max_genes": max_genes, "sources": ["opentargets", "dgidb", "chembl"], "field_chars": FIELD_CHARS,
                  "opentargets_diseases_per_gene": DISEASES_PER_GENE},
        "trials": {"default_max_results": trials_max_results, "max_results_cap": MAX_TRIALS, "field_chars": FIELD_CHARS,
                   "coordinates_in_queries": COORDINATE_RULE},
        "literature": {"default_max_results": literature_max_results, "max_results_cap": MAX_RESULTS_CAP,
                       "abstract_chars": ABSTRACT_CHARS, "peer_reviewed_only": True,
                       "coordinates_in_queries": COORDINATE_RULE},
        "retrievers": retriever_params(retrievers),
        "hpo": list(hpo),
        "hpo_source": hpo_source,
        "offline": bool(getattr(http_used, "offline", offline)),
        "cache_root": _cache_root_of(http_used),
    })
    m.counts.update({"chains_available": len(list_chains(run_dir)), "candidates_selected": 1, "reports_written": 0})

    progress("candidate: bundle")
    bundle = build_medicine_bundle(candidate, chain, run_dir, hpo, index)
    write_bundle(bundle, out_dir)
    for n in bundle.notes:
        m.note(f"{cid}: {n}")
    tools = MedicineTools(index, store, retrievers, gene_symbol=bundle.gene_symbol, gene_id=bundle.gene_id,
                          citable=bundle.record_ids, default_max_results=literature_max_results,
                          max_genes=max_genes, default_trials=trials_max_results)
    request = build_request(bundle, tools, model=model, effort=effort, max_turns=max_turns)
    write_prompt(request, out_dir, cid)
    if bundle.missing_ids:
        m.note(f"{cid}: the chain cites {len(bundle.missing_ids)} id(s) no store of the run holds; not citable here: "
               + ", ".join(bundle.missing_ids))

    if dry_run:
        index_path = store.write_index()
        m.add_output("evidence_index", index_path)
        m.counts["evidence_records"] = store.count()
        m.counts["evidence_records_added"] = 0
        m.note("dry run: bundle and prompt written; no model was called and no report was produced.")
        m.params["disclosure"] = disclosure
        manifest_path = out_dir / "manifest.json"
        m.write(manifest_path)
        return manifest_path

    if client is None:
        client = _client(provider, model, effort, progress)
    progress(f"candidate: agent ({model}, effort {effort})")
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
    except TypeError as e:
        # The SDK resolves credentials on the first request, not in the constructor,
        # and reports a missing key as a TypeError; anything else is a real bug.
        if "authentication" not in str(e).lower():
            raise
        raise ac.AgentError(f"could not call the Anthropic API: {e}") from e
    _write_json(out_dir / "transcripts" / f"{_name(cid)}.json", {**result.as_dict(), "tools": tools.log.as_dict()})

    records = citable_records(tools, index)
    scoped = EvidenceIndex.from_records(records)
    cleaned, report = check_report(result.output, bundle, records, scoped, index)
    _write_json(out_dir / "report.json", cleaned.model_dump())
    _write_json(out_dir / "validation" / f"{_name(cid)}.json", {"candidate_id": cid, **report.as_dict()})
    for r in report.rejections:
        m.note(f"{cid}: rejected {r.path}: {r.reason}")
    for n in report.notes:
        m.note(f"{cid}: {n}")
    md_path = _write_text(out_dir / "report.md", render_document(bundle, cleaned, scoped, disclosure=result.disclosure,
                                                                  model=model, effort=effort))
    index_path = store.write_index()

    m.add_output("report_json", out_dir / "report.json")
    m.add_output("report_md", md_path)
    m.add_output("evidence_index", index_path)
    m.counts.update({
        "reports_written": 1,
        "evidence_records": store.count(),
        "evidence_records_added": len(dict.fromkeys(tools.log.written)),
        "usage": asdict(result.usage),
        "mechanism_claims": len(cleaned.mechanism),
        "pathway_targets": len(cleaned.pathway_targets),
        "drug_candidates": len(cleaned.candidates),
        "drug_candidates_with_trials": sum(1 for d in cleaned.candidates if d.trial_ids),
        "follow_up_experiments": len(cleaned.follow_up_experiments),
        "validation": dict(report.counts),
        "rejections": report.counts["items_dropped"] + report.counts["literature_removed"] + report.counts["redactions"],
        "tool_calls": dict(_tool_call_counts(result)),
    })
    m.params["validation"] = {"rejections": [asdict(r) for r in report.rejections], "notes": list(report.notes)}
    m.params["tools_used"] = tools.log.as_dict()
    m.params["source_versions"] = source_versions(records)
    m.params["agent"] = {"model": result.model, "effort": result.effort, "stop_reason": result.stop_reason,
                         "disclosure": result.disclosure, "usage": asdict(result.usage), "request": dict(result.request)}
    m.params["disclosure"] = result.disclosure
    progress(f"candidate: {len(cleaned.candidates)} drug candidate(s) kept · {len(cleaned.mechanism)} mechanism claim(s) · "
             f"{m.counts['rejections']} rejected")
    manifest_path = out_dir / "manifest.json"
    m.write(manifest_path)
    return manifest_path


# ------------------------------------------------------------------------ selection

def list_chains(run_dir: Path) -> dict[str, Path]:
    """``candidate_id → chains/<file>`` for every stage-5 chain, in stage-3 priority
    order (a chain whose candidate stage 3 does not list sorts last, by id)."""
    chains_dir = Path(run_dir) / REASON_DIR / "chains"
    found: dict[str, Path] = {}
    for p in sorted(chains_dir.glob("*.json")) if chains_dir.is_dir() else []:
        doc = json.loads(p.read_text())
        cid = doc.get("candidate_id") if isinstance(doc, dict) else None
        if isinstance(cid, str) and cid and cid not in found:
            found[cid] = p
    priority = _priorities(run_dir)
    return dict(sorted(found.items(), key=lambda kv: (priority.get(kv[0], 1 << 30), kv[0])))


def select_candidate(run_dir: Path, candidate_id: str | None) -> tuple[str, Path]:
    """The candidate this run reports on and its chain file: the one asked for, else
    the chain of best stage-3 priority. Raises when stage 5 wrote no chain, or none
    for the candidate asked."""
    chains = list_chains(run_dir)
    if not chains:
        raise FileNotFoundError(f"stage 5 wrote no chain in {Path(run_dir) / REASON_DIR / 'chains'}")
    if candidate_id is None:
        cid = next(iter(chains))
        return cid, chains[cid]
    if candidate_id not in chains:
        raise KeyError(f"no stage-5 chain for candidate {candidate_id!r}; chains exist for {len(chains)} candidate(s)")
    return candidate_id, chains[candidate_id]


def load_chain(path: Path) -> EvidenceChain:
    """The stage-5 chain file as an :class:`~engine.agents.schema.EvidenceChain`. A
    file that does not fit is a ``ValueError`` naming field paths and error types —
    never values, which could be a variant key on a terminal."""
    try:
        return EvidenceChain.model_validate(json.loads(Path(path).read_text()))
    except pydantic.ValidationError as e:
        where = "; ".join(f"{'.'.join(str(x) for x in err['loc']) or '<root>'}: {err['type']}" for err in e.errors()[:6])
        raise ValueError(f"the stage-5 chain for the selected candidate does not fit EvidenceChain: "
                         f"{e.error_count()} error(s) — {where}") from e


def _priorities(run_dir: Path) -> dict[str, int]:
    path = Path(run_dir) / FILTER_DIR / "candidates.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    out: dict[str, int] = {}
    for i, c in enumerate(doc.get("candidates", []) if isinstance(doc, dict) else []):
        if isinstance(c, dict) and isinstance(c.get("candidate_id"), str):
            out[c["candidate_id"]] = int(c.get("priority", i + 1))
    return out


# --------------------------------------------------------------------------- bundle

@dataclass
class MedicineBundle:
    """What the medicine agent is given: the stage-5 bundle for the candidate's
    variants, the validated chain, and the citation vocabulary the two define."""

    candidate_id: str
    gene_symbol: str
    gene_id: str
    candidate: dict[str, Any]
    chain: dict[str, Any]
    variants: list[dict[str, Any]]
    """The stage-5 :class:`~engine.agents.bundle.VariantBundle` entries (records with payloads)."""
    rank: dict[str, Any] | None
    case_hpo: list[str]
    gene_records: list[dict[str, Any]]
    chain_record_ids: list[str]
    """Ids the chain cites that a store of the run holds."""
    missing_ids: list[str]
    """Ids the chain cites that no store holds — visible, never silently citable."""
    record_ids: list[str]
    """The citation vocabulary before any tool call."""
    text: str = ""
    notes: list[str] = field(default_factory=list)
    """What the bundle changed about the chain as read (a classification recomputed)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def build_medicine_bundle(candidate: dict[str, Any], chain: EvidenceChain, run_dir: Path, case_hpo: Iterable[str],
                          index: EvidenceIndex) -> MedicineBundle:
    """The stage-5 bundle (:func:`~engine.agents.bundle.build_bundle`) plus the chain:
    the vocabulary is the bundle's ids and every id the chain cites that the run holds.
    The classification shown per variant is recomputed from the chain's own criteria
    (ACMG/AMP combining rules); a chain file that disagrees with its criteria is used
    at the engine's value and the disagreement is noted."""
    base: Bundle = build_bundle(candidate, run_dir, case_hpo)
    chain_data = chain.model_dump()
    notes = _recompute_classifications(chain, chain_data)
    cited = cited_ids(chain_data, index)
    present = [i for i in cited if i in index]
    missing = [i for i in cited if i not in index]
    bundle = MedicineBundle(
        candidate_id=base.candidate_id,
        gene_symbol=str(candidate.get("gene_symbol") or ""),
        gene_id=str(candidate.get("gene_id") or ""),
        candidate=base.candidate,
        chain=json.loads(json.dumps(chain_data, sort_keys=True)),
        variants=[asdict(v) for v in base.variants],
        rank=base.rank,
        case_hpo=base.case_hpo,
        gene_records=base.gene_records,
        chain_record_ids=present,
        missing_ids=missing,
        record_ids=sorted(set(base.record_ids) | set(present)),
        notes=notes,
    )
    bundle.text = render_text(bundle)
    return bundle


def _recompute_classifications(chain: EvidenceChain, chain_data: dict[str, Any]) -> list[str]:
    """Set every variant's ``classification`` in ``chain_data`` to what its criteria
    combine to; one note per variant whose file said otherwise."""
    notes: list[str] = []
    for i, (v, d) in enumerate(zip(chain.variants, chain_data["variants"])):
        computed = combine_acmg(v.criteria)
        if v.classification != computed:
            notes.append(f"chain variants[{i}].classification: the file says {v.classification!r}, its criteria combine "
                         f"to {computed!r} (ACMG/AMP 2015 rules); the engine's value is used")
            d["classification"] = computed
    return notes


def write_bundle(bundle: MedicineBundle, stage_dir: Path) -> Path:
    return _write_text(Path(stage_dir) / "bundles" / f"{_name(bundle.candidate_id)}.json", bundle.to_json())


def render_text(bundle: MedicineBundle) -> str:
    """The prompt block: the candidate, one section per variant with its stage-5
    verdict and criteria, the chain's reasoning, and the citable list. Every fact is
    followed by the record id it came from."""
    c = bundle.candidate
    chain = bundle.chain
    lines = [
        f"# Candidate {bundle.candidate_id}",
        (f"gene: {bundle.gene_symbol or '-'} ({bundle.gene_id or 'no gene id'}) · model: {c.get('model') or '-'} · "
         f"priority: {c.get('priority', '-')}"),
        f"phase: {_phase(c.get('phase'))}",
        f"caveats: {'; '.join(c.get('caveats') or []) or 'none'}",
        f"exomiser: {_rank_line(bundle.rank)}" + _tag([r["record_id"] for r in bundle.gene_records]),
        f"case HPO: {', '.join(bundle.case_hpo) or 'none given'}",
    ]
    verdicts = {v["key"]: v for v in chain.get("variants", [])}
    for v in bundle.variants:
        lines.append("")
        lines.extend(_variant_lines(VariantBundle(**v), verdicts.get(v["key"])))
    lines.extend(["", "## Stage-5 reasoning (validated; classification computed by the engine)"])
    lines.append(f"mechanism hypothesis: {_one_line(chain.get('mechanism_hypothesis'))}")
    lines.append(f"phase statement: {_one_line(chain.get('phase_statement'))}")
    lines.append("limits: " + ("; ".join(_one_line(x) for x in chain.get("limits") or []) or "none stated"))
    lines.append("what would change the call: " + ("; ".join(_one_line(x) for x in chain.get("what_would_change_the_call") or []) or "none stated"))
    lines.append("literature: " + (" ".join(f"[{p}]" for p in chain.get("literature") or []) or "none"))
    if bundle.missing_ids:
        lines.append("cited by the chain but not in any store (not citable): " + ", ".join(bundle.missing_ids))
    lines.append("")
    lines.append("citable record ids: " + (" ".join(f"[{r}]" for r in bundle.record_ids) or "none"))
    return "\n".join(lines) + "\n"


def _variant_lines(v: VariantBundle, verdict: dict[str, Any] | None) -> list[str]:
    get = _getter(v)
    by_source: dict[str, list[str]] = {}
    for r in v.records:
        by_source.setdefault(r["source"], []).append(r["record_id"])
    label = (verdict or {}).get("classification") or "not classified"
    lines = [f"## Variant {v.key} — {str(label).replace('_', ' ')}"]
    mane = get("mane")
    lines.append(f"genotype {get('gt')} · {get('consequence')} ({get('impact')}) · {get('transcript_id')}"
                 f"{' (MANE ' + mane + ')' if mane != '-' else ''} · {get('hgvsc')} · {get('hgvsp')}{_tag(by_source.get('vep'))}")
    gnomad = (f"gnomAD af {_num(get('gnomad_af'))} · hom {get('gnomad_nhom')}{_tag(by_source.get('gnomad'))}"
              if by_source.get("gnomad") else f"gnomAD: no record (stage-3 AF {_num(get('af_used'))} from {get('af_source')})")
    clinvar = (f"ClinVar {get('clinvar_clnsig') if get('clinvar_clnsig') != '-' else get('clinvar_pathogenicity')} "
               f"({get('clinvar_stars')} stars) {get('clinvar_vcv')}{_tag(by_source.get('clinvar'))}"
               if by_source.get("clinvar") else "ClinVar: no record")
    lines.append(f"{gnomad} · {clinvar}")
    if verdict is not None:
        met = [c for c in verdict.get("criteria", []) if c.get("met")]
        not_met = [c["code"] for c in verdict.get("criteria", []) if not c.get("met")]
        lines.append("criteria met: " + ("; ".join(f"{c['code']} ({c['strength']}){_tag(c.get('evidence_ids'))}" for c in met) or "none")
                     + (f" · not met: {', '.join(not_met)}" if not_met else ""))
        if str(verdict.get("summary") or "").strip():
            lines.append(f"stage-5 summary: {_one_line(verdict['summary'])}")
    lines.append("records: " + (" ".join(f"[{r['record_id']}]" for r in v.records) or "none in the store"))
    return lines


def _getter(v: VariantBundle) -> Callable[[str], str]:
    def get(name: str) -> str:
        for src in (v.columns, v.fields):
            val = src.get(name)
            if val not in (None, "", [], {}):
                return str(val)
        return "-"
    return get


def _phase(p: Any) -> str:
    if not isinstance(p, dict):
        return "-"
    status = p.get("status") or "-"
    ev = p.get("evidence")
    return f"{status} — {ev}" if ev else str(status)


def _rank_line(rank: dict[str, Any] | None) -> str:
    if rank is None:
        return "no stage-4 output"
    if rank.get("exomiser_rank") is None:
        return "not ranked by Exomiser"
    parts = [f"rank {rank['exomiser_rank']}"]
    for name, label in (("exomiser_score", "combined"), ("phenotype_score", "phenotype"), ("moi", "moi")):
        if rank.get(name) is not None:
            parts.append(f"{label} {rank[name]}")
    return " · ".join(parts)


# ------------------------------------------------------------------------- request

def build_request(bundle: MedicineBundle, tools: MedicineTools, *, model: str, effort: str, max_turns: int) -> ac.AgentRequest:
    """The agent request: the frozen system prompt, the bundle as the user turn, the
    five tools, and an answer schema that pins ``candidate_id`` and ``gene_symbol`` to
    the candidate's, so the API itself refuses a report about another gene."""
    return ac.AgentRequest(
        system=SYSTEM_PROMPT,
        user=user_prompt(bundle, max_turns=max_turns),
        output_model=MedicineReport,
        tools=tools.specs(),
        max_turns=max_turns,
        model=model,
        effort=effort,
        output_schema=ac.answer_schema(MedicineReport, enum={"candidate_id": [bundle.candidate_id],
                                                             "gene_symbol": [bundle.gene_symbol]}),
    )


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
    path = _write_json(Path(stage_dir) / "prompts" / f"{_name(candidate_id)}.json", doc)
    md = (f"# Prompt — {candidate_id}\n\nmodel: {request.model} · effort: {request.effort} · max_turns: {request.max_turns} · "
          f"tools: {', '.join(t.name for t in request.tools)} · prompt version {PROMPT_VERSION}\n\n"
          f"## System\n\n{request.system}\n\n## User\n\n{request.user}\n\n## Final instruction (added by the client)\n\n"
          f"{ac.FINAL_INSTRUCTION}\n\n## Tool-budget refusal (added by the client)\n\n{ac.TOOL_BUDGET_ERROR}\n")
    _write_text(Path(stage_dir) / "prompts" / f"{_name(candidate_id)}.md", md)
    return path


# ------------------------------------------------------------------- after the model

def citable_records(tools: MedicineTools, index: EvidenceIndex) -> list[EvidenceRecord]:
    """The records this report may cite: the candidate's list and every record a tool
    returned in this conversation, each read from the run's stores."""
    records: list[EvidenceRecord] = []
    for rid in tools.citable_ids():
        rec = index.get(rid)
        if rec is None:
            raise RuntimeError(f"record {rid!r} is on the candidate's citable list but not in any store of the run")
        records.append(rec)
    return records


def check_report(output: Any, bundle: MedicineBundle, records: list[EvidenceRecord], scoped: EvidenceIndex,
                 run_index: EvidenceIndex) -> tuple[MedicineReport, ValidationReport]:
    """Everything between the model's answer and the report on disk, in order: pin the
    identity to the bundle (:func:`align_report`), the stage's own checks
    (:func:`stage_checks`), the shared validator over the candidate's scoped index, a
    note for every rejected id that exists elsewhere in the run. Every path in the
    report names the model's own positions."""
    aligned, notes = align_report(output, bundle)
    checks = stage_checks(aligned, records)
    cleaned, report = validate(checks.report, scoped, model=MedicineReport)
    for r in report.rejections + report.disputes:  # the validator saw the list after the stage's drops
        r.path = _original_path(r.path, checks.positions)
    report.rejections[:0] = checks.dropped + checks.redacted
    report.counts["items_dropped"] += len(checks.dropped)
    for name in ("redactions", "ids_checked", "ids_unknown"):
        report.counts[name] += len(checks.redacted)
    out_of_scope = [rid for rid in cited_ids(aligned, run_index) if rid not in scoped and rid in run_index]
    report.counts["citations_out_of_scope"] = len(out_of_scope)
    report.notes = notes + [f"out of scope: {rid} exists in the run's evidence but was neither in the candidate's list "
                           "nor returned by a tool in this conversation; rejected" for rid in out_of_scope] + report.notes
    return cleaned, report


def align_report(output: Any, bundle: MedicineBundle) -> tuple[dict[str, Any], list[str]]:
    """The model's report as a dict with its identity pinned to the bundle: the
    candidate id and the gene symbol are the bundle's, and a change is noted."""
    data: dict[str, Any] = output.model_dump() if hasattr(output, "model_dump") else copy.deepcopy(dict(output))
    notes: list[str] = []
    for key, value in (("candidate_id", bundle.candidate_id), ("gene_symbol", bundle.gene_symbol)):
        if data.get(key) != value:
            notes.append(f"{key}: the model wrote {data.get(key)!r}; replaced by the candidate's {value!r}")
            data[key] = value
    return data, notes


@dataclass
class StageChecks:
    """What the checks did: the report as it goes on to the validator, the drug
    candidates dropped (no cited drug-source record names the drug), the accessions
    redacted and the ChEMBL ids cleared — one :class:`~engine.agents.validator.Rejection`
    each, path as the validator spells it — and, for every surviving candidate, its
    position in the model's own list."""

    report: dict[str, Any]
    dropped: list[Rejection] = field(default_factory=list)
    redacted: list[Rejection] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)


class AccessionResolver:
    """Whether a bare accession, a ChEMBL id or a drug name is carried by one of a set
    of records — as a record's id, its URL, or in its payload. Built over the report's
    whole citable list for prose, and over one candidate's own cited records for its
    ``name`` and ``chembl_id``."""

    def __init__(self, records: Iterable[EvidenceRecord]):
        records = list(records)
        self.ids = {r.record_id for r in records}
        self.corpus = "\n".join(f"{r.record_id}\n{r.url}\n{json.dumps(r.payload, ensure_ascii=False)}" for r in records).lower()

    def carries_chembl(self, chembl_id: str) -> bool:
        return f"chembl:{chembl_id}" in self.ids or chembl_id.lower() in self.corpus

    def carries_name(self, name: Any) -> bool:
        """:data:`NAME_RULE`: the whole name, or every ``/``-separated component (a
        combination product), where a component counts if it, its text outside
        parentheses, or any parenthesised alias occurs in the corpus."""
        key = _squash(name)
        if len(key) >= MIN_NAME_CHARS and key in self.corpus:
            return True
        parts = [x for x in (_squash(x) for x in key.split("/")) if x]
        return bool(parts) and all(self._carries_part(x) for x in parts)

    def _carries_part(self, part: str) -> bool:
        alternatives = [part, _PAREN.sub(" ", part), *_PAREN.findall(part)]
        return any(len(alt) >= MIN_NAME_CHARS and alt in self.corpus for alt in (_squash(x) for x in alternatives))

    def resolves(self, m: re.Match[str]) -> bool:
        if m.group("aux"):
            return canonical_id(m.group("aux")) in self.ids
        pmid = m.group("url_pmid") or m.group("url_pmid2") or m.group("url_pmid3") or m.group("pubmed")
        if pmid:
            return f"pmid:{int(pmid)}" in self.ids
        if m.group("url_vcv"):
            return f"clinvar:VCV{int(m.group('url_vcv')):09d}" in self.ids
        if m.group("url_nct"):
            return f"nct:{m.group('url_nct').upper()}" in self.ids
        if m.group("cv_prefix"):
            acc = f"{m.group('cv_prefix').upper()}{int(m.group('cv_number')):09d}"
            return f"clinvar:{acc}" in self.ids or acc.lower() in self.corpus
        if m.group("nct"):
            return f"nct:NCT{m.group('nct')}" in self.ids or f"nct{m.group('nct')}" in self.corpus
        if m.group("chembl") or m.group("url_chembl"):
            return self.carries_chembl((m.group("url_chembl") or f"CHEMBL{m.group('chembl')}").upper())
        if m.group("url_ot") or m.group("url_dgidb"):
            return m.group(0).rstrip(_TRAIL).lower() in self.corpus
        token = m.group("doi") or (f"rs{m.group('rs')}" if m.group("rs") else f"pmc{m.group('pmc')}")
        return token.rstrip(_TRAIL).lower() in self.corpus


def stage_checks(data: dict[str, Any], records: Iterable[EvidenceRecord]) -> StageChecks:
    """Drop every drug candidate that cites no drug-source record carrying its name;
    clear a ``chembl_id`` no record the candidate cites carries; redact every
    unresolvable accession in prose. Ids are read as the validator spells them, so a
    bare ``NCT…`` in ``trial_ids`` or a ``PMID:`` prefix counts; whether every id
    resolves is still the validator's call (it sees the list after these drops)."""
    records = list(records)
    by_id = {r.record_id: r for r in records}
    resolver = AccessionResolver(records)
    out = StageChecks(copy.deepcopy(data))
    d = out.report
    for name in ("mechanism", "pathway_targets"):
        for i, claim in enumerate(_dicts(d.get(name))):
            claim["statement"] = _redact(claim.get("statement"), f"{name}[{i}].statement", resolver, out.redacted)
    kept: list[Any] = []
    for i, cand in enumerate(d.get("candidates") if isinstance(d.get("candidates"), list) else []):
        path = f"candidates[{i}]"
        if not isinstance(cand, dict):
            out.positions.append(i)
            kept.append(cand)  # not this stage's call; the validator's schema check raises on it
            continue
        own = [by_id[x] for x in cited_record_ids(cand) if x in by_id]
        drug_records = [r for r in own if r.record_id.startswith(DRUG_RECORD_SOURCES)]
        if not AccessionResolver(drug_records).carries_name(cand.get("name")):
            out.dropped.append(Rejection(path, f"drug candidate {_squash(cand.get('name'))!r} cites no record from a drug "
                                               "source that names it (a drug row, an interaction, a ChEMBL molecule, a "
                                               "trial or a paper carrying the name)"))
            continue
        chembl = str(cand.get("chembl_id") or "").strip().upper()
        if chembl and not (_CHEMBL.match(chembl) and AccessionResolver(own).carries_chembl(chembl)):
            out.redacted.append(Rejection(f"{path}.chembl_id", f"chembl_id not carried by any record the candidate cites: "
                                                                f"{cand.get('chembl_id')}; cleared"))
            cand["chembl_id"] = None
        elif chembl:
            cand["chembl_id"] = chembl
        for key in ("mechanism_of_action", "approval_status", "rationale"):
            cand[key] = _redact(cand.get(key), f"{path}.{key}", resolver, out.redacted)
        cand["counter_arguments"] = [_redact(x, f"{path}.counter_arguments[{k}]", resolver, out.redacted)
                                     for k, x in enumerate(cand.get("counter_arguments") or [])]
        out.positions.append(i)
        kept.append(cand)
    d["candidates"] = kept
    for name in ("follow_up_experiments", "limits"):
        d[name] = [_redact(x, f"{name}[{k}]", resolver, out.redacted) for k, x in enumerate(d.get(name) or [])]
    return out


def cited_record_ids(cand: dict[str, Any]) -> list[str]:
    """A drug candidate's ``evidence_ids`` and ``trial_ids`` spelled as the validator
    will spell them (prefix case-folded, whitespace stripped, a bare NCT id as
    ``nct:NCT…``), first appearance only."""
    ids = [canonical_id(str(x)) for x in _list(cand.get("evidence_ids")) if x is not None]
    for x in _list(cand.get("trial_ids")):
        t = canonical_id(str(x))
        m = _BARE_NCT.match(t)
        ids.append(f"nct:{m.group(1).upper()}" if m else t)
    return list(dict.fromkeys(x for x in ids if x))


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []  # a missing or malformed list is an empty one, as the validator reads it


def accession_tokens(text: str) -> list[str]:
    """Every bare accession in ``text`` as written, in order — for tests and audits."""
    return [m.group(0) for m in _ACCESSION.finditer(text)]


def _squash(text: Any) -> str:
    return _SPACE.sub(" ", str(text if text is not None else "")).strip().lower()


def _redact(text: Any, path: str, resolver: AccessionResolver, rejections: list[Rejection]) -> str:
    def repl(m: re.Match[str]) -> str:
        if resolver.resolves(m):
            return m.group(0)
        rejections.append(Rejection(path, f"bare accession not carried by any citable record: {m.group(0).rstrip(_TRAIL)}"))
        return REDACTED + m.group(0)[len(m.group(0).rstrip(_TRAIL)):]  # keep the sentence's punctuation
    out = _ACCESSION.sub(repl, "" if text is None else str(text))
    return out.replace(f"[{REDACTED}]", REDACTED)  # an accession the model bracketed on its own


def _dicts(items: Any) -> list[dict[str, Any]]:
    return [x for x in items if isinstance(x, dict)] if isinstance(items, list) else []


_CANDIDATE_PATH = re.compile(r"^candidates\[(\d+)\]")


def _original_path(path: str, positions: list[int]) -> str:
    """A validator path over the list the stage checks left → the model's own position."""
    m = _CANDIDATE_PATH.match(path)
    if not m:
        return path
    i = int(m.group(1))
    return f"candidates[{positions[i] if i < len(positions) else i}]" + path[m.end():]


# --------------------------------------------------------------------------- render

def render_document(bundle: MedicineBundle, report: MedicineReport, index: EvidenceIndex, *,
                    disclosure: str | None, model: str, effort: str) -> str:
    """``report.md``: a header naming the candidate and its stage-5 verdict, then the
    report in the rubric's order with References resolved against the index it was
    validated with."""
    verdicts = ", ".join(f"{v['key']} {str(v.get('classification') or 'not computed').replace('_', ' ')}"
                         for v in bundle.chain.get("variants", [])) or "no variant"
    out = [
        f"# Medicine — {bundle.candidate_id}",
        "",
        (f"candidate {bundle.candidate_id} · gene {bundle.gene_symbol or '-'} · model {bundle.candidate.get('model') or '-'} · "
         f"stage-5 classification: {verdicts} · model {model} · effort {effort}"),
        "",
        ("Hypotheses for follow-up, argued from retrieved records only — not a treatment recommendation. Every claim cites a "
         "record id and the References resolve them; a drug candidate that cited no drug-source record naming the drug, "
         "cited an unknown record or trial, or gave no counter-argument was removed by the validator (see the manifest)."),
        "",
        "---",
        "",
        render_medicine_report(report, index, disclosure=disclosure).rstrip("\n"),
    ]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- pieces

def default_retrievers(http: Http) -> MedicineRetrievers:
    """The real services over one shared ``Http`` (one cache, one limiter)."""
    return MedicineRetrievers(
        opentargets=OpenTargetsRetriever(http),
        dgidb=DgidbRetriever(http),
        chembl=ChemblRetriever(http),
        trials=TrialsRetriever(http),
        literature=LiteratureRetriever(http),
    )


def retriever_params(retrievers: MedicineRetrievers | None) -> dict[str, Any] | None:
    """Every knob the retrievers apply, for the manifest — ``None`` when none was built
    (a dry run). A retriever without ``params`` (a fake) records its class name."""
    if retrievers is None:
        return None
    out: dict[str, Any] = {}
    for name in ("opentargets", "dgidb", "chembl", "trials", "literature"):
        obj = getattr(retrievers, name)
        params = getattr(obj, "params", None)
        if callable(params):
            params = params()
        if not isinstance(params, dict):
            params = {"class": type(obj).__name__}
            for attr in ("page_size", "sort", "fields", "base_url"):
                if hasattr(obj, attr):
                    params[attr] = list(getattr(obj, attr)) if attr == "fields" else getattr(obj, attr)
        out[name] = params
    return out


def source_versions(records: Iterable[EvidenceRecord]) -> dict[str, list[str]]:
    """``source → observed versions`` over the citable records — what the manifest says
    each source was at the time a record was made."""
    out: dict[str, set[str]] = {}
    for r in records:
        out.setdefault(r.source, set()).add(r.source_version)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _optional_inputs(run_dir: Path) -> dict[str, Path]:
    """Every earlier-stage file the bundle or the index reads when it exists."""
    paths = {
        "retrieve_evidence_index": run_dir / RETRIEVE_DIR / "evidence" / "index.json",
        "filter_shortlist": run_dir / FILTER_DIR / "shortlist.tsv.gz",
        "rank_joined": run_dir / RANK_DIR / "joined.json",
        "rank_evidence_index": run_dir / RANK_DIR / "evidence" / "index.json",
        "reason_evidence_index": run_dir / REASON_DIR / "evidence" / "index.json",
    }
    return {name: p for name, p in paths.items() if p.exists()}


def _clear_outputs(out_dir: Path) -> None:
    """A rerun replaces the stage's outputs; the ``evidence/`` store stays."""
    for d in OUTPUT_DIRS:
        shutil.rmtree(out_dir / d, ignore_errors=True)
    for name in ("report.json", "report.md", "manifest.json"):
        (out_dir / name).unlink(missing_ok=True)


def _http(http: Http | None, cache_root: Path | None, offline: bool) -> Http:
    if http is not None:
        return http
    return Http(HttpCache(_cache_root(cache_root)), limiter=RateLimiter(RATE_LIMITS), offline=offline,
                backoff_floor=BACKOFF_FLOOR, retries=RETRIES)


def _client(provider: str | None, model: str, effort: str, progress: Progress) -> ac.ModelClient:
    """The real client for ``provider`` (``None``: the Anthropic SDK client, whose
    credentials are resolved on the first request — see stage 5)."""
    return providers.select_client(provider or "anthropic", model, effort, log=progress)


def _cache_root(cache_root: Path | None) -> Path:
    return Path(cache_root) if cache_root is not None else default_cache_root()


def _cache_root_of(http: Http | None) -> str | None:
    """The cache directory behind the ``Http`` the sources ran on; ``None`` when none
    ran (a dry run, injected retrievers) or the client has no cache (a stub)."""
    root = getattr(getattr(http, "cache", None), "root", None)
    return str(root) if root is not None else None


def _tool_call_counts(result: ac.AgentResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for turn in result.transcript:
        for call in turn.tool_calls:
            counts[call.name] = counts.get(call.name, 0) + 1
    return dict(sorted(counts.items()))


def _tag(ids: list[str] | None) -> str:
    return "".join(f" [{i}]" for i in (ids or []))


def _num(text: str) -> str:
    try:
        return f"{float(text):.4g}"
    except ValueError:
        return text


def _one_line(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text if text is not None else "")).strip()


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
