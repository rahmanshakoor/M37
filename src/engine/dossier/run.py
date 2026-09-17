"""The dossier step of stage 5 — ``engine dossier``.

One candidate per run: the stage-5 chain names the gene and classifies its variants;
this step fetches what the store lacks about the gene product — the reviewed UniProt
entry and a fixed set of papers — *before* any model runs, then asks the dossier
agent what is known about the protein, the disease mechanism, the regions the
variants fall in, the published genotype patterns and the functional test that would
show the defect. The engine fills each variant's residue and covering features from
the records; the model is held to the same citation rule as the chain.

The citation scope is per candidate: the stage-5 bundle's record ids (the variants'
own records), the ``uniprot:`` entry, the papers and ``pmid-search:`` records the
engine's searches wrote, and whatever the tools return in *this* conversation. The
validator is given an index of just those, so a mis-citation of any other real record
is rejected like an invented one.

What is written, and why in this form (the layout mirrors stages 5 and 6):

* ``05_reason/dossier/bundles/<cid>.json`` — what the model was given: the candidate,
  the chain as validated, the variants with the engine's residue and region, the
  UniProt record, the domain map, the papers retrieved, and the prompt text.
* ``prompts/<cid>.json`` / ``.md`` — the request as built. Written in every run; a
  ``--dry-run`` stops here (the retrieval still happens: it is the engine's, not the
  model's, and the bundle is what a reviewer wants to read).
* ``transcripts/<cid>.json`` — every turn, tool call and result; ``.failed.json`` on a
  failure.
* ``05_reason/evidence/uniprot/``, ``pmid/``, ``pmid-search/`` — the records it fetched,
  in stage 5's own store, which stage 6 already reads. A chain never cites them: stage
  5 validates each chain against its bundle and that conversation's tool returns only.
* ``<cid>.json`` — the validated :class:`~engine.dossier.schema.GeneDossier`, engine
  fields filled; ``validation/<cid>.json`` — every rejection and note; ``<cid>.md`` —
  the dossier rendered with References; ``manifest.json`` (stage ``dossier``).
* A ``## Gene dossier — <cid>`` section between ``<!-- dossier:<cid> -->`` markers at
  the end of ``05_reason/evidence_chain.md`` when that file exists — replaced on a
  rerun, never duplicated; the manifest records the file's sha256 before and after.
  ``engine reason`` rewrites the file without the section; ``engine dossier`` must
  then be rerun, and each manifest says which came last.

A run replaces the step's outputs for the candidate before it starts; the evidence
store is kept. Bundles, prompts, transcripts and the dossier sort their keys and carry
no clock; with a scripted client and a warm cache a rerun is byte-identical except for
the manifest's own timestamps.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic

from engine.agents import client as ac
from engine.agents import providers
from engine.agents.bundle import Bundle, VariantBundle, build_bundle, find_candidate
from engine.agents.schema import EvidenceChain
from engine.agents.validator import EvidenceIndex
from engine.dossier.checks import RULES, check_dossier
from engine.dossier.prompts import PROMPT_VERSION, SYSTEM_PROMPT, instructions_sha256, prompt_sha256, user_prompt
from engine.dossier.render import demote, render_dossier
from engine.dossier.schema import ENGINE_FIELDS, GeneDossier
from engine.dossier.tools import DossierRetrievers, DossierTools, fixed_queries
from engine.manifest import Manifest
from engine.medicine.run import list_chains, load_chain, select_candidate
from engine.reason.tools import ABSTRACT_CHARS, COORDINATE_RULE, MAX_RESULTS_CAP
from engine.retrieve.http import Http, HttpCache, RateLimiter, default_cache_root
from engine.retrieve.literature import LiteratureRetriever
from engine.retrieve.run import BACKOFF_FLOOR, RATE_LIMITS, RETRIES
from engine.retrieve.store import EvidenceRecord, EvidenceStore
from engine.retrieve.uniprot import (UniprotRetriever, disease_names, disease_texts, feature_map, function_text, gene_of,
                                     length_of, natural_variants_at, protein_name, reference_residue, regions_at,
                                     residue_at, residue_of)

REASON_DIR = "05_reason"
STAGE_DIR = f"{REASON_DIR}/dossier"
FILTER_DIR = "03_filter"
RANK_DIR = "04_rank"
RETRIEVE_DIR = "02_retrieve"
EVIDENCE_STAGES = (RETRIEVE_DIR, RANK_DIR, REASON_DIR)
"""Where a record this step may serve or cite can live: the stages before it and stage
5's own store, which it writes into. Stage 6's evidence is never an input here."""
OUTPUT_DIRS = ("bundles", "prompts", "transcripts", "validation")
DEFAULT_MAX_TURNS = 8
DEFAULT_LITERATURE_RESULTS = 8
"""Papers per fixed search: enough for the model to pick from, small enough to read."""
FUNCTION_CHARS = 900
DISEASE_CHARS = 400
CITATION_SCOPE = ("the stage-5 bundle's record ids, the uniprot: entry, the papers and pmid-search: records the engine's "
                  "fixed searches wrote, and the records returned by this candidate's own tool calls")
SECTION_START = "<!-- dossier:{cid} -->"
SECTION_END = "<!-- /dossier:{cid} -->"

Progress = Callable[[str], None]

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._:-]+")


# ------------------------------------------------------------------------------ run

def run_dossier(
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
    retrievers: DossierRetrievers | None = None,
    provider: str | None = None,
    literature_max_results: int = DEFAULT_LITERATURE_RESULTS,
    progress: Progress = lambda s: None,
) -> Path:
    """Run the dossier step into ``run_dir/05_reason/dossier`` for one candidate;
    returns the manifest path. ``candidate_id`` defaults to the stage-5 chain of the
    best stage-3 priority. ``client`` defaults to the ``provider``'s client, built
    when needed (never in a dry run); ``retrievers`` (tests) or ``http``/
    ``cache_root``/``offline`` shape the UniProt and Europe PMC services — which are
    asked in every run, dry or not. A gene UniProt has no reviewed human entry for
    ends the run with a manifest and no dossier (an absence, recorded, exit 0).
    Raises :class:`~engine.agents.client.AgentError` when the model or a tool's
    service fails — a dossier is never written around a failure."""
    run_dir = Path(run_dir)
    out_dir = run_dir / STAGE_DIR
    chains_dir = run_dir / REASON_DIR / "chains"
    if not chains_dir.is_dir():
        raise FileNotFoundError(f"stage 5 has not run in {run_dir} ({chains_dir} missing)")
    if max_turns < 1:
        raise ValueError(f"max_turns must be at least 1, got {max_turns}")
    if not 1 <= literature_max_results <= MAX_RESULTS_CAP:
        raise ValueError(f"literature_max_results must be between 1 and {MAX_RESULTS_CAP}, got {literature_max_results}")
    model = providers.provider_model(provider or "anthropic", model)
    cid, chain_path = select_candidate(run_dir, candidate_id)
    chain = load_chain(chain_path)
    candidate = find_candidate(run_dir, cid)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = Manifest(stage="dossier")
    m.add_input("chain", chain_path)
    m.add_input("candidates", run_dir / FILTER_DIR / "candidates.json")
    for name, path in _optional_inputs(run_dir).items():
        m.add_input(name, path)
    m.tools["anthropic-sdk"] = anthropic.__version__

    _clear_outputs(out_dir, cid)
    store = EvidenceStore(run_dir / REASON_DIR / "evidence")  # stage 5's store; exists before the index is built
    index = EvidenceIndex.from_run(run_dir, stages=EVIDENCE_STAGES)
    http_used: Http | None = None
    if retrievers is None:
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
        "tools": ["get_record", "search_literature", "get_paper"],
        "output_schema": "GeneDossier without protein_position/region; keys pinned to the chain's, candidate_id, "
                         "gene_symbol and uniprot_accession pinned",
        "citation_scope": CITATION_SCOPE,
        "evidence_stages": list(EVIDENCE_STAGES),
        "stage_checks": dict(RULES),
        "literature": {"max_results_per_query": literature_max_results, "max_results_cap": MAX_RESULTS_CAP,
                       "abstract_chars": ABSTRACT_CHARS, "peer_reviewed_only": True,
                       "coordinates_in_queries": COORDINATE_RULE},
        "uniprot": _params_of(retrievers.uniprot),
        "offline": bool(getattr(http_used, "offline", offline)),
        "cache_root": _cache_root_of(http_used),
        "evidence_chain_md": None,
    })
    m.counts.update({"chains_available": len(list_chains(run_dir)), "candidates_selected": 1, "dossiers_written": 0,
                     "papers_retrieved": 0, "features_in_map": 0, "natural_variants_at_positions": 0})

    # -- engine-driven retrieval, before any model
    gene = str(candidate.get("gene_symbol") or "")
    progress("candidate: UniProt entry")
    acc = retrievers.uniprot.accession_for_symbol(gene) if gene else None
    m.params["accession"] = acc
    m.params["uniprot_release"] = retrievers.uniprot.version() if hasattr(retrievers.uniprot, "version") else None
    m.params["uniprot_fields"] = getattr(retrievers.uniprot, "fields", None)
    if acc is None:
        m.note("UniProt has no reviewed human entry for the symbol" if gene else "the candidate names no gene symbol")
        m.note("no dossier: nothing to build it on; the run stops here (an absence, recorded).")
        index_path = store.write_index()
        m.add_output("evidence_index", index_path)
        m.counts.update({"evidence_records": store.count(), "evidence_records_added": 0})
        m.params["disclosure"] = disclosure or providers.disclosure(provider or "anthropic", model, effort)
        manifest_path = out_dir / "manifest.json"
        m.write(manifest_path)
        return manifest_path
    entry = retrievers.uniprot.entry(acc)
    written: list[str] = []
    if not store.exists(entry.record_id):
        written.append(entry.record_id)
    store.put(entry)

    progress("candidate: literature")
    queries: list[dict[str, Any]] = []
    papers: dict[str, EvidenceRecord] = {}  # first appearance wins; a paper two searches return is one record
    for query in fixed_queries(gene, disease_names(entry)):
        result = retrievers.literature.search(query, max_results=literature_max_results)
        for rec in [result.record, *result.papers]:
            held = index.get(rec.record_id)  # a store already holding the record is the one shown and cited
            if held is None:
                written.append(rec.record_id)
                store.put(rec)
            if rec.record_id != result.record.record_id and rec.record_id not in papers:
                papers[rec.record_id] = held or rec
        queries.append({"query": query, "sent": result.record.payload["sent"]["query"], "hit_count": result.record.payload["hitCount"],
                        "returned": len(result.papers), "search_record": result.record.record_id,
                        "pmids": [p.record_id for p in result.papers]})
    m.params["queries"] = queries

    progress("candidate: bundle")
    bundle = build_dossier_bundle(candidate, chain, run_dir, entry, list(papers.values()), [q["search_record"] for q in queries],
                                  literature=retrievers.literature)
    write_bundle(bundle, out_dir)
    for n in bundle.notes:
        m.note(f"{cid}: {n}")
    m.counts.update({"papers_retrieved": len(papers), "features_in_map": len(bundle.feature_map),
                     "natural_variants_at_positions": sum(v["n_natural_variants"] for v in bundle.variants)})
    tools = DossierTools(index, store, retrievers.literature, citable=bundle.record_ids, default_max_results=literature_max_results)
    request = build_request(bundle, tools, model=model, effort=effort, max_turns=max_turns)
    write_prompt(request, out_dir, cid)

    if dry_run:
        index_path = store.write_index()
        m.add_output("evidence_index", index_path)
        m.counts.update({"evidence_records": store.count(), "evidence_records_added": len(dict.fromkeys(written))})
        m.note("dry run: the UniProt entry and the fixed searches were retrieved, the bundle and the prompt written; "
               "no model was called and no dossier was produced.")
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
        # The SDK resolves credentials on the first request and reports a missing key
        # as a TypeError; anything else is a real bug (see stage 6).
        if "authentication" not in str(e).lower():
            raise
        raise ac.AgentError(f"could not call the Anthropic API: {e}") from e
    _write_json(out_dir / "transcripts" / f"{_name(cid)}.json", {**result.as_dict(), "tools": tools.log.as_dict()})
    written.extend(tools.log.written)

    records = citable_records(tools, index)
    scoped = EvidenceIndex.from_records(records)
    cleaned, report = check_dossier(result.output, bundle, records, scoped)
    _write_json(out_dir / f"{_name(cid)}.json", cleaned.model_dump())
    _write_json(out_dir / "validation" / f"{_name(cid)}.json", {"candidate_id": cid, **report.as_dict()})
    for r in report.rejections:
        m.note(f"{cid}: rejected {r.path}: {r.reason}")
    for n in report.notes:
        m.note(f"{cid}: {n}")
    md = render_dossier(cleaned, scoped, disclosure=result.disclosure, rejections=report.rejections)
    md_path = _write_text(out_dir / f"{_name(cid)}.md", md)
    m.params["evidence_chain_md"] = splice_evidence_chain(run_dir / REASON_DIR / "evidence_chain.md", cid, md)
    index_path = store.write_index()

    m.add_output("dossier_json", out_dir / f"{_name(cid)}.json")
    m.add_output("dossier_md", md_path)
    m.add_output("evidence_index", index_path)
    claims_kept = sum(len(getattr(cleaned, name)) for name in ("protein", "mechanism_of_disease", "region_knowledge",
                                                              "genotype_patterns", "functional_test"))
    m.counts.update({
        "dossiers_written": 1,
        "evidence_records": store.count(),
        "evidence_records_added": len(dict.fromkeys(written)),
        "usage": asdict(result.usage),
        "claims_kept": claims_kept,
        "variant_positions": len(cleaned.variant_positions),
        "items_dropped": report.counts["items_dropped"],
        "positions_replaced": report.counts.get("positions_replaced", 0),
        "redactions": report.counts["redactions"],
        "validation": dict(report.counts),
        "rejections": report.counts["items_dropped"] + report.counts["literature_removed"] + report.counts["redactions"],
        "tool_calls": dict(_tool_call_counts(result)),
    })
    m.params["validation"] = {"rejections": [asdict(r) for r in report.rejections], "notes": list(report.notes)}
    m.params["tools_used"] = tools.log.as_dict()
    m.params["source_versions"] = _source_versions(records)
    m.params["agent"] = {"model": result.model, "effort": result.effort, "stop_reason": result.stop_reason,
                         "disclosure": result.disclosure, "usage": asdict(result.usage), "request": dict(result.request)}
    m.params["disclosure"] = result.disclosure
    progress(f"candidate: {claims_kept} claim(s) kept · {len(cleaned.variant_positions)} variant position(s) · "
             f"{m.counts['rejections']} rejected")
    manifest_path = out_dir / "manifest.json"
    m.write(manifest_path)
    return manifest_path


# --------------------------------------------------------------------------- bundle

@dataclass
class DossierBundle:
    """What the dossier agent is given: the candidate and its validated chain, the
    variants with the engine's residue and region, the UniProt record and its domain
    map, the papers the engine retrieved, and the citation vocabulary."""

    candidate_id: str
    gene_symbol: str
    accession: str
    candidate: dict[str, Any]
    chain: dict[str, Any]
    variants: list[dict[str, Any]]
    """Per chain variant: ``key, hgvsc, hgvsp, consequence, exon, classification, points,
    residue, reference_residue, region, n_natural_variants`` — ``residue`` and ``region``
    are what the engine fills into the dossier."""
    uniprot: dict[str, Any]
    """The ``uniprot:`` record (``asdict``)."""
    feature_map: list[dict[str, Any]]
    papers: list[dict[str, Any]]
    """Summaries (``record_id, title, year, journal``) of the papers the fixed searches returned."""
    search_records: list[str]
    record_ids: list[str]
    """The citation vocabulary before any tool call."""
    text: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def build_dossier_bundle(candidate: dict[str, Any], chain: EvidenceChain, run_dir: Path, entry: EvidenceRecord,
                         papers: list[EvidenceRecord], search_records: list[str], *,
                         literature: Any = None) -> DossierBundle:
    """The stage-5 bundle (:func:`~engine.agents.bundle.build_bundle`, for the variant
    columns) joined with the chain's verdicts, the UniProt entry and the papers."""
    base: Bundle = build_bundle(candidate, run_dir, case_hpo=())
    chain_data = json.loads(json.dumps(chain.model_dump(), sort_keys=True))
    verdicts = {v["key"]: v for v in chain_data.get("variants", [])}
    variants = [variant_entry(v, verdicts.get(v.key), entry) for v in base.variants]
    notes = [v.pop("note") for v in variants if v.get("note")]
    for v in variants:
        v.pop("note", None)
    lit = literature or LiteratureRetriever(None)  # extract() is a pure projection
    summaries = [_paper_summary(p, lit) for p in papers]
    ids = set(base.record_ids) | {entry.record_id} | {p["record_id"] for p in summaries} | set(search_records)
    bundle = DossierBundle(
        candidate_id=base.candidate_id,
        gene_symbol=str(candidate.get("gene_symbol") or ""),
        accession=str(entry.payload.get("primaryAccession") or entry.record_id.split(":", 1)[1]),
        candidate=base.candidate,
        chain=chain_data,
        variants=variants,
        uniprot=asdict(entry),
        feature_map=feature_map(entry),
        papers=summaries,
        search_records=list(search_records),
        record_ids=sorted(ids),
        notes=notes,
    )
    bundle.text = render_text(bundle)
    return bundle


def variant_entry(v: VariantBundle, verdict: dict[str, Any] | None, entry: EvidenceRecord) -> dict[str, Any]:
    """One chain variant with the engine's residue and region: the residue from the
    hgvsp column of the ``vep:`` record, the features covering it and the natural
    variants at it from the UniProt entry — unless the HGVS reference amino acid is
    not what the canonical sequence carries at that residue (the transcript's protein
    is another isoform), in which case the map is not applied and the region says
    so; a residue no feature covers says that too."""
    get = _getter(v)
    hgvsp = get("hgvsp")
    residue = residue_of(hgvsp)
    region: list[str] = []
    note = ""
    n_variants = 0
    if residue is not None:
        expected, actual = reference_residue(hgvsp), residue_at(entry, residue)
        if expected and actual and expected != actual and expected != "*":
            region = [f"residue {residue}: the transcript's reference amino acid ({expected}) is not the UniProt "
                      f"canonical sequence's ({actual}); the feature map is numbered on another isoform and is not applied"]
            note = f"{v.key}: hgvsp reference residue {expected}{residue} differs from the canonical sequence ({actual}); region not applied"
        else:
            covering = regions_at(entry, residue)
            variants_here = natural_variants_at(entry, residue)
            n_variants = len(variants_here)
            region = (covering or [f"no annotated feature covers residue {residue}"]) + variants_here
    return {
        "key": v.key,
        "hgvsc": get("hgvsc"),
        "hgvsp": hgvsp,
        "consequence": get("consequence"),
        "impact": get("impact"),
        "exon": get("exon"),
        "gt": get("gt"),
        "classification": (verdict or {}).get("classification"),
        "points": (verdict or {}).get("points"),
        "summary": (verdict or {}).get("summary") or "",
        "residue": residue,
        "reference_residue": reference_residue(hgvsp),
        "region": region,
        "n_natural_variants": n_variants,
        "record_ids": [r["record_id"] for r in v.records],
        "note": note,
    }


def write_bundle(bundle: DossierBundle, stage_dir: Path) -> Path:
    return _write_text(Path(stage_dir) / "bundles" / f"{_name(bundle.candidate_id)}.json", bundle.to_json())


def render_text(bundle: DossierBundle) -> str:
    """The prompt block: the candidate, the protein with its FUNCTION text, the domain
    map, one section per variant with the engine's residue and covering features, the
    DISEASE texts, the papers retrieved, and the citable list. Every fact is followed
    by the record id it came from."""
    c = bundle.candidate
    uid = bundle.uniprot["record_id"]
    tag = f" [{uid}]"
    lines = [
        f"# Candidate {bundle.candidate_id}",
        f"gene: {bundle.gene_symbol or '-'} ({c.get('gene_id') or 'no gene id'}) · model: {c.get('model') or '-'} · priority: {c.get('priority', '-')}",
        "",
        "## Protein",
        f"{bundle.accession} {protein_name(bundle.uniprot) or '-'} · {length_of(bundle.uniprot) or '?'} aa · "
        f"UniProt entry {bundle.uniprot['payload'].get('uniProtkbId') or '-'} · gene {gene_of(bundle.uniprot) or '-'}{tag}",
        f"function: {_short(function_text(bundle.uniprot), FUNCTION_CHARS) or 'no FUNCTION comment'}{tag}",
        "",
        "## Domain map (every feature the record annotates; residue numbering of the canonical isoform)",
    ]
    if bundle.feature_map:
        lines.extend(f"{f['type']} {f['start']}–{f['end']} {f['description']}".rstrip() + tag for f in bundle.feature_map)
    else:
        lines.append(f"no Domain, Region, Motif, Topological domain, Transmembrane, Binding site, Active site or Site feature{tag}")
    lines.extend(["", "## Variants (residue and covering features computed by the engine from the records; do not restate them from memory)"])
    for v in bundle.variants:
        vep = f" [vep:{v['key']}]"
        label = str(v.get("classification") or "not classified").replace("_", " ")
        points = f" ({v['points']:+d} SVI points)" if isinstance(v.get("points"), int) else ""
        lines.append(f"### Variant {v['key']} — {label}{points}")
        lines.append(f"genotype {v['gt']} · {v['consequence']} ({v['impact']}) · {v['hgvsc']} · {v['hgvsp']} · exon {v['exon']}{vep}")
        lines.append(f"residue (engine, from hgvsp): {v['residue'] if v['residue'] is not None else 'none — the protein change names no residue'}")
        for r in v["region"]:
            lines.append(f"- {r}{tag}")
        if v.get("summary"):
            lines.append(f"stage-5 summary: {_one_line(v['summary'])}")
        lines.append("records: " + (" ".join(f"[{r}]" for r in v["record_ids"]) or "none in the store"))
        lines.append("")
    lines.append("## Disease (UniProt)")
    texts = disease_texts(bundle.uniprot)
    if texts:
        lines.extend(f"- {_short(t, DISEASE_CHARS)}{tag}" for t in texts)
    else:
        lines.append(f"- no DISEASE comment{tag}")
    lines.extend(["", "## Papers retrieved (by the engine's fixed searches; read an abstract with get_paper before citing a result)"])
    if bundle.papers:
        lines.extend(f"- [{p['record_id']}] {p['year'] or '-'} · {p['title'] or '(no title)'}" for p in bundle.papers)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("citable record ids: " + (" ".join(f"[{r}]" for r in bundle.record_ids) or "none"))
    return "\n".join(lines) + "\n"


def _paper_summary(rec: EvidenceRecord, lit: Any) -> dict[str, Any]:
    cols = lit.extract(rec)
    return {"record_id": rec.record_id, "title": cols.get("title", ""), "year": cols.get("year", ""),
            "journal": cols.get("journal", ""), "cited_by": cols.get("cited_by", ""), "url": rec.url}


# ------------------------------------------------------------------------- request

def build_request(bundle: DossierBundle, tools: DossierTools, *, model: str, effort: str, max_turns: int) -> ac.AgentRequest:
    """The agent request: the frozen system prompt, the bundle as the user turn, the
    three tools, and an answer schema that omits the engine's fields and pins the
    identity and the variant keys, so the API itself refuses a foreign key."""
    enums: dict[str, list[str]] = {"candidate_id": [bundle.candidate_id], "gene_symbol": [bundle.gene_symbol],
                                   "uniprot_accession": [bundle.accession]}
    if bundle.variants:  # an empty enum is not a schema
        enums["key"] = [v["key"] for v in bundle.variants]
    return ac.AgentRequest(
        system=SYSTEM_PROMPT,
        user=user_prompt(bundle, max_turns=max_turns),
        output_model=GeneDossier,
        tools=tools.specs(),
        max_turns=max_turns,
        model=model,
        effort=effort,
        output_schema=ac.answer_schema(GeneDossier, drop=ENGINE_FIELDS, enum=enums),
    )


def write_prompt(request: ac.AgentRequest, stage_dir: Path, candidate_id: str) -> Path:
    """``prompts/<cid>.json`` (the request as built) and ``.md`` (for reading)."""
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

def citable_records(tools: DossierTools, index: EvidenceIndex) -> list[EvidenceRecord]:
    """The records this dossier may cite: the bundle's list and every record a tool
    returned in this conversation, each read from the run's stores."""
    records: list[EvidenceRecord] = []
    for rid in tools.citable_ids():
        rec = index.get(rid)
        if rec is None:
            raise RuntimeError(f"record {rid!r} is on the candidate's citable list but not in any store of the run")
        records.append(rec)
    return records


def splice_evidence_chain(path: Path, candidate_id: str, dossier_md: str) -> dict[str, str] | None:
    """Append — or replace, between the candidate's markers — the dossier section at
    the end of ``evidence_chain.md``; ``None`` when the file does not exist (stage 5
    ran dry or has not run). Returns the file's sha256 before and after; a rerun with
    the same dossier leaves the bytes unchanged."""
    path = Path(path)
    if not path.exists():
        return None
    before = path.read_text()
    section = "\n".join([
        SECTION_START.format(cid=candidate_id),
        f"## Gene dossier — {candidate_id}",
        "",
        f"Written by `engine dossier` after the chains above (`05_reason/dossier/{_name(candidate_id)}.md`); validated "
        "like a chain, engine-filled residue and region.",
        "",
        demote(dossier_md).rstrip("\n"),
        SECTION_END.format(cid=candidate_id),
        "",
    ])
    pattern = re.compile(re.escape(SECTION_START.format(cid=candidate_id)) + r".*?" + re.escape(SECTION_END.format(cid=candidate_id)) + r"\n?",
                         re.DOTALL)
    after = pattern.sub(lambda m: section, before, count=1) if pattern.search(before) else before.rstrip("\n") + "\n\n" + section
    _write_text(path, after)
    return {"sha256_before": _sha256(before), "sha256_after": _sha256(after)}


# --------------------------------------------------------------------------- pieces

def default_retrievers(http: Http) -> DossierRetrievers:
    """The real services over one shared ``Http`` (one cache, one limiter)."""
    return DossierRetrievers(uniprot=UniprotRetriever(http), literature=LiteratureRetriever(http))


def _params_of(obj: Any) -> dict[str, Any]:
    params = getattr(obj, "params", None)
    if callable(params):
        params = params()
    return params if isinstance(params, dict) else {"class": type(obj).__name__}


def _source_versions(records: list[EvidenceRecord]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for r in records:
        out.setdefault(r.source, set()).add(r.source_version)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _optional_inputs(run_dir: Path) -> dict[str, Path]:
    """Every earlier file the bundle or the index reads when it exists."""
    paths = {
        "retrieve_evidence_index": run_dir / RETRIEVE_DIR / "evidence" / "index.json",
        "filter_shortlist": run_dir / FILTER_DIR / "shortlist.tsv.gz",
        "rank_joined": run_dir / RANK_DIR / "joined.json",
        "rank_evidence_index": run_dir / RANK_DIR / "evidence" / "index.json",
        "reason_evidence_index": run_dir / REASON_DIR / "evidence" / "index.json",
        "evidence_chain_md": run_dir / REASON_DIR / "evidence_chain.md",
    }
    return {name: p for name, p in paths.items() if p.exists()}


def _clear_outputs(out_dir: Path, candidate_id: str) -> None:
    """A rerun replaces the step's outputs for the candidate — per-candidate files in
    each output directory, the dossier, the manifest. Another candidate's dossier
    stays; the evidence store is stage 5's and is kept."""
    stem = _name(candidate_id)
    for d in OUTPUT_DIRS:
        for p in (out_dir / d).glob(f"{stem}.*"):
            p.unlink()
    for name in (f"{stem}.json", f"{stem}.md", "manifest.json"):
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
    root = getattr(getattr(http, "cache", None), "root", None)
    return str(root) if root is not None else None


def _tool_call_counts(result: ac.AgentResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for turn in result.transcript:
        for call in turn.tool_calls:
            counts[call.name] = counts.get(call.name, 0) + 1
    return dict(sorted(counts.items()))


def _getter(v: VariantBundle) -> Callable[[str], str]:
    def get(name: str) -> str:
        for src in (v.columns, v.fields):
            val = src.get(name)
            if val not in (None, "", [], {}):
                return str(val)
        return "-"
    return get


def _short(text: str, n: int) -> str:
    text = _one_line(text)
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def _one_line(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text if text is not None else "")).strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


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
