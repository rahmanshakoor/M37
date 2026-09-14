"""JSON views over a run directory — what the report renders and the UI serves.

Why views, and why pure: the report and the live UI must show the same facts, so
both read them through these functions and nothing else. Each view is a pure
function of the files in the run directory (no clock, no network, no recomputation):
a number in a view is a number a stage wrote, quoted as written — the ranking scores
keep the text of ``ranking.tsv``, the allele frequencies the text of
``candidates.json``. What a view adds is *joins* (a stage-3 candidate with its
stage-4 rank, a criterion's evidence ids with the URLs of the records they name) and
*flags* (whether the blind ranker agrees, whether a chain is one the stage-5 manifest
claims), never arithmetic.

Every view tolerates a missing stage: a run that stopped after stage 3 gives
``present: False`` for the later views instead of an error, so the report can be
rendered at any point of a run. A file that is present but unreadable is reported the
same way, with the reason.

Evidence ids resolve against the union of every stage's ``evidence/index.json``
(stages 2, 4, 5, 6): a record is written once by the stage that fetched it, and the
index entry carries the URL a reader opens. An id that resolves nowhere is returned
with ``url: None`` — the validator should have rejected it, and the report shows it
as unresolved rather than hiding it.

Two things a run directory can hold are easy to misread and are named explicitly
rather than folded into "absent": a stage-4 join older than the shortlist it joins
(stage 3 rerun after stage 4 — ``join_status``, and ``join`` per candidate), and a live
stage 5 or 6 that failed on the model (``transcripts/<candidate>.failed.json``, the
only file such a run leaves — ``failures`` in the chain and medicine views).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from engine.agents.validator import KNOWN_SOURCES, citation_tokens
from engine.manifest import sha256_file

STAGES: tuple[tuple[str, str], ...] = (
    ("01_ingest", "ingest"),
    ("02_retrieve", "retrieve"),
    ("03_filter", "filter"),
    ("04_rank", "rank"),
    ("05_reason", "reason"),
    ("06_medicine", "medicine"),
)
"""Stage directories in pipeline order, with the stage name each manifest carries."""

EVIDENCE_STAGES = ("02_retrieve", "04_rank", "05_reason", "06_medicine")
"""Where an ``evidence/index.json`` may live, in the order records were written."""

DEFAULT_TOP_N = 20
"""Ranked genes shown in the blind-ranking view (plus every shortlist gene beyond it)."""

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._:-]+")


# --------------------------------------------------------------------------- files

def read_json(path: Path) -> Any:
    """The parsed file, or ``None`` when it is absent or not JSON (a view never raises
    on a file a stage did not finish writing)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def manifest(run_dir: Path, stage_dir: str) -> dict[str, Any] | None:
    doc = read_json(Path(run_dir) / stage_dir / "manifest.json")
    return doc if isinstance(doc, dict) else None


def evidence_index(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Record id → index entry (``source``, ``source_version``, ``url``, ``retrieved_at``,
    ``path``, ``sha256``) plus ``stage`` (the directory it was read from), over every
    stage store. The first stage to hold an id wins; a store never rewrites an
    earlier stage's record."""
    run_dir = Path(run_dir)
    merged: dict[str, dict[str, Any]] = {}
    for stage_dir in EVIDENCE_STAGES:
        doc = read_json(run_dir / stage_dir / "evidence" / "index.json")
        if not isinstance(doc, dict):
            continue
        for rid, entry in doc.items():
            if rid not in merged and isinstance(entry, dict):
                merged[rid] = {**entry, "stage": stage_dir}
    return merged


def resolve(ids: list[Any], index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """``[{id, url, source, stage}]`` for each id, ``url: None`` when it resolves nowhere."""
    out = []
    for rid in ids:
        rid = str(rid)
        entry = index.get(rid)
        out.append({
            "id": rid,
            "url": entry.get("url") if entry else None,
            "source": entry.get("source") if entry else rid.split(":", 1)[0],
            "stage": entry.get("stage") if entry else None,
        })
    return out


def cited_in(texts: list[str], index: dict[str, dict[str, Any]]) -> list[str]:
    """Every record id cited inline in the texts — the validator's own tokenizer, widened
    to the sources the run's stores hold — in order of first appearance."""
    sources = set(KNOWN_SOURCES) | {rid.split(":", 1)[0] for rid in index}
    found: list[str] = []
    for text in texts:
        for rid in citation_tokens(str(text), sources):
            if rid not in found:
                found.append(rid)
    return found


def candidate_file_name(candidate_id: str) -> str:
    """The file stem stages 5 and 6 use for a candidate id (``CFTR:comphet.json``)."""
    return _SAFE_NAME.sub("_", candidate_id) or "candidate"


# ----------------------------------------------------------------------- summary

def run_summary(run_dir: Path) -> dict[str, Any]:
    """The run header: the sample the VCF was read for (the stage-1 manifest; the case
    file's proband id is not written into any manifest), the VCF's checksum, the case
    HPO terms as the first stage that recorded them saw them, and one line per stage:
    present or not, engine version, when it started and finished, the duration those
    two timestamps span, whether it was a dry run, the candidates a live stage 5 or 6
    ``failed`` on (see :func:`failures`), and the counts that summarise it."""
    run_dir = Path(run_dir)
    stages = [_stage_summary(run_dir, stage_dir, name) for stage_dir, name in STAGES]
    ingest = manifest(run_dir, "01_ingest") or {}
    hpo, hpo_source = _hpo(run_dir)
    vcf = (ingest.get("inputs") or {}).get("vcf") or {}
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "sample": (ingest.get("params") or {}).get("sample"),
        "vcf": {"path": vcf.get("path"), "bytes": vcf.get("bytes"), "sha256": vcf.get("sha256")} if vcf else None,
        "hpo": hpo,
        "hpo_source": hpo_source,
        "stages": stages,
        "stages_present": [s["dir"] for s in stages if s["present"]],
        "engine_versions": sorted({s["engine_version"] for s in stages if s.get("engine_version")}),
    }


def _stage_summary(run_dir: Path, stage_dir: str, name: str) -> dict[str, Any]:
    present = (run_dir / stage_dir).is_dir()
    m = manifest(run_dir, stage_dir)
    out: dict[str, Any] = {"dir": stage_dir, "stage": name, "present": present, "manifest": m is not None}
    if present and name in ("reason", "medicine"):
        out["failed"] = [f["candidate_id"] for f in failures(run_dir / stage_dir)]
    if m is None:
        return out
    params = m.get("params") or {}
    counts = m.get("counts") or {}
    out.update({
        "engine_version": m.get("engine_version"),
        "platform": m.get("platform"),
        "started_at": m.get("started_at"),
        "finished_at": m.get("finished_at"),
        "duration_s": _duration(m.get("started_at"), m.get("finished_at")),
        "wall_time_s": params.get("wall_time_s"),
        "dry_run": params.get("dry_run"),
        "tools": dict(m.get("tools") or {}),
        "counts": counts,
        "notes": list(m.get("notes") or []),
        "headline": _headline(name, params, counts),
    })
    return out


def _duration(started: Any, finished: Any) -> float | None:
    """Seconds between the manifest's own two timestamps; ``None`` if either is missing."""
    try:
        a = datetime.fromisoformat(str(started))
        b = datetime.fromisoformat(str(finished))
    except (TypeError, ValueError):
        return None
    return round((b - a).total_seconds(), 1)


def _headline(name: str, params: dict[str, Any], counts: dict[str, Any]) -> list[tuple[str, Any]]:
    """The few counts that say what a stage did, as (label, value) pairs — all read
    from the manifest, none derived."""
    if name == "ingest":
        return [("rows out", counts.get("rows_out")), ("sites in", counts.get("sites_in")), ("flagged", counts.get("flagged_rows"))]
    if name == "retrieve":
        http = counts.get("http") or {}
        return [("variants", counts.get("unique_variants")), ("records", counts.get("evidence_records")),
                ("live requests", http.get("live_requests")), ("cache hits", http.get("hits"))]
    if name == "filter":
        return [("rows in", counts.get("rows_in")), ("kept", counts.get("kept")), ("dropped", counts.get("dropped")),
                ("candidates", counts.get("candidates"))]
    if name == "rank":
        return [("genes ranked", counts.get("genes_ranked")), ("candidates ranked", counts.get("candidates_ranked")),
                ("exomiser-only", counts.get("exomiser_only")), ("exit code", params.get("exomiser_exit_code"))]
    if name == "reason":
        return [("selected", counts.get("candidates_selected")), ("chains written", counts.get("chains_written")),
                ("rejections", counts.get("rejections")), ("model", params.get("model") if not params.get("dry_run") else "dry run")]
    if name == "medicine":
        return [("candidate", params.get("candidate")), ("reports written", counts.get("reports_written")),
                ("model", params.get("model") if not params.get("dry_run") else "dry run")]
    return []


def _hpo(run_dir: Path) -> tuple[list[str], str | None]:
    """The case HPO terms from the first file that recorded them: stage 4's
    ``joined.json`` (what Exomiser was given), else the stage-5 or stage-6 manifest."""
    joined = read_json(run_dir / "04_rank" / "joined.json")
    if isinstance(joined, dict) and isinstance(joined.get("hpo"), list):
        return [str(t) for t in joined["hpo"]], "04_rank/joined.json"
    for stage_dir in ("05_reason", "06_medicine"):
        m = manifest(run_dir, stage_dir)
        if m and isinstance((m.get("params") or {}).get("hpo"), list):
            return [str(t) for t in m["params"]["hpo"]], f"{stage_dir}/manifest.json"
    return [], None


# -------------------------------------------------------------------- candidates

def candidates_view(run_dir: Path) -> dict[str, Any]:
    """Stage 3's candidates in priority order, each joined with stage 4's view of its
    gene (rank, scores, MOI, the ``exomiser:`` record) when stage 4 ran.

    ``agreement`` per candidate compares two orderings of the candidates Exomiser
    ranked: the shortlist's own (by ``priority``) and the blind ranker's (by
    ``exomiser_rank``). A candidate whose position is the same in both ``agrees``;
    otherwise it ``disagrees``; a gene Exomiser did not rank is ``unranked``.
    ``top_agreement`` says whether the shortlist's first candidate is also Exomiser's
    rank 1 overall — the stronger claim, since Exomiser ranks every gene in the VCF.
    Every rule hit, caveat and variant field is passed through as stage 3 wrote it.

    ``join`` per candidate says what stage 4's ``joined.json`` holds for it: ``ranked``,
    ``unranked`` (joined, but Exomiser ranked no gene for it), ``missing`` (not in the
    join at all — the shortlist was written after stage 4 joined it), or ``None`` when
    stage 4 has not run. A ``missing`` candidate is not "unranked": ``ranking.tsv`` may
    well hold its gene, and ``join_check`` (see :func:`join_status`) says the join is
    stale rather than letting the two be confused."""
    run_dir = Path(run_dir)
    doc = read_json(run_dir / "03_filter" / "candidates.json")
    if not isinstance(doc, dict):
        return {"present": False, "candidates": [], "counts": {}, "config": {}, "rank_present": False, "join_check": None}
    index = evidence_index(run_dir)
    joined = read_json(run_dir / "04_rank" / "joined.json")
    rank_present = isinstance(joined, dict)
    by_id = {str(c.get("candidate_id")): c for c in (joined or {}).get("candidates", [])} if rank_present else {}

    cands: list[dict[str, Any]] = []
    for c in sorted(doc.get("candidates", []), key=lambda c: (c.get("priority", 1 << 30), str(c.get("candidate_id")))):
        j = by_id.get(str(c.get("candidate_id"))) if rank_present else None
        entry = {
            "candidate_id": c.get("candidate_id"),
            "gene_symbol": c.get("gene_symbol"),
            "gene_id": c.get("gene_id"),
            "model": c.get("model"),
            "priority": c.get("priority"),
            "phase": dict(c.get("phase") or {}),
            "rule_hits": list(c.get("rule_hits") or []),
            "caveats": list(c.get("caveats") or []),
            "clinvar_plp": any(str(v.get("clinvar_pathogenicity") or "") in ("Pathogenic", "Likely_pathogenic")
                               for v in c.get("variants", [])),
            "variants": [{**v, "evidence": resolve(list(v.get("evidence_ids") or []), index)} for v in c.get("variants", [])],
            "exomiser": _exomiser_join(j, index),
            "join": _join_kind(j) if rank_present else None,
        }
        cands.append(entry)
    _mark_agreement(cands)
    compared = [c for c in cands if c["agreement"] in ("agrees", "disagrees")]
    return {
        "present": True,
        "rank_present": rank_present,
        "counts": dict(doc.get("counts") or {}),
        "config": dict(doc.get("config") or {}),
        "candidates": cands,
        "top_agreement": (cands[0]["exomiser"] or {}).get("rank") == 1 if rank_present and cands else None,
        "order_agreement": all(c["agreement"] == "agrees" for c in compared) if compared else None,
        "join_check": join_status(run_dir) if rank_present else None,
    }


def _join_kind(j: dict[str, Any] | None) -> str:
    """What ``joined.json`` holds for a shortlist candidate (see :func:`candidates_view`)."""
    if j is None:
        return "missing"
    return "ranked" if j.get("exomiser_rank") is not None else "unranked"


def _exomiser_join(j: dict[str, Any] | None, index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Stage 4's fields for one candidate, or ``None`` when ``joined.json`` holds no
    rank for it — because Exomiser ranked no gene for it, or because the candidate is
    not in the join at all; the caller tells those apart with :func:`_join_kind`."""
    if not j or j.get("exomiser_rank") is None:
        return None
    rid = j.get("exomiser_evidence_id")
    return {
        "rank": j.get("exomiser_rank"),
        "score": j.get("exomiser_score"),
        "phenotype_score": j.get("phenotype_score"),
        "variant_score": j.get("variant_score"),
        "moi": j.get("exomiser_moi"),
        "gene_symbol": j.get("exomiser_gene_symbol"),
        "match": j.get("exomiser_match"),
        "variants": list(j.get("exomiser_variants") or []),
        "variants_matched": list(j.get("exomiser_variants_matched") or []),
        "by_moi": dict(j.get("exomiser_by_moi") or {}),
        "evidence": resolve([rid], index)[0] if rid else None,
    }


def _mark_agreement(cands: list[dict[str, Any]]) -> None:
    """``agreement`` per candidate: the shortlist's order of the ranked candidates
    against the blind ranker's order of the same candidates (see :func:`candidates_view`).
    A candidate the join does not hold is ``not_joined`` — nothing is known about the
    blind ranker's view of it, which is not the same as ``unranked``."""
    ranked = [c for c in cands if c.get("exomiser")]
    by_exomiser = sorted(ranked, key=lambda c: (c["exomiser"]["rank"], str(c["candidate_id"])))
    for c in cands:
        if c["exomiser"] is None:
            c["agreement"] = "not_joined" if c.get("join") == "missing" else "unranked"
        else:
            c["agreement"] = "agrees" if ranked.index(c) == by_exomiser.index(c) else "disagrees"


def join_status(run_dir: Path) -> dict[str, Any] | None:
    """Whether stage 4's join still describes stage 3's shortlist. ``None`` unless both
    ``03_filter/candidates.json`` and ``04_rank/joined.json`` are present; otherwise
    the shortlist candidates the join does not hold (``missing`` — stage 3 wrote the
    shortlist after stage 4 joined it), the joined candidates the shortlist no longer
    holds (``dropped``), the sha256 of ``candidates.json`` the stage-4 manifest recorded
    as its input beside the file's current one, and ``stale`` when any of them differ.
    A stale join is repaired by ``engine rank --join-only``; nothing here re-joins."""
    run_dir = Path(run_dir)
    doc = read_json(run_dir / "03_filter" / "candidates.json")
    joined = read_json(run_dir / "04_rank" / "joined.json")
    if not isinstance(doc, dict) or not isinstance(joined, dict):
        return None
    now = [str(c.get("candidate_id")) for c in doc.get("candidates", [])]
    then = [str(c.get("candidate_id")) for c in joined.get("candidates", [])]
    recorded = (((manifest(run_dir, "04_rank") or {}).get("inputs") or {}).get("candidates") or {}).get("sha256")
    try:
        current: str | None = sha256_file(run_dir / "03_filter" / "candidates.json")
    except OSError:
        current = None
    missing = [c for c in now if c not in then]
    dropped = [c for c in then if c not in now]
    return {
        "missing": missing,
        "dropped": dropped,
        "candidates_sha256_recorded": recorded,
        "candidates_sha256_current": current,
        "stale": bool(missing or dropped or (recorded and current and recorded != current)),
    }


# ----------------------------------------------------------------------- ranking

def ranking_view(run_dir: Path, top_n: int = DEFAULT_TOP_N) -> dict[str, Any]:
    """Stage 4's blind ranking: the top ``top_n`` rows of ``ranking.tsv`` plus every
    row whose gene is on the stage-3 shortlist, each marked ``in_shortlist`` with the
    candidate id it belongs to and ``joined`` (whether ``joined.json`` holds that
    candidate), and the ``exomiser_only`` genes ``joined.json`` recorded (ranked genes
    the shortlist did not contain when stage 4 joined). ``in_shortlist`` follows the
    shortlist as ``03_filter/candidates.json`` holds it now — ``joined.json``'s copy of
    it supplies the symbol Exomiser matched a candidate under, and stands alone only
    when stage 3's file is gone. Scores are the file's own text."""
    run_dir = Path(run_dir)
    stage = run_dir / "04_rank"
    joined = read_json(stage / "joined.json")
    rows = _ranking_rows(stage / "ranking.tsv")
    if not isinstance(joined, dict) and rows is None:
        return {"present": False, "rows": [], "exomiser_only": [], "counts": {}, "join_check": None}
    joined = joined if isinstance(joined, dict) else {}
    index = evidence_index(run_dir)
    shortlist = _shortlist(run_dir)
    joined_ids = {str(c.get("candidate_id")) for c in joined.get("candidates", [])}
    by_gene: dict[str, str] = {}
    for c in joined.get("candidates", []):
        for symbol in (c.get("exomiser_gene_symbol"), c.get("gene_symbol")):
            if symbol and symbol not in by_gene:
                by_gene[str(symbol)] = str(c.get("candidate_id"))
    out_rows = []
    for i, r in enumerate(rows or []):
        symbol = r.get("gene_symbol", "")
        cid = by_gene.get(symbol)
        if shortlist is not None:
            cid = cid if cid in shortlist.values() else shortlist.get(symbol)
        if i >= top_n and cid is None:
            continue
        rid = f"exomiser:{symbol}"
        out_rows.append({**r, "variants": [v for v in str(r.get("variants", "")).split(";") if v],
                         "in_shortlist": cid is not None, "candidate_id": cid,
                         "joined": (cid in joined_ids) if cid is not None else None,
                         "evidence": resolve([rid], index)[0]})
    return {
        "present": True,
        "join_check": join_status(run_dir),
        "exomiser_version": joined.get("exomiser_version"),
        "data_version": joined.get("data_version"),
        "analysis_yaml_sha256": joined.get("analysis_yaml_sha256"),
        "hpo": list(joined.get("hpo") or []),
        "counts": dict(joined.get("counts") or {}),
        "top_n": top_n,
        "rows_total": len(rows or []),
        "rows": out_rows,
        "exomiser_only": [{**g, "evidence": resolve([g.get("evidence_id") or f"exomiser:{g.get('gene_symbol')}"], index)[0]}
                          for g in joined.get("exomiser_only", [])],
    }


def _shortlist(run_dir: Path) -> dict[str, str] | None:
    """Gene symbol → candidate id of the shortlist as ``03_filter/candidates.json``
    holds it now (the first candidate of a symbol wins); ``None`` without stage 3."""
    doc = read_json(run_dir / "03_filter" / "candidates.json")
    if not isinstance(doc, dict):
        return None
    out: dict[str, str] = {}
    for c in doc.get("candidates", []):
        symbol = c.get("gene_symbol")
        if symbol and str(symbol) not in out:
            out[str(symbol)] = str(c.get("candidate_id"))
    return out


def _ranking_rows(path: Path) -> list[dict[str, str]] | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    if not lines:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:] if line]


# ------------------------------------------------------------------------ chains

def chain_view(run_dir: Path, candidate_id: str | None = None) -> dict[str, Any]:
    """Stage 5: every chain in ``05_reason/chains/`` (or the one named), each criterion's
    evidence ids resolved to URLs, the engine-computed classification as the chain
    carries it, the validator's report for the candidate (``validation/``) with its
    rejections and disputes in full, and whether the stage-5 manifest claims the chain
    — a chain copied in beside a dry run (the public demo) is shown as such. A live
    run that failed on the model leaves no manifest and no chain, only
    ``transcripts/<candidate>.failed.json``; those are ``failures``."""
    run_dir = Path(run_dir)
    stage = run_dir / "05_reason"
    m = manifest(run_dir, "05_reason")
    params = (m or {}).get("params") or {}
    chains_dir = stage / "chains"
    present = stage.is_dir()
    files = sorted(chains_dir.glob("*.json")) if chains_dir.is_dir() else []
    index = evidence_index(run_dir)
    candidates = {c.get("candidate_id"): c for c in (read_json(run_dir / "03_filter" / "candidates.json") or {}).get("candidates", [])}
    chains = []
    for path in files:
        doc = read_json(path)
        if not isinstance(doc, dict):
            continue
        cid = str(doc.get("candidate_id") or path.stem)
        if candidate_id is not None and cid != candidate_id:
            continue
        chains.append(_chain(doc, path, stage, index, candidates.get(cid), m))
    chains.sort(key=lambda c: (c.get("priority") if c.get("priority") is not None else 1 << 30, c["candidate_id"]))
    return {
        "present": present,
        "manifest": m is not None,
        "failures": failures(stage),
        "dry_run": params.get("dry_run"),
        "model": params.get("model"),
        "effort": params.get("effort"),
        "disclosure": params.get("disclosure"),
        "prompt_version": params.get("prompt_version"),
        "candidates_selected": list(params.get("candidates") or []),
        "chains_written": ((m or {}).get("counts") or {}).get("chains_written"),
        "chains": chains,
        "notes": list((m or {}).get("notes") or []),
    }


def _chain(doc: dict[str, Any], path: Path, stage: Path, index: dict[str, dict[str, Any]],
           candidate: dict[str, Any] | None, m: dict[str, Any] | None) -> dict[str, Any]:
    cid = str(doc.get("candidate_id") or path.stem)
    params = (m or {}).get("params") or {}
    variants = []
    for v in doc.get("variants", []):
        variants.append({
            "key": v.get("key"),
            "classification": v.get("classification"),
            "summary": v.get("summary", ""),
            "criteria": [{
                "code": c.get("code"), "strength": c.get("strength"), "met": c.get("met"),
                "justification": c.get("justification", ""),
                "evidence": resolve(list(c.get("evidence_ids") or []), index),
            } for c in v.get("criteria", [])],
        })
    texts = [doc.get("phase_statement", ""), doc.get("mechanism_hypothesis", "")]
    texts += list(doc.get("limits") or []) + list(doc.get("what_would_change_the_call") or [])
    texts += [v["summary"] for v in variants] + [c["justification"] for v in variants for c in v["criteria"]]
    cited = cited_in(texts, index)
    for v in variants:
        for c in v["criteria"]:
            for e in c["evidence"]:
                if e["id"] not in cited:
                    cited.append(e["id"])
    for rid in doc.get("literature") or []:
        if rid not in cited:
            cited.append(str(rid))
    validation = read_json(stage / "validation" / f"{candidate_file_name(cid)}.json")
    claimed = bool(m) and not params.get("dry_run") and cid in (params.get("candidates") or [])
    return {
        "candidate_id": cid,
        "file": str(path.relative_to(stage.parent)),
        "gene_symbol": (candidate or {}).get("gene_symbol"),
        "model": (candidate or {}).get("model"),
        "priority": (candidate or {}).get("priority"),
        "variants": variants,
        "phase_statement": doc.get("phase_statement", ""),
        "mechanism_hypothesis": doc.get("mechanism_hypothesis", ""),
        "limits": list(doc.get("limits") or []),
        "what_would_change_the_call": list(doc.get("what_would_change_the_call") or []),
        "literature": resolve(list(doc.get("literature") or []), index),
        "references": resolve(sorted(cited), index),
        "validation": _validation(validation),
        "claimed_by_manifest": claimed,
        "manifest_note": None if claimed else (
            "the stage-5 manifest records a dry run and claims no chain; this file was placed in chains/ by other means"
            if params.get("dry_run") else "the stage-5 manifest does not list this candidate" if m else
            "stage 5 wrote no manifest"),
    }


def failures(stage: Path) -> list[dict[str, Any]]:
    """The ``transcripts/<candidate>.failed.json`` records a live stage 5 or 6 writes
    when the model call raises (``AgentError``: overloaded, timed out, over the turn
    limit, no schema-valid answer) and then stops without a manifest — so this file
    is the run's only record of what happened. Each: ``candidate_id``, the error
    text (which never quotes the model), the API request id, the turns completed
    before the failure, and the file. A rerun of the stage clears them."""
    out = []
    suffix = ".failed.json"
    for path in sorted((Path(stage) / "transcripts").glob(f"*{suffix}")):
        doc = read_json(path)
        doc = doc if isinstance(doc, dict) else {}
        out.append({
            "candidate_id": str(doc.get("candidate_id") or path.name[:-len(suffix)]),
            "error": str(doc.get("error") or "failure record is not readable"),
            "request_id": doc.get("request_id"),
            "turns": len(doc.get("transcript") or []),
            "file": str(path.relative_to(Path(stage).parent)),
        })
    return out


def _validation(doc: Any) -> dict[str, Any] | None:
    if not isinstance(doc, dict):
        return None
    return {
        "rejections": [dict(r) for r in doc.get("rejections") or []],
        "disputes": [dict(d) for d in doc.get("disputes") or []],
        "notes": list(doc.get("notes") or []),
        "counts": dict(doc.get("counts") or {}),
        "rules": dict(doc.get("rules") or {}),
        "thresholds": dict(doc.get("thresholds") or {}),
    }


# ---------------------------------------------------------------------- medicine

def medicine_view(run_dir: Path) -> dict[str, Any]:
    """Stage 6 in the rubric's order — mechanism, pathway targets, drug candidates
    with their counter-arguments, follow-up experiments, limits — every evidence and
    trial id resolved to its URL, plus the validator's report and the stage-5 verdict
    the report was written against (from the stage-6 bundle). A dry run has a bundle
    and a prompt but no report; that is said, not hidden — as is a live run that
    failed on the model (``failures``, see :func:`failures`)."""
    run_dir = Path(run_dir)
    stage = run_dir / "06_medicine"
    m = manifest(run_dir, "06_medicine")
    params = (m or {}).get("params") or {}
    report = read_json(stage / "report.json")
    if not stage.is_dir():
        return {"present": False, "report": None}
    index = evidence_index(run_dir)
    failed = failures(stage)
    cid = params.get("candidate") or (report or {}).get("candidate_id") or (failed[0]["candidate_id"] if failed else None)
    bundle = read_json(stage / "bundles" / f"{candidate_file_name(str(cid))}.json") if cid else None
    verdicts = [{"key": v.get("key"), "classification": v.get("classification")}
                for v in ((bundle or {}).get("chain") or {}).get("variants", [])]
    validation = read_json(stage / "validation" / f"{candidate_file_name(str(cid))}.json") if cid else None
    out: dict[str, Any] = {
        "present": True,
        "manifest": m is not None,
        "failures": failed,
        "dry_run": params.get("dry_run"),
        "model": params.get("model"),
        "effort": params.get("effort"),
        "disclosure": params.get("disclosure"),
        "candidate_id": cid,
        "gene_symbol": (bundle or {}).get("gene_symbol") or (report or {}).get("gene_symbol"),
        "stage5_verdicts": verdicts,
        "notes": list((m or {}).get("notes") or []),
        "validation": _validation(validation),
        "report": None,
    }
    if not isinstance(report, dict):
        return out
    texts: list[str] = []

    def claims(items: Any) -> list[dict[str, Any]]:
        return [{"statement": c.get("statement", ""), "evidence": resolve(list(c.get("evidence_ids") or []), index)}
                for c in items or []]

    drugs = []
    for i, d in enumerate(report.get("candidates") or [], 1):
        drugs.append({
            "n": i, "name": d.get("name"), "chembl_id": d.get("chembl_id"),
            "chembl": resolve([f"chembl:{d['chembl_id']}"], index)[0] if d.get("chembl_id") and f"chembl:{d['chembl_id']}" in index else None,
            "mechanism_of_action": d.get("mechanism_of_action", ""), "approval_status": d.get("approval_status", ""),
            "rationale": d.get("rationale", ""), "counter_arguments": list(d.get("counter_arguments") or []),
            "evidence": resolve(list(d.get("evidence_ids") or []), index), "trials": resolve(list(d.get("trial_ids") or []), index),
        })
        texts += [d.get("rationale", ""), d.get("mechanism_of_action", ""), d.get("approval_status", "")] + list(d.get("counter_arguments") or [])
    mechanism = claims(report.get("mechanism"))
    pathway = claims(report.get("pathway_targets"))
    texts += [c["statement"] for c in mechanism + pathway]
    texts += list(report.get("follow_up_experiments") or []) + list(report.get("limits") or [])
    cited = cited_in(texts, index)
    for group in (mechanism, pathway):
        for c in group:
            cited += [e["id"] for e in c["evidence"] if e["id"] not in cited]
    for d in drugs:
        cited += [e["id"] for e in d["evidence"] + d["trials"] if e["id"] not in cited]
        if d["chembl"] and d["chembl"]["id"] not in cited:
            cited.append(d["chembl"]["id"])
    cited += [str(r) for r in report.get("literature") or [] if str(r) not in cited]
    out["report"] = {
        "candidate_id": report.get("candidate_id"),
        "gene_symbol": report.get("gene_symbol"),
        "mechanism": mechanism,
        "pathway_targets": pathway,
        "candidates": drugs,
        "follow_up_experiments": list(report.get("follow_up_experiments") or []),
        "limits": list(report.get("limits") or []),
        "literature": resolve(list(report.get("literature") or []), index),
        "references": resolve(sorted(dict.fromkeys(cited)), index),
    }
    return out


# -------------------------------------------------------------------- provenance

def provenance_view(run_dir: Path) -> dict[str, Any]:
    """Every manifest in the run, in stage order: inputs (path, bytes, sha256), tools,
    params, counts, outputs and notes, exactly as written."""
    run_dir = Path(run_dir)
    out = []
    for stage_dir, name in STAGES:
        m = manifest(run_dir, stage_dir)
        if m is None:
            out.append({"dir": stage_dir, "stage": name, "present": (run_dir / stage_dir).is_dir(), "manifest": None})
            continue
        out.append({
            "dir": stage_dir, "stage": name, "present": True,
            "manifest": {
                "stage": m.get("stage"), "engine_version": m.get("engine_version"), "platform": m.get("platform"),
                "started_at": m.get("started_at"), "finished_at": m.get("finished_at"),
                "inputs": [{"name": k, **(v or {})} for k, v in (m.get("inputs") or {}).items()],
                "tools": dict(m.get("tools") or {}),
                "params": m.get("params") or {},
                "counts": m.get("counts") or {},
                "outputs": [{"name": k, **(v or {})} for k, v in (m.get("outputs") or {}).items()],
                "notes": list(m.get("notes") or []),
            },
        })
    return {"run_dir": str(run_dir), "stages": out}
