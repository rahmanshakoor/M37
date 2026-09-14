"""What the model sees — and a file that proves it.

The reasoning agent must argue only from retrieved records, so the prompt it receives
is built here from the run directory and nothing else: every evidence record the
store holds for the candidate's variants (full payloads, so a number the model quotes
can be checked), the columns stage 2 projected from them (as stage 3 saw them), the
stage-4 phenotype rank when there is one, and the case's HPO terms. The same content
is rendered twice — as JSON, written to ``05_reason/bundles/<candidate_id>.json`` so a
judge can open exactly what the model was given, and as a compact text block for the
prompt, with every fact followed by the record id it came from (``[gnomad:…]``) so
the model learns the ids it may cite by reading them.

Deterministic on purpose: records are sorted by id, keys are sorted in the JSON,
nothing here reads a clock. Same run directory, same bytes.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from engine.retrieve.clinvar import ClinvarRetriever
from engine.retrieve.gnomad import GnomadRetriever, unaskable, variant_id
from engine.retrieve.store import EvidenceRecord, EvidenceStore, parse_key
from engine.retrieve.vep import VepRetriever

RETRIEVE_DIR = "02_retrieve"
FILTER_DIR = "03_filter"
RANK_DIR = "04_rank"
BUNDLE_DIR = "bundles"

RANK_FIELDS = {
    "exomiser_rank": ("exomiser_rank",),
    "exomiser_score": ("exomiser_score",),
    "phenotype_score": ("phenotype_score",),
    "variant_score": ("variant_score",),
    "moi": ("exomiser_moi", "moi"),
    "evidence_id": ("exomiser_evidence_id",),
}
"""What the bundle keeps from a ``joined.json`` entry: bundle key → the entry keys it
is read from, first present wins (``engine.rank.run`` spells the MOI ``exomiser_moi``
and names the citable ``exomiser:<gene>`` record in ``exomiser_evidence_id``)."""


@dataclass
class VariantBundle:
    key: str
    fields: dict[str, Any]
    """The candidate's variant entry from ``candidates.json`` — stage 3's projection."""
    columns: dict[str, str] = field(default_factory=dict)
    """The stage-2 projection of ``records`` (each retriever's own ``extract``), with the
    run's annotated row (``03_filter/shortlist.tsv.gz``) laid over it when there is one."""
    records: list[dict[str, Any]] = field(default_factory=list)
    """Every evidence record for this variant, full payload, sorted by ``record_id``."""
    missing_ids: list[str] = field(default_factory=list)
    """Ids the candidate lists that the store does not hold — visible, never silent."""


@dataclass
class Bundle:
    candidate_id: str
    candidate: dict[str, Any]
    variants: list[VariantBundle]
    rank: dict[str, Any] | None
    """Stage-4 join for this candidate; ``None`` when stage 4 has not run."""
    case_hpo: list[str]
    record_ids: list[str]
    """Every record id in the bundle — the citation vocabulary."""
    gene_records: list[dict[str, Any]] = field(default_factory=list)
    """Records about the candidate's gene rather than one variant: the stage-4
    ``exomiser:<gene>`` record when the rank names one and the store holds it."""
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=1, ensure_ascii=False) + "\n"


# ----------------------------------------------------------------------------- build

def build_bundle(candidate: dict[str, Any] | str, run_dir: str | Path, case_hpo: Iterable[str] = ()) -> Bundle:
    """``candidate`` is an entry of ``03_filter/candidates.json`` (or its
    ``candidate_id``, looked up there). Reads ``02_retrieve/evidence`` (required),
    ``03_filter/shortlist.tsv.gz`` and ``04_rank/joined.json`` (both optional)."""
    run_dir = Path(run_dir)
    if isinstance(candidate, str):
        candidate = find_candidate(run_dir, candidate)
    cid = str(candidate["candidate_id"])
    evidence_dir = run_dir / RETRIEVE_DIR / "evidence"
    if not evidence_dir.is_dir():
        raise FileNotFoundError(f"stage 2 has not run in {run_dir} ({evidence_dir} missing)")
    store = EvidenceStore(evidence_dir)

    keys = [str(v["key"]) for v in candidate.get("variants", [])]
    rows = read_shortlist_rows(run_dir, keys)
    variants = [_variant_bundle(v, store, rows.get(str(v["key"]), {})) for v in candidate.get("variants", [])]
    hpo = sorted(dict.fromkeys(str(h) for h in case_hpo))
    rank = load_rank(run_dir, cid, str(candidate.get("gene_symbol", "")))
    gene_records = [asdict(r) for r in _rank_records(run_dir, rank)]
    bundle = Bundle(
        candidate_id=cid,
        candidate=json.loads(json.dumps(candidate, sort_keys=True)),
        variants=variants,
        rank=rank,
        case_hpo=hpo,
        record_ids=sorted({r["record_id"] for v in variants for r in v.records} | {r["record_id"] for r in gene_records}),
        gene_records=gene_records,
    )
    bundle.text = render_text(bundle)
    return bundle


def find_candidate(run_dir: Path, candidate_id: str) -> dict[str, Any]:
    path = Path(run_dir) / FILTER_DIR / "candidates.json"
    doc = json.loads(path.read_text())
    for c in doc.get("candidates", []):
        if c.get("candidate_id") == candidate_id:
            return c
    raise KeyError(f"no candidate {candidate_id!r} in {path}")


def write_bundle(bundle: Bundle, stage_dir: str | Path) -> Path:
    """``<stage_dir>/bundles/<candidate_id>.json``. Unchanged bytes are left alone."""
    out = Path(stage_dir) / BUNDLE_DIR / f"{bundle.candidate_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = bundle.to_json()
    if not (out.exists() and out.read_text() == text):
        out.write_text(text)
    return out


def _variant_bundle(v: dict[str, Any], store: EvidenceStore, columns: dict[str, str]) -> VariantBundle:
    key = str(v["key"])
    wanted = list(dict.fromkeys(str(i) for i in v.get("evidence_ids", []) if i))
    for rid in _canonical_ids(key):  # what stage 2 would have written, if the list is short
        if rid not in wanted and store.exists(rid):
            wanted.append(rid)
    records: dict[str, EvidenceRecord] = {}
    missing: list[str] = []
    for rid in wanted:
        rec = store.get(rid)
        if rec is None:
            missing.append(rid)
        else:
            records[rid] = rec
    cols = project_columns(records.values())
    cols.update({k: v for k, v in columns.items() if v not in (None, "")})
    return VariantBundle(
        key=key,
        fields=json.loads(json.dumps(v, sort_keys=True)),
        columns=dict(sorted(cols.items())),
        records=[asdict(records[rid]) for rid in sorted(records)],
        missing_ids=sorted(missing),
    )


def project_columns(records: Iterable[EvidenceRecord]) -> dict[str, str]:
    """The table columns stage 2 derives from these records, by the retrievers' own
    ``extract`` (pure projections; no request is made). Sources without a projection
    here (``pmid``, ``nct``, …) contribute nothing."""
    by_source: dict[str, list[EvidenceRecord]] = {}
    for rec in sorted(records, key=lambda r: r.record_id):
        by_source.setdefault(rec.source, []).append(rec)
    out: dict[str, str] = {}
    if by_source.get("vep"):
        out.update(VepRetriever(None).extract(by_source["vep"]))
    if by_source.get("clinvar"):
        # extract() reads only ``columns`` from self; a VCF is not needed for a projection
        out.update(ClinvarRetriever.extract(SimpleNamespace(columns=ClinvarRetriever.columns), by_source["clinvar"]))
    if by_source.get("gnomad"):
        out.update(GnomadRetriever(None).extract(by_source["gnomad"]))
    return {k: v for k, v in out.items() if v not in (None, "")}


def _canonical_ids(key: str) -> list[str]:
    try:
        k = parse_key(key)
    except ValueError:
        return []
    ids = [f"vep:{key}"]
    if not unaskable(k):
        ids.append(f"gnomad:{variant_id(k)}")
    return ids


# ------------------------------------------------------------------- run-dir readers

def read_shortlist_rows(run_dir: Path, keys: Iterable[str]) -> dict[str, dict[str, str]]:
    """The annotated rows of ``keys`` from ``03_filter/shortlist.tsv.gz`` (kept rows,
    every stage-2 column). ``{}`` when stage 3 has not written one."""
    path = Path(run_dir) / FILTER_DIR / "shortlist.tsv.gz"
    wanted = set(keys)
    out: dict[str, dict[str, str]] = {}
    if not path.exists() or not wanted:
        return out
    with gzip.open(path, "rt", encoding="utf-8") as f:
        cols = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(cols, line.rstrip("\n").split("\t")))
            key = f"{row.get('chrom')}:{row.get('pos')}:{row.get('ref')}:{row.get('alt')}"
            if key in wanted and key not in out:
                out[key] = row
    return out


def load_rank(run_dir: Path, candidate_id: str, gene_symbol: str) -> dict[str, Any] | None:
    """This candidate's entry in ``04_rank/joined.json`` reduced to :data:`RANK_FIELDS`;
    ``{"exomiser_rank": None}`` when stage 4 ran but did not rank the gene (its entry
    says so); ``None`` when stage 4 has not run. Stage 4 writes every stage-3
    candidate, so a ``joined.json`` with no entry for this one — or in a shape this
    reader does not know — raises rather than quietly reporting "not ranked": an
    absence in the run directory is not a finding about the gene."""
    path = Path(run_dir) / RANK_DIR / "joined.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    entries = doc.get("candidates") if isinstance(doc, dict) else doc
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ValueError(f"{path}: expected a 'candidates' list of objects (or a top-level list)")
    match = next((e for e in entries if e.get("candidate_id") == candidate_id), None)
    if match is None:
        match = next((e for e in entries if e.get("gene_symbol") == gene_symbol and "candidate_id" not in e), None)
    if match is None:
        raise ValueError(f"{path}: no entry for candidate {candidate_id!r} (gene {gene_symbol!r}) — "
                         "stage 4 did not run over this stage-3 shortlist")
    out: dict[str, Any] = {}
    for name, sources in RANK_FIELDS.items():
        for src in sources:
            if src in match:
                out[name] = match[src]
                break
    return out or {"exomiser_rank": None}


def _rank_records(run_dir: Path, rank: dict[str, Any] | None) -> list[EvidenceRecord]:
    """The ``exomiser:`` record the rank names, from ``04_rank/evidence``, when it exists."""
    rid = (rank or {}).get("evidence_id")
    evidence_dir = Path(run_dir) / RANK_DIR / "evidence"
    if not isinstance(rid, str) or not rid or not evidence_dir.is_dir():
        return []
    rec = EvidenceStore(evidence_dir).get(rid)
    return [rec] if rec is not None else []


# ----------------------------------------------------------------------------- text

def render_text(bundle: Bundle) -> str:
    """The prompt block: one candidate header, then one section per variant with every
    fact tagged by its record id. Compact, fixed order, no payloads (the model has
    ``get_record`` for those)."""
    c = bundle.candidate
    lines = [
        f"# Candidate {bundle.candidate_id}",
        (f"gene: {c.get('gene_symbol') or '-'} ({c.get('gene_id') or 'no gene id'}) · model: {c.get('model') or '-'} · "
         f"priority: {c.get('priority', '-')}"),
        f"phase: {_phase(c.get('phase'))}",
        f"rule hits: {'; '.join(c.get('rule_hits') or []) or 'none'}",
        f"caveats: {'; '.join(c.get('caveats') or []) or 'none'}",
        f"exomiser: {_rank_line(bundle.rank)}" + _tag([r["record_id"] for r in bundle.gene_records]),
        f"case HPO: {', '.join(bundle.case_hpo) or 'none given'}",
    ]
    for v in bundle.variants:
        lines.append("")
        lines.extend(_variant_lines(v))
    lines.append("")
    lines.append("citable record ids: " + (" ".join(f"[{r}]" for r in bundle.record_ids) or "none"))
    return "\n".join(lines) + "\n"


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
    for name, label in (("exomiser_score", "combined"), ("phenotype_score", "phenotype"),
                        ("variant_score", "variant"), ("moi", "moi")):
        if rank.get(name) is not None:
            parts.append(f"{label} {rank[name]}")
    return " · ".join(parts)


def _variant_lines(v: VariantBundle) -> list[str]:
    get = _getter(v)
    by_source = _ids_by_source(v)
    vep = _tag(by_source.get("vep"))
    lines = [f"## Variant {v.key}"]
    flag = get("quality_flag")
    lines.append(f"genotype: {get('gt')} · AD {get('ad')} · DP {get('dp')} · GQ {get('gq')} · "
                 f"filter: {'PASS' if flag == '-' else flag}")
    mane = get("mane")
    lines.append(f"consequence: {get('consequence')} ({get('impact')}) · {get('transcript_id')}"
                 f"{' (MANE ' + mane + ')' if mane and mane != '-' else ''} · {get('hgvsc')} · {get('hgvsp')}{vep}")
    lines.append(f"in silico: SIFT {get('sift_pred')} · PolyPhen {get('polyphen_pred')} · "
                 f"SpliceAI ds_max {get('spliceai_ds_max')} · CADD {get('cadd_phred')}{vep}")
    lines.append(_gnomad_line(v, get, by_source))
    lines.append(f"stage-3 AF: {_num(get('af_used'))} from {get('af_source') if get('af_source') != '-' else 'nothing (absent everywhere = 0)'}")
    lines.append(_clinvar_line(v, get, by_source))
    if v.fields.get("caveats"):
        lines.append("caveats: " + "; ".join(str(x) for x in v.fields["caveats"]))
    lines.append("records: " + (" ".join(f"[{r['record_id']}]" for r in v.records) or "none in the store"))
    if v.missing_ids:
        lines.append("listed but not in the store: " + ", ".join(v.missing_ids))
    return lines


def _gnomad_line(v: VariantBundle, get: Any, by_source: dict[str, list[str]]) -> str:
    ids = by_source.get("gnomad")
    if not ids:
        af, src = get("af_used"), get("af_source")
        return f"gnomAD: no record in the store (af_used {af} from {src or 'nothing'})"
    parts = [f"af {_num(get('gnomad_af'))}"]
    if get("gnomad_ac") != "-" and get("gnomad_an") != "-":
        parts[0] += f" (ac {get('gnomad_ac')} / an {get('gnomad_an')})"
    parts.append(f"hom {get('gnomad_nhom')}")
    if get("gnomad_grpmax_af") != "-":
        parts.append(f"grpmax {_num(get('gnomad_grpmax_af'))} ({get('gnomad_grpmax_pop')})")
    if get("gnomad_faf95_popmax") != "-":
        parts.append(f"faf95 {_num(get('gnomad_faf95_popmax'))} ({get('gnomad_faf95_pop')})")
    parts.append(f"filters {get('gnomad_filters') if get('gnomad_filters') != '-' else 'PASS'}")
    return "gnomAD: " + " · ".join(parts) + _tag(ids)


def _clinvar_line(v: VariantBundle, get: Any, by_source: dict[str, list[str]]) -> str:
    ids = by_source.get("clinvar")
    if not ids:
        return "ClinVar: no record in the store"
    sig = get("clinvar_clnsig") if get("clinvar_clnsig") != "-" else get("clinvar_pathogenicity")
    parts = [f"{sig}", f"{get('clinvar_stars')} stars"]
    if get("clinvar_revstat") != "-":
        parts[-1] += f" ({get('clinvar_revstat')})"
    if get("clinvar_conditions") != "-":
        parts.append(_short(get("clinvar_conditions").replace("_", " ").replace("|", "; ")))
    parts.append(get("clinvar_vcv"))
    return "ClinVar: " + " · ".join(parts) + _tag(ids)


def _getter(v: VariantBundle) -> Any:
    """Column lookup: the annotated row first (every stage-2 column), then the
    candidate's projection; ``-`` when neither has it."""

    def get(name: str) -> str:
        for src in (v.columns, v.fields):
            val = src.get(name)
            if val not in (None, "", [], {}):
                return str(val)
        return "-"
    return get


def _ids_by_source(v: VariantBundle) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in v.records:
        out.setdefault(r["source"], []).append(r["record_id"])
    return out


def _tag(ids: list[str] | None) -> str:
    return "".join(f" [{i}]" for i in (ids or []))


def _num(text: str) -> str:
    """A frequency to four significant digits; anything non-numeric unchanged."""
    try:
        return f"{float(text):.4g}"
    except ValueError:
        return text


def _short(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
