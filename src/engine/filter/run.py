"""Stage 3 orchestrator — ``engine filter``.

Streams the stage-2 table once through the per-row rules, keeps only the survivors
in memory (a few thousand rows at most; the decisions for every row are small),
resolves the gene-level models, and writes three views of the same result:

* ``decisions.tsv.gz`` — every input row with the rule that decided it, so a judge
  can answer "why is variant X not on the list" without rerunning anything;
* ``shortlist.tsv.gz`` — the kept rows with every annotated column plus the model
  they support;
* ``candidates.json`` — the gene-level candidates in priority order, the shape the
  ranking and reasoning stages consume.

Everything is a pure function of the table and ``configs/filter.yaml``: no network,
no clock in the data files (gzip ``mtime=0``, sorted JSON keys), so a rerun over the
same inputs is byte-identical and the manifest is the only thing that carries a time.
"""

from __future__ import annotations

import gzip
import io
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Iterable

from engine.filter.config import DEFAULT_CONFIG, FilterConfig
from engine.filter.rules import (
    DECISION_COLUMNS, MODEL_ORDER, REQUIRED_COLUMNS, SHORTLIST_EXTRA_COLUMNS, Candidate, Decision, Survivor,
    candidate_view, phase_counts, resolve_models, rule_family, screen_rows,
)
from engine.manifest import Manifest
from engine.retrieve.clinvar import ClinvarRetriever
from engine.retrieve.gnomad import GnomadRetriever
from engine.retrieve.run import STAGE_DIR as RETRIEVE_DIR
from engine.retrieve.run import read_annotated
from engine.retrieve.store import key_str
from engine.retrieve.vep import VepRetriever

STAGE_DIR = "03_filter"

SOURCE_COLUMNS: dict[str, tuple[str, ...]] = {
    "vep": tuple(VepRetriever.columns),
    "clinvar": tuple(ClinvarRetriever.columns),
    "gnomad": tuple(GnomadRetriever.columns),
}
"""What each retriever contributes to the table — read from the classes, echoed into
the manifest so the column contract this stage ran against is on record."""

RULE_ORDER = ("duplicate", "genotype", "sex", "consequence", "clinvar_benign", "rarity", "quality(caveats)", "model", "phase")
PRIORITY_KEY = ("clinvar_plp desc", "model rank", "dense_cluster asc", "best impact_any_coding desc", "max af_used asc", "gene", "candidate_id")


def table_header(run_dir: Path) -> list[str]:
    """The stage-2 header. Stage 2 writes the header with its first row, so a run
    with no variant in scope leaves an empty file: that is refused here with its
    real cause rather than reported as a missing AF column."""
    path = Path(run_dir) / RETRIEVE_DIR / "variants.annotated.tsv.gz"
    with gzip.open(path, "rt") as f:
        line = f.readline().rstrip("\n")
    if not line:
        raise ValueError(f"{path} is empty: stage 2 wrote no rows (no variant in scope), so there is no table to filter")
    return line.split("\t")


def _open_gz_text(path: Path) -> io.TextIOWrapper:
    # mtime=0: no timestamp in the gzip header, so identical content is an identical file.
    return io.TextIOWrapper(gzip.GzipFile(path, "wb", mtime=0), encoding="utf-8", newline="\n")


def write_decisions(path: Path, decisions: Iterable[Decision]) -> int:
    n = 0
    with _open_gz_text(path) as out:
        out.write("\t".join(DECISION_COLUMNS) + "\n")
        for d in decisions:
            out.write("\t".join(d.to_row()) + "\n")
            n += 1
    return n


def write_shortlist(path: Path, header: list[str], survivors: Iterable[Survivor]) -> int:
    """Kept rows in input order: every annotated column, then the model columns."""
    n = 0
    with _open_gz_text(path) as out:
        out.write("\t".join(list(header) + list(SHORTLIST_EXTRA_COLUMNS)) + "\n")
        for d, row in survivors:
            if not d.kept:
                continue
            extra = [d.model, ";".join(key_str(k) for k in d.partner_keys), d.phase, ";".join(d.caveats),
                     d.af_used, d.af_source, d.candidate_id]
            out.write("\t".join([row.get(c, "") or "" for c in header] + extra) + "\n")
            n += 1
    return n


def write_candidates(path: Path, cands: list[Candidate], cfg: FilterConfig, counts: dict) -> None:
    doc = {"candidates": [candidate_view(c) for c in cands], "config": cfg.as_params(), "counts": counts}
    path.write_text(json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def sample_sex(run_dir: Path) -> tuple[str, str]:
    """The sex stage 1 recorded (stated in the case file, else inferred from the
    calls), and a sentence for the manifest. ``unknown`` when stage 1 predates the
    field or could not tell."""
    p = run_dir / "01_ingest" / "manifest.json"
    if not p.exists():
        return "unknown", ""
    params = json.loads(p.read_text()).get("params", {})
    sex = params.get("sex") or "unknown"
    stated, inferred = params.get("sex_stated", "unknown"), (params.get("sex_inference") or {}).get("inferred", "unknown")
    if sex == "unknown":
        return sex, ("Sample sex unknown (not stated, not inferable from the calls): the sex rule is off, so X "
                     "heterozygotes can pair as comphet and Y calls are kept.")
    how = "stated in the case file" if stated == sex else "inferred from the calls (X het fraction, Y calls)"
    return sex, f"Sample sex {sex} ({how}); genotypes a {sex} karyotype cannot carry are dropped as sex:*."


def run_filter(run_dir: Path, config_path: Path = DEFAULT_CONFIG) -> Path:
    """Run stage 3 for ``run_dir`` into ``run_dir/03_filter``. Returns the manifest path."""
    run_dir = Path(run_dir)
    config_path = Path(config_path)
    table_path = run_dir / RETRIEVE_DIR / "variants.annotated.tsv.gz"
    if not table_path.exists():
        raise FileNotFoundError(f"stage 2 has not run in {run_dir} ({table_path} missing)")
    cfg = FilterConfig.load(config_path)
    out_dir = run_dir / STAGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    m = Manifest(stage="filter")
    m.add_input("variants_annotated", table_path)
    m.add_input("config", config_path)
    retrieve_manifest = run_dir / RETRIEVE_DIR / "manifest.json"
    if retrieve_manifest.exists():
        m.add_input("retrieve_manifest", retrieve_manifest, checksum=False)
    m.tools["python"] = platform.python_version()

    sex, sex_note = sample_sex(run_dir)
    m.params["sex"] = sex
    if sex_note:
        m.note(sex_note)

    header = table_header(run_dir)
    present = set(header)
    missing_required = [c for c in REQUIRED_COLUMNS if c not in present]
    if missing_required:
        # Rules would read '' for these and, e.g. without gt, drop every row as a non-carrier.
        raise ValueError(f"{table_path.name} lacks stage-1 column(s) {missing_required}; not the contract's table")
    for src, cols in SOURCE_COLUMNS.items():
        missing = [c for c in cols if c not in present]
        if missing:
            m.note(f"{src}: {len(missing)} of {len(cols)} columns absent from the table "
                   f"(source not run in stage 2?): rules read '' for them.")
    af_present = [c for c in cfg.rarity.af_source_order if c in present]
    if not af_present:
        # Every row would read AF 0 and pass the rarity rule: a wrong config, not a result.
        raise ValueError(f"none of rarity.af_source_order {cfg.rarity.af_source_order} is a column of "
                         f"{table_path.name}; the table has no allele frequency to filter on")
    if len(af_present) < len(cfg.rarity.af_source_order):
        m.note(f"af_source_order columns absent from the table: "
               f"{[c for c in cfg.rarity.af_source_order if c not in present]}")
    m.params.update({
        "config_path": str(config_path),
        **cfg.as_params(),
        "rule_order": list(RULE_ORDER),
        "model_rank": list(MODEL_ORDER),
        "priority_key": list(PRIORITY_KEY),
        "columns_by_source": {src: list(cols) for src, cols in SOURCE_COLUMNS.items()},
        "af_source_columns_present": af_present,
        "required_columns": list(REQUIRED_COLUMNS),
        "decisions_columns": list(DECISION_COLUMNS),
        "shortlist_extra_columns": list(SHORTLIST_EXTRA_COLUMNS),
    })

    # ---- pass 1: per-row rules, streaming; only survivors keep their full row
    decisions: list[Decision] = []
    survivors: list[Survivor] = []
    for d, row in screen_rows(read_annotated(run_dir), cfg, sex=sex):
        decisions.append(d)
        if d.rule == "":
            survivors.append((d, row))
    # ---- pass 2: gene models and priority (marks the survivors' decisions)
    cands = resolve_models(survivors, cfg, sex)

    dropped_by_rule = Counter(rule_family(d.rule) for d in decisions if not d.kept)
    phase = phase_counts(cands, survivors)
    counts = {
        "rows_in": len(decisions),
        "rows_after_row_rules": len(survivors),
        "kept": sum(1 for d in decisions if d.kept),
        "dropped": sum(1 for d in decisions if not d.kept),
        "dropped_by_rule": dict(sorted(dropped_by_rule.items())),
        "candidates": len(cands),
        "candidates_by_model": {mdl: n for mdl in MODEL_ORDER if (n := sum(1 for c in cands if c.model == mdl))},
        "genes_with_candidates": len({c.gene_symbol or c.gene_id for c in cands}),
        "rows_with_caveats": sum(1 for d in decisions if d.kept and d.caveats),
        "rarity": {
            "rows_by_af_source": dict(sorted(Counter(d.af_source for d in decisions).items())),
            "af_fallback_first_masked_rows": sum(1 for d in decisions if d.af_masked),
        },
        "phase": phase,
    }
    if counts["rarity"]["af_fallback_first_masked_rows"]:
        m.note(f"{counts['rarity']['af_fallback_first_masked_rows']} row(s) read a rare AF from the first non-empty "
               f"fallback column while another fallback column was above recessive_max_af "
               f"(rarity.af_fallback_pick=first); set it to max to let the commoner copy decide.")
    if phase["shared_pid_opposite_pgt_pairs"] and not cfg.phase.trust_pgt:
        m.note(f"{phase['shared_pid_opposite_pgt_pairs']} heterozygous pair(s) share a PID with opposite PGT "
               f"(the caller phased them in trans) but phase.trust_pgt is false, so the PID-only rule read them "
               f"as cis. Review decisions.tsv.gz rows with rule phase:cis before trusting a missing comphet.")
    if phase["cis_rows_dropped_het_single_eligible"]:
        m.note(f"{phase['cis_rows_dropped_het_single_eligible']} heterozygous row(s) dropped as phase:cis would have "
               f"qualified as het_single alone (HIGH or ClinVar P/LP at AF <= dominant_max_af); they carry the hit "
               f"model:het_single_eligible in decisions.tsv.gz. A dominant candidate can hide behind a phased neighbour.")

    decisions_path = out_dir / "decisions.tsv.gz"
    shortlist_path = out_dir / "shortlist.tsv.gz"
    candidates_path = out_dir / "candidates.json"
    write_decisions(decisions_path, decisions)
    write_shortlist(shortlist_path, header, survivors)
    write_candidates(candidates_path, cands, cfg, counts)

    m.counts.update(counts)
    m.add_output("decisions", decisions_path)
    m.add_output("shortlist", shortlist_path)
    m.add_output("candidates", candidates_path)
    manifest_path = out_dir / "manifest.json"
    m.write(manifest_path)
    return manifest_path


def read_candidates(run_dir: Path) -> dict:
    """``candidates.json`` of a run — for later stages and tests."""
    return json.loads((Path(run_dir) / STAGE_DIR / "candidates.json").read_text())


def read_decisions(run_dir: Path) -> Iterable[dict[str, str]]:
    """``decisions.tsv.gz`` of a run as dicts — for later stages and tests."""
    path = Path(run_dir) / STAGE_DIR / "decisions.tsv.gz"
    with gzip.open(path, "rt") as f:
        cols = f.readline().rstrip("\n").split("\t")
        for line in f:
            yield dict(zip(cols, line.rstrip("\n").split("\t")))


def read_shortlist(run_dir: Path) -> Iterable[dict[str, str]]:
    """``shortlist.tsv.gz`` of a run as dicts — for later stages and tests."""
    path = Path(run_dir) / STAGE_DIR / "shortlist.tsv.gz"
    with gzip.open(path, "rt") as f:
        cols = f.readline().rstrip("\n").split("\t")
        for line in f:
            yield dict(zip(cols, line.rstrip("\n").split("\t")))
