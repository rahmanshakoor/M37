"""Stage 8 — ``engine submit``: the Track 1 CSV, written the way the scorer reads it.

The scorer (the challenge's ``evaluation.py``) compares chromosome strings exactly,
matches a compound-heterozygous answer only when both alleles sit in ONE row, allows
at most ten rows, sorts rows by ``epcr`` descending with ties broken by file order,
and gives half credit for one allele of a pair. Every one of those is a way to score
zero with the right answer, so this stage owns them:

* chromosomes go through :func:`engine.contigs.to_submission` (``15`` → ``chr15``);
* a compound-het candidate is one row with both alleles;
* EPCRs are unique and strictly decreasing, so no tie can pull a wrong variant into
  the same threshold bucket as the right one — and, since EPCR means an estimated
  probability, the lead carries 0.95 and the backups descend from 0.10;
* candidates whose every allele sits in a dense cluster of rare calls (stage 3's
  ``dense_cluster`` caveat: a read pile from a paralog or a divergent haplotype, as
  in SERPINA1 or the HLA genes) are ordered with the artefact families, last;
* candidates the stage-5 chain classified benign or likely benign are left out, as
  are pairs whose every allele ClinVar calls benign or likely benign (a rare
  frameshift pair that ClinVar has already looked at is not a ranked answer), and
  compound heterozygotes on X when the sample is male (a haploid chromosome has no
  second allele; stage 3 now refuses them, this guards older runs);
* ``finding_type`` follows the template's meaning — ``secondary`` is an incidental
  finding: under a model that does not explain this presentation (a dominant single
  heterozygote), a variant the stage-5 chain classified pathogenic or likely
  pathogenic, or one ClinVar calls P/LP at two stars or better. A lone heterozygote
  nobody has classified is a weak ``primary`` candidate, not an incidental finding;
* the file is dry-run through the scorer against the top row as a hypothetical key,
  so a format error is caught before a submission is spent.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.contigs import to_submission
from engine.manifest import Manifest
from engine.retrieve.store import parse_key

STAGE_DIR = "08_submit"
COLUMNS = ("proband_id", "chrom_1", "pos_1", "ref_1", "alt_1", "chrom_2", "pos_2", "ref_2", "alt_2",
           "epcr", "finding_type", "notes")
MAX_ROWS = 10
TOP_EPCR = 0.95
BACKUP_TOP_EPCR = 0.10
EPCR_STEP = 0.01
"""EPCR is the template's *estimated probability of causal relationship*. The lead row
carries the engine's belief (0.95); every row below it is a backup the engine does
not believe is the answer, so backups descend from 0.10 in steps of 0.01 — strictly
decreasing (the scorer breaks ties by file order and the F-max threshold is the lead's
EPCR, so the score is unchanged) and honest about what they are."""
BENIGN = {"benign", "likely_benign"}
PLP = {"pathogenic", "likely_pathogenic"}
CLINVAR_BENIGN = {"Benign", "Likely_benign"}
SECONDARY_MODELS = {"het_single"}
"""Models that do not explain a recessive presentation: a P/LP variant under one is
an incidental (``secondary``) finding."""
SECONDARY_MIN_CLINVAR_STARS = 2
"""A lone heterozygote ClinVar calls P/LP at this review level or better is an
incidental finding whatever the engine's own chain concludes about it — the template
asks for incidental findings, not for the engine's agreement with ClinVar."""
ARTEFACT_FAMILIES = ("HLA-", "MUC", "KIR", "NBPF", "PRAMEF", "TAS2R", "OR", "LILR", "FCGB", "GOLGA", "USP17L", "TBC1D3", "NPIPA", "NPIPB")
"""Gene families whose apparent rare compound-hets are mostly mapping artefacts in
short-read WGS (paralogs, high polymorphism). Ordered last, never removed."""


def is_artefact_family(symbol: str) -> bool:
    s = symbol.upper()
    return any(s.startswith(fam) and (fam != "OR" or (len(s) > 2 and s[2].isdigit())) for fam in ARTEFACT_FAMILIES)


def is_clustered(candidate: dict[str, Any]) -> bool:
    """Every allele carries stage 3's ``dense_cluster`` caveat (a read pile, not a
    genotype): ordered with the artefact families, last."""
    vs = candidate.get("variants") or []
    return bool(vs) and all(any(str(cv).startswith("dense_cluster:") for cv in (v.get("caveats") or [])) for v in vs)


@dataclass
class Row:
    variants: list[tuple[str, int, str, str]]
    epcr: float
    finding_type: str
    notes: str
    candidate_id: str

    def as_csv(self, proband_id: str) -> dict[str, str]:
        a = self.variants[0]
        b = self.variants[1] if len(self.variants) > 1 else None
        return {
            "proband_id": proband_id,
            "chrom_1": to_submission(a[0]), "pos_1": str(a[1]), "ref_1": a[2], "alt_1": a[3],
            "chrom_2": to_submission(b[0]) if b else "", "pos_2": str(b[1]) if b else "",
            "ref_2": b[2] if b else "", "alt_2": b[3] if b else "",
            "epcr": f"{self.epcr:.2f}", "finding_type": self.finding_type, "notes": self.notes,
        }


@dataclass
class SubmitPlan:
    rows: list[Row]
    skipped: list[dict[str, str]] = field(default_factory=list)


def _load(run_dir: Path, rel: str) -> Any:
    p = run_dir / rel
    return json.loads(p.read_text()) if p.exists() else None


def _classifications(run_dir: Path) -> dict[str, dict[str, str]]:
    """candidate_id → {variant key → stage-5 classification}, when stage 5 has run."""
    out: dict[str, dict[str, str]] = {}
    chains = run_dir / "05_reason" / "chains"
    if chains.exists():
        for p in sorted(chains.glob("*.json")):
            ch = json.loads(p.read_text())
            out[ch["candidate_id"]] = {v["key"]: (v.get("classification") or "") for v in ch.get("variants", [])}
    return out


def _ranks(run_dir: Path) -> dict[str, int | None]:
    j = _load(run_dir, "04_rank/joined.json") or {}
    return {c["candidate_id"]: c.get("exomiser_rank") for c in j.get("candidates", [])}


def _sex(run_dir: Path) -> str:
    m = _load(run_dir, "01_ingest/manifest.json") or {}
    return (m.get("params") or {}).get("sex") or "unknown"


def _clinvar_benign(v: dict[str, Any]) -> bool:
    members = [m.split(",")[0] for m in (v.get("clinvar_pathogenicity") or "").split("/") if m]
    return bool(members) and all(m in CLINVAR_BENIGN for m in members)


def _skip_reason(c: dict[str, Any], cl: dict[str, str], sex: str) -> str:
    """Why a candidate takes no row, or ``''``."""
    if cl and any(cl.values()) and all(cl.get(v["key"], "") in BENIGN for v in c["variants"] if cl.get(v["key"])):
        return "stage-5 chain classified benign/likely benign"
    if all(_clinvar_benign(v) for v in c["variants"]):
        return "every allele ClinVar Benign/Likely_benign"
    if c["model"] == "comphet" and sex == "male" and all(v["key"].startswith("X:") for v in c["variants"]):
        return "compound heterozygote on X in a male"
    return ""


def _clinvar_plp(v: dict[str, Any], min_stars: int = SECONDARY_MIN_CLINVAR_STARS) -> bool:
    members = [m.split(",")[0] for m in (v.get("clinvar_pathogenicity") or "").split("/") if m]
    try:
        stars = int(v.get("clinvar_stars") or 0)
    except ValueError:
        stars = 0
    return bool(members) and all(m in ("Pathogenic", "Likely_pathogenic") for m in members) and stars >= min_stars


def _finding_type(c: dict[str, Any], cl: dict[str, str]) -> str:
    """``secondary`` = an incidental finding: under a model that does not explain the
    presentation, a variant the engine's chain classified P/LP, or one ClinVar calls
    P/LP at ≥ SECONDARY_MIN_CLINVAR_STARS — the chain's own verdict is stated in the
    notes either way."""
    if c["model"] in SECONDARY_MODELS and any(cl.get(v["key"], "") in PLP or _clinvar_plp(v) for v in c["variants"]):
        return "secondary"
    return "primary"


def _pair_for(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """The two alleles a compound-het row should carry: the two most consequential
    (HIGH before MODERATE, then lowest frequency). More than two rare hets in a gene
    is possible; the scorer wants exactly the causal pair."""
    order = {"HIGH": 0, "MODERATE": 1, "SPLICE": 1, "LOW": 2, "MODIFIER": 3, "": 4}
    vs = sorted(candidate["variants"], key=lambda v: (order.get(v.get("impact", ""), 4), _af(v)))
    return vs[:2]


def _af(v: dict[str, Any]) -> float:
    try:
        return float(v.get("af_used") or 0.0)
    except ValueError:
        return 0.0


def _notes(candidate: dict[str, Any], vs: list[dict[str, Any]], classes: dict[str, str], rank: int | None) -> str:
    parts = [f"{candidate['gene_symbol']} {candidate['model']}"]
    for v in vs:
        hgvs = (v.get("hgvsp") or v.get("hgvsc") or "").split(":")[-1]
        bits = [hgvs or v.get("consequence", "")]
        if classes.get(v["key"]):
            bits.append(f"ACMG {classes[v['key']]}")
        if v.get("clinvar_pathogenicity"):
            bits.append(f"ClinVar {v['clinvar_pathogenicity']}")
        parts.append(" ".join(b for b in bits if b))
    if candidate["model"] == "comphet":
        parts.append(f"phase {candidate.get('phase', {}).get('status', 'unknown')}")
    if rank:
        parts.append(f"blind Exomiser rank {rank}")
    return "; ".join(parts).replace(",", " ")[:250]


def plan(run_dir: Path, *, also_pairs: list[tuple[str, str]] | None = None, max_rows: int = MAX_ROWS) -> SubmitPlan:
    cands = (_load(run_dir, "03_filter/candidates.json") or {}).get("candidates", [])
    if not cands:
        raise FileNotFoundError(f"no 03_filter/candidates.json under {run_dir}")
    classes = _classifications(run_dir)
    ranks = _ranks(run_dir)
    sex = _sex(run_dir)
    rows: list[Row] = []
    skipped: list[dict[str, str]] = []
    seen: set[frozenset] = set()

    def add(variants, finding_type, notes, cid):
        key = frozenset(variants)
        if key in seen:
            return
        seen.add(key)
        rows.append(Row(variants=list(variants), epcr=0.0, finding_type=finding_type, notes=notes, candidate_id=cid))

    # The lead row is stage 3's #1. Every row below it earns points only if it IS the
    # answer, so the backups are ordered by the blind ranker's phenotype rank (the
    # evidence stage 3 does not use), then by stage-3 priority for unranked genes.
    ordered = sorted(cands, key=lambda c: (0 if c is cands[0] else 1, is_artefact_family(c["gene_symbol"]) or is_clustered(c),
                                           ranks.get(c["candidate_id"]) or 10**9, c.get("priority", 10**9)))
    first = True
    for c in ordered:
        cl = classes.get(c["candidate_id"], {})
        if why := _skip_reason(c, cl, sex):
            skipped.append({"candidate_id": c["candidate_id"], "why": why})
            continue
        vs = _pair_for(c) if c["model"] == "comphet" else c["variants"][:1]
        ft = _finding_type(c, cl)
        add([parse_key(v["key"]) for v in vs], ft, _notes(c, vs, cl, ranks.get(c["candidate_id"])), c["candidate_id"])
        if first and also_pairs:
            # Alternative partners for the lead allele: a different second hit the
            # funnel may not have carried (e.g. a deep-intronic candidate).
            for a, b in also_pairs:
                add([parse_key(a), parse_key(b)], "primary", "alternative pair for the lead candidate", c["candidate_id"] + ":alt")
        first = False
        if len(rows) >= max_rows:
            break
    rows = rows[:max_rows]
    for i, r in enumerate(rows):
        r.epcr = TOP_EPCR if i == 0 else round(BACKUP_TOP_EPCR - (i - 1) * EPCR_STEP, 2)
    return SubmitPlan(rows=rows, skipped=skipped)


def write_csv(plan_: SubmitPlan, proband_id: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in plan_.rows:
            w.writerow(r.as_csv(proband_id))
    return out


def dry_run(csv_path: Path, scorer: Path, proband_id: str, truth: list[tuple[str, int, str, str]]) -> dict[str, Any]:
    """Score the CSV with the challenge's own ``evaluation.py`` against ``truth``
    (canonical keys; converted to the scorer's spelling here)."""
    spec = importlib.util.spec_from_file_location("challenge_evaluation", scorer)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["challenge_evaluation"] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    sub = mod.load_submission(str(csv_path))
    rows = sub[proband_id]
    key = frozenset((to_submission(c), p, r.upper(), a.upper()) for c, p, r, a in truth)
    res = mod.score_proband(proband_id, rows, key)
    return {"rank_points": res.rank_points, "f_max": res.f_max, "full_match_rank": res.full_match_rank,
            "partial_match_rank": res.partial_match_rank, "f_max_threshold": res.f_max_threshold,
            "rows": len(rows)}


def run_submit(run_dir: Path, *, proband_id: str, out: Path | None = None, scorer: Path | None = None,
               also_pairs: list[tuple[str, str]] | None = None) -> Path:
    run_dir = Path(run_dir)
    out_dir = run_dir / STAGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out or (out_dir / "track1_submission.csv")
    m = Manifest(stage="submit")
    m.add_input("candidates", run_dir / "03_filter" / "candidates.json")
    p = plan(run_dir, also_pairs=also_pairs)
    write_csv(p, proband_id, csv_path)
    m.params.update({"proband_id": proband_id, "top_epcr": TOP_EPCR, "backup_top_epcr": BACKUP_TOP_EPCR,
                     "epcr_step": EPCR_STEP, "max_rows": MAX_ROWS, "artefact_families": list(ARTEFACT_FAMILIES),
                     "also_pairs": [list(x) for x in (also_pairs or [])],
                     "rows": [{"candidate_id": r.candidate_id, "epcr": r.epcr, "finding_type": r.finding_type,
                               "n_variants": len(r.variants)} for r in p.rows],
                     "skipped": p.skipped})
    m.counts.update({"rows": len(p.rows), "primary": sum(r.finding_type == "primary" for r in p.rows),
                     "secondary": sum(r.finding_type == "secondary" for r in p.rows), "skipped": len(p.skipped)})
    checks: dict[str, Any] = {}
    if scorer and Path(scorer).exists() and p.rows:
        m.add_input("scorer", Path(scorer))
        lead = p.rows[0].variants
        checks["if_lead_is_the_key"] = dry_run(csv_path, Path(scorer), proband_id, lead)
        if len(lead) == 2:
            checks["if_only_first_allele_is_right"] = dry_run(csv_path, Path(scorer), proband_id, [lead[0], ("1", 1, "N", "N")])
        # a key the file does not contain must score zero, proving the scorer ran
        checks["if_key_is_absent"] = dry_run(csv_path, Path(scorer), proband_id, [("1", 1, "N", "N")])
    m.params["dry_run"] = checks
    m.add_output("csv", csv_path)
    m.write(out_dir / "manifest.json")
    return csv_path
