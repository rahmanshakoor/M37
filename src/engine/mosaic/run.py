"""``engine mosaic`` — does the VCF carry a chromosome-level signal of mosaic aneuploidy?

Mosaic variegated aneuploidy is defined by whole-chromosome gains and losses in a
fraction of cells. A blood-derived short-read genome sees that as two things a VCF
already contains: **allele balance** — in a cell fraction *f* carrying a trisomy, a
heterozygous SNP's alternate fraction moves from 1/2 to (1+f)/(2+f) or 1/(2+f)
depending on which homolog was gained, so the chromosome's heterozygotes spread away
from 0.5 — and **depth**, which rises by f/2 for a gain and falls by f/2 for a loss.
Neither needs the BAM: the sample's AD and DP fields carry both.

The scan streams the VCF with ``bcftools query``, keeps PASS heterozygous SNVs with
depth ≥ ``min_dp``, and reports per contig the number of heterozygotes, the median
depth relative to the autosomal median, the variance of allele balance about 0.5 (a
trisomy in cell fraction *f* adds δ² with δ = f / (2(2+f)) to the binomial variance),
the mean absolute deviation, the fraction of heterozygotes outside ``[0.5 − skew,
0.5 + skew]``, and a z-score of each against the other autosomes. A contig whose
depth ratio, variance or skewed fraction is ``z_flag`` standard deviations or more
from the rest is flagged. Sensitivity is stated empirically: the smallest trisomic
fraction whose depth shift (f/2) or variance shift (δ²) would reach ``z_flag`` times
the observed spread among the autosomes — so systematic per-chromosome effects
(mapping and reference bias) are already priced in. It is a screen for a clonal or
high-fraction aneuploidy in the sequenced tissue, not a karyotype. Everything here is
arithmetic on the proband's own calls; nothing leaves the machine.
"""

from __future__ import annotations

import csv
import json
import platform
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from engine.config import CaseConfig
from engine.contigs import AUTOSOMES, canonical
from engine.ingest import read_header
from engine.manifest import Manifest
from engine.sex import in_par

STAGE_DIR = "mosaic"
MIN_DP = 20
SKEW = 0.15
"""A heterozygote is *skewed* when its allele balance lies outside 0.5 ± SKEW."""
Z_FLAG = 3.0
MIN_HETS = 100
"""An autosome with fewer heterozygotes than this is reported but not used as a reference."""
QUERY_FORMAT = "%CHROM\t%POS\t%REF\t%ALT\t%FILTER[\t%GT\t%AD\t%DP]\n"


@dataclass
class ContigTally:
    n_sites: int = 0
    n_het: int = 0
    depths: list[int] = field(default_factory=list)
    abs_dev_sum: float = 0.0
    sq_dev_sum: float = 0.0
    n_skewed: int = 0
    n_low: int = 0
    """Skewed heterozygotes with allele balance below 0.5 − skew (the reference-biased side)."""
    n_het_par: int = 0
    """Heterozygotes inside a pseudoautosomal region (X/Y only), counted apart."""

    def add_het(self, ab: float, dp: int, *, par: bool = False, skew: float = SKEW) -> None:
        if par:
            self.n_het_par += 1
            return
        self.n_het += 1
        self.depths.append(dp)
        d = abs(ab - 0.5)
        self.abs_dev_sum += d
        self.sq_dev_sum += d * d
        if d > skew:
            self.n_skewed += 1
            if ab < 0.5:
                self.n_low += 1

    @property
    def median_dp(self) -> float | None:
        return statistics.median(self.depths) if self.depths else None

    @property
    def mean_abs_dev(self) -> float | None:
        return self.abs_dev_sum / self.n_het if self.n_het else None

    @property
    def var_ab(self) -> float | None:
        """Variance of allele balance about 0.5: binomial noise plus any mosaic shift."""
        return self.sq_dev_sum / self.n_het if self.n_het else None

    @property
    def frac_skewed(self) -> float | None:
        return self.n_skewed / self.n_het if self.n_het else None

    @property
    def low_share(self) -> float | None:
        """Share of the skewed heterozygotes that lie on the low side. A trisomy splits
        allele balance symmetrically (about one half); reference and mapping bias pile
        up on the low side only."""
        return self.n_low / self.n_skewed if self.n_skewed else None


def parse_line(line: str) -> tuple[str, int, str, str, str, str, str, str] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 8:
        return None
    chrom, pos, ref, alt, filt, gt, ad, dp = parts[:8]
    try:
        return chrom, int(pos), ref, alt, filt, gt, ad, dp
    except ValueError:
        return None


def het_allele_balance(gt: str, ad: str, *, min_dp: int) -> tuple[float, int] | None:
    """``(alt fraction, AD depth)`` for a biallelic heterozygous call with usable AD;
    ``None`` for anything else."""
    alleles = gt.replace("|", "/").split("/")
    if sorted(alleles) != ["0", "1"]:
        return None
    counts = ad.split(",")
    if len(counts) != 2 or not all(c.isdigit() for c in counts):
        return None
    ref_n, alt_n = int(counts[0]), int(counts[1])
    total = ref_n + alt_n
    if total < max(min_dp, 1):
        return None
    return alt_n / total, total


def tally(lines: Iterable[str], *, min_dp: int = MIN_DP, skew: float = SKEW,
          pass_only: bool = True, snv_only: bool = True) -> dict[str, ContigTally]:
    out: dict[str, ContigTally] = defaultdict(ContigTally)
    for line in lines:
        rec = parse_line(line)
        if rec is None:
            continue
        chrom_raw, pos, ref, alt, filt, gt, ad, _dp = rec
        chrom = canonical(chrom_raw)
        if chrom is None or chrom == "MT":
            continue
        if pass_only and filt not in ("PASS", "."):
            continue
        if snv_only and (len(ref) != 1 or len(alt) != 1):
            continue
        t = out[chrom]
        t.n_sites += 1
        h = het_allele_balance(gt, ad, min_dp=min_dp)
        if h is None:
            continue
        ab, depth = h
        t.add_het(ab, depth, par=chrom in ("X", "Y") and in_par(chrom, pos), skew=skew)
    return dict(out)


def _z(value: float | None, others: list[float]) -> float | None:
    if value is None or len(others) < 3:
        return None
    mu = statistics.mean(others)
    sd = statistics.pstdev(others)
    if sd == 0:
        return 0.0
    return (value - mu) / sd


def summarize(tallies: dict[str, ContigTally], *, z_flag: float = Z_FLAG,
              min_hets: int = MIN_HETS) -> tuple[list[dict[str, object]], dict[str, object]]:
    """One row per contig, each autosome compared against the other autosomes, and
    the empirical sensitivity of the scan."""
    autos = [c for c in AUTOSOMES if c in tallies and tallies[c].n_het >= min_hets]
    genome_median_dp = statistics.median(tallies[c].median_dp for c in autos) if autos else None
    rows: list[dict[str, object]] = []
    for chrom in list(AUTOSOMES) + ["X", "Y"]:
        t = tallies.get(chrom)
        if t is None:
            continue
        others = [c for c in autos if c != chrom]
        dp_ratio = (t.median_dp / genome_median_dp) if (t.median_dp and genome_median_dp) else None
        dev, var, skewed = t.mean_abs_dev, t.var_ab, t.frac_skewed
        z_dp = _z(dp_ratio, [tallies[c].median_dp / genome_median_dp for c in others]) if dp_ratio else None
        z_var = _z(var, [tallies[c].var_ab for c in others]) if var is not None else None
        z_skew = _z(skewed, [tallies[c].frac_skewed for c in others]) if skewed is not None else None
        flags = []
        symmetric = t.low_share is not None and 0.35 <= t.low_share <= 0.65
        if chrom in AUTOSOMES and t.n_het >= min_hets:
            if z_dp is not None and abs(z_dp) >= z_flag:
                flags.append("depth")
            if z_var is not None and z_var >= z_flag:
                flags.append("allele_balance_variance" if symmetric else "allele_balance_variance(one_sided_tail)")
            if z_skew is not None and z_skew >= z_flag:
                flags.append("skewed_fraction" if symmetric else "skewed_fraction(one_sided_tail)")
        rows.append({
            "contig": chrom, "sites": t.n_sites, "hets": t.n_het, "hets_par": t.n_het_par,
            "median_dp": t.median_dp, "depth_ratio": None if dp_ratio is None else round(dp_ratio, 4),
            "mean_abs_dev": None if dev is None else round(dev, 4),
            "var_ab": None if var is None else round(var, 5),
            "frac_skewed": None if skewed is None else round(skewed, 4),
            "low_share": None if t.low_share is None else round(t.low_share, 3),
            "z_depth": None if z_dp is None else round(z_dp, 2),
            "z_var_ab": None if z_var is None else round(z_var, 2),
            "z_skewed": None if z_skew is None else round(z_skew, 2),
            "flags": ";".join(flags),
        })
    sensitivity: dict[str, object] = {}
    if len(autos) >= 3 and genome_median_dp:
        sd_dp = statistics.pstdev([tallies[c].median_dp / genome_median_dp for c in autos])
        sd_var = statistics.pstdev([tallies[c].var_ab for c in autos])
        # a trisomy in cell fraction f raises depth by f/2 and the AB variance by δ², δ = f/(2(2+f))
        f_depth = min(2 * z_flag * sd_dp, 1.0)
        delta = (z_flag * sd_var) ** 0.5
        f_var = min(4 * delta / (1 - 2 * delta), 1.0) if delta < 0.5 else 1.0
        sensitivity = {
            "autosome_sd_depth_ratio": round(sd_dp, 4), "autosome_sd_var_ab": round(sd_var, 6),
            "min_trisomic_fraction_by_depth": round(f_depth, 3),
            "min_trisomic_fraction_by_allele_balance": round(f_var, 3),
            "note": (f"the smallest whole-chromosome trisomic cell fraction that would stand {z_flag:g} SD from the other "
                     "autosomes; a monosomy shifts depth by the same amount and shows as lost heterozygosity"),
        }
    return rows, sensitivity


def run_mosaic(case: CaseConfig, run_dir: Path, *, min_dp: int = MIN_DP, skew: float = SKEW,
               z_flag: float = Z_FLAG) -> Path:
    run_dir = Path(run_dir)
    out_dir = run_dir / STAGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    m = Manifest(stage="mosaic")
    m.add_tool("bcftools")
    m.add_input("vcf", case.vcf)
    m.tools["python"] = platform.python_version()
    header = read_header(case.vcf)
    sample = case.sample or (header.samples[0] if len(header.samples) == 1 else None)
    if sample is None:
        raise ValueError(f"VCF has {len(header.samples)} samples; set `sample:` in the case file")
    cmd = ["bcftools", "query", "-s", sample, "-f", QUERY_FORMAT, str(case.vcf)]
    m.params.update({"sample": sample, "min_dp": min_dp, "skew": skew, "z_flag": z_flag, "min_hets": MIN_HETS,
                     "pass_only": True, "snv_only": True, "query_command": cmd,
                     "method": ("per-contig allele balance (AD-derived) and median depth of PASS heterozygous SNVs, "
                                "z-scored against the other autosomes; X/Y reported, never flagged")})
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    tallies = tally(proc.stdout, min_dp=min_dp, skew=skew)
    _, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"bcftools query failed: {err.strip()}")
    rows, sensitivity = summarize(tallies, z_flag=z_flag)
    tsv = out_dir / "per_contig.tsv"
    with open(tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["contig"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    autos = [r for r in rows if r["contig"] in AUTOSOMES and int(r["hets"]) >= MIN_HETS]
    flagged = [str(r["contig"]) for r in rows if r["flags"]]
    trisomy_like = [str(r["contig"]) for r in rows if r["flags"] and "one_sided_tail" not in str(r["flags"])]
    m.counts.update({
        "contigs": len(rows), "hets_total": sum(int(r["hets"]) for r in rows), "sites_total": sum(int(r["sites"]) for r in rows),
        "autosomes_scored": len(autos),
        "typical_hets_per_autosome": int(statistics.median(int(r["hets"]) for r in autos)) if autos else 0,
        "genome_median_dp": statistics.median(float(r["median_dp"]) for r in autos) if autos else None,
        "flagged": flagged, "flagged_trisomy_like": trisomy_like,
        "x_hets_nonpar": next((int(r["hets"]) for r in rows if r["contig"] == "X"), 0),
        "x_hets_par": next((int(r["hets_par"]) for r in rows if r["contig"] == "X"), 0),
    })
    m.params["sensitivity"] = sensitivity
    if trisomy_like:
        m.note(f"Contig(s) {', '.join(trisomy_like)} stand {z_flag:g} SD or more from the other autosomes in depth or in a "
               "symmetric spread of allele balance: a chromosome-level signal worth a karyotype, not a diagnosis.")
    elif flagged:
        m.note(f"Contig(s) {', '.join(flagged)} stand {z_flag:g} SD or more from the other autosomes only through a one-sided "
               "low tail of allele balance (the signature of reference and mapping bias, not of a trisomy, which splits "
               "allele balance symmetrically); no autosome shows the joint depth-plus-symmetric-spread pattern of a mosaic gain.")
    else:
        seen = ""
        if sensitivity:
            floor = min(float(sensitivity["min_trisomic_fraction_by_depth"]),
                        float(sensitivity["min_trisomic_fraction_by_allele_balance"]))
            seen = f" (a trisomic cell fraction below about {floor:.0%} would not be seen)"
        m.note("No autosome stands out in depth or allele balance: no clonal or high-fraction aneuploidy in the "
               f"sequenced tissue{seen}. Mosaicism confined to other tissues, or below that fraction, is not excluded.")
    m.add_output("per_contig", tsv)
    summary = out_dir / "summary.md"
    summary.write_text(render_summary(rows, m))
    m.add_output("summary", summary)
    m.write(out_dir / "manifest.json")
    return out_dir / "manifest.json"


def render_summary(rows: list[dict[str, object]], m: Manifest) -> str:
    out = ["# Mosaic-aneuploidy scan (from the VCF)", "", m.notes[-1] if m.notes else "", "",
           "| Contig | Het SNVs | Median depth | Depth ratio | Var(AB) | Skewed fraction | Low share | z depth | z Var(AB) | z skewed | Flags |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['contig']} | {int(r['hets']):,} | {r['median_dp']} | {r['depth_ratio']} | {r['var_ab']} | "
                   f"{r['frac_skewed']} | {r['low_share']} | {r['z_depth']} | {r['z_var_ab']} | {r['z_skewed']} | {r['flags'] or '—'} |")
    out += ["", f"Sensitivity: {json.dumps(m.params.get('sensitivity'))}", ""]
    return "\n".join(out)
