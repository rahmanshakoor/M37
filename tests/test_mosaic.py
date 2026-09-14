"""engine mosaic: a synthetic VCF where one autosome carries a mosaic trisomy."""

import json
import random
import subprocess
from pathlib import Path

import pytest

from engine.config import CaseConfig
from engine.mosaic.run import ContigTally, het_allele_balance, run_mosaic, summarize, tally

CONTIGS = [str(i) for i in range(1, 23)] + ["X"]


def _vcf_lines(seed: int = 7, mosaic: dict[str, float] | None = None, depth: int = 40, n: int = 400) -> list[str]:
    """PASS heterozygous SNVs on every contig; on the contigs in ``mosaic`` a fraction
    f of cells is trisomic (depth ×(1+f/2), allele balance split to (1+f)/(2+f) or 1/(2+f))."""
    rng = random.Random(seed)
    lines = ["##fileformat=VCFv4.2", "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"\">",
             "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"\">", "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"\">"]
    lines += [f"##contig=<ID={c},length=200000000>" for c in CONTIGS]
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1")
    for c in CONTIGS:
        f = (mosaic or {}).get(c, 0.0)
        offset = rng.uniform(-0.6, 0.6)  # a small systematic per-contig depth difference, as real genomes have
        for i in range(n):
            pos = 10_000_000 + i * 1000
            dp = max(1, round(rng.gauss(depth * (1 + f / 2) + offset, 5)))
            p = 0.5 if f == 0 else ((1 + f) / (2 + f) if i % 2 else 1 / (2 + f))
            alt = sum(1 for _ in range(dp) if rng.random() < p)
            gt = "0/1" if c != "X" else "1/1"
            ad = f"{dp - alt},{alt}" if c != "X" else f"0,{dp}"
            lines.append(f"{c}\t{pos}\t.\tA\tG\t50\tPASS\t.\tGT:AD:DP\t{gt}:{ad}:{dp}")
    return lines


def _case(tmp: Path, lines: list[str]) -> CaseConfig:
    raw = tmp / "s.vcf"
    raw.write_text("\n".join(lines) + "\n")
    gz = tmp / "s.vcf.gz"
    subprocess.run(["bgzip", "-c", str(raw)], stdout=open(gz, "wb"), check=True)
    subprocess.run(["tabix", "-p", "vcf", str(gz)], check=True)
    (tmp / "case.yaml").write_text(f"proband_id: T\nvcf: {gz.name}\n")
    return CaseConfig.load(tmp / "case.yaml")


def test_allele_balance_parsing():
    assert het_allele_balance("0/1", "20,20", min_dp=20) == (0.5, 40)
    assert het_allele_balance("1|0", "10,30", min_dp=20) == (0.75, 40)
    assert het_allele_balance("0/1", "5,5", min_dp=20) is None      # too shallow
    assert het_allele_balance("1/1", "0,40", min_dp=20) is None     # not a heterozygote
    assert het_allele_balance("0/1", "20,20,3", min_dp=20) is None  # not biallelic AD
    t = ContigTally()
    t.add_het(0.5, 40); t.add_het(0.8, 40)
    assert t.n_het == 2 and t.n_skewed == 1 and t.var_ab == pytest.approx((0.0 + 0.09) / 2)


def test_clean_genome_flags_nothing(tmp_path: Path):
    case = _case(tmp_path, _vcf_lines())
    m = json.loads(run_mosaic(case, tmp_path / "run").read_text())
    assert m["counts"]["flagged"] == [] and m["counts"]["autosomes_scored"] == 22
    assert m["counts"]["x_hets_nonpar"] == 0  # a male-like X: homozygous calls only
    assert any(n.startswith("No autosome stands out") for n in m["notes"])
    s = m["params"]["sensitivity"]
    assert 0 < s["min_trisomic_fraction_by_depth"] <= 1 and 0 < s["min_trisomic_fraction_by_allele_balance"] <= 1
    rows = (tmp_path / "run" / "mosaic" / "per_contig.tsv").read_text().splitlines()
    assert rows[0].startswith("contig\tsites\thets") and len(rows) == 1 + 23


def test_a_clonal_trisomy_is_flagged_by_depth_and_allele_balance(tmp_path: Path):
    case = _case(tmp_path, _vcf_lines(mosaic={"21": 0.6}))
    m = json.loads(run_mosaic(case, tmp_path / "run").read_text())
    assert m["counts"]["flagged"] == ["21"]
    row = next(r for r in (tmp_path / "run" / "mosaic" / "per_contig.tsv").read_text().splitlines() if r.startswith("21\t"))
    assert "depth" in row and "allele_balance_variance" in row
    assert any(n.startswith("Contig(s) 21 stand") for n in m["notes"])


def test_tally_ignores_filtered_indels_and_mt():
    lines = ["1\t100\tA\tG\tPASS\t0/1\t20,20\t40", "1\t200\tA\tG\tLowQual\t0/1\t20,20\t40",
             "1\t300\tAT\tA\tPASS\t0/1\t20,20\t40", "MT\t400\tA\tG\tPASS\t0/1\t20,20\t40",
             "chr2\t500\tA\tG\t.\t0|1\t10,30\t40", "X\t2000000\tA\tG\tPASS\t0/1\t20,20\t40"]
    t = tally(lines)
    assert t["1"].n_het == 1 and t["1"].n_sites == 1 and "MT" not in t
    assert t["2"].n_het == 1 and t["2"].n_skewed == 1              # chr2 → 2; phased het counts; 0.75 is skewed
    assert t["X"].n_het == 0 and t["X"].n_het_par == 1            # PAR1 het counted apart
    rows, sens = summarize(t)
    assert [r["contig"] for r in rows] == ["1", "2", "X"] and sens == {}  # too few autosomes to score
