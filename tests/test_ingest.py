import json
import shutil
import subprocess
from pathlib import Path

import pytest

from engine.config import CaseConfig
from engine.ingest import COLUMNS, read_variants, run_ingest

pytestmark = pytest.mark.skipif(
    not (shutil.which("bcftools") and shutil.which("bgzip") and shutil.which("tabix")),
    reason="bcftools/htslib not installed",
)

HEADER = """##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=LowQual,Description="Low quality">
##FILTER=<ID=QD2,Description="QD < 2.0">
##INFO=<ID=DP,Number=1,Type=Integer,Description="depth">
##INFO=<ID=QD,Number=1,Type=Float,Description="qd">
##INFO=<ID=MQ,Number=1,Type=Float,Description="mq">
##INFO=<ID=FS,Number=1,Type=Float,Description="fs">
##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="ad">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="dp">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="gq">
##FORMAT=<ID=PGT,Number=1,Type=String,Description="pgt">
##FORMAT=<ID=PID,Number=1,Type=String,Description="pid">
##reference=file:///fake/GRCh38_no_chr.fa
"""

# (chrom, pos, id, ref, alt, qual, filter, info, sample) — chrom spelled per style below.
RECORDS = [
    ("1",   100, ".",   "A",   "G",   "50", "PASS",        "DP=30;QD=10;MQ=60;FS=1", "0/1:15,15:30:99:.:."),
    ("1",   200, "rs1", "C",   "T",   "10", "LowQual;QD2", "DP=8;QD=1;MQ=55;FS=2",   "0/1:5,3:8:20:.:."),
    ("1",   250, ".",   "G",   "*",   "40", "PASS",        "DP=20;QD=9;MQ=60;FS=0",  "0/1:10,10:20:99:.:."),
    ("15",  300, ".",   "T",   "G,A", "60", "PASS",        "DP=22;QD=12;MQ=60;FS=0", "1/2:0,10,12:22:99:.:."),
    ("15",  400, ".",   "ATT", "A",   "60", "PASS",        "DP=40;QD=15;MQ=60;FS=0", "1/1:0,40:40:99:.:."),
    ("X",   700, ".",   "AC",  "GT",  "45", "PASS",        "DP=33;QD=11;MQ=60;FS=0", "0/1:16,17:33:99:0|1:700_AC_GT"),
    ("M",   500, ".",   "A",   "G",   "99", "PASS",        "DP=900;QD=30;MQ=60;FS=0","1/1:0,900:900:99:.:."),
    ("1_KI270706v1_random", 100, ".", "A", "C", "30", "PASS", "DP=10;QD=5;MQ=30;FS=0", "0/1:5,5:10:50:.:."),
]
CONTIG_LENGTHS = {"1": 248956422, "15": 101991189, "X": 156040895, "M": 16569, "1_KI270706v1_random": 175055}


def _spell(chrom: str, style: str) -> str:
    return f"chr{chrom}" if style == "ucsc" else chrom


def make_vcf(tmp: Path, style: str = "ensembl") -> Path:
    contigs = "".join(f"##contig=<ID={_spell(c, style)},length={n}>\n" for c, n in CONTIG_LENGTHS.items())
    body = "".join(
        f"{_spell(c, style)}\t{p}\t{i}\t{r}\t{a}\t{q}\t{f}\t{info}\tGT:AD:DP:GQ:PGT:PID\t{s}\n"
        for c, p, i, r, a, q, f, info, s in RECORDS
    )
    raw = tmp / "tiny.vcf"
    raw.write_text(HEADER + contigs + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n" + body)
    gz = tmp / "tiny.vcf.gz"
    subprocess.run(["bgzip", "-c", str(raw)], stdout=open(gz, "wb"), check=True)
    subprocess.run(["tabix", "-p", "vcf", str(gz)], check=True)
    return gz


def make_case(tmp: Path, vcf: Path, **extra) -> CaseConfig:
    case = tmp / "case.yaml"
    lines = [f"proband_id: TEST01", f"vcf: {vcf.name}"]
    for k, v in extra.items():
        lines.append(f"{k}: {v}")
    case.write_text("\n".join(lines) + "\n")
    return CaseConfig.load(case)


def test_whole_genome_ingest(tmp_path: Path):
    vcf = make_vcf(tmp_path)
    case = make_case(tmp_path, vcf)
    run = tmp_path / "run"
    manifest_path = run_ingest(case, run, threads=1)

    rows = list(read_variants(run))
    assert list(rows[0].keys()) == list(COLUMNS)

    # 8 records in: 1 dropped (random contig), 1 dropped (* allele), 1 split into 2 → 7 rows.
    assert len(rows) == 7
    by_key = {(r["chrom"], r["pos"], r["alt"]): r for r in rows}

    # Filtered record kept, with the caveat on it.
    flagged = by_key[("1", "200", "T")]
    assert flagged["quality_flag"] == "LowQual;QD2"
    assert flagged["filter"] == "LowQual;QD2"
    assert by_key[("1", "100", "G")]["quality_flag"] == ""

    # Multi-allelic 1/2 site became two single-ALT rows, each het for its own allele.
    g, a = by_key[("15", "300", "G")], by_key[("15", "300", "A")]
    for r in (g, a):
        assert r["gt"].count("1") == 1, r["gt"]
        assert r["split_from"].startswith("15|300|T|G,A|")
    assert g["ad"] == "0,10" and a["ad"] == "0,12"

    # Types, mito naming, phasing fields carried through untouched.
    assert by_key[("15", "400", "A")]["vtype"] == "indel"
    assert by_key[("X", "700", "GT")]["vtype"] == "mnp"
    assert by_key[("X", "700", "GT")]["pid"] == "700_AC_GT"
    assert by_key[("MT", "500", "G")]["chrom_source"] == "M"
    assert by_key[("1", "100", "G")]["info_dp"] == "30" and by_key[("1", "100", "G")]["qd"] == "10"

    m = json.loads(manifest_path.read_text())
    c = m["counts"]
    assert c["rows_out"] == 7
    assert c["sites_in"] == 6
    assert c["multiallelic_sites_split"] == 1 and c["rows_from_splits"] == 2
    assert c["flagged_rows"] == 1
    assert c["dropped_star_allele"] == 0 and c["dropped_star_sites"] == 1
    assert c["rows_per_contig"] == {"1": 2, "15": 3, "X": 1, "MT": 1}
    # Reconciled against our own index: every record in the file is accounted for.
    assert c["records_in_file"] == 8
    assert c["records_excluded_non_primary"] == 1
    assert c["records_on_primary_contigs"] == 7 == c["sites_in"] + c["dropped_star_sites"]
    assert any(n.startswith("Consistency check passed") for n in m["notes"])
    assert c["records_per_contig_in_file"] == {"1": 3, "15": 2, "X": 1, "MT": 1}
    assert (run / "01_ingest" / "input.csi").exists()
    assert c["per_type"] == {"snv": 5, "indel": 1, "mnp": 1}
    assert m["params"]["naming_style"] == "ensembl"
    assert m["params"]["sample"] == "S1"
    assert m["params"]["left_aligned"] is False
    assert m["params"]["vcf_header_reference"] == "file:///fake/GRCh38_no_chr.fa"
    assert m["inputs"]["vcf"]["sha256"] and m["outputs"]["variants"]["sha256"]
    assert "bcftools" in m["tools"] and m["tools"]["bcftools"].startswith("bcftools")
    assert any("not left-aligned" in n for n in m["notes"])
    # sex: not stated, one X call -> too thin to infer, recorded as such
    assert m["params"]["sex_stated"] == "unknown" and m["params"]["sex"] == "unknown"
    assert m["params"]["sex_inference"]["x_nonpar_carrier_calls"] == 1  # X:700 is outside PAR1 (starts at 10,001)
    assert m["params"]["sex_inference"]["inferred"] == "unknown"


def test_stated_sex_is_recorded_and_wins(tmp_path: Path):
    vcf = make_vcf(tmp_path)
    case = make_case(tmp_path, vcf, sex="male")
    assert case.sex == "male"
    m = json.loads(run_ingest(case, tmp_path / "run", threads=1).read_text())
    assert m["params"]["sex_stated"] == "male" and m["params"]["sex"] == "male"
    assert any(n.startswith("Sex stated male; too few X calls") for n in m["notes"])
    with pytest.raises(ValueError, match="sex must be"):
        make_case(tmp_path, vcf, sex="boy")


def test_ucsc_named_vcf_is_canonicalised(tmp_path: Path):
    vcf = make_vcf(tmp_path, style="ucsc")
    case = make_case(tmp_path, vcf)
    run = tmp_path / "run"
    manifest_path = run_ingest(case, run, threads=1)
    rows = list(read_variants(run))
    assert {r["chrom"] for r in rows} == {"1", "15", "X", "MT"}
    assert {r["chrom_source"] for r in rows} == {"chr1", "chr15", "chrX", "chrM"}
    assert json.loads(manifest_path.read_text())["params"]["naming_style"] == "ucsc"


def test_regions_restrict(tmp_path: Path):
    vcf = make_vcf(tmp_path)
    bed = tmp_path / "r.bed"
    bed.write_text("15\t250\t450\tGENE\n")
    case = make_case(tmp_path, vcf)
    run = tmp_path / "run"
    run_ingest(case, run, regions=bed, threads=1)
    rows = list(read_variants(run))
    assert {(r["chrom"], r["pos"]) for r in rows} == {("15", "300"), ("15", "400")}
    m = json.loads((run / "01_ingest" / "manifest.json").read_text())
    assert m["params"]["regions_mode"] == "bed"
    assert "regions" in m["inputs"]


def test_multisample_requires_sample(tmp_path: Path):
    vcf = make_vcf(tmp_path)
    # Fake a second sample by rewriting the header line and body.
    txt = subprocess.run(["bcftools", "view", str(vcf)], capture_output=True, text=True, check=True).stdout
    out = []
    for line in txt.splitlines():
        if line.startswith("#CHROM"):
            out.append(line + "\tS2")
        elif not line.startswith("#"):
            out.append(line + "\t" + line.split("\t")[9])
        else:
            out.append(line)
    raw = tmp_path / "two.vcf"
    raw.write_text("\n".join(out) + "\n")
    gz = tmp_path / "two.vcf.gz"
    subprocess.run(["bgzip", "-c", str(raw)], stdout=open(gz, "wb"), check=True)
    subprocess.run(["tabix", "-p", "vcf", str(gz)], check=True)

    with pytest.raises(ValueError, match="2 samples"):
        run_ingest(make_case(tmp_path, gz), tmp_path / "run_a", threads=1)
    run_ingest(make_case(tmp_path, gz, sample="S2"), tmp_path / "run_b", threads=1)
    assert len(list(read_variants(tmp_path / "run_b"))) == 7


def test_case_config_validation(tmp_path: Path):
    vcf = make_vcf(tmp_path)
    with pytest.raises(ValueError, match="HPO"):
        make_case(tmp_path, vcf, hpo="[HP:1, HP:0000121]")
    (tmp_path / "case.yaml").write_text("proband_id: X\nvcf: missing.vcf.gz\n")
    with pytest.raises(FileNotFoundError):
        CaseConfig.load(tmp_path / "case.yaml")
