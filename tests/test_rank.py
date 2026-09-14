"""Stage 4 — Exomiser configuration, invocation, output parsing, the join, and the
bundle downloader. Network-free: Docker is replaced by a fake runner that drops the
recorded outputs into the results directory, and the downloader is exercised against
an in-memory server. The recorded outputs under ``fixtures/rank/`` are the verbatim
files Exomiser 14.0.0 (pinned image, 2406 data) wrote for the public three-variant VCF
(MTHFR rs1801133, CFTR p.Phe508del, TP53 p.Arg175His; HPO HP:0002205, HP:0001738,
HP:0011947): CFTR ranked 1 (AR), TP53 ranked 2 (AD), MTHFR filtered out by frequency."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import urllib.error
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from engine.cli import main
from engine.rank import download as dl
from engine.rank import run as rank_run
from engine.rank.exomiser import (
    ANALYSIS_FILENAME, DEFAULT_CONFIG, PROPERTIES_FILENAME, REGIONS_VCF_FILENAME, Bundle, ExomiserConfig, GeneRow,
    bed_interval_count, build_ranking, check_template, container_vcf_name, contig_renames, contributing_from_json,
    contributing_from_tsv, count_records, docker_command, docker_load, gene_identifiers_from_json, heap_bytes, heap_for,
    image_digest, load_json_genes, mito_contig, parse_genes_tsv, parse_mem, parse_variants_tsv, rename_contigs,
    render_analysis, render_properties, sha256_text, subset_regions, vcf_header_contigs, vcf_record_count,
    version_from_log,
)
from engine.rank.run import (
    ExomiserFailed, RANKING_COLUMNS, RUN_FILENAME, analysis_from_yaml, carries_digest, gene_url, hpo_from_analysis,
    image_digest_used, join_candidates, read_joined, run_rank,
)

FIX = Path(__file__).parent / "fixtures" / "rank"
PUBLIC = Path(__file__).parent / "fixtures" / "public_case"

HPO = ["HP:0002205", "HP:0001738", "HP:0011947"]  # public CF terms, as in the recorded run

F508 = "7:117559590:ATCT:A"
R175H = "17:7675088:C:T"

BANNER = (
    "Picked up JAVA_TOOL_OPTIONS: -Xmx2816m\n"
    "Welcome to:\n The Exomiser - A Tool to Annotate and Prioritize Exome Variants     v14.0.0\n"
    "INFO o.monarchinitiative.exomiser.cli.Main : Starting Main using Java 17.0.6 with PID 1\n"
)


@pytest.fixture
def cfg() -> ExomiserConfig:
    return ExomiserConfig.load(DEFAULT_CONFIG)


# ---------------------------------------------------------------- the pin

def test_pin_is_complete(cfg: ExomiserConfig):
    assert cfg.image_tag == "14.0.0" and cfg.image_version == "14.0.0"
    assert cfg.image_digest.startswith("sha256:") and len(cfg.image_digest) == 71
    assert cfg.image_ref == f"docker.io/exomiser/exomiser-cli@{cfg.image_digest}"
    assert set(cfg.platform_digests) == {"linux/arm64", "linux/amd64"}
    assert cfg.data_version == "2406"
    assert set(cfg.bundles) == {"hg38", "phenotype"}
    hg38, pheno = cfg.bundles["hg38"], cfg.bundles["phenotype"]
    assert hg38.url == "https://data.monarchinitiative.org/exomiser/latest/2406_hg38.zip"
    assert hg38.bytes == 23507791558 and len(hg38.md5) == 32
    # both zips pinned by sha256 as well as size + md5
    assert hg38.sha256 and len(hg38.sha256) == 64 and pheno.sha256 and len(pheno.sha256) == 64
    assert "2406_hg38_variants.mv.db" in hg38.required_files
    # the hg38 members are pinned individually and checked against the list the bundle ships
    assert hg38.checksum_list == "2406_hg38.sha256"
    assert set(hg38.member_sha256) == {"2406_hg38_clinvar.mv.db", "2406_hg38_genome.mv.db", "2406_hg38_transcripts_ensembl.ser",
                                       "2406_hg38_transcripts_refseq.ser", "2406_hg38_transcripts_ucsc.ser", "2406_hg38_variants.mv.db"}
    assert all(len(v) == 64 for v in hg38.member_sha256.values())
    # an empty 2406_phenotype/ must not pass as complete; its three stores are pinned by content
    assert set(pheno.required_files) == {"2406_phenotype.mv.db", "rw_string_10.mv", "hp.obo"}
    assert set(pheno.member_sha256) == set(pheno.required_files) and all(len(v) == 64 for v in pheno.member_sha256.values())
    assert pheno.checksum_list is None  # the phenotype bundle ships no list
    assert cfg.output_formats == ("TSV_GENE", "TSV_VARIANT", "JSON")
    # every threshold the analysis applies is in the template
    assert cfg.analysis["inheritanceModes"]["AUTOSOMAL_RECESSIVE_COMP_HET"] == 2.0
    assert {"frequencyFilter": {"maxFrequency": 2.0}} in cfg.analysis["steps"]
    assert "REMM" not in cfg.analysis["pathogenicitySources"]  # needs a 17 GB file the pin does not ship
    assert len(cfg.sha256) == 64


def test_empty_phenotype_dir_is_incomplete(cfg: ExomiserConfig, tmp_path: Path):
    for b in cfg.bundles.values():
        b.extracted_path(tmp_path).mkdir(parents=True)
    missing = cfg.missing_data(tmp_path)
    assert set(missing) == {"hg38", "phenotype"} and "2406_phenotype.mv.db" in missing["phenotype"]


# ---------------------------------------------------------------- per-run files

def test_analysis_yaml_carries_only_hpo_and_vcf_from_the_case(cfg: ExomiserConfig):
    text = render_analysis(cfg, HPO, "/vcf/case.vcf.gz", "/results")
    doc = yaml.safe_load(text)
    a = doc["analysis"]
    assert list(a) == ["genomeAssembly", "vcf", "ped", "proband", "hpoIds", "inheritanceModes", "analysisMode",
                       "frequencySources", "pathogenicitySources", "steps"]
    assert a["genomeAssembly"] == "hg38" and a["vcf"] == "/vcf/case.vcf.gz"
    assert a["ped"] is None and a["proband"] is None
    assert a["hpoIds"] == HPO
    # no gene list, seed genes or candidate gene anywhere in the file
    step_names = [next(iter(s)) for s in a["steps"]]
    assert "genePanelFilter" not in step_names and "exomeWalkerPrioritiser" not in step_names
    assert "geneSymbols" not in text and "seedGeneIds" not in text and "candidateGeneSymbol" not in text
    assert step_names[:2] == ["hiPhivePrioritiser", "priorityScoreFilter"]  # the memory-saving order
    o = doc["outputOptions"]
    assert o["outputDirectory"] == "/results" and o["outputFileName"] == "exomiser"
    assert o["outputFormats"] == ["TSV_GENE", "TSV_VARIANT", "JSON"] and o["numGenes"] == 0
    # byte-identical for the same case
    assert render_analysis(cfg, HPO, "/vcf/case.vcf.gz", "/results") == text
    assert render_analysis(cfg, HPO, "/vcf/case.vcf.gz", "/results", ("JSON", "TSV_GENE")) != text
    assert hpo_from_analysis(text) == HPO


def test_analysis_yaml_rejects_bad_inputs(cfg: ExomiserConfig):
    with pytest.raises(ValueError, match="HPO"):
        render_analysis(cfg, [], "/vcf/case.vcf.gz", "/results")
    with pytest.raises(ValueError, match="unknown"):
        render_analysis(cfg, HPO, "/vcf/case.vcf.gz", "/results", ("TSV_GENE", "PARQUET"))
    with pytest.raises(ValueError, match="TSV_GENE"):
        render_analysis(cfg, HPO, "/vcf/case.vcf.gz", "/results", ("JSON",))


def test_template_cannot_override_the_case_or_steer_the_ranking(cfg: ExomiserConfig):
    """The blind-ranking invariant is enforced, not assumed: a template that carries
    hpoIds/vcf would silently replace the case values (they are copied after them)."""
    steered = ExomiserConfig(**{**cfg.__dict__, "analysis": {**cfg.analysis, "hpoIds": ["HP:0000001"], "vcf": "/tmp/other.vcf"}})
    with pytest.raises(ValueError, match="case keys \\['vcf', 'hpoIds'\\]"):
        render_analysis(steered, HPO, "/vcf/case.vcf.gz", "/results")
    panel = ExomiserConfig(**{**cfg.__dict__, "analysis": {**cfg.analysis, "steps": [{"genePanelFilter": {"geneSymbols": ["CFTR"]}}]}})
    with pytest.raises(ValueError, match="genePanelFilter"):
        render_analysis(panel, HPO, "/vcf/case.vcf.gz", "/results")
    with pytest.raises(ValueError, match="exomeWalkerPrioritiser"):
        check_template({"steps": [{"exomeWalkerPrioritiser": {"seedGeneIds": [1080]}}]})
    check_template(cfg.analysis)  # the shipped template passes


def test_properties_name_the_data_release_and_only_hg38(cfg: ExomiserConfig):
    text = render_properties(cfg)
    assert text.splitlines() == [
        "exomiser.data-directory=/exomiser-data",
        "exomiser.hg38.data-version=2406",
        "exomiser.phenotype.data-version=2406",
    ]
    assert "hg19" not in text


def test_heap_is_sized_from_what_the_daemon_has_left(cfg: ExomiserConfig):
    assert heap_for(cfg, 4109217792) == "2816m"       # this Mac's Docker Desktop: 3.83 GiB - 1 GiB, rounded to 256 MiB
    assert heap_for(cfg, 4109217792, 0) == "2816m"
    # a dozen resident database containers (≈1.3 GiB, as observed here) come off the top:
    # sized from the total alone, the 2816m heap was OOM-killed (exit 137)
    assert heap_for(cfg, 4109217792, 1_400_000_000) == "1536m"
    assert heap_for(cfg, 4109217792, 3_500_000_000) == "1024m"   # floor
    assert heap_for(cfg, 8 * (1 << 30)) == "7168m"
    assert heap_for(cfg, 1 << 30) == "1024m"          # floor
    assert heap_for(cfg, None) == "2g"                # daemon unreachable → fallback
    assert heap_for(cfg, None, 1 << 30) == "2g"
    fixed = ExomiserConfig(**{**cfg.__dict__, "docker": {**cfg.docker, "heap": "6g"}})
    assert heap_for(fixed, 4109217792) == "6g"
    assert heap_bytes("2816m") == 2816 << 20 and heap_bytes("5g") == 5 << 30 and heap_bytes("512k") == 512 << 10
    assert heap_bytes("1024") == 1024 and heap_bytes("lots") is None
    assert parse_mem("163MiB") == 163 << 20 and parse_mem("3.827GiB") == int(3.827 * (1 << 30))
    assert parse_mem("512kB") == 512_000 and parse_mem("0B") == 0 and parse_mem("--") is None


def test_docker_load_sums_running_containers_and_spots_an_exomiser(cfg: ExomiserConfig):
    fake = FakeDocker(cfg.image_digest, containers=[("supabase_db", "public.ecr.aws/supabase/postgres:17", "163MiB"),
                                                     ("rankrev2", f"exomiser/exomiser-cli@{cfg.image_digest}", "1.5GiB")])
    load = docker_load(fake)
    assert load.in_use_bytes == (163 << 20) + int(1.5 * (1 << 30))
    assert [c["name"] for c in load.containers] == ["rankrev2", "supabase_db"] and load.exomiser_containers == ["rankrev2"]
    assert load.containers[1]["image"].startswith("public.ecr.aws") and load.containers[1]["mem_bytes"] == 163 << 20
    down = docker_load(FakeDocker(cfg.image_digest, info_ok=False))
    assert down.in_use_bytes is None and down.containers == () and down.exomiser_containers == []
    assert docker_load(FakeDocker(cfg.image_digest)).in_use_bytes == 0


def test_docker_command_mounts_and_argument_order(cfg: ExomiserConfig, tmp_path: Path):
    vcf = tmp_path / "proband.vcf.gz"
    vcf.write_bytes(b"")
    (tmp_path / "proband.vcf.gz.tbi").write_bytes(b"")
    cmd = docker_command(cfg, image=cfg.image_ref, data_dir=tmp_path / "data", config_dir=tmp_path / "cfg",
                         results_dir=tmp_path / "res", vcf=vcf, vcf_container_name="case.vcf.gz", heap="2816m")
    assert cmd[:3] == ["docker", "run", "--rm"]
    mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
    assert f"{tmp_path / 'data'}:/exomiser-data:ro" in mounts
    assert f"{tmp_path / 'cfg'}:/exomiser:ro" in mounts
    assert f"{tmp_path / 'res'}:/results" in mounts
    assert f"{vcf}:/vcf/case.vcf.gz:ro" in mounts
    assert f"{vcf}.tbi:/vcf/case.vcf.gz.tbi:ro" in mounts
    assert "JAVA_TOOL_OPTIONS=-Xmx2816m" in cmd
    assert cmd[cmd.index(cfg.image_ref) + 1:] == ["--analysis", f"/exomiser/{ANALYSIS_FILENAME}",
                                                  f"--spring.config.location=/exomiser/{PROPERTIES_FILENAME}"]
    assert "proband" not in " ".join(cmd[cmd.index(cfg.image_ref):])  # the file name stays outside the container
    assert container_vcf_name(Path("x.VCF.GZ")) == "case.vcf.gz" and container_vcf_name(Path("x.vcf")) == "case.vcf"


def test_version_and_digest_helpers():
    assert version_from_log(BANNER) == "14.0.0"
    assert version_from_log("no banner here") is None
    pinned = "sha256:" + "a" * 64
    other = "sha256:" + "b" * 64
    assert carries_digest({"id": pinned, "repo_digests": []}, pinned)
    assert carries_digest({"id": other, "repo_digests": [f"exomiser/exomiser-cli@{pinned}"]}, pinned)
    assert not carries_digest({"id": other, "repo_digests": []}, pinned)
    # the digest named in the evidence is the one that ran, never the pin by default
    assert image_digest_used({"id": other, "repo_digests": [f"exomiser/exomiser-cli@{pinned}"]}, pinned) == pinned
    assert image_digest_used({"id": other, "repo_digests": [f"exomiser/exomiser-cli@{other}"]}, pinned) == other
    assert image_digest_used({"id": other, "repo_digests": []}, pinned) == other

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout='{"repo_digests": ["r@' + pinned + '"], "id": "x", "os": "linux", "arch": "arm64"}\n', stderr="")
    assert image_digest("r@" + pinned, fake_run)["arch"] == "arm64"
    with pytest.raises(RuntimeError, match="inspect"):
        image_digest("nope", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No such image"))


def test_gene_url_is_always_openable():
    assert gene_url("1080", "ENSG00000001626", "CFTR") == "https://www.ncbi.nlm.nih.gov/gene/1080"
    assert gene_url("", "ENSG00000001626", "CFTR") == "https://www.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000001626"
    assert gene_url("", "", "CFTR") == "https://www.genenames.org/tools/search/#!/?query=CFTR"


def test_vcf_header_contigs(tmp_path: Path):
    vcf = tmp_path / "a.vcf.gz"
    with gzip.open(vcf, "wt") as f:
        f.write("##fileformat=VCFv4.2\n##contig=<ID=1,length=248956422>\n##contig=<ID=M,length=16569>\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n1\t11796321\trs1801133\tG\tA\t50\tPASS\t.\tGT\t0/1\n")
    contigs, first = vcf_header_contigs(vcf)
    assert contigs == ["1", "M"] and first is None
    assert mito_contig(contigs) == "M" and contig_renames(contigs) == {"M": "MT"}
    assert contig_renames(["1", "MT"]) == {} and contig_renames(["chr1", "chrM"]) == {}  # both resolve in svart
    bare = tmp_path / "b.vcf"
    bare.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\nchr17\t7675088\t.\tC\tT\t50\tPASS\t.\tGT\t0/1\n")
    assert vcf_header_contigs(bare) == ([], "chr17")


# ---------------------------------------------------------------- output parsing (real 14.0.0 output)

def _tsv_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def test_parse_genes_tsv_by_header_name():
    genes = parse_genes_tsv(FIX / "exomiser.genes.tsv")
    assert [(g.gene_symbol, g.moi, g.rank) for g in genes] == [("CFTR", "AR", 1), ("TP53", "AD", 2)]
    cftr, tp53 = genes
    assert cftr.entrez_id == "1080" and cftr.combined_score == 0.9894 and cftr.phenotype_score == 0.8311
    assert cftr.variant_score == 1.0 and cftr.p_value == 0.0
    assert tp53.entrez_id == "7157" and tp53.combined_score == 0.9828 and tp53.p_value == 0.0002
    assert cftr.raw["HUMAN_PHENO_EVIDENCE"].startswith("Bronchiectasis with or without elevated sweat chloride 1, modifier of (OMIM:211400)")
    assert cftr.raw["HUMAN_PPI_EVIDENCE"].startswith("Proximity to SLC26A9") and cftr.raw["FISH_PPI_EVIDENCE"] == ""


def test_parse_variants_tsv_contributing_and_contigs(tmp_path: Path):
    rows = parse_variants_tsv(FIX / "exomiser.variants.tsv")
    assert len(rows) == 2
    by = {(r.key, r.moi): r for r in rows}
    f508 = by[(("7", 117559590, "ATCT", "A"), "AR")]
    assert f508.contributing and f508.genotype == "1/1" and f508.variant_score == 1.0
    assert f508.raw["CLINVAR_VARIATION_ID"] == "7105"  # the 14.0.0 spelling, not CLINVAR_VARIANT_ID
    assert f508.raw["FUNCTIONAL_CLASS"] == "disruptive_inframe_deletion" and f508.raw["WHITELIST_VARIANT"] == "1"
    r175h = by[(("17", 7675088, "C", "T"), "AD")]
    assert r175h.contributing and r175h.genotype == "0/1" and r175h.raw["RS_ID"] == "rs28934578"
    assert contributing_from_tsv(rows) == {("CFTR", "AR"): [("7", 117559590, "ATCT", "A")], ("TP53", "AD"): [("17", 7675088, "C", "T")]}
    # a non-contributing row is kept but not counted; an unplaced contig is skipped
    header, cftr_row, _ = _tsv_lines(FIX / "exomiser.variants.tsv")
    cols = header.split("\t")
    non = cftr_row.split("\t")
    non[cols.index("CONTRIBUTING_VARIANT")] = "0"
    unplaced = list(non)
    unplaced[cols.index("CONTIG")] = "1_KI270706v1_random"
    p = tmp_path / "v.tsv"
    p.write_text("\n".join([header, cftr_row, "\t".join(non), "\t".join(unplaced)]) + "\n")
    rows = parse_variants_tsv(p)
    assert len(rows) == 2 and [r.contributing for r in rows] == [True, False]
    assert contributing_from_tsv(rows) == {("CFTR", "AR"): [("7", 117559590, "ATCT", "A")]}


def test_tsv_reader_applies_no_quoting(tmp_path: Path):
    """Both 14.0.0 writers emit unquoted fields (CSVFormat.newFormat('\\t').withQuote(null)):
    a disease name beginning with a double quote must not swallow the following rows."""
    header, cftr_row, tp53_row = _tsv_lines(FIX / "exomiser.genes.tsv")
    cols = header.split("\t")
    a = cftr_row.split("\t")
    a[cols.index("HUMAN_PHENO_EVIDENCE")] = '"Unbalanced (OMIM:000000): starts with a quote'
    b = tp53_row.split("\t")
    b[cols.index("HUMAN_PHENO_EVIDENCE")] = '"Quoted" disease (OMIM:000001)'
    p = tmp_path / "g.tsv"
    p.write_text("\n".join([header, "\t".join(a), "\t".join(b)]) + "\n")
    genes = parse_genes_tsv(p)
    assert [g.gene_symbol for g in genes] == ["CFTR", "TP53"]
    assert genes[0].raw["HUMAN_PHENO_EVIDENCE"] == '"Unbalanced (OMIM:000000): starts with a quote'
    assert genes[1].raw["HUMAN_PHENO_EVIDENCE"] == '"Quoted" disease (OMIM:000001)'


def test_tsv_parser_rejects_foreign_or_truncated_files(tmp_path: Path):
    p = tmp_path / "x.tsv"
    p.write_text("RANK\tGENE\n1\tCFTR\n")
    with pytest.raises(ValueError, match="#RANK"):
        parse_genes_tsv(p)
    p.write_text("#RANK\tID\tGENE_SYMBOL\n1\tCFTR_AR\n")
    with pytest.raises(ValueError, match="truncated"):
        parse_genes_tsv(p)


def test_json_fallback_agrees_with_the_variants_tsv():
    from_json = contributing_from_json(FIX / "exomiser.json")
    from_tsv = contributing_from_tsv(parse_variants_tsv(FIX / "exomiser.variants.tsv"))
    assert from_json == from_tsv  # only MOIs with contributing variants are reported by either
    ids = gene_identifiers_from_json(FIX / "exomiser.json")
    # the stage parses the JSON once and hands the same object to both readers
    genes = load_json_genes(FIX / "exomiser.json")
    assert contributing_from_json(genes) == from_json and gene_identifiers_from_json(genes) == ids
    with pytest.raises(ValueError, match="top-level array"):
        load_json_genes(FIX / "candidates.json")
    assert ids["CFTR"] == {"ensembl_id": "ENSG00000001626", "entrez_id": "1080", "hgnc_id": ""}  # 14.0.0 writes no hgncId
    assert ids["TP53"]["ensembl_id"] == "ENSG00000141510"
    # the shapes 14.0.0 really serialises: one geneScores entry per inheritance mode, most
    # without contributing variants; frequencies[] and clinVarData under the variant
    doc = json.loads((FIX / "exomiser.json").read_text())
    cftr = doc[0]
    assert [gs["modeOfInheritance"] for gs in cftr["geneScores"]] == ["AUTOSOMAL_DOMINANT", "AUTOSOMAL_RECESSIVE", "X_RECESSIVE", "X_DOMINANT", "MITOCHONDRIAL"]
    assert [len(gs.get("contributingVariants", [])) for gs in cftr["geneScores"]] == [0, 1, 0, 0, 0]
    v = cftr["geneScores"][1]["contributingVariants"][0]
    assert "frequencies" in v["frequencyData"] and "knownFrequencies" not in v["frequencyData"]
    assert v["pathogenicityData"]["clinVarData"]["reviewStatus"] == "PRACTICE_GUIDELINE"
    assert "hgncId" not in cftr["geneIdentifier"]


def test_build_ranking_one_row_per_gene_best_moi():
    genes = parse_genes_tsv(FIX / "exomiser.genes.tsv")
    contributing = contributing_from_tsv(parse_variants_tsv(FIX / "exomiser.variants.tsv"))
    ranking = build_ranking(genes, contributing)
    assert [(g.gene_symbol, g.rank, g.moi) for g in ranking] == [("CFTR", 1, "AR"), ("TP53", 2, "AD")]
    cftr = ranking[0]
    assert cftr.variants == (F508,) and cftr.all_variants() == {F508}
    d = cftr.as_dict()
    assert d["n_variants"] == 1 and d["variants"] == [F508] and d["exomiser_score"] == 0.9894
    assert "p_value" not in d and all("p_value" not in moi for moi in d["by_moi"].values())  # sampled per run; not derived
    # the same contributing variants from the JSON give the same ranking
    assert build_ranking(genes, contributing_from_json(FIX / "exomiser.json")) == ranking
    # a gene scoring equally under two MOIs: recessive first, both kept
    row = genes[1]
    tie = [GeneRow(**{**row.__dict__, "moi": "AD"}), GeneRow(**{**row.__dict__, "moi": "AR"})]
    both = build_ranking(tie, {("TP53", "AD"): [("17", 7675088, "C", "T")]})
    assert len(both) == 1 and both[0].moi == "AR" and set(both[0].by_moi) == {"AR", "AD"}
    assert both[0].variants == () and both[0].all_variants() == {R175H}


# ---------------------------------------------------------------- the join

def test_join_candidates_symbol_ensembl_and_unranked():
    genes = parse_genes_tsv(FIX / "exomiser.genes.tsv")
    ranking = build_ranking(genes, contributing_from_tsv(parse_variants_tsv(FIX / "exomiser.variants.tsv")))
    gene_ids = gene_identifiers_from_json(FIX / "exomiser.json")
    cands = json.loads((FIX / "candidates.json").read_text())
    joined, counts = join_candidates(cands, ranking, gene_ids, top_n=20)
    by_id = {c["candidate_id"]: c for c in joined["candidates"]}
    cftr = by_id["CFTR:hom"]
    assert cftr["exomiser_rank"] == 1 and cftr["exomiser_score"] == 0.9894 and cftr["phenotype_score"] == 0.8311
    assert cftr["exomiser_match"] == "symbol" and cftr["exomiser_variants_matched"] == [F508]
    assert cftr["exomiser_evidence_id"] == "exomiser:CFTR" and set(cftr["exomiser_by_moi"]) == {"AR"}
    assert cftr["variants"][0]["evidence_ids"] == ["vep:7:117559590:ATCT:A", "clinvar:VCV000007105", "gnomad:7-117559590-ATCT-A"]
    mthfr = by_id["MTHFR:het_single"]
    assert mthfr["exomiser_rank"] is None and mthfr["exomiser_score"] is None and mthfr["phenotype_score"] is None
    assert mthfr["exomiser_match"] is None and mthfr["exomiser_evidence_id"] is None
    assert [g["gene_symbol"] for g in joined["exomiser_only"]] == ["TP53"]
    assert joined["exomiser_only"][0]["evidence_id"] == "exomiser:TP53"
    assert counts == {"candidates": 2, "candidates_ranked": 1, "candidates_unranked": 1, "candidates_with_variant_match": 1,
                      "exomiser_only": 1, "exomiser_genes_not_in_shortlist": 1}
    # an old alias in the shortlist is matched on its Ensembl id — only when the JSON supplied the ids
    alias = json.loads(json.dumps(cands))
    alias["candidates"].append({"candidate_id": "P53:het_single", "gene_symbol": "P53", "gene_id": "ENSG00000141510",
                                "model": "het_single", "priority": 3, "variants": [{"key": R175H}]})
    j2, c2 = join_candidates(alias, ranking, gene_ids, top_n=20)
    p53 = {c["candidate_id"]: c for c in j2["candidates"]}["P53:het_single"]
    assert p53["exomiser_match"] == "ensembl_id" and p53["exomiser_rank"] == 2 and p53["exomiser_moi"] == "AD"
    assert p53["exomiser_gene_symbol"] == "TP53" and p53["exomiser_variants_matched"] == [R175H]
    assert j2["exomiser_only"] == [] and c2["candidates_ranked"] == 2
    j3, c3 = join_candidates(alias, ranking, {}, top_n=0)
    assert j3["exomiser_only"] == [] and c3["exomiser_genes_not_in_shortlist"] == 1  # no JSON → P53 unmatched
    empty, c4 = join_candidates(None, ranking, {}, top_n=1)
    assert empty["candidates"] == [] and [g["gene_symbol"] for g in empty["exomiser_only"]] == ["CFTR"] and c4["exomiser_only"] == 1


# ---------------------------------------------------------------- the whole stage, Docker faked

def make_case(tmp: Path, hpo: list[str] = HPO, mito: str = "MT") -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    vcf = tmp / "public.vcf.gz"
    with gzip.open(vcf, "wt") as f:
        f.write("##fileformat=VCFv4.2\n##reference=GRCh38\n##contig=<ID=1,length=248956422>\n##contig=<ID=7,length=159345973>\n"
                f"##contig=<ID=17,length=83257441>\n##contig=<ID={mito},length=16569>\n"
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSYNTHETIC\n"
                "1\t11796321\trs1801133\tG\tA\t500\tPASS\t.\tGT\t0/1\n"
                "7\t117559590\t.\tATCT\tA\t500\tPASS\t.\tGT\t1/1\n"
                "17\t7675088\t.\tC\tT\t500\tPASS\t.\tGT\t0/1\n")
    (tmp / "public.vcf.gz.tbi").write_bytes(b"")  # CaseConfig only checks that an index exists
    case = tmp / "case.yaml"
    case.write_text("proband_id: PUBLIC01\nvcf: public.vcf.gz\nhpo: [" + ", ".join(hpo) + "]\n")
    return case


def make_data_dir(tmp: Path, cfg: ExomiserConfig) -> Path:
    data = tmp / "exomiser-data"
    for b in cfg.bundles.values():
        d = b.extracted_path(data)
        d.mkdir(parents=True)
        for f in b.required_files:
            (d / f).write_bytes(b"stub")
    return data


def make_run(tmp: Path, with_candidates: bool = True) -> Path:
    run = tmp / "run"
    run.mkdir(parents=True, exist_ok=True)
    if with_candidates:
        (run / "03_filter").mkdir(exist_ok=True)
        shutil.copy(FIX / "candidates.json", run / "03_filter" / "candidates.json")
    return run


FIXTURE_FILES = {"genes": "exomiser.genes.tsv", "variants": "exomiser.variants.tsv", "json": "exomiser.json"}


class FakeDocker:
    """Answers the three docker calls the stage makes; ``run`` drops the recorded outputs
    (or ``files`` overrides) into the mounted results directory, like the container would."""

    def __init__(self, digest: str, outputs: tuple[str, ...] = ("genes", "variants", "json"), exit_code: int = 0,
                 log: str = BANNER, files: dict[str, Path] | None = None, repo_digests: list[str] | None = None,
                 info_ok: bool = True, containers: list[tuple[str, str, str]] | None = None, bcftools_ok: bool = True,
                 index_counts: bool = True):
        self.digest = digest
        self.outputs = outputs
        self.exit_code = exit_code
        self.log = log
        self.files = files or {}
        self.repo_digests = repo_digests if repo_digests is not None else [f"exomiser/exomiser-cli@{digest}"]
        self.info_ok = info_ok
        self.containers = containers or []  # (name, image, mem usage) per running container, as docker prints them
        self.bcftools_ok = bcftools_ok
        self.index_counts = index_counts
        """False: the case VCF's index answers `bcftools index -n` like a provider's tabix index without counts."""
        self.calls: list[list[str]] = []
        self.vcf_mounted: list[tuple[Path, list[str]]] = []
        """The VCF the container was given and its header contigs, as they were at run time."""

    def __call__(self, cmd: list[str], **kw: Any) -> subprocess.CompletedProcess:
        self.calls.append(list(cmd))
        if cmd[0] == "bcftools":  # the contig rename and the regions subset are real bcftools calls; the daemon is not involved
            if not self.bcftools_ok:
                return subprocess.CompletedProcess(cmd, 255, stdout="", stderr="[E::hts_open_format] fail to open file")
            if not self.index_counts and cmd[1:3] == ["index", "-n"] and "##idx##" not in cmd[3] and not cmd[3].endswith(REGIONS_VCF_FILENAME):
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=f"index of {cmd[3]} does not contain any count metadata. "
                                                   "Please re-index with a newer version of htslib.")
            return subprocess.run(cmd, **kw)
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(
                {"repo_digests": self.repo_digests, "id": self.digest, "os": "linux", "arch": "arm64"}) + "\n", stderr="")
        if not self.info_ok and cmd[:2] in (["docker", "info"], ["docker", "ps"], ["docker", "stats"]):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Cannot connect to the Docker daemon")
        if cmd[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="4109217792\n", stderr="")
        if cmd[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="".join(f"{n}\t{i}\n" for n, i, _ in self.containers), stderr="")
        if cmd[:2] == ["docker", "stats"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="".join(f"{n}\t{m} / 3.827GiB\n" for n, _, m in self.containers), stderr="")
        if cmd[:2] == ["docker", "run"]:
            mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
            results = Path(next(m for m in mounts if m.endswith(":/results")).rsplit(":", 1)[0])
            vcf = Path(next(m for m in mounts if ":/vcf/" in m and not m.endswith((".tbi:ro", ".csi:ro"))).rsplit(":", 2)[0])
            self.vcf_mounted.append((vcf, vcf_header_contigs(vcf)[0]))
            for name in self.outputs:
                shutil.copy(self.files.get(name, FIX / FIXTURE_FILES[name]), results / FIXTURE_FILES[name])
            kw["stdout"].write(self.log)
            return subprocess.CompletedProcess(cmd, self.exit_code)
        raise AssertionError(f"unexpected docker call: {cmd}")


def _index(out: Path) -> dict[str, Any]:
    return json.loads((out / "evidence" / "index.json").read_text())


def _record(out: Path, rid: str) -> dict[str, Any]:
    return json.loads((out / "evidence" / _index(out)[rid]["path"]).read_text())


def _snapshot(out: Path) -> dict[str, bytes]:
    files = [out / "ranking.tsv", out / "joined.json", out / "evidence" / "index.json"] + sorted((out / "evidence" / "exomiser").glob("*.json"))
    return {str(p.relative_to(out)): p.read_bytes() for p in files}


def test_run_rank_end_to_end_and_join_only_is_byte_identical(tmp_path: Path, cfg: ExomiserConfig):
    case = make_case(tmp_path)
    run = make_run(tmp_path)
    data = make_data_dir(tmp_path, cfg)
    fake = FakeDocker(cfg.image_digest)
    messages: list[str] = []

    manifest = run_rank(run, case, data, progress=messages.append, run=fake)
    out = run / "04_rank"
    raw = out / "exomiser"
    # the generated files, and the raw outputs where the container left them
    analysis = yaml.safe_load((raw / "config" / ANALYSIS_FILENAME).read_text())
    assert analysis["analysis"]["hpoIds"] == HPO and analysis["analysis"]["vcf"] == "/vcf/case.vcf.gz"
    assert (raw / "config" / PROPERTIES_FILENAME).read_text().startswith("exomiser.data-directory=/exomiser-data")
    assert (raw / "exomiser.genes.tsv").exists() and (raw / "exomiser.variants.tsv").exists() and (raw / "exomiser.json").exists()
    log = (raw / "exomiser.log").read_text()
    assert "v14.0.0" in log and log.rstrip().endswith("[engine] exit code 0")
    facts = json.loads((raw / RUN_FILENAME).read_text())
    assert facts["exomiser_exit_code"] == 0 and facts["image_matches_pin"] is True and facts["heap"] == "2816m"
    assert facts["image_digest_used"] == cfg.image_digest and facts["hpo"] == HPO and facts["docker_command"][:2] == ["docker", "run"]
    # the identity of what the container ranked, and of the config that shaped the run
    vcf_sha = hashlib.sha256((tmp_path / "public.vcf.gz").read_bytes()).hexdigest()
    assert facts["vcf_sha256"] == vcf_sha == facts["vcf_container_sha256"] and facts["vcf_bytes"] == facts["vcf_container_bytes"]
    assert facts["vcf_contig_renames"] == {} and facts["vcf_mito_contig"] == "MT"
    assert facts["config_sha256"] == cfg.sha256 and facts["config_path"] == str(cfg.path) and facts["data_version"] == "2406"
    assert facts["candidates_sha256"] == hashlib.sha256((run / "03_filter" / "candidates.json").read_bytes()).hexdigest()
    assert facts["docker_memory_bytes"] == 4109217792 and facts["docker_memory_in_use_bytes"] == 0 and facts["docker_containers_running"] == 0
    assert facts["analysis_yaml_sha256"] == sha256_text((raw / "config" / ANALYSIS_FILENAME).read_text())
    assert not (out / "input").exists()  # no renamed copy was needed
    run_cmd = next(c for c in fake.calls if c[:2] == ["docker", "run"])
    assert run_cmd[-1] == f"--spring.config.location=/exomiser/{PROPERTIES_FILENAME}" and cfg.image_ref in run_cmd
    assert "JAVA_TOOL_OPTIONS=-Xmx2816m" in run_cmd
    # ranking.tsv
    lines = (out / "ranking.tsv").read_text().splitlines()
    assert lines[0].split("\t") == list(RANKING_COLUMNS)
    assert lines[1].split("\t") == ["1", "CFTR", "0.9894", "0.8311", "1.0000", "AR", "1", F508]
    assert lines[2].split("\t") == ["2", "TP53", "0.9828", "0.7838", "1.0000", "AD", "1", R175H]
    # joined.json
    j = read_joined(run)
    assert j["hpo"] == HPO and j["exomiser_version"] == "14.0.0" and j["data_version"] == "2406"
    by_id = {c["candidate_id"]: c for c in j["candidates"]}
    assert by_id["CFTR:hom"]["exomiser_rank"] == 1 and by_id["MTHFR:het_single"]["exomiser_rank"] is None
    assert [g["gene_symbol"] for g in j["exomiser_only"]] == ["TP53"]
    assert "p_value" not in (out / "joined.json").read_text()
    # evidence records: one per gene in joined.json, citable ids, the raw slice minus P-VALUE
    index = _index(out)
    assert set(index) == {"exomiser:CFTR", "exomiser:TP53"}
    rec = _record(out, "exomiser:CFTR")
    assert rec["source_version"] == f"exomiser-cli 14.0.0 · data 2406 · image {cfg.image_digest}"
    assert rec["query"]["hpoIds"] == HPO and rec["payload"]["ranking"]["rank"] == 1
    assert rec["retrieved_at"] == facts["finished_at"]  # when the evidence came into being, not when it was joined
    assert rec["url"] == "https://www.ncbi.nlm.nih.gov/gene/1080"
    raw_slice = rec["payload"]["raw"]
    assert list(raw_slice["genes_tsv"]) == ["AR"]
    assert raw_slice["genes_tsv"]["AR"]["HUMAN_PHENO_EVIDENCE"].startswith("Bronchiectasis")
    assert raw_slice["genes_tsv"]["AR"]["EXOMISER_GENE_COMBINED_SCORE"] == "0.9894"
    assert len(raw_slice["variants_tsv"]) == 1 and raw_slice["variants_tsv"][0]["HGVS"] == "CFTR:ENST00000003084.11:c.1521_1523del:p.(Phe508del)"
    assert "P-VALUE" not in raw_slice["genes_tsv"]["AR"] and "P-VALUE" not in raw_slice["variants_tsv"][0]
    assert "P-VALUE" in rec["payload"]["omitted_columns"]
    assert _record(out, "exomiser:TP53")["url"] == "https://www.ncbi.nlm.nih.gov/gene/7157"
    # manifest
    m = json.loads(manifest.read_text())
    assert m["stage"] == "rank"
    p, c = m["params"], m["counts"]
    assert p["analysis_yaml_sha256"] == sha256_text((raw / "config" / ANALYSIS_FILENAME).read_text())
    assert p["hpo"] == HPO and p["heap"] == "2816m" and p["image_matches_pin"] is True
    assert p["image_observed"]["arch"] == "arm64" and p["image_digest_used"] == cfg.image_digest and p["data_version"] == "2406"
    assert p["vcf_contig_style"] == "ensembl" and p["vcf_mito_contig"] == "MT"
    assert p["analysis"]["inheritanceModes"]["AUTOSOMAL_DOMINANT"] == 0.1  # every threshold echoed
    assert p["exomiser_exit_code"] == 0 and isinstance(p["wall_time_s"], float) and p["docker_memory_bytes"] == 4109217792
    assert p["data_bundles"]["hg38"]["members"]["pinned"] == 6 and p["data_bundles"]["hg38"]["members"]["hashed"] == 0
    assert any("pinned member files" in n for n in m["notes"])  # the stub data dir was never hashed
    assert m["tools"]["exomiser"] == "14.0.0"
    assert c["genes_ranked"] == 2 and c["gene_moi_rows"] == 2 and c["variant_rows"] == 2
    assert c["candidates"] == 2 and c["candidates_ranked"] == 1 and c["exomiser_only"] == 1
    assert c["evidence_records"] == 2 == len(index)
    assert m["inputs"]["vcf"]["sha256"] == vcf_sha and m["inputs"]["candidates"]["sha256"] and m["inputs"]["case"]["sha256"]
    assert m["outputs"]["exomiser_genes"]["sha256"] and m["outputs"]["joined"]["sha256"] and m["outputs"]["exomiser_run"]["sha256"]
    assert p["vcf_container_sha256"] == vcf_sha and p["docker_memory_in_use_bytes"] == 0 and p["docker_exomiser_containers_running"] == []
    # progress never prints an HPO term, a variant, or the VCF's path
    assert messages and not any("HP:" in msg for msg in messages)
    assert not any(pos in msg for msg in messages for pos in ("117559590", "7675088", "11796321", "ATCT"))
    assert not any("public.vcf" in msg or str(tmp_path) in msg for msg in messages)

    # --join-only rebuilds the derived files without Docker, produces the same bytes, and
    # keeps the facts of the container run instead of dropping them
    before = _snapshot(out)
    fake2 = FakeDocker(cfg.image_digest)
    run_rank(run, case, data, join_only=True, run=fake2)
    assert fake2.calls == []
    assert _snapshot(out) == before
    m2 = json.loads(manifest.read_text())
    p2 = m2["params"]
    assert p2["join_only"] is True and any(n.startswith("join-only") for n in m2["notes"])
    assert p2["exomiser_exit_code"] == 0 and p2["image_matches_pin"] is True and p2["image_observed"] == p["image_observed"]
    assert p2["wall_time_s"] == p["wall_time_s"] and p2["heap"] == "2816m" and p2["docker_command"] == p["docker_command"]
    assert p2["run_facts_source"].endswith(RUN_FILENAME)
    assert p2["analysis"] == p["analysis"] and p2["output_formats"] == p["output_formats"] and p2["config_sha256"] == cfg.sha256
    assert m2["inputs"]["config"]["sha256"] == cfg.sha256 and m2["inputs"]["vcf"]["sha256"] == vcf_sha
    assert not any("could not be verified" in n or "is not the one the run used" in n for n in m2["notes"])


def test_join_only_uses_the_terms_and_image_of_the_run_as_it_was(tmp_path: Path, cfg: ExomiserConfig, monkeypatch):
    case = make_case(tmp_path)
    run = make_run(tmp_path)
    data = make_data_dir(tmp_path, cfg)
    run_rank(run, case, data, run=FakeDocker(cfg.image_digest))
    out = run / "04_rank"
    first = _record(out, "exomiser:CFTR")
    # the case file changes after the run; the raw outputs are still from the old terms
    case.write_text("proband_id: PUBLIC01\nvcf: public.vcf.gz\nhpo: [HP:0000001]\n")
    manifest = run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest))
    m = json.loads(manifest.read_text())
    assert m["params"]["hpo"] == HPO and m["params"]["n_hpo"] == 3
    assert any("case.yaml now lists 1 HPO terms but the raw run used 3" in n for n in m["notes"])
    assert read_joined(run)["hpo"] == HPO
    rec = _record(out, "exomiser:CFTR")
    assert rec["query"]["hpoIds"] == HPO and rec == first  # byte-identical: the same raw run
    # a fresh run with the new terms is a different question, and different evidence: its
    # records are stamped with that container's finish time
    monkeypatch.setattr(rank_run, "_now", lambda: "2030-01-01T00:00:00+00:00")
    run_rank(run, case, data, run=FakeDocker(cfg.image_digest))
    rec2 = _record(out, "exomiser:CFTR")
    assert rec2["query"]["hpoIds"] == ["HP:0000001"] and rec2["payload"] == first["payload"]
    assert rec2["retrieved_at"] == "2030-01-01T00:00:00+00:00" != first["retrieved_at"]
    assert json.loads((out / "exomiser" / RUN_FILENAME).read_text())["finished_at"] == "2030-01-01T00:00:00+00:00"


def test_join_only_refuses_a_missing_or_failed_run(tmp_path: Path, cfg: ExomiserConfig):
    case = make_case(tmp_path)
    run = make_run(tmp_path)
    data = make_data_dir(tmp_path, cfg)
    with pytest.raises(FileNotFoundError, match="join-only"):  # nothing to join yet
        run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest))
    # Exomiser writes the genes TSV before the writer that can crash: a genes TSV alone is not a run
    crash = BANNER + "java.lang.ArrayIndexOutOfBoundsException: Index 0 out of bounds for length 0\n\tat FrequencyData.maxFrequency\n"
    with pytest.raises(ExomiserFailed, match="issue #565"):
        run_rank(run, case, data, run=FakeDocker(cfg.image_digest, outputs=("genes",), exit_code=1, log=crash))
    raw = run / "04_rank" / "exomiser"
    assert (raw / "exomiser.genes.tsv").exists() and json.loads((raw / RUN_FILENAME).read_text())["exomiser_exit_code"] == 1
    assert (raw / "exomiser.log").read_text().rstrip().endswith("[engine] exit code 1")
    with pytest.raises(ExomiserFailed, match="exited with code 1"):
        run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest))
    (raw / RUN_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match=RUN_FILENAME):
        run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest))
    assert not (run / "04_rank" / "manifest.json").exists()


def test_stale_evidence_records_are_removed_on_rerun(tmp_path: Path, cfg: ExomiserConfig):
    case = make_case(tmp_path)
    run = make_run(tmp_path)
    data = make_data_dir(tmp_path, cfg)
    out = run / "04_rank"
    run_rank(run, case, data, run=FakeDocker(cfg.image_digest))
    assert set(_index(out)) == {"exomiser:CFTR", "exomiser:TP53"}
    cftr_only = tmp_path / "cftr_only.genes.tsv"
    cftr_only.write_text("\n".join(_tsv_lines(FIX / "exomiser.genes.tsv")[:2]) + "\n")
    manifest = run_rank(run, case, data, run=FakeDocker(cfg.image_digest, files={"genes": cftr_only}))
    m = json.loads(manifest.read_text())
    j = read_joined(run)
    cited = {c["exomiser_evidence_id"] for c in j["candidates"] if c["exomiser_evidence_id"]} | {g["evidence_id"] for g in j["exomiser_only"]}
    assert cited == {"exomiser:CFTR"} and set(_index(out)) == cited
    assert m["counts"]["evidence_records"] == 1 == len(_index(out))
    assert not list((out / "evidence" / "exomiser").glob("*TP53*"))


def test_non_pinned_image_is_named_in_the_evidence_and_manifest(tmp_path: Path, cfg: ExomiserConfig):
    case = make_case(tmp_path)
    run = make_run(tmp_path, with_candidates=False)
    data = make_data_dir(tmp_path, cfg)
    other = "sha256:" + "b" * 64
    fake = FakeDocker(other, repo_digests=[f"exomiser/exomiser-cli@{other}"])
    manifest = run_rank(run, case, data, image="exomiser/exomiser-cli:latest", run=fake)
    m = json.loads(manifest.read_text())
    assert m["params"]["image_matches_pin"] is False and m["params"]["image_digest_used"] == other
    assert any("does not carry the pinned digest" in n for n in m["notes"])
    rec = _record(run / "04_rank", "exomiser:CFTR")
    assert rec["source_version"].endswith(f"image {other}") and cfg.image_digest not in rec["source_version"]
    # --join-only keeps naming the image that ran, even though the pin has not changed
    run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest))
    assert _record(run / "04_rank", "exomiser:CFTR")["source_version"].endswith(f"image {other}")
    assert json.loads(manifest.read_text())["params"]["image_digest_used"] == other


def test_run_rank_without_stage3_and_without_variants_tsv(tmp_path: Path, cfg: ExomiserConfig):
    case = make_case(tmp_path, mito="chrM")  # svart resolves chrM: no rename
    run = make_run(tmp_path, with_candidates=False)
    data = make_data_dir(tmp_path, cfg)
    fake = FakeDocker(cfg.image_digest, outputs=("genes", "json"))
    manifest = run_rank(run, case, data, image="exomiser/exomiser-cli:14.0.0", heap="6g", run=fake)
    m = json.loads(manifest.read_text())
    assert m["params"]["heap"] == "6g" and m["params"]["image_requested"] == "exomiser/exomiser-cli:14.0.0"
    assert m["params"]["contributing_variants_source"] == "json"
    assert any("issue #565" in n for n in m["notes"])
    assert any("stage 3 has not run" in n for n in m["notes"])
    assert m["params"]["vcf_mito_contig"] == "chrM" and m["params"]["vcf_contig_renames"] == {}
    assert not any("mitochondrion" in n for n in m["notes"]) and not (run / "04_rank" / "input").exists()
    j = read_joined(run)
    assert j["candidates"] == [] and [g["gene_symbol"] for g in j["exomiser_only"]] == ["CFTR", "TP53"]
    assert j["exomiser_only"][0]["variants"] == [F508]  # contributing variants from the JSON
    rec = _record(run / "04_rank", "exomiser:TP53")
    assert rec["payload"]["raw"]["variants_tsv"] == [] and list(rec["payload"]["raw"]["genes_tsv"]) == ["AD"]


def test_run_rank_refuses_incomplete_data_and_reports_failures(tmp_path: Path, cfg: ExomiserConfig):
    case = make_case(tmp_path)
    run = make_run(tmp_path, with_candidates=False)
    data = make_data_dir(tmp_path, cfg)
    with pytest.raises(FileNotFoundError, match="exomiser-download"):
        run_rank(run, case, tmp_path / "empty", run=FakeDocker(cfg.image_digest))
    oom = BANNER + "java.lang.OutOfMemoryError: Java heap space\n"
    with pytest.raises(ExomiserFailed, match="out of memory"):
        run_rank(run, case, data, run=FakeDocker(cfg.image_digest, outputs=(), exit_code=1, log=oom))
    with pytest.raises(ExomiserFailed, match="exited 0 but wrote no genes TSV"):
        run_rank(run, case, data, run=FakeDocker(cfg.image_digest, outputs=("json",)))
    assert not (run / "04_rank" / "manifest.json").exists()  # a failed run leaves the log, not a manifest
    no_hpo = make_case(tmp_path / "nohpo", hpo=[])
    with pytest.raises(ValueError, match="HPO"):
        run_rank(run, no_hpo, data, run=FakeDocker(cfg.image_digest))


def test_unreadable_docker_info_is_noted_not_silent(tmp_path: Path, cfg: ExomiserConfig):
    case = make_case(tmp_path)
    run = make_run(tmp_path, with_candidates=False)
    data = make_data_dir(tmp_path, cfg)
    manifest = run_rank(run, case, data, run=FakeDocker(cfg.image_digest, info_ok=False))
    m = json.loads(manifest.read_text())
    assert m["params"]["heap"] == "2g" and m["params"]["docker_memory_bytes"] is None
    assert m["params"]["docker_memory_in_use_bytes"] is None and m["params"]["docker_containers_running"] == 0
    assert any("heap fell back to 2g" in n for n in m["notes"])
    manifest = run_rank(run, case, data, heap="5g", run=FakeDocker(cfg.image_digest, info_ok=False))
    m = json.loads(manifest.read_text())
    assert m["params"]["heap"] == "5g" and not any("fell back" in n for n in m["notes"])


def test_a_failed_rerun_leaves_no_stale_results(tmp_path: Path, cfg: ExomiserConfig):
    """A success, then a rerun with new terms that is OOM-killed: 04_rank/ must not
    present the old manifest, joined.json, ranking.tsv or evidence next to the failed
    run's analysis.yml and run.json (stage 5 reads joined.json directly)."""
    case = make_case(tmp_path)
    run = make_run(tmp_path)
    data = make_data_dir(tmp_path, cfg)
    out = run / "04_rank"
    run_rank(run, case, data, run=FakeDocker(cfg.image_digest))
    assert (out / "manifest.json").exists() and (out / "joined.json").exists() and (out / "evidence" / "index.json").exists()
    case.write_text("proband_id: PUBLIC01\nvcf: public.vcf.gz\nhpo: [HP:0000001]\n")
    with pytest.raises(ExomiserFailed, match="out of memory"):
        run_rank(run, case, data, run=FakeDocker(cfg.image_digest, outputs=("genes",), exit_code=137))
    for name in ("manifest.json", "joined.json", "ranking.tsv"):
        assert not (out / name).exists(), name
    assert not (out / "evidence").exists()
    raw = out / "exomiser"
    assert json.loads((raw / RUN_FILENAME).read_text())["exomiser_exit_code"] == 137
    assert hpo_from_analysis((raw / "config" / ANALYSIS_FILENAME).read_text()) == ["HP:0000001"]
    assert (raw / "exomiser.log").read_text().rstrip().endswith("[engine] exit code 137")
    with pytest.raises(ExomiserFailed, match="exited with code 137"):
        run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest))


def test_join_only_refuses_an_edited_analysis_or_a_swapped_vcf(tmp_path: Path, cfg: ExomiserConfig):
    """--join-only vouches for what it joins: the analysis.yml must still be the file
    the run used and the case VCF must still be the file the run ranked."""
    case = make_case(tmp_path)
    run = make_run(tmp_path)
    data = make_data_dir(tmp_path, cfg)
    run_rank(run, case, data, run=FakeDocker(cfg.image_digest))
    raw = run / "04_rank" / "exomiser"
    analysis = raw / "config" / ANALYSIS_FILENAME
    original = analysis.read_text()
    analysis.write_text(original.replace("HP:0011947", "HP:0000001"))
    with pytest.raises(ExomiserFailed, match="edited after the run"):
        run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest))
    analysis.write_text(original)
    run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest))  # restored: fine again
    # the VCF is replaced by another (public, one-variant) file after the run
    vcf = tmp_path / "public.vcf.gz"
    with gzip.open(vcf, "wt") as f:
        f.write("##fileformat=VCFv4.2\n##contig=<ID=17,length=83257441>\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
                "17\t7675088\t.\tC\tT\t50\tPASS\t.\tGT\t0/1\n")
    before = _snapshot(run / "04_rank") | {"manifest.json": (run / "04_rank" / "manifest.json").read_bytes()}
    with pytest.raises(ExomiserFailed, match="is not the file the raw run ranked"):
        run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest))
    assert _snapshot(run / "04_rank") | {"manifest.json": (run / "04_rank" / "manifest.json").read_bytes()} == before  # a refused join writes nothing
    # run.json from an engine version that recorded no VCF identity: joined, but said so
    facts = json.loads((raw / RUN_FILENAME).read_text())
    for k in ("vcf_sha256", "vcf_bytes", "vcf_path"):
        facts.pop(k)
    (raw / RUN_FILENAME).write_text(json.dumps(facts, sort_keys=True, indent=1) + "\n")
    m = json.loads(run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest)).read_text())
    assert any("could not be verified" in n for n in m["notes"])


def test_join_only_manifest_describes_the_run_not_the_current_config(tmp_path: Path, cfg: ExomiserConfig):
    """The thresholds, output formats and config identity in a --join-only manifest are
    those of the run being joined, read back from the analysis.yml Exomiser saw and from
    run.json — not the config and options passed now."""
    case = make_case(tmp_path)
    run = make_run(tmp_path)
    data = make_data_dir(tmp_path, cfg)
    run_rank(run, case, data, run=FakeDocker(cfg.image_digest))
    edited = tmp_path / "exomiser.edited.yaml"
    edited.write_text(DEFAULT_CONFIG.read_text().replace("frequencyFilter: {maxFrequency: 2.0}", "frequencyFilter: {maxFrequency: 1.0}"))
    edited_cfg = ExomiserConfig.load(edited)
    assert {"frequencyFilter": {"maxFrequency": 1.0}} in edited_cfg.analysis["steps"] and edited_cfg.sha256 != cfg.sha256
    manifest = run_rank(run, case, data, config_path=edited, output_formats=("JSON", "TSV_GENE"), join_only=True,
                        run=FakeDocker(cfg.image_digest))
    m = json.loads(manifest.read_text())
    p = m["params"]
    assert {"frequencyFilter": {"maxFrequency": 2.0}} in p["analysis"]["steps"]  # the run's, not the edited config's
    assert p["analysis"] == cfg.analysis and "hpoIds" not in p["analysis"] and "vcf" not in p["analysis"]
    assert p["output_formats"] == ["TSV_GENE", "TSV_VARIANT", "JSON"] and p["output_options"]["numGenes"] == 0
    assert p["config_sha256"] == cfg.sha256 == m["inputs"]["config"]["sha256"] and m["inputs"]["config"]["path"] == str(cfg.path)
    assert p["data_version"] == "2406"
    assert any("is not the one the run used" in n for n in m["notes"])
    assert any("output formats ['JSON', 'TSV_GENE'] were requested but the run wrote" in n for n in m["notes"])
    assert any("analysis template differs" in n for n in m["notes"])
    # the same on-disk file read back agrees with what the config rendered
    template, options = analysis_from_yaml((run / "04_rank" / "exomiser" / "config" / ANALYSIS_FILENAME).read_text())
    assert template == cfg.analysis and options["outputFormats"] == ["TSV_GENE", "TSV_VARIANT", "JSON"]


def test_auto_heap_subtracts_running_containers_and_refuses_a_second_exomiser(tmp_path: Path, cfg: ExomiserConfig):
    case = make_case(tmp_path)
    run = make_run(tmp_path, with_candidates=False)
    data = make_data_dir(tmp_path, cfg)
    resident = [("supabase_db", "public.ecr.aws/supabase/postgres:17", "163MiB"),
                ("supabase_analytics", "public.ecr.aws/supabase/logflare:1.0", "367.3MiB"),
                ("gram-api-postgres-1", "postgres:16", "16.27MiB")]
    fake = FakeDocker(cfg.image_digest, containers=resident)
    m = json.loads(run_rank(run, case, data, run=fake).read_text())
    in_use = (163 << 20) + int(367.3 * (1 << 20)) + int(16.27 * (1 << 20))
    assert m["params"]["docker_memory_in_use_bytes"] == in_use and m["params"]["docker_containers_running"] == 3
    assert m["params"]["heap"] == heap_for(cfg, 4109217792, in_use) == "2304m"  # not the 2816m the total alone gives
    assert m["params"]["docker_exomiser_containers_running"] == [] and not any("exceeds" in n for n in m["notes"])
    assert "JAVA_TOOL_OPTIONS=-Xmx2304m" in next(c for c in fake.calls if c[:2] == ["docker", "run"])
    # an explicit heap the daemon cannot back is run as asked, but the manifest says so
    m = json.loads(run_rank(run, case, data, heap="3g", run=FakeDocker(cfg.image_digest, containers=resident)).read_text())
    assert m["params"]["heap"] == "3g" and any("Heap 3g exceeds what the daemon can back" in n for n in m["notes"])
    # another exomiser-cli on the daemon: refused before anything is written — the previous
    # run's results stay as they are — unless --heap is explicit
    busy = resident + [("rankrev2", f"docker.io/exomiser/exomiser-cli@{cfg.image_digest}", "1.9GiB")]
    fake = FakeDocker(cfg.image_digest, containers=busy)
    before = _snapshot(run / "04_rank") | {"manifest.json": (run / "04_rank" / "manifest.json").read_bytes()}
    with pytest.raises(ExomiserFailed, match="another exomiser-cli container is running .*rankrev2"):
        run_rank(run, case, data, run=fake)
    assert not any(c[:2] == ["docker", "run"] for c in fake.calls)
    assert _snapshot(run / "04_rank") | {"manifest.json": (run / "04_rank" / "manifest.json").read_bytes()} == before
    assert json.loads((run / "04_rank" / "exomiser" / RUN_FILENAME).read_text())["heap"] == "3g"  # the previous run's
    m = json.loads(run_rank(run, case, data, heap="1g", run=FakeDocker(cfg.image_digest, containers=busy)).read_text())
    assert m["params"]["docker_exomiser_containers_running"] == ["rankrev2"]
    assert any("Another exomiser-cli container was running" in n for n in m["notes"])


needs_bcftools = pytest.mark.skipif(shutil.which("bcftools") is None, reason="bcftools not installed")


def make_case_with_mito(tmp: Path, mito: str) -> Path:
    """The public three variants plus the public MT m.3243A>G (rs199474657, ClinVar
    pathogenic/likely pathogenic), with the mitochondrion spelled ``mito``."""
    case = make_case(tmp, mito=mito)
    vcf = tmp / "public.vcf.gz"
    with gzip.open(vcf, "rt") as f:
        text = f.read()
    with gzip.open(vcf, "wt") as f:
        f.write(text + f"{mito}\t3243\trs199474657\tA\tG\t500\tPASS\t.\tGT\t1/1\n")
    return case


@needs_bcftools
def test_bare_M_is_renamed_to_MT_for_the_container(tmp_path: Path, cfg: ExomiserConfig):
    """Exomiser 14.0.0 drops every record on a contig it cannot resolve, and bare 'M' is
    one (verified live: MT-ND1 ranked with 'MT', silently absent with 'M'). The container
    gets a renamed copy; the manifest names both files; the copy does not outlive the run."""
    case = make_case_with_mito(tmp_path, "M")
    run = make_run(tmp_path, with_candidates=False)
    data = make_data_dir(tmp_path, cfg)
    out = run / "04_rank"
    fake = FakeDocker(cfg.image_digest)
    manifest = run_rank(run, case, data, run=fake)
    (mounted, contigs), = fake.vcf_mounted
    assert mounted == out / "input" / "case.vcf.gz" and contigs == ["1", "7", "17", "MT"]
    assert not (out / "input").exists()  # removed after the run, success or not
    m = json.loads(manifest.read_text())
    p = m["params"]
    original_sha = hashlib.sha256((tmp_path / "public.vcf.gz").read_bytes()).hexdigest()
    assert m["inputs"]["vcf"]["sha256"] == original_sha and p["vcf_container_sha256"] != original_sha
    assert p["vcf_contig_renames"] == {"M": "MT"} and p["vcf_mito_contig"] == "M" and p["vcf_container_bytes"] > 0
    assert m["tools"]["bcftools"].startswith("bcftools") and any("renamed in the header and records" in n for n in m["notes"])
    facts = json.loads((out / "exomiser" / RUN_FILENAME).read_text())
    assert facts["vcf_sha256"] == original_sha and facts["vcf_container_sha256"] == p["vcf_container_sha256"]
    assert facts["vcf_contig_renames"] == {"M": "MT"}
    # the mount was the copy, read-only, under the fixed container name; the index is not needed
    run_cmd = next(c for c in fake.calls if c[:2] == ["docker", "run"])
    assert f"{mounted}:/vcf/case.vcf.gz:ro" in run_cmd and not any(".tbi" in a for a in run_cmd)
    rename_cmd = next(c for c in fake.calls if c[0] == "bcftools")
    assert rename_cmd[1:4] == ["annotate", "--no-version", "--rename-chrs"] and rename_cmd[-1] == str(tmp_path / "public.vcf.gz")
    # the same input gives the same copy (bcftools --no-version: no date line), so the recorded sha256 is reproducible
    again = rename_contigs(tmp_path / "public.vcf.gz", {"M": "MT"}, tmp_path / "again" / "case.vcf.gz")
    assert hashlib.sha256(again.read_bytes()).hexdigest() == p["vcf_container_sha256"]
    with gzip.open(again, "rt") as f:
        body = [line for line in f if not line.startswith("#")]
    assert body[-1].startswith("MT\t3243\t") and not any(line.startswith("M\t") for line in body)
    # --join-only: the VCF check is against the original (which is what case.yaml names)
    m2 = json.loads(run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest)).read_text())
    assert m2["params"]["vcf_container_sha256"] == p["vcf_container_sha256"] and m2["params"]["vcf_contig_renames"] == {"M": "MT"}
    assert any("ranked a copy of the VCF with contigs" in n for n in m2["notes"])
    # 'MT' as is: no copy, no bcftools, the original is mounted
    plain = make_case_with_mito(tmp_path / "mt", "MT")
    fake = FakeDocker(cfg.image_digest)
    run_rank(run, plain, data, run=fake)
    assert fake.vcf_mounted[0][0] == tmp_path / "mt" / "public.vcf.gz" and not any(c[0] == "bcftools" for c in fake.calls)


def test_a_failed_rename_stops_the_run_before_docker(tmp_path: Path, cfg: ExomiserConfig):
    case = make_case(tmp_path, mito="M")
    run = make_run(tmp_path, with_candidates=False)
    data = make_data_dir(tmp_path, cfg)
    fake = FakeDocker(cfg.image_digest, bcftools_ok=False)
    with pytest.raises(RuntimeError, match="rename-chrs failed"):
        run_rank(run, case, data, run=fake)
    assert not any(c[:2] == ["docker", "run"] for c in fake.calls)
    assert not (run / "04_rank" / "manifest.json").exists()


# ---------------------------------------------------------------- --regions: the subset the container sees

CFTR_BED = "7\t117480000\t117670000\n"  # CFTR on GRCh38, 0-based half-open; seven of the public VCF's twelve records
"""A BED with one interval, as `engine funnel` would write for a one-gene panel."""


def make_public_case(tmp: Path, hpo: list[str] = HPO) -> Path:
    """The public twelve-variant fixture (bgzipped and indexed, so bcftools can subset it) with a case file."""
    tmp.mkdir(parents=True, exist_ok=True)
    for name in ("public.vcf.gz", "public.vcf.gz.tbi"):
        shutil.copy(PUBLIC / name, tmp / name)
    case = tmp / "case.yaml"
    case.write_text("proband_id: PUBLIC01\nvcf: public.vcf.gz\nhpo: [" + ", ".join(hpo) + "]\n")
    return case


def make_bgzipped_case_with_mito(tmp: Path, mito: str) -> Path:
    """The public three variants plus the public MT m.3243A>G, with the mitochondrion
    spelled ``mito`` — bgzipped by bcftools and tabix-indexed, so ``-R`` can read it."""
    make_case_with_mito(tmp, mito)
    plain = tmp / "public.vcf.gz"
    text = plain.with_suffix(".txt.vcf")
    with gzip.open(plain, "rt") as f:
        text.write_text(f.read())
    subprocess.run(["bcftools", "view", "--no-version", "-Oz", "-o", str(plain), str(text)], check=True)
    subprocess.run(["bcftools", "index", "-t", "-f", str(plain)], check=True)
    text.unlink()
    return tmp / "case.yaml"


def _bed(tmp: Path, text: str, name: str = "funnel.bed") -> Path:
    bed = tmp / name
    bed.write_text(text)
    return bed


def _records(vcf: Path) -> list[list[str]]:
    with gzip.open(vcf, "rt") as f:
        return [line.rstrip("\n").split("\t") for line in f if not line.startswith("#")]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@needs_bcftools
def test_regions_subsets_the_vcf_and_the_manifest_says_so(tmp_path: Path, cfg: ExomiserConfig):
    """``--regions``: the container gets 04_rank/input.regions.vcf.gz (+ .tbi), the records
    of the case VCF inside the BED, and the manifest carries the BED's sha256, the counts
    before and after, the bcftools lines and the statement that the ranking is gene-blind
    but not genome-wide. The subset is reproducible and survives the run; --join-only
    describes the run's regions, not the flag's; a plain rerun removes the subset."""
    case = make_public_case(tmp_path)
    bed = _bed(tmp_path, "# CFTR\n" + CFTR_BED)
    run = make_run(tmp_path)
    data = make_data_dir(tmp_path, cfg)
    out = run / "04_rank"
    subset = out / REGIONS_VCF_FILENAME
    fake = FakeDocker(cfg.image_digest)
    messages: list[str] = []

    manifest = run_rank(run, case, data, regions=bed, progress=messages.append, run=fake)
    # the subset: written, indexed, seven records all on chromosome 7, the file the container was given
    assert subset.exists() and subset.with_name(subset.name + ".tbi").exists()
    rows = _records(subset)
    assert len(rows) == 7 and {r[0] for r in rows} == {"7"} and all(117480000 < int(r[1]) <= 117670000 for r in rows)
    (mounted, contigs), = fake.vcf_mounted
    assert mounted == subset and contigs == ["1", "7", "15", "17"]
    run_cmd = next(c for c in fake.calls if c[:2] == ["docker", "run"])
    assert f"{subset}:/vcf/case.vcf.gz:ro" in run_cmd and f"{subset}.tbi:/vcf/case.vcf.gz.tbi:ro" in run_cmd
    assert not (out / "input").exists()  # the genome was never copied
    view_cmd = next(c for c in fake.calls if c[:2] == ["bcftools", "view"])
    assert view_cmd[2:8] == ["--no-version", "-R", str(bed), "--regions-overlap", "1", "-Oz"] and view_cmd[-1] == str(tmp_path / "public.vcf.gz")
    assert ["bcftools", "index", "-t", "-f", str(subset)] in fake.calls and ["bcftools", "index", "-n", str(subset)] in fake.calls
    assert not any(c[:2] == ["bcftools", "annotate"] for c in fake.calls)  # MT is already MT
    # the manifest
    m = json.loads(manifest.read_text())
    p, c = m["params"], m["counts"]
    bed_sha, vcf_sha, subset_sha = _sha(bed), _sha(tmp_path / "public.vcf.gz"), _sha(subset)
    assert m["inputs"]["regions"] == {"path": str(bed), "bytes": bed.stat().st_size, "sha256": bed_sha}
    assert m["inputs"]["vcf"]["sha256"] == vcf_sha and m["tools"]["bcftools"].startswith("bcftools")
    r = p["regions"]
    assert r["bed"] == str(bed) and r["bed_sha256"] == bed_sha and r["bed_intervals"] == 1
    assert r["records_before"] == 12 and r["records_after"] == 7 and r["records_before_source"].startswith("bcftools index -n")
    assert r["vcf"] == str(subset) and r["vcf_sha256"] == subset_sha == p["vcf_container_sha256"] != vcf_sha
    assert r["vcf_bytes"] == subset.stat().st_size == p["vcf_container_bytes"] and r["vcf_index"] == str(subset) + ".tbi"
    assert r["contig_renames"] == {} == p["vcf_contig_renames"] and r["regions_overlap"] == 1
    assert [cmd[:2] for cmd in r["commands"]] == [["bcftools", "view"], ["bcftools", "index"], ["bcftools", "index"]]
    assert c["vcf_records"] == 12 and c["regions_records"] == 7 and c["genes_ranked"] == 2  # the join is untouched
    assert m["outputs"]["regions_vcf"]["sha256"] == subset_sha and m["outputs"]["regions_vcf_index"]["path"] == str(subset) + ".tbi"
    note = next(n for n in m["notes"] if n.startswith("Restricted to regions"))
    assert "7 of 12 records" in note and bed_sha in note and "gene-blind" in note and "not genome-wide" in note
    assert p["vcf_container_path"] == "/vcf/case.vcf.gz" and p["join_only"] is False
    # run.json keeps the same facts for --join-only; the original VCF is what case.yaml names
    facts = json.loads((out / "exomiser" / RUN_FILENAME).read_text())
    assert facts["regions"] == r and facts["vcf_sha256"] == vcf_sha and facts["vcf_container_sha256"] == subset_sha
    # progress: counts, never a path or a variant
    assert any(msg == "regions: 7 of 12 records kept (1 intervals)" for msg in messages)
    assert not any(str(tmp_path) in msg or "public.vcf" in msg or "117559590" in msg for msg in messages)
    # the same inputs give the same subset bytes (bcftools --no-version)
    again = subset_regions(tmp_path / "public.vcf.gz", bed, tmp_path / "again" / REGIONS_VCF_FILENAME)
    assert _sha(again.path) == subset_sha and again.records == 7 and again.renames == {}

    # --join-only without the flag: params.regions, the counts, the note and inputs.regions describe the run as it was
    before = _snapshot(out)
    m2 = json.loads(run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest)).read_text())
    assert _snapshot(out) == before and subset.exists()
    assert m2["params"]["regions"] == r and m2["counts"]["vcf_records"] == 12 and m2["counts"]["regions_records"] == 7
    assert m2["inputs"]["regions"]["sha256"] == bed_sha and m2["inputs"]["regions"]["as_of"].startswith("the run being joined")
    assert m2["outputs"]["regions_vcf"]["sha256"] == subset_sha and any(n.startswith("Restricted to regions") for n in m2["notes"])
    assert not any("--regions" in n for n in m2["notes"])
    # --join-only with another BED: noted, not applied
    other = _bed(tmp_path, "17\t7660000\t7690000\n", name="other.bed")
    m3 = json.loads(run_rank(run, case, data, join_only=True, regions=other, run=FakeDocker(cfg.image_digest)).read_text())
    assert m3["params"]["regions"] == r and m3["inputs"]["regions"]["sha256"] == bed_sha
    assert any(f"--regions names {other}" in n and "describe the run as it was" in n for n in m3["notes"])
    # --join-only refuses a subset that is no longer the one the container ranked
    shutil.copy(tmp_path / "public.vcf.gz", subset)
    with pytest.raises(ExomiserFailed, match="not the regions subset the raw run ranked"):
        run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest))
    subset.unlink()
    m4 = json.loads(run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest)).read_text())
    assert any("no longer on disk" in n for n in m4["notes"]) and "regions_vcf" not in m4["outputs"]

    # a plain rerun: the whole VCF is mounted, nothing of the subset remains, nothing regions-shaped in the manifest
    fake = FakeDocker(cfg.image_digest)
    m5 = json.loads(run_rank(run, case, data, run=fake).read_text())
    assert fake.vcf_mounted[0][0] == tmp_path / "public.vcf.gz" and not list(out.glob("input.regions*"))
    assert m5["params"]["regions"] is None and "regions" not in m5["inputs"] and "vcf_records" not in m5["counts"]
    assert not any("Restricted to regions" in n for n in m5["notes"]) and not any(c[0] == "bcftools" for c in fake.calls)
    assert json.loads((out / "exomiser" / RUN_FILENAME).read_text())["regions"] is None
    m6 = json.loads(run_rank(run, case, data, join_only=True, regions=bed, run=FakeDocker(cfg.image_digest)).read_text())
    assert m6["params"]["regions"] is None and any("ranked the whole VCF" in n for n in m6["notes"])


@needs_bcftools
def test_regions_subset_renames_bare_M_to_MT(tmp_path: Path, cfg: ExomiserConfig):
    """A VCF that spells the mitochondrion 'M': the subset — not the genome — is
    rewritten with M → MT, indexed, and the rename is recorded; the MT record is in
    the file the container saw."""
    case = make_bgzipped_case_with_mito(tmp_path, "M")
    bed = _bed(tmp_path, CFTR_BED + "M\t0\t16569\n")
    run = make_run(tmp_path, with_candidates=False)
    data = make_data_dir(tmp_path, cfg)
    out = run / "04_rank"
    subset = out / REGIONS_VCF_FILENAME
    fake = FakeDocker(cfg.image_digest)

    manifest = run_rank(run, case, data, regions=bed, run=fake)
    rows = _records(subset)
    assert [(r[0], r[1]) for r in rows] == [("7", "117559590"), ("MT", "3243")]
    assert not any(r[0] == "M" for r in rows)
    (mounted, contigs), = fake.vcf_mounted
    assert mounted == subset and contigs == ["1", "7", "17", "MT"] and subset.with_name(subset.name + ".tbi").exists()
    assert not (out / "input").exists()
    mapping = subset.with_name(subset.name + ".rename.tsv")
    assert mapping.read_text() == "M\tMT\n"
    annotate = next(c for c in fake.calls if c[:2] == ["bcftools", "annotate"])
    assert annotate[2:5] == ["--no-version", "--rename-chrs", str(mapping)] and annotate[-1] == str(subset)
    order = [(c[1], c[2]) for c in fake.calls if c[0] == "bcftools"]
    assert order == [("view", "--no-version"), ("annotate", "--no-version"), ("index", "-t"), ("index", "-n"), ("index", "-n")]
    assert ["bcftools", "index", "-n", str(tmp_path / "public.vcf.gz")] in fake.calls  # the count before, from the original's index
    m = json.loads(manifest.read_text())
    p = m["params"]
    r = p["regions"]
    assert r["contig_renames"] == {"M": "MT"} == p["vcf_contig_renames"] and p["vcf_mito_contig"] == "M"
    assert r["records_before"] == 4 and r["records_after"] == 2 and r["bed_intervals"] == 2
    assert [cmd[:2] for cmd in r["commands"]] == [["bcftools", "view"], ["bcftools", "annotate"], ["bcftools", "index"], ["bcftools", "index"]]
    assert p["vcf_container_sha256"] == _sha(subset) != m["inputs"]["vcf"]["sha256"]
    assert any("renamed in the header and records" in n and "on the subset only" in n for n in m["notes"])
    assert any(n.startswith("Restricted to regions") for n in m["notes"])
    facts = json.loads((out / "exomiser" / RUN_FILENAME).read_text())
    assert facts["regions"]["contig_renames"] == {"M": "MT"} and facts["vcf_contig_renames"] == {"M": "MT"}
    # --join-only says which file was ranked, once
    m2 = json.loads(run_rank(run, case, data, join_only=True, run=FakeDocker(cfg.image_digest)).read_text())
    assert m2["params"]["regions"] == r and m2["params"]["vcf_contig_renames"] == {"M": "MT"}
    assert not any("ranked a copy of the VCF" in n for n in m2["notes"])
    # 'MT' as is: subset, no rename, no map
    plain = make_bgzipped_case_with_mito(tmp_path / "mt", "MT")
    fake = FakeDocker(cfg.image_digest)
    run_rank(run, plain, data, regions=_bed(tmp_path / "mt", CFTR_BED + "MT\t0\t16569\n"), run=fake)
    assert not any(c[:2] == ["bcftools", "annotate"] for c in fake.calls) and not mapping.exists()
    assert [(r[0], r[1]) for r in _records(subset)] == [("7", "117559590"), ("MT", "3243")]


@needs_bcftools
def test_regions_that_select_nothing_are_refused_before_docker(tmp_path: Path, cfg: ExomiserConfig):
    """A BED that spells contigs 'chr7' against a VCF that says '7' selects nothing;
    Exomiser is not started on an empty file, the reason names the naming style, and
    nothing of the subset or a manifest is left behind."""
    case = make_public_case(tmp_path)
    run = make_run(tmp_path, with_candidates=False)
    data = make_data_dir(tmp_path, cfg)
    fake = FakeDocker(cfg.image_digest)
    with pytest.raises(ExomiserFailed, match=r"selects none of the VCF's 12 records \(1 intervals\).*ensembl"):
        run_rank(run, case, data, regions=_bed(tmp_path, "chr7\t117480000\t117670000\n"), run=fake)
    assert not any(c[:2] == ["docker", "run"] for c in fake.calls)
    assert not list((run / "04_rank").glob("input.regions*")) and not (run / "04_rank" / "manifest.json").exists()
    # bcftools itself failing stops the run the same way, and leaves nothing of the subset
    fake = FakeDocker(cfg.image_digest, bcftools_ok=False)
    with pytest.raises(RuntimeError, match="bcftools view -R failed"):
        run_rank(run, case, data, regions=_bed(tmp_path, CFTR_BED), run=fake)
    assert not any(c[:2] == ["docker", "run"] for c in fake.calls) and not list((run / "04_rank").glob("input.regions*"))
    calls: list[list[str]] = []

    def index_fails(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess:
        calls.append(cmd)
        if cmd[:3] == ["bcftools", "index", "-t"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="[E::hts_idx_push] Unsorted positions")
        return subprocess.run(cmd, **kw)

    with pytest.raises(RuntimeError, match="bcftools index -t failed on the regions subset"):
        subset_regions(tmp_path / "public.vcf.gz", _bed(tmp_path, CFTR_BED), tmp_path / "sub" / REGIONS_VCF_FILENAME, run=index_fails)
    assert [c[1] for c in calls] == ["view", "index"] and not list((tmp_path / "sub").glob("*"))


@needs_bcftools
def test_records_before_come_from_an_index_that_counts_else_one_pass(tmp_path: Path, cfg: ExomiserConfig):
    """The count before subsetting: stage 1's own CSI when it indexed this very file,
    else the index beside the VCF, else — a provider's index without counts — one pass
    with bcftools query, which prints nothing of the records."""
    case = make_public_case(tmp_path)
    bed = _bed(tmp_path, CFTR_BED)
    run = make_run(tmp_path, with_candidates=False)
    data = make_data_dir(tmp_path, cfg)
    vcf = tmp_path / "public.vcf.gz"

    def source(**kw: Any) -> tuple[int, str]:
        fake = FakeDocker(cfg.image_digest, **kw)
        r = json.loads(run_rank(run, case, data, regions=bed, run=fake).read_text())["params"]["regions"]
        return r["records_before"], r["records_before_source"], fake

    n, src, fake = source(index_counts=False)
    assert n == 12 and src.startswith("bcftools query") and ["bcftools", "query", "-f", "\\n", str(vcf)] in fake.calls
    n, src, fake = source()
    assert n == 12 and src == "bcftools index -n, the index beside the VCF" and not any(c[1] == "query" for c in fake.calls)
    # stage 1's CSI, only when its manifest says it indexed this file
    ingest = run / "01_ingest"
    ingest.mkdir()
    subprocess.run(["bcftools", "index", "-c", "-o", str(ingest / "input.csi"), str(vcf)], check=True)
    (ingest / "manifest.json").write_text(json.dumps({"inputs": {"vcf": {"sha256": "0" * 64}}}))
    n, src, _ = source()
    assert n == 12 and src == "bcftools index -n, the index beside the VCF"
    (ingest / "manifest.json").write_text(json.dumps({"inputs": {"vcf": {"sha256": _sha(vcf)}}}))
    n, src, fake = source(index_counts=False)
    assert n == 12 and src.startswith("bcftools index -n, stage 1's index") and ["bcftools", "index", "-n", f"{vcf}##idx##{ingest / 'input.csi'}"] in fake.calls
    # the helpers on their own
    assert vcf_record_count(vcf) == 12 and vcf_record_count(vcf, index=ingest / "input.csi") == 12
    assert vcf_record_count(tmp_path / "missing.vcf.gz") is None and count_records(vcf) == 12
    assert bed_interval_count(_bed(tmp_path, "# a comment\n\n" + CFTR_BED + "17\t1\t2\n", name="c.bed")) == 2


@needs_bcftools
def test_cli_rank_regions(tmp_path: Path, cfg: ExomiserConfig, monkeypatch):
    """``engine rank --regions <bed>``: the BED is named on the terminal, the counts are,
    a variant never is; a BED that does not exist is a usage error before anything runs."""
    case = make_public_case(tmp_path)
    bed = _bed(tmp_path, CFTR_BED)
    data = make_data_dir(tmp_path, cfg)
    run = make_run(tmp_path)
    fake = FakeDocker(cfg.image_digest)
    real_run_rank = rank_run.run_rank
    monkeypatch.setattr(rank_run, "run_rank", lambda *a, **kw: real_run_rank(*a, run=fake, **kw))
    args = ["rank", "--run", str(run), "--case", str(case), "--exomiser-data", str(data), "--heap", "2g"]

    result = CliRunner().invoke(main, args + ["--regions", str(bed)])
    assert result.exit_code == 0, result.output
    assert f"rank · {run} · exomiser-data={data} · regions={bed}" in result.output
    assert "regions: 7 of 12 records kept (1 intervals)" in result.output
    assert f"restricted to regions: 7 of 12 records (1 intervals of {bed}) — gene-blind, not genome-wide" in result.output
    assert "genes ranked: 2 · candidates: 2 (1 ranked by Exomiser)" in result.output
    for secret in (F508, R175H, "117559590", "7675088", "CFTR", "TP53", "MTHFR"):
        assert secret not in result.output, secret
    m = json.loads((run / "04_rank" / "manifest.json").read_text())
    assert m["params"]["regions"]["records_after"] == 7 and m["inputs"]["regions"]["sha256"] == _sha(bed)
    assert (run / "04_rank" / REGIONS_VCF_FILENAME).exists()
    # --join-only keeps the restriction on the terminal too
    result = CliRunner().invoke(main, args + ["--join-only"])
    assert result.exit_code == 0 and "restricted to regions: 7 of 12 records" in result.output and "JOIN ONLY" in result.output
    # a missing BED is refused by the option itself
    n_calls = len(fake.calls)
    result = CliRunner().invoke(main, args + ["--regions", str(tmp_path / "nope.bed")])
    assert result.exit_code == 2 and "Invalid value for '--regions'" in result.output and len(fake.calls) == n_calls


# ---------------------------------------------------------------- the downloader, against an in-memory server

class FakeServer:
    """A GCS-like object store: HEAD with x-goog-hash, GET with Range → 206, and an
    optional connection drop after N bytes of the first GET (``drop_every``: of every GET)."""

    def __init__(self, blob: bytes, *, drop_after: int | None = None, drop_every: bool = False, honour_range: bool = True):
        self.blob = blob
        self.drop_after = drop_after
        self.drop_every = drop_every
        self.honour_range = honour_range
        self.requests: list[tuple[str, dict[str, str]]] = []

    def __call__(self, req, timeout=None):
        headers = {k.lower(): v for k, v in req.header_items()}
        self.requests.append((req.get_method(), headers))
        md5_b64 = __import__("base64").b64encode(hashlib.md5(self.blob).digest()).decode()
        if req.get_method() == "HEAD":
            return _Resp(200, b"", {"Content-Length": str(len(self.blob)), "ETag": f'"{hashlib.md5(self.blob).hexdigest()}"',
                                    "Last-Modified": "Tue, 29 Oct 2024 01:21:09 GMT", "Accept-Ranges": "bytes"},
                         multi={"x-goog-hash": ["crc32c=AAAA", f"md5={md5_b64}"]})
        start = 0
        rng = headers.get("range")
        if rng and self.honour_range:
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= len(self.blob):
                raise urllib.error.HTTPError(req.full_url, 416, "Range Not Satisfiable", {}, None)
            body = self.blob[start:]
            hdrs = {"Content-Length": str(len(body)), "Content-Range": f"bytes {start}-{len(self.blob) - 1}/{len(self.blob)}"}
            status = 206
        else:
            body, hdrs, status = self.blob, {"Content-Length": str(len(self.blob))}, 200
        drop = self.drop_after if (self.drop_every or (self.drop_after is not None and start == 0)) else None
        if drop is not None and not self.drop_every:
            self.drop_after = None  # only the first attempt breaks
        return _Resp(status, body, hdrs, drop_after=drop)


class _Headers(dict):
    def __init__(self, single: dict[str, str], multi: dict[str, list[str]] | None = None):
        super().__init__(single)
        self.multi = multi or {}

    def get(self, k, default=None):  # case-insensitive, like http.client.HTTPMessage
        for kk, v in self.items():
            if kk.lower() == k.lower():
                return v
        return default

    def get_all(self, k):
        return self.multi.get(k) or ([self.get(k)] if self.get(k) else None)


class _Resp:
    def __init__(self, status: int, body: bytes, headers: dict[str, str], *, multi=None, drop_after: int | None = None):
        self.status = status
        self.headers = _Headers(headers, multi)
        self._buf = io.BytesIO(body)
        self._drop_after = drop_after
        self._sent = 0

    def read(self, n: int = -1) -> bytes:
        exhausted = self._buf.tell() >= len(self._buf.getbuffer())
        if self._drop_after is not None and self._sent >= self._drop_after and not exhausted:
            raise ConnectionResetError("connection dropped by the fake server")
        chunk = self._buf.read(n if self._drop_after is None else min(n, self._drop_after - self._sent) or n)
        self._sent += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_bundle(blob: bytes, **over: Any) -> Bundle:
    return Bundle(**{"name": "test", "filename": "test_bundle.zip", "bytes": len(blob), "md5": hashlib.md5(blob).hexdigest(),
                     "sha256": None, "crc32c": None, "extracted_dir": "test_bundle", "required_files": (),
                     "url": "https://data.example.org/exomiser/latest/test_bundle.zip", **over})


def test_head_reads_size_and_goog_md5():
    blob = os.urandom(1000)
    srv = FakeServer(blob)
    info = dl.head("https://data.example.org/x.zip", srv)
    assert info.bytes == 1000 and info.md5 == hashlib.md5(blob).hexdigest() and info.accept_ranges
    dl.check_remote(make_bundle(blob), info)
    with pytest.raises(dl.PinMismatch, match="bytes"):
        dl.check_remote(make_bundle(blob, bytes=999), info)
    with pytest.raises(dl.PinMismatch, match="md5"):
        dl.check_remote(make_bundle(blob, md5="0" * 32), info)


def test_download_resumes_with_range_and_verifies(tmp_path: Path):
    blob = os.urandom(3 * dl.CHUNK + 12345)
    srv = FakeServer(blob, drop_after=dl.CHUNK + 7)  # first GET dies mid-way
    b = make_bundle(blob)
    sleeps: list[float] = []
    dest, hashes, info = dl.download(b, tmp_path, opener=srv, sleep=sleeps.append)
    assert dest == tmp_path / b.filename and dest.read_bytes() == blob
    assert hashes.md5 == b.md5 and hashes.sha256 == hashlib.sha256(blob).hexdigest() and hashes.bytes == len(blob)
    gets = [h for m, h in srv.requests if m == "GET"]
    assert len(gets) == 2 and "range" not in gets[0] and gets[1]["range"].startswith("bytes=")
    assert int(gets[1]["range"][6:-1]) > dl.CHUNK  # resumed from what was already on disk, not from zero
    assert sleeps and not (tmp_path / (b.filename + ".part")).exists()
    side = dl.read_verified(dest)
    assert side["md5"] == b.md5 and side["sha256"] == hashes.sha256 and side["server"]["etag"] == b.md5
    assert dl.is_verified(b, tmp_path)
    # a second call is a no-op transfer: the verified copy is trusted
    assert dl.fetch_all(_cfg_with(b), tmp_path, do_extract=False, opener=srv)["test"]["status"] == "verified"


def test_resume_budget_counts_stalls_not_drops(tmp_path: Path):
    """A link that drops after every chunk must not exhaust the budget as long as the
    file keeps growing; a server that never delivers a byte must."""
    blob = os.urandom(15 * dl.CHUNK)
    srv = FakeServer(blob, drop_after=dl.CHUNK, drop_every=True)  # exactly one chunk per GET
    b = make_bundle(blob)
    sleeps: list[float] = []
    dest, hashes, _ = dl.download(b, tmp_path, opener=srv, retries=3, sleep=sleeps.append)
    assert dest.read_bytes() == blob and hashes.md5 == b.md5
    assert sum(1 for m, _ in srv.requests if m == "GET") == 15 and len(sleeps) == 14
    stall = FakeServer(blob, drop_after=0, drop_every=True)
    with pytest.raises(RuntimeError, match="without progress"):
        dl.download(b, tmp_path / "stall", opener=stall, retries=3, sleep=lambda s: None)
    assert sum(1 for m, _ in stall.requests if m == "GET") == 4  # the first try plus three retries


def test_download_restarts_when_server_ignores_range(tmp_path: Path):
    blob = os.urandom(2 * dl.CHUNK)
    srv = FakeServer(blob, honour_range=False)
    part = tmp_path / "test_bundle.zip.part"
    part.write_bytes(blob[: dl.CHUNK // 2])
    dest, hashes, _ = dl.download(make_bundle(blob), tmp_path, opener=srv)
    assert dest.read_bytes() == blob and hashes.md5 == hashlib.md5(blob).hexdigest()


def test_download_refuses_a_republished_bundle_before_transfer(tmp_path: Path):
    blob = os.urandom(4096)
    srv = FakeServer(blob)
    with pytest.raises(dl.PinMismatch):
        dl.download(make_bundle(blob, md5="f" * 32), tmp_path, opener=srv)
    assert [m for m, _ in srv.requests] == ["HEAD"]  # nothing fetched


def test_verify_rejects_wrong_bytes(tmp_path: Path):
    blob = b"exomiser" * 100
    good = dl.hash_file(_write(tmp_path / "a.zip", blob))
    dl.verify(make_bundle(blob), good)
    with pytest.raises(dl.PinMismatch, match="md5"):
        dl.verify(make_bundle(blob, md5="0" * 32), good)
    with pytest.raises(dl.PinMismatch, match="sha256"):
        dl.verify(make_bundle(blob, sha256="0" * 64), good)
    with pytest.raises(dl.PinMismatch, match="bytes"):
        dl.verify(make_bundle(blob, bytes=1), good)


def _make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_extract_places_required_files_and_verifies_members(tmp_path: Path):
    sha_a, sha_b = hashlib.sha256(b"aaaa").hexdigest(), hashlib.sha256(b"bbbb").hexdigest()
    shipped = f"{sha_a}  test_bundle/a.ser\n{sha_b}  test_bundle/sub/b.db\n"
    z = _make_zip(tmp_path / "test_bundle.zip", {"test_bundle/a.ser": b"aaaa", "test_bundle/sub/b.db": b"bbbb",
                                                 "test_bundle/test_bundle.sha256": shipped.encode()})
    b = make_bundle(z.read_bytes(), required_files=("a.ser", "sub/b.db"), checksum_list="test_bundle.sha256",
                    member_sha256={"a.ser": sha_a})
    out = dl.extract(b, z, tmp_path / "data")
    assert out == tmp_path / "data" / "test_bundle"
    assert (out / "a.ser").read_bytes() == b"aaaa" and (out / "sub" / "b.db").read_bytes() == b"bbbb"
    assert b.missing_files(tmp_path / "data") == []
    assert not (tmp_path / "data" / ".extract-test_bundle").exists()  # the temp dir never masquerades as final
    # every member hashed in the same pass; compared with the pin and with the shipped list
    members = dl.read_verified(tmp_path / "data" / b.filename)["members"]
    assert members["a.ser"] == {"bytes": 4, "sha256": sha_a, "matches_pin": True, "matches_checksum_list": True}
    assert members["sub/b.db"] == {"bytes": 4, "sha256": sha_b, "matches_pin": None, "matches_checksum_list": True}
    assert members["test_bundle.sha256"]["matches_checksum_list"] is None
    status = dl.members_status(b, tmp_path / "data")
    assert status["hashed"] == 3 and status["pinned"] == 1 and status["pinned_verified"] == 1 and status["all_pinned_verified"]
    assert status["pinned_mismatched"] == []
    # a pin added after the extraction is checked against the recorded hashes, not the flag written then
    later = make_bundle(z.read_bytes(), member_sha256={"a.ser": sha_a, "sub/b.db": sha_b})
    assert dl.members_status(later, tmp_path / "data")["all_pinned_verified"] is True
    wrong = make_bundle(z.read_bytes(), member_sha256={"a.ser": sha_a, "sub/b.db": "0" * 64})
    assert dl.members_status(wrong, tmp_path / "data")["pinned_mismatched"] == ["sub/b.db"]
    # a required file the zip lacks
    with pytest.raises(ValueError, match="missing"):
        dl.extract(make_bundle(z.read_bytes(), required_files=("c.ser",)), z, tmp_path / "data2")
    assert not (tmp_path / "data2" / ".extract-test_bundle").exists()


def test_extract_refuses_a_wrong_member_hash_and_cleans_up(tmp_path: Path):
    z = _make_zip(tmp_path / "test_bundle.zip", {"test_bundle/a.ser": b"aaaa"})
    bad_pin = make_bundle(z.read_bytes(), member_sha256={"a.ser": "0" * 64})
    with pytest.raises(dl.PinMismatch, match="a.ser sha256"):
        dl.extract(bad_pin, z, tmp_path / "data")
    assert not (tmp_path / "data" / "test_bundle").exists() and not (tmp_path / "data" / ".extract-test_bundle").exists()
    wrong_list = _make_zip(tmp_path / "wrong.zip", {"test_bundle/a.ser": b"aaaa", "test_bundle/list.sha256": f"{'1' * 64}  test_bundle/a.ser\n".encode()})
    with pytest.raises(dl.PinMismatch, match="list.sha256"):
        dl.extract(make_bundle(wrong_list.read_bytes(), checksum_list="list.sha256"), wrong_list, tmp_path / "data")
    absent = make_bundle(z.read_bytes(), member_sha256={"never.ser": "0" * 64})
    with pytest.raises(dl.PinMismatch, match="not in the bundle"):
        dl.extract(absent, z, tmp_path / "data")


def test_extract_checks_free_space_before_writing(tmp_path: Path):
    z = _make_zip(tmp_path / "test_bundle.zip", {"test_bundle/a.ser": b"a" * 1000})
    b = make_bundle(z.read_bytes(), required_files=("a.ser",))

    def usage(free: int):
        return lambda path: _Usage(free)

    with pytest.raises(RuntimeError, match="not enough free space"):
        dl.extract(b, z, tmp_path / "data", disk_usage=usage(dl.EXTRACT_MARGIN + 999))  # one byte short
    assert not (tmp_path / "data" / ".extract-test_bundle").exists() and not (tmp_path / "data" / "test_bundle").exists()
    assert dl.extract(b, z, tmp_path / "data", disk_usage=usage(dl.EXTRACT_MARGIN + 1000)).exists()


class _Usage:
    """The one field of ``shutil.disk_usage`` the extractor reads."""

    def __init__(self, free: int):
        self.free = free


def test_verify_extracted_hashes_an_existing_extraction(tmp_path: Path):
    sha_a = hashlib.sha256(b"aaaa").hexdigest()
    z = _make_zip(tmp_path / "test_bundle.zip", {"test_bundle/a.ser": b"aaaa", "test_bundle/sub/b.db": b"bbbb"})
    b = make_bundle(z.read_bytes(), required_files=("a.ser",))
    dl.extract(b, z, tmp_path / "data")
    sidecar = tmp_path / "data" / (b.filename + dl.VERIFIED_SUFFIX)
    sidecar.unlink()  # as if extracted before member hashing existed
    pinned = make_bundle(z.read_bytes(), required_files=("a.ser",), member_sha256={"a.ser": sha_a})
    assert dl.members_status(pinned, tmp_path / "data") == {"hashed": 0, "pinned": 1, "pinned_verified": 0, "pinned_mismatched": [],
                                                            "all_pinned_verified": False, "verified_at": None}
    report = dl.verify_extracted(pinned, tmp_path / "data")
    assert report["a.ser"]["matches_pin"] is True and report["sub/b.db"]["matches_pin"] is None
    assert dl.members_status(pinned, tmp_path / "data")["all_pinned_verified"] is True
    with pytest.raises(dl.PinMismatch):
        dl.verify_extracted(make_bundle(z.read_bytes(), required_files=("a.ser",), member_sha256={"a.ser": "0" * 64}), tmp_path / "data")


def test_fetch_all_downloads_extracts_and_can_delete_the_zip(tmp_path: Path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("test_bundle/a.ser", b"aaaa")
    blob = buf.getvalue()
    b = make_bundle(blob, required_files=("a.ser",), member_sha256={"a.ser": hashlib.sha256(b"aaaa").hexdigest()})
    srv = FakeServer(blob)
    report = dl.fetch_all(_cfg_with(b), tmp_path, delete_zip=True, opener=srv)
    assert report["test"]["status"] == "downloaded" and report["test"]["zip_deleted"] is True
    assert report["test"]["members"]["a.ser"]["matches_pin"] is True
    assert (tmp_path / "test_bundle" / "a.ser").exists() and not (tmp_path / b.filename).exists()
    side = dl.read_verified(tmp_path / b.filename)
    assert side["sha256"] == hashlib.sha256(blob).hexdigest() and side["members"]["a.ser"]["sha256"] == hashlib.sha256(b"aaaa").hexdigest()
    again = dl.fetch_all(_cfg_with(b), tmp_path, opener=srv)
    assert again["test"]["status"] == "extracted"  # nothing re-fetched
    assert sum(1 for m, _ in srv.requests if m == "GET") == 1


def test_fetch_all_verifies_an_extraction_that_has_neither_zip_nor_sidecar(tmp_path: Path):
    """Unzipped by hand (or the sidecar lost): hash the members against the pin rather
    than fetch 22 GB again; a member that does not match the pin is refused."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("test_bundle/a.ser", b"aaaa")
    blob = buf.getvalue()
    b = make_bundle(blob, required_files=("a.ser",), member_sha256={"a.ser": hashlib.sha256(b"aaaa").hexdigest()})
    (tmp_path / "test_bundle").mkdir()
    (tmp_path / "test_bundle" / "a.ser").write_bytes(b"aaaa")
    srv = FakeServer(blob)
    report = dl.fetch_all(_cfg_with(b), tmp_path, opener=srv)
    assert report["test"]["status"] == "extracted-unverified-zip" and report["test"]["members"]["a.ser"]["matches_pin"] is True
    assert srv.requests == [] and not (tmp_path / b.filename).exists()  # nothing fetched
    side = dl.read_verified(tmp_path / b.filename)
    assert side["members"]["a.ser"]["sha256"] == hashlib.sha256(b"aaaa").hexdigest() and "sha256" not in side  # the zip was never seen
    assert dl.fetch_all(_cfg_with(b), tmp_path, opener=srv)["test"]["status"] == "extracted"  # the sidecar now vouches for it
    bad = make_bundle(blob, required_files=("a.ser",), member_sha256={"a.ser": "0" * 64})
    (tmp_path / (b.filename + dl.VERIFIED_SUFFIX)).unlink()
    with pytest.raises(dl.PinMismatch, match="a.ser sha256"):
        dl.fetch_all(_cfg_with(bad), tmp_path, opener=srv)
    assert srv.requests == []


def test_fetch_all_hashes_a_complete_zip_without_sidecar(tmp_path: Path):
    """A zip fetched by other means (curl) is hashed in place, not transferred again;
    one with the wrong bytes is replaced."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("test_bundle/a.ser", b"aaaa")
    blob = buf.getvalue()
    b = make_bundle(blob, required_files=("a.ser",))
    srv = FakeServer(blob)
    (tmp_path / b.filename).write_bytes(blob)
    report = dl.fetch_all(_cfg_with(b), tmp_path, opener=srv)
    assert report["test"]["status"] == "hashed" and report["test"]["sha256"] == hashlib.sha256(blob).hexdigest()
    assert not any(m == "GET" for m, _ in srv.requests) and dl.is_verified(b, tmp_path)
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    (wrong / b.filename).write_bytes(blob + b"trailing garbage")
    report = dl.fetch_all(_cfg_with(b), wrong, do_extract=False, opener=srv)
    assert report["test"]["status"] == "downloaded" and (wrong / b.filename).read_bytes() == blob
    assert sum(1 for m, _ in srv.requests if m == "GET") == 1


def _write(p: Path, data: bytes) -> Path:
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------- the two commands, Docker and the server faked

def test_cli_rank_runs_the_stage_and_prints_counts_only(tmp_path: Path, cfg: ExomiserConfig, monkeypatch):
    """``engine rank`` end to end over the fake daemon: the pinned image by digest, the
    heap as asked, counts and the manifest path on the terminal — never a variant or a
    gene — then ``--join-only`` without Docker, and a failed Exomiser as exit 1."""
    case = make_case(tmp_path)
    data = make_data_dir(tmp_path, cfg)
    run = make_run(tmp_path)
    fake = FakeDocker(cfg.image_digest)
    real_run_rank = rank_run.run_rank
    monkeypatch.setattr(rank_run, "run_rank", lambda *a, **kw: real_run_rank(*a, run=fake, **kw))
    args = ["rank", "--run", str(run), "--case", str(case), "--exomiser-data", str(data), "--heap", "2g"]

    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert f"rank · {run} · exomiser-data={data}" in result.output
    assert "exomiser 14.0.0 · data 2406 · heap 2g · 3 HPO terms" in result.output
    assert "exomiser finished in" in result.output
    assert "genes ranked: 2 · candidates: 2 (1 ranked by Exomiser) · exomiser-only shown: 1 · wall" in result.output
    assert f"manifest: {run / '04_rank' / 'manifest.json'}" in result.output
    for secret in (F508, R175H, "117559590", "7675088", "CFTR", "TP53", "MTHFR"):
        assert secret not in result.output, secret
    m = json.loads((run / "04_rank" / "manifest.json").read_text())
    assert m["params"]["heap"] == "2g" and m["params"]["image_digest_used"] == cfg.image_digest and m["params"]["join_only"] is False
    assert any(c[:2] == ["docker", "run"] for c in fake.calls)

    # --join-only: the same outputs from the raw files, no docker run
    fake2 = FakeDocker(cfg.image_digest)
    monkeypatch.setattr(rank_run, "run_rank", lambda *a, **kw: real_run_rank(*a, run=fake2, **kw))
    result = CliRunner().invoke(main, args + ["--join-only"])
    assert result.exit_code == 0 and "JOIN ONLY" in result.output and "genes ranked: 2" in result.output
    assert not any(c[:2] == ["docker", "run"] for c in fake2.calls)
    assert json.loads((run / "04_rank" / "manifest.json").read_text())["params"]["join_only"] is True

    # a crash inside the container is reported as FAILED and exit 1, with the log path, not a traceback
    crash = FakeDocker(cfg.image_digest, outputs=("genes",), exit_code=137)
    monkeypatch.setattr(rank_run, "run_rank", lambda *a, **kw: real_run_rank(*a, run=crash, **kw))
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 1 and "FAILED: Exomiser exited with code 137" in result.output and "out of memory" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)

    # an unverified data directory is refused before Docker is touched
    result = CliRunner().invoke(main, ["rank", "--run", str(run), "--case", str(case), "--exomiser-data", str(tmp_path)])
    assert result.exit_code == 1 and "FAILED:" in result.output and "exomiser-download" in result.output


def test_cli_exomiser_download_fetches_verifies_extracts_and_reports(tmp_path: Path, monkeypatch):
    """``engine exomiser-download`` against a pinned config whose one bundle the fake
    server holds: fetched, verified, extracted, the sidecar written and the summary line
    printed; a second run transfers nothing; a pin that does not match is exit 1."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("2406_hg38/a.ser", b"aaaa")
    blob = buf.getvalue()
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["data"]["bundles"] = {"hg38": {
        "filename": "2406_hg38.zip", "bytes": len(blob), "md5": hashlib.md5(blob).hexdigest(),
        "sha256": hashlib.sha256(blob).hexdigest(), "extracted_dir": "2406_hg38", "required_files": ["a.ser"],
        "member_sha256": {"a.ser": hashlib.sha256(b"aaaa").hexdigest()},
    }}
    config = tmp_path / "exomiser.yaml"
    config.write_text(yaml.safe_dump(raw))
    srv = FakeServer(blob)
    real_fetch_all = dl.fetch_all
    monkeypatch.setattr(dl, "fetch_all", lambda *a, **kw: real_fetch_all(*a, opener=srv, **kw))
    ref = tmp_path / "ref"

    result = CliRunner().invoke(main, ["exomiser-download", "--config", str(config), "--ref-dir", str(ref), "--bundle", "hg38"])
    assert result.exit_code == 0, result.output
    assert f"exomiser-download · data release 2406 · 0.0 GiB of zips → {ref}" in result.output
    assert f"hg38: downloaded · sha256 {hashlib.sha256(blob).hexdigest()} · members hashed 1 (verified against a pin or list: 1)" in result.output
    assert (ref / "2406_hg38" / "a.ser").read_bytes() == b"aaaa" and (ref / "2406_hg38.zip").exists()
    assert dl.read_verified(ref / "2406_hg38.zip")["sha256"] == hashlib.sha256(blob).hexdigest()
    assert sum(1 for m, _ in srv.requests if m == "GET") == 1

    result = CliRunner().invoke(main, ["exomiser-download", "--config", str(config), "--ref-dir", str(ref)])
    assert result.exit_code == 0 and "already present and verified" in result.output and "hg38: verified · sha256" in result.output
    assert sum(1 for m, _ in srv.requests if m == "GET") == 1  # nothing re-fetched

    raw["data"]["bundles"]["hg38"]["md5"] = "0" * 32
    config.write_text(yaml.safe_dump(raw))
    result = CliRunner().invoke(main, ["exomiser-download", "--config", str(config), "--ref-dir", str(tmp_path / "ref2")])
    assert result.exit_code == 1 and "FAILED:" in result.output and "md5" in result.output


def _cfg_with(b: Bundle) -> ExomiserConfig:
    base = ExomiserConfig.load(DEFAULT_CONFIG)
    return ExomiserConfig(**{**base.__dict__, "bundles": {b.name: b}})


# ---------------------------------------------------------------- live: the image itself (no data bundle needed)

@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to run against Docker and Monarch")
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_live_pinned_image_and_bundle_pin(cfg: ExomiserConfig):
    observed = image_digest(cfg.image_ref)
    assert carries_digest(observed, cfg.image_digest), observed
    out = subprocess.run(["docker", "run", "--rm", cfg.image_ref, "--help"], capture_output=True, text=True, check=True)
    assert "--analysis <file>" in out.stdout and "--output-format" in out.stdout
    assert "analyse" not in out.stdout.split("usage:")[1][:200]  # 14.0.0 has no subcommand (15.x syntax)
    for b in cfg.bundles.values():
        info = dl.head(b.url)
        dl.check_remote(b, info)
        assert info.accept_ranges
