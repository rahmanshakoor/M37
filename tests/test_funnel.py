import gzip
import hashlib
import json
import os
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from engine.contigs import PRIMARY
from engine.retrieve.funnel import (
    DEFAULT_CONFIG,
    GRCH38_LENGTHS,
    GRCH38_PRIMARY_LENGTH,
    BsdSum,
    ScanCounts,
    bsd_sum,
    build_funnel_bed,
    download_gtf,
    iter_exons,
    load_config,
    manifest_path_for,
    merge_padded,
    observe_checksums,
    parse_attributes,
    parse_checksums,
    sidecar_path,
)
from engine.retrieve.http import Http, HttpCache, RateLimiter, Response
from engine.retrieve.intervals import IntervalIndex

FIXTURES = Path(__file__).parent / "fixtures" / "funnel"
# Recorded live from ftp.ensembl.org on 2026-09-12 (public annotation files, no variants).
CHECKSUMS = json.loads((FIXTURES / "CHECKSUMS.json").read_text())
README = json.loads((FIXTURES / "README.json").read_text())
# Real lines of the pinned GTF for the public textbook genes; see public_genes.json.
PUBLIC_GTF = FIXTURES / "public_genes.gtf.gz"
PUBLIC_PROVENANCE = json.loads((FIXTURES / "public_genes.json").read_text())


class StubHttp:
    """Stands in for :class:`engine.retrieve.http.Http`: replays recorded responses by URL."""

    def __init__(self, responses: dict[str, dict], *, offline: bool = False):
        self.responses = responses
        self.offline = offline
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, url: str, **kw) -> Response:
        self.calls.append((method, url))
        rec = self.responses[url]
        return Response(rec["status"], rec["text"], rec["retrieved_at"], False, "stub", rec.get("headers", {}))

    def get(self, url: str, **kw) -> Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, json_body=None, **kw) -> Response:
        return self.request("POST", url, **kw)


def _stub() -> StubHttp:
    return StubHttp({CHECKSUMS["url"]: CHECKSUMS, README["url"]: README})


def _gz(tmp: Path, name: str = "tiny.gtf.gz") -> Path:
    p = tmp / name
    with gzip.open(p, "wb") as f:
        f.write((FIXTURES / "tiny.gtf").read_bytes())
    return p


def _cfg_for(gtf_gz: Path):
    """The real config, re-pinned to the fixture's own checksum so the build accepts it."""
    return replace(load_config(DEFAULT_CONFIG), checksum=bsd_sum(gtf_gz))


# ---------------------------------------------------------------- config

def test_config_pins_release_116_chr_file():
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.release == 116
    assert cfg.url == "https://ftp.ensembl.org/pub/release-116/gtf/homo_sapiens/Homo_sapiens.GRCh38.116.chr.gtf.gz"
    assert cfg.checksums_url == "https://ftp.ensembl.org/pub/release-116/gtf/homo_sapiens/CHECKSUMS"
    assert cfg.checksum == "31909 137737"
    assert cfg.gene_biotypes == ("protein_coding", "Mt_tRNA", "Mt_rRNA")
    assert cfg.transcript_biotypes == ("protein_coding", "nonsense_mediated_decay", "protein_coding_LoF",
                                       "Mt_tRNA", "Mt_rRNA")
    assert cfg.gene_names_always == ("RNU4ATAC", "RMRP", "TERC", "RNU12", "RNU4-2", "RNU7-1", "SNORD118")
    assert cfg.pad_bp == 50
    assert cfg.genome_length == GRCH38_PRIMARY_LENGTH == 3_088_286_401
    # The pinned entry is exactly what the recorded live CHECKSUMS file says.
    assert parse_checksums(CHECKSUMS["text"])[cfg.filename] == cfg.checksum


def test_genome_length_is_the_sum_of_all_25_primary_contigs():
    """The denominator must be what its description says: 1-22, X, Y *and* MT."""
    assert set(GRCH38_LENGTHS) == set(PRIMARY) and len(GRCH38_LENGTHS) == 25
    assert sum(GRCH38_LENGTHS.values()) == GRCH38_PRIMARY_LENGTH == 3_088_286_401
    assert GRCH38_LENGTHS["MT"] == 16_569
    assert GRCH38_PRIMARY_LENGTH - GRCH38_LENGTHS["MT"] == 3_088_269_832  # the nuclear-only sum


def test_config_rejects_bad_checksum_format(tmp_path: Path):
    bad = tmp_path / "f.yaml"
    bad.write_text(DEFAULT_CONFIG.read_text().replace('checksum: "31909 137737"', 'checksum: "deadbeef"'))
    with pytest.raises(ValueError, match="BSD sum"):
        load_config(bad)


def test_config_allow_list_is_optional_but_never_blank(tmp_path: Path):
    text = DEFAULT_CONFIG.read_text()
    start, end = text.index("gene_names_always:"), text.index("# --- padding")
    without = tmp_path / "without.yaml"
    without.write_text(text[:start] + text[end:])
    assert load_config(without).gene_names_always == ()
    blank = tmp_path / "blank.yaml"
    blank.write_text(text[:start] + 'gene_names_always: ["RMRP", ""]\n' + text[end:])
    with pytest.raises(ValueError, match="empty name"):
        load_config(blank)


# ---------------------------------------------------------------- BSD sum

def test_bsd_sum_reproduces_ensembl_readme_entry(tmp_path: Path):
    """Ensembl publishes ``56461 11`` for README; the pure-Python sum must agree byte for byte."""
    readme = tmp_path / "README"
    readme.write_bytes(README["text"].encode("utf-8"))
    assert readme.stat().st_size == int(README["headers"]["content-length"])
    assert bsd_sum(readme) == parse_checksums(CHECKSUMS["text"])["README"] == "56461 11"
    # Chunk boundaries must not change the result (the download streams in pieces).
    data = readme.read_bytes()
    h = BsdSum()
    for i in range(0, len(data), 997):
        h.update(data[i:i + 997])
    assert str(h) == "56461 11"
    assert bsd_sum(readme, chunk=1) == "56461 11"


def test_bsd_sum_edge_cases(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert bsd_sum(empty) == "0 0"
    one = tmp_path / "one"
    one.write_bytes(b"\xff")
    assert bsd_sum(one) == "255 1"
    # 1025 bytes is two blocks; 1024 rotations of a 16-bit word are 64 full cycles.
    two = tmp_path / "two"
    two.write_bytes(b"\x01" + b"\x00" * 1024)
    assert bsd_sum(two) == "1 2"
    h = BsdSum()
    h.update(b"\x01")
    h.update(b"\x00")
    assert h.checksum == 1 << 15


def test_parse_checksums_recorded_file():
    entries = parse_checksums(CHECKSUMS["text"])
    assert entries == {
        "Homo_sapiens.GRCh38.116.abinitio.gtf.gz": "20333 3243",
        "Homo_sapiens.GRCh38.116.chr.gtf.gz": "31909 137737",
        "Homo_sapiens.GRCh38.116.chr_patch_hapl_scaff.gtf.gz": "35479 142363",
        "Homo_sapiens.GRCh38.116.gtf.gz": "49151 137815",
        "README": "56461 11",
    }
    assert parse_checksums("garbage\n\n1 2\n") == {}


# ---------------------------------------------------------------- attributes

KAZN_LINE = (
    'gene_id "ENSG00000189337"; gene_version "17"; transcript_id "ENST00000376030"; transcript_version "7"; '
    'exon_number "1"; gene_name "KAZN"; gene_source "ensembl_havana"; gene_biotype "protein_coding"; '
    'transcript_name "KAZN-203"; transcript_source "havana"; transcript_biotype "protein_coding"; tag "CCDS"; '
    'ccds_id "CCDS152"; exon_id "ENSE00003586727"; exon_version "2"; tag "gencode_basic"; tag "gencode_primary"; '
    'tag "MANE_Select"; tag "Ensembl_canonical"; transcript_support_level "5 (assigned to previous version 6)";'
)


def test_parse_attributes_keeps_every_repeated_tag():
    a = parse_attributes(KAZN_LINE)
    assert a["tag"] == ["CCDS", "gencode_basic", "gencode_primary", "MANE_Select", "Ensembl_canonical"]
    assert a["gene_name"] == ["KAZN"] and a["gene_id"] == ["ENSG00000189337"]
    assert a["transcript_support_level"] == ["5 (assigned to previous version 6)"]  # spaces inside a value
    assert "tag" not in parse_attributes('gene_id "ENSG1"; gene_biotype "lncRNA";')


# ---------------------------------------------------------------- scan + build (handmade fixture)

def test_iter_exons_applies_biotype_and_contig_rules():
    cfg = load_config(DEFAULT_CONFIG)
    counts = ScanCounts()
    with open(FIXTURES / "tiny.gtf") as f:
        exons = list(iter_exons(f, cfg, counts))
    assert counts.lines == 37 and counts.exon_lines == 13 and counts.kept == 9 and counts.kept_by_name == 0
    assert counts.dropped_non_primary == 1  # GENEC on KI270728.1 qualifies by biotype but has no canonical name
    assert counts.header[0] == "#!genome-build GRCh38.p14" and len(counts.header) == 5
    by_tx = {}
    for e in exons:
        by_tx.setdefault(e.transcript_id, []).append(e)
    assert set(by_tx) == {"ENST00000361390", "ENST00000000006", "ENST00000000001", "ENST00001000002",
                          "ENST00000000004", "ENST00000000007"}
    # ENST00001... ids are real; retained_intron, CDS_not_defined and lncRNA transcripts are not here.
    assert [e.exon_number for e in by_tx["ENST00000000001"]] == [1, 2, 3]
    assert by_tx["ENST00000000001"][0].tags == ("CCDS", "gencode_basic", "gencode_primary", "MANE_Select", "Ensembl_canonical")
    assert by_tx["ENST00000000001"][0].strand == "-"
    unnamed = by_tx["ENST00000000006"][0]
    assert unnamed.gene_name == unnamed.gene_id == "ENSG00000000004"
    assert by_tx["ENST00000361390"][0].chrom == "MT"


def test_iter_exons_allow_list_bypasses_biotypes_by_gene_name():
    """GENEF has only a protein_coding_CDS_not_defined transcript: out by biotype, in by name."""
    cfg = replace(load_config(DEFAULT_CONFIG), gene_names_always=("GENEF", "NOT-IN-FILE"))
    counts = ScanCounts()
    with open(FIXTURES / "tiny.gtf") as f:
        exons = list(iter_exons(f, cfg, counts))
    assert counts.kept == 10 and counts.kept_by_name == 1
    (genef,) = [e for e in exons if e.gene_name == "GENEF"]
    assert (genef.chrom, genef.start, genef.end, genef.transcript_biotype) == ("1", 6001, 6100, "protein_coding_CDS_not_defined")
    # The unnamed lncRNA gene cannot be allow-listed: the list is keyed on gene_name.
    assert not any(e.gene_id == "ENSG00000000002" for e in exons)


def test_merge_padded_pads_sorts_merges_and_joins_names():
    ivs = [(2001, 2100, "A"), (1001, 1300, "A"), (2150, 2200, "A"), (1250, 1400, "G"), (1001, 1300, "A")]
    assert merge_padded(ivs, 50) == [(950, 1450, "A,G"), (1950, 2250, "A")]
    assert merge_padded([(30, 80, "D")], 50) == [(0, 130, "D")]  # never below 0
    assert merge_padded([(10, 20, "X"), (21, 30, "X")], 0) == [(9, 30, "X")]  # touching intervals merge
    assert merge_padded([(10, 20, "X"), (22, 30, "Y")], 0) == [(9, 20, "X"), (21, 30, "Y")]
    assert merge_padded([], 50) == []


EXPECTED_BED = (
    "1\t950\t1450\tGENEA,GENEG\n"   # exon 3 of T1 + GENEG's exon, bridged by padding
    "1\t1950\t2250\tGENEA\n"        # exon 2 of T1 + the NMD-only alternative exon
    "1\t2950\t3250\tGENEA\n"        # exon 1, shared by T1 and the NMD isoform: once
    "1\t4950\t5150\tGENEA\n"        # the protein_coding_LoF exon
    "7\t0\t130\tENSG00000000004\n"  # no gene_name -> gene_id; pad clamped at 0
    "MT\t3256\t4312\tMT-ND1\n"
)


def test_build_funnel_bed_from_tiny_gtf(tmp_path: Path):
    gz = _gz(tmp_path)
    cfg = _cfg_for(gz)
    out = tmp_path / "out" / "funnel.bed"
    stats = build_funnel_bed(gz, out, cfg)

    assert out.read_text() == EXPECTED_BED
    assert stats.n_transcripts == 6
    assert stats.n_exon_lines == 9
    assert stats.n_exons == 8            # 3001-3200 is shared by two transcripts
    assert stats.n_genes == 4
    assert stats.n_intervals == 6
    assert stats.total_bp == 500 + 300 + 300 + 200 + 130 + 1056 == 2486
    assert stats.genome_fraction == round(2486 / 3_088_286_401, 6)
    assert stats.dropped_non_primary == 1
    assert stats.gtf_lines == 37

    # The retained_intron exon (1001-3200) must NOT have bridged the gene into one interval,
    # and the lncRNA / CDS_not_defined / scaffold exons are absent.
    idx = IntervalIndex.from_bed(out)
    assert idx.lookup("1", 2500) is None
    assert idx.lookup("1", 4050) is None and idx.lookup("1", 6050) is None
    assert idx.lookup("1", 1000) == "GENEA,GENEG"   # 1-based 951..1450 covered
    assert idx.lookup("1", 950) is None
    assert idx.lookup("7", 1) == "ENSG00000000004"
    assert idx.lookup("MT", 4312) == "MT-ND1" and idx.lookup("MT", 4313) is None

    m = json.loads(manifest_path_for(out).read_text())
    assert manifest_path_for(out).name == "funnel.bed.manifest.json"
    assert m["stage"] == "funnel"
    assert m["params"]["release"] == 116
    assert m["params"]["checksum_pinned"] == m["params"]["checksum_observed"] == cfg.checksum
    assert m["params"]["config"]["pad_bp"] == 50
    assert m["params"]["config"]["transcript_biotypes"] == list(cfg.transcript_biotypes)
    assert m["params"]["config"]["gene_names_always"] == list(cfg.gene_names_always)
    assert m["params"]["config"]["genome_length"] == 3_088_286_401
    assert m["params"]["gtf_header"][4] == "#!genebuild-last-updated 2025-11"
    assert m["counts"]["n_intervals"] == 6 and m["counts"]["gtf_exon_lines"] == 13
    assert m["counts"]["n_exon_lines_by_name"] == 0
    assert m["inputs"]["gtf"]["sha256"] and m["outputs"]["bed"]["sha256"]
    assert "checksums" not in m["inputs"]  # no sidecar: the file did not come from download_gtf


def test_build_is_deterministic(tmp_path: Path):
    gz = _gz(tmp_path)
    cfg = _cfg_for(gz)
    a, b = tmp_path / "a.bed", tmp_path / "b.bed"
    assert build_funnel_bed(gz, a, cfg) == build_funnel_bed(gz, b, cfg)
    assert a.read_bytes() == b.read_bytes()


def test_build_refuses_a_file_that_is_not_the_pinned_one(tmp_path: Path):
    gz = _gz(tmp_path)
    with pytest.raises(ValueError, match="pins"):
        build_funnel_bed(gz, tmp_path / "x.bed", load_config(DEFAULT_CONFIG))
    assert not (tmp_path / "x.bed").exists()


def test_build_refuses_an_empty_funnel(tmp_path: Path):
    """A misspelled biotype must be an error, not a 0-byte BED that sends nothing anywhere."""
    gz = _gz(tmp_path)
    cfg = replace(_cfg_for(gz), transcript_biotypes=("protein_codng",))
    out = tmp_path / "empty.bed"
    with pytest.raises(ValueError, match="no exon in .* matched .*protein_codng"):
        build_funnel_bed(gz, out, cfg)
    assert not out.exists() and not manifest_path_for(out).exists()


def test_pad_zero_keeps_exons_separate(tmp_path: Path):
    gz = _gz(tmp_path)
    cfg = replace(_cfg_for(gz), pad_bp=0)
    stats = build_funnel_bed(gz, tmp_path / "p0.bed", cfg)
    lines = (tmp_path / "p0.bed").read_text().splitlines()
    assert lines[0] == "1\t1000\t1400\tGENEA,GENEG"  # 1001-1300 and 1250-1400 overlap even unpadded
    assert lines[1] == "1\t2000\t2100\tGENEA" and lines[2] == "1\t2149\t2200\tGENEA"
    assert stats.n_intervals == 7


# ---------------------------------------------------------------- build (real lines, public genes)

def test_public_genes_fixture_is_the_recorded_slice():
    with gzip.open(PUBLIC_GTF, "rb") as f:
        data = f.read()
    assert hashlib.sha256(data).hexdigest() == PUBLIC_PROVENANCE["sha256_uncompressed"]
    assert data.count(b"\n") == PUBLIC_PROVENANCE["lines"] == 3193
    assert len(data) == PUBLIC_PROVENANCE["bytes_uncompressed"]
    assert PUBLIC_PROVENANCE["source"]["checksum_bsd"] == load_config(DEFAULT_CONFIG).checksum
    ids = set(PUBLIC_PROVENANCE["genes"])
    for line in data.decode("utf-8").splitlines():
        if not line.startswith("#"):
            assert parse_attributes(line.split("\t", 8)[8])["gene_id"][0] in ids


def test_public_variants_land_in_the_funnel(tmp_path: Path):
    """The three public textbook variants (GRCh38) resolve to their gene, in intervals that are
    byte-identical to the ones the full-genome build gives (checked against that BED once, live)."""
    cfg = _cfg_for(PUBLIC_GTF)
    out = tmp_path / "public.bed"
    stats = build_funnel_bed(PUBLIC_GTF, out, cfg)
    assert (stats.n_genes, stats.n_transcripts, stats.n_exons, stats.n_exon_lines) == (12, 101, 185, 1315)
    assert (stats.n_intervals, stats.total_bp, stats.dropped_non_primary, stats.gtf_lines) == (77, 35827, 0, 3193)
    bed = set(out.read_text().splitlines())
    idx = IntervalIndex.from_bed(out)

    assert idx.lookup("1", 11796321) == "MTHFR"                      # rs1801133, 1:11796321 G>A
    assert "1\t11796155\t11796595\tMTHFR" in bed
    assert idx.lookup("7", 117559590) == "CFTR"                      # p.Phe508del, 7:117559590 ATCT>A
    assert idx.overlaps("7", 117559590, "ATCT") and "7\t117559413\t117559963\tCFTR" in bed
    assert idx.lookup("17", 7675088) == "TP53"                       # p.Arg175His, 17:7675088 C>T
    assert "17\t7674808\t7675543\tTP53" in bed

    # Splice edge: MANE exon 4 of CFTR is 117530899-117531114; the canonical donor +1
    # (c.489+1G>T, 7:117531115) is in, 50 bp past the exon is in, 51 bp is out.
    assert "7\t117530848\t117531164\tCFTR" in bed
    assert idx.lookup("7", 117531115) == "CFTR"
    assert idx.lookup("7", 117531164) == "CFTR" and idx.lookup("7", 117531165) is None
    assert idx.lookup("7", 117540000) is None                        # deep CFTR intron

    # Mt_tRNA/Mt_rRNA by biotype; the non-coding disease genes by name, whatever their biotype.
    assert idx.lookup("MT", 3243) == "MT-TL1" and "MT\t3179\t3354\tMT-TL1" in bed      # m.3243A>G
    assert idx.lookup("MT", 1555) == "MT-RNR1" and "MT\t597\t1651\tMT-RNR1" in bed    # m.1555A>G
    for chrom, pos, name in [("2", 121530900, "RNU4ATAC"), ("9", 35657800, "RMRP"), ("3", 169764700, "TERC"),
                             ("22", 42615300, "RNU12"), ("12", 120291800, "RNU4-2"), ("12", 6943850, "RNU7-1"),
                             ("17", 8173500, "SNORD118")]:
        assert idx.lookup(chrom, pos) == name
    m = json.loads(manifest_path_for(out).read_text())
    assert m["counts"]["n_exon_lines_by_name"] == 7 and m["counts"]["gtf_exon_lines"] == 1400

    # Without the allow-list only the biotype-qualified genes remain.
    plain = build_funnel_bed(PUBLIC_GTF, tmp_path / "plain.bed", replace(cfg, gene_names_always=()))
    assert plain.n_genes == 5 and plain.n_intervals == 70
    assert IntervalIndex.from_bed(tmp_path / "plain.bed").lookup("9", 35657800) is None


# ---------------------------------------------------------------- download

def test_observe_checksums_uses_http_and_response_time():
    cfg = load_config(DEFAULT_CONFIG)
    http = _stub()
    obs = observe_checksums(cfg, http)
    assert obs.entry == "31909 137737"
    assert obs.retrieved_at == CHECKSUMS["retrieved_at"] and obs.from_cache is False
    assert http.calls == [("GET", cfg.checksums_url)]


def test_observe_checksums_detects_drift():
    cfg = load_config(DEFAULT_CONFIG)
    drifted = dict(CHECKSUMS, text=CHECKSUMS["text"].replace("31909 137737", "31910 137737"))
    with pytest.raises(ValueError, match="pins '31909 137737'"):
        observe_checksums(cfg, StubHttp({CHECKSUMS["url"]: drifted}))
    unlisted = dict(CHECKSUMS, text="56461 11 README\n")
    with pytest.raises(ValueError, match="not listed"):
        observe_checksums(cfg, StubHttp({CHECKSUMS["url"]: unlisted}))


def _readme_cfg(remote_dir: Path):
    """The real config pointed at README (the one CHECKSUMS entry we hold the bytes for)
    on a file:// base URL, so the streaming path runs without a network."""
    (remote_dir / "README").write_bytes(README["text"].encode("utf-8"))
    (remote_dir / "CHECKSUMS").write_text(CHECKSUMS["text"])
    cfg = load_config(DEFAULT_CONFIG)
    base = remote_dir.resolve().as_uri() + "/"
    return replace(cfg, base_url=base, filename="README", checksum="56461 11")


def test_download_gtf_streams_verifies_and_writes_sidecar(tmp_path: Path):
    remote = tmp_path / "remote"
    remote.mkdir()
    cfg = _readme_cfg(remote)
    http = StubHttp({cfg.checksums_url: CHECKSUMS})
    dest_dir = tmp_path / "ref"
    msgs: list[str] = []

    got = download_gtf(cfg, dest_dir, http, progress=msgs.append)
    assert got == dest_dir / "README" and got.read_bytes() == (remote / "README").read_bytes()
    assert not (dest_dir / "README.part").exists()
    assert any(s.startswith("funnel: streaming " + cfg.url) for s in msgs)  # the one uncounted live access
    side_bytes = sidecar_path(got).read_bytes()
    side = json.loads(side_bytes)
    assert side["entry"] == side["observed"] == "56461 11"
    assert side["retrieved_at"] == CHECKSUMS["retrieved_at"]
    assert side["file_url"] == cfg.url and side["checksums_url"] == cfg.checksums_url
    assert side["gtf_bytes"] == 10315 and side["text"] == CHECKSUMS["text"]
    assert "from_cache" not in side  # a fact about the run, not the source
    assert http.calls == [("GET", cfg.checksums_url)]

    # Present and verified: no second stream, CHECKSUMS re-observed, same path back,
    # and the provenance bytes untouched (not even rewritten).
    mtime = got.stat().st_mtime_ns
    side_mtime = sidecar_path(got).stat().st_mtime_ns
    assert download_gtf(cfg, dest_dir, http, progress=msgs.append) == got
    assert got.stat().st_mtime_ns == mtime
    assert sidecar_path(got).read_bytes() == side_bytes and sidecar_path(got).stat().st_mtime_ns == side_mtime
    assert http.calls == [("GET", cfg.checksums_url)] * 2
    assert sum(s.startswith("funnel: streaming") for s in msgs) == 1


def test_download_gtf_honours_offline_mode(tmp_path: Path):
    remote = tmp_path / "remote"
    remote.mkdir()
    cfg = _readme_cfg(remote)
    dest_dir = tmp_path / "ref"
    offline = StubHttp({cfg.checksums_url: CHECKSUMS}, offline=True)
    with pytest.raises(RuntimeError, match="offline"):
        download_gtf(cfg, dest_dir, offline)
    assert not (dest_dir / "README").exists() and not (dest_dir / "README.part").exists()
    # A verified copy needs nothing from outside, so offline mode is satisfied.
    got = download_gtf(cfg, dest_dir, StubHttp({cfg.checksums_url: CHECKSUMS}))
    assert download_gtf(cfg, dest_dir, offline) == got


def test_download_gtf_replaces_a_corrupt_local_copy(tmp_path: Path):
    remote = tmp_path / "remote"
    remote.mkdir()
    cfg = _readme_cfg(remote)
    dest_dir = tmp_path / "ref"
    dest_dir.mkdir()
    (dest_dir / "README").write_bytes(b"truncated")
    msgs: list[str] = []
    got = download_gtf(cfg, dest_dir, StubHttp({cfg.checksums_url: CHECKSUMS}), progress=msgs.append)
    assert bsd_sum(got) == "56461 11"
    assert any("downloading again" in s for s in msgs)


def test_download_gtf_rejects_bytes_that_do_not_match_checksums(tmp_path: Path):
    remote = tmp_path / "remote"
    remote.mkdir()
    cfg = _readme_cfg(remote)
    (remote / "README").write_bytes(README["text"].encode("utf-8") + b"\n")  # one byte off
    dest_dir = tmp_path / "ref"
    with pytest.raises(ValueError, match="BSD sum"):
        download_gtf(cfg, dest_dir, StubHttp({cfg.checksums_url: CHECKSUMS}))
    assert not (dest_dir / "README").exists() and not (dest_dir / "README.part").exists()


class _TruncatingHandler(BaseHTTPRequestHandler):
    """Promises the whole README, sends 4000 bytes, closes — as a dropped FTP mirror would."""
    body = README["text"].encode("utf-8")

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body[:4000])

    def log_message(self, *args):
        pass


def test_download_gtf_names_a_truncated_stream(tmp_path: Path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _TruncatingHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = replace(load_config(DEFAULT_CONFIG), base_url=f"http://127.0.0.1:{srv.server_port}/",
                      filename="README", checksum="56461 11")
        dest_dir = tmp_path / "ref"
        with pytest.raises(ValueError, match="truncated: got 4000 of 10315 bytes"):
            download_gtf(cfg, dest_dir, StubHttp({cfg.checksums_url: CHECKSUMS}))
        assert not (dest_dir / "README").exists() and not (dest_dir / "README.part").exists()
    finally:
        srv.shutdown()
        srv.server_close()


def test_downloaded_file_provenance_reaches_the_manifest(tmp_path: Path):
    """A sidecar written by download_gtf is carried into the BED manifest."""
    gz = _gz(tmp_path)
    cfg = _cfg_for(gz)
    sidecar_path(gz).write_text(json.dumps({"entry": cfg.checksum, "observed": cfg.checksum,
                                            "retrieved_at": CHECKSUMS["retrieved_at"],
                                            "checksums_url": cfg.checksums_url, "file_url": cfg.url,
                                            "text": CHECKSUMS["text"], "gtf_bytes": gz.stat().st_size}))
    out = tmp_path / "f.bed"
    build_funnel_bed(gz, out, cfg)
    m = json.loads(manifest_path_for(out).read_text())
    assert m["inputs"]["checksums"]["retrieved_at"] == CHECKSUMS["retrieved_at"]
    assert m["inputs"]["checksums"]["entry"] == cfg.checksum
    assert m["inputs"]["checksums"]["gtf_bytes"] == m["inputs"]["gtf"]["bytes"]


# ---------------------------------------------------------------- live

@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ENGINE_LIVE_TESTS"), reason="set ENGINE_LIVE_TESTS=1 to hit ftp.ensembl.org")
def test_live_checksums_and_readme_download(tmp_path: Path):
    """Real Ensembl FTP: the live CHECKSUMS still pins our entry, and streaming README
    through download_gtf reproduces Ensembl's own BSD sum for it. No variant is sent."""
    cfg = load_config(DEFAULT_CONFIG)
    http = Http(HttpCache(tmp_path / "cache"), limiter=RateLimiter(default_per_second=1.0))
    obs = observe_checksums(cfg, http)
    assert obs.entry == "31909 137737" and not obs.from_cache
    readme_cfg = replace(cfg, filename="README", checksum=parse_checksums(obs.text)["README"])
    got = download_gtf(readme_cfg, tmp_path / "ref", http)
    assert got.name == "README" and bsd_sum(got) == "56461 11"
    assert "Homo_sapiens" in got.read_text()
    assert json.loads(sidecar_path(got).read_text())["retrieved_at"] == obs.retrieved_at
    # Warm cache + verified file: a rerun is hermetic and leaves the provenance bytes alone.
    side = sidecar_path(got).read_bytes()
    assert download_gtf(readme_cfg, tmp_path / "ref", Http(http.cache, offline=True)) == got
    assert sidecar_path(got).read_bytes() == side
