"""Stage 1 — ingest and normalize.

Deterministic and model-free. One bcftools pipeline, one Python pass:

    bcftools norm  (primary contigs only · split multi-allelics · left-align if a
                    reference is configured · tag every modified record)
      | bcftools query  (flat rows: site, quality, sample genotype fields)
      | python          (canonical chromosome names · quality flag · counts)
      | bgzip           (variants.tsv.gz)

Records that failed the caller's hard filters are **kept** with their filter string in
``quality_flag`` — the filters were blunt, and a real variant in difficult sequence
should surface with its weakness stated rather than vanish here. Non-primary contigs
(unplaced, random, alt, decoy) are dropped and counted; nothing on them can be
reported anyway.
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterator

from engine.config import CaseConfig
from engine.contigs import canonical, naming_style, primary_names
from engine.sex import SexTally, effective_sex
from engine.manifest import Manifest

STAGE_DIR = "01_ingest"
OLD_REC_TAG = "ORIGINAL"

# Output columns, in order. Downstream stages read by name, never by position.
COLUMNS = (
    "chrom",         # canonical: 1..22, X, Y, MT
    "chrom_source",  # as spelled in the VCF (e.g. chr15, 15, M)
    "pos",
    "id",            # dbSNP rsID if the caller wrote one, else '.'
    "ref",
    "alt",           # exactly one allele after splitting
    "vtype",         # snv | indel | mnp
    "qual",
    "filter",        # caller FILTER column verbatim
    "quality_flag",  # '' when PASS, else the filter string — a caveat, not an exclusion
    "info_dp",
    "qd",
    "mq",
    "fs",
    "gt",
    "ad",
    "dp",
    "gq",
    "pgt",
    "pid",
    "split_from",    # original multi-allelic/realigned record, or ''
)

INFO_TAGS = ("DP", "QD", "MQ", "FS")
FORMAT_TAGS = ("GT", "AD", "DP", "GQ", "PGT", "PID")


@dataclass
class VcfHeader:
    contigs: list[str]
    info: set[str]
    formats: set[str]
    samples: list[str]
    reference: str | None
    lines: int


def read_header(vcf: Path) -> VcfHeader:
    out = subprocess.run(["bcftools", "view", "-h", str(vcf)], capture_output=True, text=True, check=True)
    contigs: list[str] = []
    info: set[str] = set()
    formats: set[str] = set()
    samples: list[str] = []
    reference: str | None = None
    n = 0
    for line in out.stdout.splitlines():
        n += 1
        if line.startswith("##contig=<ID="):
            contigs.append(line[13:].split(",", 1)[0].rstrip(">"))
        elif line.startswith("##INFO=<ID="):
            info.add(line[11:].split(",", 1)[0])
        elif line.startswith("##FORMAT=<ID="):
            formats.add(line[13:].split(",", 1)[0])
        elif line.startswith("##reference="):
            reference = line[12:]
        elif line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
    return VcfHeader(contigs, info, formats, samples, reference, n)


def build_index(vcf: Path, out_dir: Path, threads: int) -> Path:
    """Build our own CSI index beside the stage output (the input is never touched).

    Provider indexes often lack record statistics and may be older than the data
    file; ours gives exact per-contig record counts in a couple of seconds.
    """
    csi = out_dir / "input.csi"
    subprocess.run(["bcftools", "index", "-c", "--threads", str(threads), "-o", str(csi), str(vcf)], check=True)
    return csi


def index_stats(vcf: Path, csi: Path) -> dict[str, int]:
    """Records per contig, for every contig that has any, from ``bcftools index -s``."""
    out = subprocess.run(["bcftools", "index", "-s", f"{vcf}##idx##{csi}"], capture_output=True, text=True, check=True)
    stats: dict[str, int] = {}
    for line in out.stdout.splitlines():
        contig, _length, n = line.split("\t")
        stats[contig] = int(n)
    return stats


def query_format(header: VcfHeader) -> str:
    """Build the bcftools query format, substituting a literal '.' for absent tags so
    the column layout is identical for every input."""
    site = ["%CHROM", "%POS", "%ID", "%REF", "%ALT", "%QUAL", "%FILTER"]
    site += [f"%INFO/{t}" if t in header.info else "." for t in INFO_TAGS]
    site.append(f"%INFO/{OLD_REC_TAG}")
    fmt = [f"%{t}" if t in header.formats else "." for t in FORMAT_TAGS]
    return "\t".join(site) + "[\t" + "\t".join(fmt) + "]\n"


def norm_command(
    vcf: Path,
    *,
    index: Path | None = None,
    regions_bed: Path | None,
    primary: list[str],
    reference: Path | None,
    regions_overlap: int,
    threads: int,
) -> list[str]:
    cmd = ["bcftools", "norm", "-m", "-any", "--old-rec-tag", OLD_REC_TAG, "--threads", str(threads), "-Ou"]
    if reference is not None:
        cmd += ["-f", str(reference), "-c", "w"]
    if regions_bed is not None:
        cmd += ["-R", str(regions_bed), "--regions-overlap", str(regions_overlap)]
    else:
        cmd += ["-r", ",".join(primary)]
    cmd.append(f"{vcf}##idx##{index}" if index else str(vcf))
    return cmd


def query_command(fmt: str, sample: str | None) -> list[str]:
    cmd = ["bcftools", "query", "-f", fmt]
    if sample:
        cmd += ["-s", sample]
    return cmd


def _vtype(ref: str, alt: str) -> str:
    if len(ref) == 1 and len(alt) == 1:
        return "snv"
    if len(ref) != len(alt):
        return "indel"
    return "mnp"


@dataclass
class IngestCounts:
    rows_out: int = 0
    sites_in: int = 0
    per_contig: Counter = field(default_factory=Counter)
    per_type: Counter = field(default_factory=Counter)
    per_filter: Counter = field(default_factory=Counter)
    per_gt: Counter = field(default_factory=Counter)
    split_rows: int = 0
    split_sites: int = 0
    realigned_rows: int = 0
    dropped_non_primary: int = 0
    dropped_star_allele: int = 0   # '*' rows split off a multi-allelic site; the site survives
    dropped_star_sites: int = 0    # standalone '*' records; the site is gone

    def as_dict(self) -> dict:
        return {
            "sites_in": self.sites_in,
            "rows_out": self.rows_out,
            "pass_rows": self.per_filter.get("PASS", 0),
            "flagged_rows": self.rows_out - self.per_filter.get("PASS", 0),
            "multiallelic_sites_split": self.split_sites,
            "rows_from_splits": self.split_rows,
            "rows_realigned": self.realigned_rows,
            "dropped_non_primary": self.dropped_non_primary,
            "dropped_star_allele": self.dropped_star_allele,
            "dropped_star_sites": self.dropped_star_sites,
            "per_type": dict(self.per_type),
            "per_filter": dict(self.per_filter.most_common()),
            "per_gt": dict(self.per_gt.most_common()),
            "rows_per_contig": {k: self.per_contig[k] for k in sorted(self.per_contig, key=_contig_sort)},
        }


def _contig_sort(c: str) -> tuple[int, str]:
    return (int(c), "") if c.isdigit() else (100, c)


def transform(lines: Iterator[str], counts: IngestCounts) -> Iterator[str]:
    """Turn bcftools query rows into output rows. Pure; the counts object is filled
    as a side effect so the caller can write the manifest."""
    canon: dict[str, str | None] = {}
    split_seen: set[str] = set()
    for line in lines:
        f = line.rstrip("\n").split("\t")
        chrom_src = f[0]
        c = canon.get(chrom_src)
        if c is None and chrom_src not in canon:
            c = canon[chrom_src] = canonical(chrom_src)
        if c is None:
            counts.dropped_non_primary += 1
            continue
        ref, alt = f[3], f[4]
        filt = f[6]
        orig = f[11]
        if alt == "*" or alt == ".":
            if orig != "." and "," in orig.split("|")[3]:
                counts.dropped_star_allele += 1
            else:
                counts.dropped_star_sites += 1
            continue
        if orig != ".":
            # ORIGINAL is "CHROM|POS|REF|ALT|k" with k the allele index — a comma in
            # ALT means it was a split; the site key is the record without k.
            parts = orig.split("|")
            if "," in parts[3]:
                counts.split_rows += 1
                site = "|".join(parts[:4])
                if site not in split_seen:
                    split_seen.add(site)
                    counts.split_sites += 1
            else:
                counts.realigned_rows += 1
        vt = _vtype(ref, alt)
        counts.rows_out += 1
        counts.per_contig[c] += 1
        counts.per_type[vt] += 1
        counts.per_filter[filt] += 1
        counts.per_gt[f[12]] += 1
        quality_flag = "" if filt in ("PASS", ".") else filt
        split_from = "" if orig == "." else orig
        yield "\t".join((
            c, chrom_src, f[1], f[2], ref, alt, vt, f[5], filt, quality_flag,
            f[7], f[8], f[9], f[10],
            f[12], f[13], f[14], f[15], f[16], f[17],
            split_from,
        )) + "\n"
    counts.sites_in = counts.rows_out - counts.split_rows + counts.split_sites


def _open_bgzip(path: Path, threads: int, log: IO[str]) -> tuple[IO[str], subprocess.Popen | None]:
    if shutil.which("bgzip"):
        p = subprocess.Popen(
            ["bgzip", "--threads", str(threads), "-c"],
            stdin=subprocess.PIPE, stdout=open(path, "wb"), stderr=log, text=True,
        )
        assert p.stdin is not None
        return p.stdin, p
    return gzip.open(path, "wt"), None


def run_ingest(
    case: CaseConfig,
    run_dir: Path,
    *,
    regions: Path | None = None,
    regions_overlap: int = 1,
    threads: int = 2,
) -> Path:
    """Run stage 1 for ``case`` into ``run_dir/01_ingest``. Returns the manifest path."""
    out_dir = run_dir / STAGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    variants_path = out_dir / "variants.tsv.gz"
    manifest_path = out_dir / "manifest.json"
    log_path = out_dir / "ingest.log"

    m = Manifest(stage="ingest")
    m.add_tool("bcftools")
    m.add_tool("bgzip")
    m.add_input("vcf", case.vcf)
    for ext in (".tbi", ".csi"):
        idx = case.vcf.with_name(case.vcf.name + ext)
        if idx.exists():
            m.add_input("vcf_index", idx, checksum=False)
            break

    csi = build_index(case.vcf, out_dir, threads)
    stats = index_stats(case.vcf, csi)
    header = read_header(case.vcf)
    style = naming_style(header.contigs)
    primary = primary_names(header.contigs)
    sample = case.sample
    if sample is None:
        if len(header.samples) != 1:
            raise ValueError(f"VCF has {len(header.samples)} samples; set `sample:` in the case file")
        sample = header.samples[0]
    elif sample not in header.samples:
        raise ValueError(f"sample {sample!r} not in VCF samples {header.samples}")

    regions_bed = regions or case.regions
    if regions_bed is not None:
        m.add_input("regions", regions_bed)
    if case.reference_fasta is not None:
        _check_reference_names(case.reference_fasta, header.contigs)
        m.add_input("reference_fasta", case.reference_fasta, checksum=False)
    else:
        m.note("No reference FASTA configured: multi-allelic sites were split but indels were "
               "not left-aligned and REF alleles were not checked against the assembly.")

    fmt = query_format(header)
    ncmd = norm_command(case.vcf, index=csi, regions_bed=regions_bed, primary=primary,
                        reference=case.reference_fasta, regions_overlap=regions_overlap, threads=threads)
    qcmd = query_command(fmt, sample)

    m.params.update({
        "sample": sample,
        "naming_style": style,
        "primary_contigs_in_header": primary,
        "contigs_in_header": len(header.contigs),
        "regions_mode": "bed" if regions_bed else "primary_contigs",
        "regions_overlap": regions_overlap if regions_bed else None,
        "left_aligned": case.reference_fasta is not None,
        "multiallelics": "split (-m -any)",
        "vcf_header_reference": header.reference,
        "info_tags_present": [t for t in INFO_TAGS if t in header.info],
        "format_tags_present": [t for t in FORMAT_TAGS if t in header.formats],
        "norm_command": ncmd,
        "query_command": qcmd,
    })

    counts = IngestCounts()
    tally = SexTally()
    gt_col, chrom_col, pos_col = COLUMNS.index("gt"), COLUMNS.index("chrom"), COLUMNS.index("pos")
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(ncmd) + "\n$ " + " ".join(qcmd) + "\n\n")
        log.flush()
        norm = subprocess.Popen(ncmd, stdout=subprocess.PIPE, stderr=log)
        query = subprocess.Popen(qcmd, stdin=norm.stdout, stdout=subprocess.PIPE, stderr=log, text=True)
        assert norm.stdout is not None and query.stdout is not None
        norm.stdout.close()  # let norm receive SIGPIPE if query exits
        out, bgz = _open_bgzip(variants_path, threads, log)
        try:
            out.write("\t".join(COLUMNS) + "\n")
            for row in transform(query.stdout, counts):
                out.write(row)
                cells = row.split("\t")
                if cells[chrom_col] in ("X", "Y"):
                    tally.add(cells[chrom_col], int(cells[pos_col]), cells[gt_col])
        finally:
            out.close()
            if bgz is not None:
                bgz.wait()
        qrc = query.wait()
        nrc = norm.wait()
    if nrc != 0 or qrc != 0 or (bgz is not None and bgz.returncode != 0):
        raise RuntimeError(f"bcftools pipeline failed (norm={nrc}, query={qrc}); see {log_path}")
    if counts.rows_out == 0:
        raise RuntimeError(f"ingest produced no rows — check contig naming and regions; see {log_path}")

    on_primary = {canonical(c): n for c, n in stats.items() if canonical(c) is not None}
    n_primary = sum(on_primary.values())
    n_file = sum(stats.values())
    m.counts.update({
        "records_in_file": n_file,
        "contigs_with_records": len(stats),
        "records_on_primary_contigs": n_primary,
        "records_excluded_non_primary": n_file - n_primary,
    })
    m.counts.update(counts.as_dict())
    m.counts["records_per_contig_in_file"] = {k: on_primary[k] for k in sorted(on_primary, key=_contig_sort)}
    if regions_bed is None and counts.sites_in + counts.dropped_star_sites != n_primary:
        m.note(f"Consistency check FAILED: {counts.sites_in} sites ingested + {counts.dropped_star_sites} "
               f"star-only records ≠ {n_primary} records on primary contigs per the index.")
    elif regions_bed is None:
        m.note(f"Consistency check passed: every one of the {n_primary} primary-contig records is accounted for.")
    inferred = tally.inferred()
    m.params["sex_stated"] = case.sex
    m.params["sex_inference"] = tally.as_dict()
    m.params["sex"] = effective_sex(case.sex, inferred)
    if case.sex in ("male", "female") and inferred in ("male", "female") and inferred != case.sex:
        m.note(f"Sex check FAILED: the case file says {case.sex} but the calls look {inferred} "
               f"(X non-PAR het fraction {tally.x_het_fraction:.3f}, {tally.y_nonpar_carrier} Y calls). "
               "The stated sex is used; check the sample.")
    elif case.sex == "unknown" and inferred != "unknown":
        m.note(f"Sex not stated in the case file; inferred {inferred} from the calls "
               f"(X non-PAR het fraction {tally.x_het_fraction:.3f}, {tally.y_nonpar_carrier} Y calls).")
    elif case.sex in ("male", "female"):
        m.note(f"Sex check passed: stated {case.sex}, calls agree" if inferred == case.sex
               else f"Sex stated {case.sex}; too few X calls to check ({tally.x_nonpar_carrier}).")
    m.add_output("index", csi)
    m.add_output("variants", variants_path)
    m.write(manifest_path)
    return manifest_path


def _check_reference_names(fasta: Path, vcf_contigs: list[str]) -> None:
    fai = fasta.with_name(fasta.name + ".fai")
    if not fai.exists():
        raise FileNotFoundError(f"reference is not indexed: {fai} (run: samtools faidx {fasta})")
    ref_names = {line.split("\t", 1)[0] for line in fai.read_text().splitlines()}
    missing = [c for c in primary_names(vcf_contigs) if c not in ref_names]
    if missing:
        raise ValueError(
            "reference FASTA contig names do not match the VCF "
            f"(missing {missing[:5]}{'…' if len(missing) > 5 else ''}). "
            "The VCF and reference must use the same spelling (e.g. both '15', or both 'chr15')."
        )


def read_variants(run_dir: Path) -> Iterator[dict[str, str]]:
    """Iterate stage-1 rows as dicts. Used by later stages and tests."""
    path = run_dir / STAGE_DIR / "variants.tsv.gz"
    with gzip.open(path, "rt") as f:
        cols = f.readline().rstrip("\n").split("\t")
        for line in f:
            yield dict(zip(cols, line.rstrip("\n").split("\t")))
