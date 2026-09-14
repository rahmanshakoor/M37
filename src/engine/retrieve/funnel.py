"""Tier A funnel — exon/splice-proximal regions from a release-pinned Ensembl GTF.

A whole genome carries millions of variants and every public source downstream is
rate-limited, so stage 2 sends only the variants that a coding or splicing mechanism
could explain: those inside, or within a few dozen bases of, an exon of a
protein-coding transcript, of a mitochondrial tRNA/rRNA gene, or of a short list of
clinically established non-coding genes. This module builds that region set once, as
a BED, from one release-pinned annotation file — never from a live REST endpoint —
so the same VCF always yields the same set of variants sent to the sources.

The build is a single streaming pass over the gzip (the file is ~140 MB compressed,
~1.4 GB of text; it is never held in memory). Facts about the release-116 file that
the parser depends on, all observed by probing it:

* the ``tag`` attribute repeats on one line (``tag "CCDS"; tag "MANE_Select"; …``),
  so attributes are parsed to lists, never to a dict that keeps the last value;
* ``gene_name`` is absent on a third of lines — regions are keyed on ``gene_id`` and
  named by ``gene_name`` when present, else ``gene_id``;
* transcript ids are ``ENST`` + 11 digits and no longer all start ``ENST00000``;
* the file is not coordinate-sorted (genes appear in internal-id order), so every
  chromosome is sorted before merging;
* chromosomes are spelled ``1``…``22``, ``X``, ``Y``, ``MT`` (no ``chr``); anything
  else is dropped with a count, as at ingest.

Ensembl's CHECKSUMS files use BSD ``sum(1)``, not md5 — the algorithm is implemented
here so verification does not depend on a ``sum`` binary. The BED is 0-based
half-open with canonical chromosome names, padded and merged; its manifest records
the release, the pinned and observed checksums, the GTF header, the config and the
statistics, so a judge can re-derive every interval. An empty funnel is an error,
never a result: it would silently send nothing to any source.
"""

from __future__ import annotations

import gzip
import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import yaml

from engine.contigs import PRIMARY, canonical
from engine.manifest import Manifest
from engine.retrieve.http import DEFAULT_USER_AGENT, Http

GRCH38_LENGTHS: dict[str, int] = {
    "1": 248_956_422, "2": 242_193_529, "3": 198_295_559, "4": 190_214_555,
    "5": 181_538_259, "6": 170_805_979, "7": 159_345_973, "8": 145_138_636,
    "9": 138_394_717, "10": 133_797_422, "11": 135_086_622, "12": 133_275_309,
    "13": 114_364_328, "14": 107_043_718, "15": 101_991_189, "16": 90_338_345,
    "17": 83_257_441, "18": 80_373_285, "19": 58_617_616, "20": 64_444_167,
    "21": 46_709_983, "22": 50_818_468, "X": 156_040_895, "Y": 57_227_415,
    "MT": 16_569,
}
"""Length of each of the 25 primary GRCh38 contigs, as rest.ensembl.org/info/assembly/homo_sapiens
reports them (GRCh38.p14; patch releases never alter the primary chromosomes)."""

GRCH38_PRIMARY_LENGTH = sum(GRCH38_LENGTHS.values())
"""3,088,286,401 bp: 1–22, X, Y and MT together — the genome_fraction denominator."""

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "funnel.yaml"

Progress = Callable[[str], None]


# ---------------------------------------------------------------- config

@dataclass(frozen=True)
class FunnelConfig:
    release: int
    base_url: str
    filename: str
    checksum: str
    """Pinned BSD-sum entry for ``filename``, e.g. ``"31909 137737"``."""
    gene_biotypes: tuple[str, ...]
    transcript_biotypes: tuple[str, ...]
    pad_bp: int
    gene_names_always: tuple[str, ...] = ()
    """Genes kept whatever their biotypes — every exon of every transcript of a gene
    whose ``gene_name`` is listed. For clinically established non-coding genes."""
    genome_length: int = GRCH38_PRIMARY_LENGTH

    @property
    def url(self) -> str:
        return self.base_url + self.filename

    @property
    def checksums_url(self) -> str:
        return self.base_url + "CHECKSUMS"


_BSD_ENTRY = re.compile(r"^\d{1,5} \d+$")


def load_config(path: Path = DEFAULT_CONFIG) -> FunnelConfig:
    """Read ``configs/funnel.yaml``; ``{release}`` placeholders pin URL and filename to one number."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    release = int(raw["release"])
    cfg = FunnelConfig(
        release=release,
        base_url=str(raw["base_url"]).format(release=release),
        filename=str(raw["filename"]).format(release=release),
        checksum=str(raw["checksum"]).strip(),
        gene_biotypes=tuple(str(b) for b in raw["gene_biotypes"]),
        transcript_biotypes=tuple(str(b) for b in raw["transcript_biotypes"]),
        pad_bp=int(raw["pad_bp"]),
        gene_names_always=tuple(str(n) for n in raw.get("gene_names_always") or ()),
        genome_length=int(raw.get("genome_length", GRCH38_PRIMARY_LENGTH)),
    )
    if not cfg.base_url.endswith("/"):
        raise ValueError(f"base_url must end with '/': {cfg.base_url!r}")
    if not _BSD_ENTRY.match(cfg.checksum):
        raise ValueError(f"checksum must be a BSD sum entry '<checksum> <blocks>', got {cfg.checksum!r}")
    if not cfg.gene_biotypes or not cfg.transcript_biotypes:
        raise ValueError("gene_biotypes and transcript_biotypes must not be empty")
    if any(not n.strip() for n in cfg.gene_names_always):
        raise ValueError("gene_names_always must not contain an empty name")
    if cfg.pad_bp < 0 or cfg.genome_length <= 0:
        raise ValueError("pad_bp must be >= 0 and genome_length > 0")
    return cfg


# ---------------------------------------------------------------- BSD sum

# rotr16 of every value the running sum can hold (< 65536 + 256), so the per-byte
# step is one lookup and one add with no masking: ~3 s for a 141 MB file.
_ROTR = tuple((((x & 0xFFFF) >> 1) | ((x & 1) << 15)) for x in range(0x10000 + 0x100))


class BsdSum:
    """Streaming BSD ``sum(1)``: a 16-bit right-rotating checksum plus a 1 KiB block count.

    For every byte ``s = rotr16(s) + byte (mod 2**16)``; the block count is
    ``ceil(size / 1024)``. This is what Ensembl's CHECKSUMS files contain
    (``sum README`` → ``56461 11``, matching the published entry).
    """

    def __init__(self) -> None:
        self._s = 0
        self.size = 0

    def update(self, data: bytes) -> None:
        s = self._s
        rot = _ROTR
        for b in data:
            s = rot[s] + b
        self._s = s & 0xFFFF
        self.size += len(data)

    @property
    def checksum(self) -> int:
        return self._s

    @property
    def blocks(self) -> int:
        return -(-self.size // 1024)

    def __str__(self) -> str:
        return f"{self.checksum} {self.blocks}"


def bsd_sum(path: Path, chunk: int = 1 << 20) -> str:
    """``"<checksum> <blocks>"`` of a file, as ``sum`` would print it."""
    h = BsdSum()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return str(h)


def parse_checksums(text: str) -> dict[str, str]:
    """``{filename: "<checksum> <blocks>"}`` from an Ensembl CHECKSUMS file."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            out[parts[2]] = f"{parts[0]} {parts[1]}"
    return out


# ---------------------------------------------------------------- download

@dataclass(frozen=True)
class ChecksumObservation:
    """What the live CHECKSUMS file said, and when — written beside the GTF."""
    url: str
    retrieved_at: str
    from_cache: bool
    entry: str
    text: str


def observe_checksums(cfg: FunnelConfig, http: Http) -> ChecksumObservation:
    """Fetch CHECKSUMS through ``http`` and confirm it still lists the pinned entry for ``cfg.filename``.

    A mismatch means the file changed under the release or the config is stale;
    either way the build must not proceed on a different file than the one pinned.
    ``retrieved_at`` is the Response's, so a warm cache replays the original observation.
    """
    resp = http.get(cfg.checksums_url)
    entries = parse_checksums(resp.text)
    if cfg.filename not in entries:
        raise ValueError(f"{cfg.filename} is not listed in {cfg.checksums_url}")
    entry = entries[cfg.filename]
    if entry != cfg.checksum:
        raise ValueError(
            f"CHECKSUMS lists {entry!r} for {cfg.filename} but the config pins {cfg.checksum!r}: "
            f"the file changed under release {cfg.release}, or the config is stale"
        )
    return ChecksumObservation(cfg.checksums_url, resp.retrieved_at, resp.from_cache, entry, resp.text)


def sidecar_path(gtf: Path) -> Path:
    return gtf.with_name(gtf.name + ".CHECKSUMS.json")


def _write_sidecar(gtf: Path, obs: ChecksumObservation, observed: str, url: str) -> None:
    """Provenance beside the GTF: what CHECKSUMS said, when, and what the bytes summed to.

    Only facts about the source go in — not whether this run hit the cache — so a
    re-run with a warm cache produces the same bytes, and an unchanged sidecar is
    left untouched rather than rewritten.
    """
    rec = {
        "file_url": url,
        "gtf_bytes": gtf.stat().st_size,
        "observed": observed,
        "entry": obs.entry,
        "checksums_url": obs.url,
        "retrieved_at": obs.retrieved_at,
        "text": obs.text,
    }
    data = json.dumps(rec, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    p = sidecar_path(gtf)
    if p.exists() and p.read_text(encoding="utf-8") == data:
        return
    p.write_text(data, encoding="utf-8")


def download_gtf(
    cfg: FunnelConfig,
    dest_dir: Path,
    http: Http,
    *,
    progress: Progress | None = None,
    timeout: float = 120.0,
) -> Path:
    """Stream the pinned GTF into ``dest_dir`` and verify it; skip when a verified copy exists.

    The CHECKSUMS file goes through ``http`` (cached, retried, replayable offline). The
    GTF cannot: ``Http`` decodes every body to text, which would corrupt a gzip, so it
    is streamed with urllib straight to disk, checksummed on the way, and renamed into
    place only once the sum matches. That stream is the one network access
    ``Http.live_requests`` does not count, so it honours ``http.offline`` and announces
    itself through ``progress``. Any HTTP failure raises. Returns the verified path.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / cfg.filename
    obs = observe_checksums(cfg, http)

    if dest.exists():
        seen = bsd_sum(dest)
        if seen == obs.entry:
            _write_sidecar(dest, obs, seen, cfg.url)
            return dest
        if progress:
            progress(f"funnel: {dest.name} present but checksum {seen} != {obs.entry}; downloading again")

    # A stub standing in for Http implements only get/post/request, hence getattr.
    if getattr(http, "offline", False):
        raise RuntimeError(
            f"offline: no verified {cfg.filename} in {dest_dir} and offline mode forbids streaming {cfg.url}"
        )
    if progress:
        progress(f"funnel: streaming {cfg.url} (live; not cached by Http)")
    part = dest.with_name(dest.name + ".part")
    h = BsdSum()
    ua = getattr(http, "user_agent", DEFAULT_USER_AGENT)
    req = urllib.request.Request(cfg.url, headers={"User-Agent": ua})
    reported = 0
    with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "wb") as out:
        expected = r.headers.get("Content-Length")
        while chunk := r.read(1 << 20):
            out.write(chunk)
            h.update(chunk)
            if progress and h.size - reported >= 32 << 20:
                reported = h.size
                progress(f"funnel: downloaded {h.size >> 20} MiB of {cfg.filename}")
    # http.client does not raise when the server closes early: read() goes short, then empty.
    if expected is not None and h.size != int(expected):
        part.unlink(missing_ok=True)
        raise ValueError(f"download of {cfg.url} truncated: got {h.size} of {expected} bytes")
    seen = str(h)
    if seen != obs.entry:
        part.unlink(missing_ok=True)
        raise ValueError(f"downloaded {cfg.url} has BSD sum {seen!r}, CHECKSUMS says {obs.entry!r}")
    part.replace(dest)
    _write_sidecar(dest, obs, seen, cfg.url)
    return dest


# ---------------------------------------------------------------- GTF parsing

_ATTR = re.compile(r'(\S+) "([^"]*)"')


def parse_attributes(text: str) -> dict[str, list[str]]:
    """GTF column 9 as ``{key: [value, ...]}`` in file order.

    Every key maps to a list because ``tag`` legitimately repeats on one line and a
    plain dict would silently keep only the last tag. Values are the quoted strings
    verbatim (``transcript_support_level`` can be ``"5 (assigned to previous version 6)"``).
    """
    out: dict[str, list[str]] = {}
    for k, v in _ATTR.findall(text):
        out.setdefault(k, []).append(v)
    return out


@dataclass(slots=True)
class Exon:
    chrom: str
    """Canonical chromosome name."""
    start: int
    """1-based inclusive, as in the GTF."""
    end: int
    strand: str
    gene_id: str
    gene_name: str
    """``gene_name`` attribute, or ``gene_id`` when the line has none."""
    transcript_id: str
    transcript_biotype: str
    exon_number: int
    tags: tuple[str, ...]


@dataclass
class ScanCounts:
    lines: int = 0
    exon_lines: int = 0
    kept: int = 0
    kept_by_name: int = 0
    """Of ``kept``, exon lines admitted only by ``gene_names_always``."""
    dropped_non_primary: int = 0
    header: list[str] = field(default_factory=list)


def iter_exons(
    lines: Iterable[str],
    cfg: FunnelConfig,
    counts: ScanCounts,
    *,
    progress: Progress | None = None,
) -> Iterator[Exon]:
    """Yield the exons of qualifying transcripts on primary contigs from GTF lines.

    A transcript qualifies by biotypes (gene and transcript both listed) or because
    its gene is named in ``gene_names_always``. Pure over ``lines``; ``counts`` is
    filled as a side effect. A line is parsed in full only after cheap substring
    tests, which is what keeps a pass over ~11M lines in seconds.
    """
    gene_needles = tuple(f'gene_biotype "{b}"' for b in cfg.gene_biotypes)
    tx_needles = tuple(f'transcript_biotype "{b}"' for b in cfg.transcript_biotypes)
    name_needles = tuple(f'gene_name "{n}"' for n in cfg.gene_names_always)
    canon: dict[str, str | None] = {}
    for line in lines:
        counts.lines += 1
        if progress and counts.lines % 5_000_000 == 0:
            progress(f"funnel: {counts.lines:,} GTF lines scanned, {counts.kept:,} exons kept")
        if line[:1] == "#":
            if line.startswith("#!"):
                counts.header.append(line.rstrip("\n"))
            continue
        f = line.split("\t", 8)
        if len(f) < 9 or f[2] != "exon":
            continue
        counts.exon_lines += 1
        attrs = f[8]
        maybe_biotype = any(n in attrs for n in gene_needles) and any(n in attrs for n in tx_needles)
        if not maybe_biotype and not any(n in attrs for n in name_needles):
            continue
        chrom = canon.get(f[0])
        if chrom is None and f[0] not in canon:
            chrom = canon[f[0]] = canonical(f[0])
        if chrom is None:
            counts.dropped_non_primary += 1
            continue
        a = parse_attributes(attrs)
        # The substring tests were a pre-filter; decide on the parsed values.
        if not _biotypes_match(a, cfg):
            if _first(a, "gene_name") not in cfg.gene_names_always:
                continue
            counts.kept_by_name += 1
        gene_id, transcript_id = _first(a, "gene_id"), _first(a, "transcript_id")
        if not gene_id or not transcript_id:
            raise ValueError(f"GTF line {counts.lines}: exon without gene_id/transcript_id")
        counts.kept += 1
        yield Exon(
            chrom=chrom,
            start=int(f[3]),
            end=int(f[4]),
            strand=f[6],
            gene_id=gene_id,
            gene_name=_first(a, "gene_name") or gene_id,
            transcript_id=transcript_id,
            transcript_biotype=_first(a, "transcript_biotype"),
            exon_number=int(_first(a, "exon_number") or 0),
            tags=tuple(a.get("tag", ())),
        )


def _first(a: dict[str, list[str]], key: str, default: str = "") -> str:
    v = a.get(key)
    return v[0] if v else default


def _biotypes_match(a: dict[str, list[str]], cfg: FunnelConfig) -> bool:
    return (_first(a, "gene_biotype") in cfg.gene_biotypes
            and _first(a, "transcript_biotype") in cfg.transcript_biotypes)


# ---------------------------------------------------------------- BED

def merge_padded(exons: Iterable[tuple[int, int, str]], pad: int) -> list[tuple[int, int, str]]:
    """Pad 1-based closed ``(start, end, name)`` exons, convert to 0-based half-open,
    sort, and merge overlapping or touching intervals. The merged name is the unique
    names comma-joined in coordinate order (a gene contributes its name once however
    many of its exons merge)."""
    padded = sorted((max(0, s - 1 - pad), e + pad, n) for s, e, n in exons)
    out: list[tuple[int, int, list[str]]] = []
    for s, e, n in padded:
        if out and s <= out[-1][1]:
            ps, pe, names = out[-1]
            if n not in names:
                names.append(n)
            out[-1] = (ps, max(pe, e), names)
        else:
            out.append((s, e, [n]))
    return [(s, e, ",".join(names)) for s, e, names in out]


@dataclass(frozen=True)
class FunnelStats:
    n_genes: int
    n_transcripts: int
    n_exons: int
    """Distinct (chrom, start, end) exon coordinates before padding."""
    n_exon_lines: int
    """Qualifying exon lines, i.e. exon × transcript pairs."""
    n_intervals: int
    total_bp: int
    genome_fraction: float
    dropped_non_primary: int
    gtf_lines: int


def manifest_path_for(out_bed: Path) -> Path:
    return Path(str(out_bed) + ".manifest.json")


def build_funnel_bed(
    gtf_gz: Path,
    out_bed: Path,
    cfg: FunnelConfig,
    *,
    progress: Progress | None = None,
) -> FunnelStats:
    """Write the merged, padded, sorted funnel BED (plus ``<out_bed>.manifest.json``) from the GTF.

    The file must carry the pinned checksum — the BED's provenance is the whole point.
    Same file and config give byte-identical BED output. Raises before touching
    ``out_bed`` if nothing qualifies (a misspelled biotype would otherwise yield an
    empty funnel and a retrieval that quietly sends no variant anywhere).
    """
    gtf_gz, out_bed = Path(gtf_gz), Path(out_bed)
    m = Manifest(stage="funnel")
    m.add_input("gtf", gtf_gz)
    observed = bsd_sum(gtf_gz)
    if observed != cfg.checksum:
        raise ValueError(f"{gtf_gz} has BSD sum {observed!r} but the config pins {cfg.checksum!r}")
    sidecar = sidecar_path(gtf_gz)
    if sidecar.exists():
        m.inputs["checksums"] = json.loads(sidecar.read_text(encoding="utf-8"))

    scan = ScanCounts()
    per_chrom: dict[str, set[tuple[int, int, str]]] = {}
    genes: set[str] = set()
    transcripts: set[str] = set()
    with gzip.open(gtf_gz, "rt", encoding="utf-8") as f:
        for ex in iter_exons(f, cfg, scan, progress=progress):
            per_chrom.setdefault(ex.chrom, set()).add((ex.start, ex.end, ex.gene_name))
            genes.add(ex.gene_id)
            transcripts.add(ex.transcript_id)
    if scan.kept == 0:
        raise ValueError(
            f"no exon in {gtf_gz} matched gene_biotypes={cfg.gene_biotypes} "
            f"transcript_biotypes={cfg.transcript_biotypes} gene_names_always={cfg.gene_names_always}: "
            "an empty funnel would send nothing to any source"
        )

    n_intervals = total_bp = 0
    out_bed.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bed, "w", encoding="utf-8") as out:
        for chrom in sorted(per_chrom, key=PRIMARY.index):
            for s, e, name in merge_padded(per_chrom[chrom], cfg.pad_bp):
                out.write(f"{chrom}\t{s}\t{e}\t{name}\n")
                n_intervals += 1
                total_bp += e - s

    stats = FunnelStats(
        n_genes=len(genes),
        n_transcripts=len(transcripts),
        n_exons=sum(len({(s, e) for s, e, _n in v}) for v in per_chrom.values()),
        n_exon_lines=scan.kept,
        n_intervals=n_intervals,
        total_bp=total_bp,
        genome_fraction=round(total_bp / cfg.genome_length, 6),
        dropped_non_primary=scan.dropped_non_primary,
        gtf_lines=scan.lines,
    )
    m.params.update({
        "release": cfg.release,
        "url": cfg.url,
        "checksums_url": cfg.checksums_url,
        "checksum_format": "bsd_sum",
        "checksum_pinned": cfg.checksum,
        "checksum_observed": observed,
        "config": asdict(cfg),
        "gtf_header": scan.header,
        "bed_format": "0-based half-open; canonical chromosomes 1..22,X,Y,MT; "
                      "name = gene_name or gene_id, comma-joined when merged",
    })
    m.counts.update(asdict(stats))
    m.counts["gtf_exon_lines"] = scan.exon_lines
    m.counts["n_exon_lines_by_name"] = scan.kept_by_name
    m.add_output("bed", out_bed)
    m.write(manifest_path_for(out_bed))
    if progress:
        progress(f"funnel: {stats.n_intervals:,} intervals, {stats.total_bp:,} bp "
                 f"({stats.genome_fraction:.2%} of the genome) from {stats.n_transcripts:,} transcripts")
    return stats
