"""ClinVar, from a release-pinned local VCF.

Why a downloaded release rather than the E-utilities API: the API is a live index
(rebuilt daily, not pinnable), it is position-indexed rather than allele-indexed, and
its canonical-SPDI search conflates haplotypes with the simple allele they contain.
The weekly GRCh38 VCF is one file with one row per VariationID, an md5 beside it and
a ``##fileDate`` inside it — the same bytes give the same answer forever, and every
lookup is a local ``bcftools query``.

Matching is exact on CHROM + POS + REF + ALT after canonicalising the chromosome. A
region query returns every record whose span touches the position — large deletions
that merely overlap, other ALTs at the same POS — and none of those is the variant
asked about. ``CLNHGVS`` is never used for matching: it is 3'-shifted for indels while
the VCF row is left-anchored (CFTR F508del is the row 7:117559590 ATCT>A but
CLNHGVS ``g.117559592_117559594del``).

Three ways a lookup can silently look like "absent from ClinVar" (empty output,
exit 0), and what guards each:

* chromosome spelling — queries use the file's own sequence names, read from its
  index (the raw ClinVar VCF declares no ``##contig`` lines; htslib synthesises them
  from the ``.tbi``), so there is nothing to misspell. A primary chromosome missing
  from the index has no rows, and a key on it is simply absent.
* lower-case alleles — upper-cased for matching; the query keeps what was asked.
* an index that belongs to a different file — nothing at query time can tell. The
  release pin is the guard: :meth:`ClinvarRelease.download` verifies the VCF md5 and
  the ``.tbi`` size, and :class:`ClinvarRetriever` re-hashes the VCF at start-up
  against the md5 recorded at download.

Why ``urllib`` rather than :class:`~engine.retrieve.http.Http` for the release
itself: it is a 190 MB binary stream (Http keeps response text in a JSON cache) and
its ``Last-Modified`` header is needed, which Http drops. Only the directory listing
goes through Http. ``retrieved_at`` is therefore not a lookup time but the
``Last-Modified`` NCBI serves for the release file — the instant the source stamped
on these bytes, checkable with one HEAD — recorded once by ``download()`` in a
``.retrieved.json`` sidecar beside the VCF. Every copy of the release (cp, rsync,
checkout) then yields byte-identical records, and a lookup over ``https://`` agrees
with one over the downloaded file.

The query positions are handed to bcftools on stdin, never through a file: nothing
of a lookup touches disk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import yaml

from engine.contigs import canonical
from engine.retrieve.http import DEFAULT_USER_AGENT, Http
from engine.retrieve.store import EvidenceRecord, VariantKey, key_str

log = logging.getLogger(__name__)

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/"
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "clinvar.yaml"
VARIATION_URL = "https://www.ncbi.nlm.nih.gov/clinvar/variation/{id}/"
RETRIEVED_SUFFIX = ".retrieved.json"
"""Sidecar written beside the VCF by :meth:`ClinvarRelease.download`: ``retrieved_at``
(what every record cites), ``last_modified`` (the raw header), ``recorded_at``,
``url`` and ``md5`` (the hash the download verified, re-checked at start-up)."""

# INFO tags copied into every payload — enough to re-check every extracted column and
# the haplotype-inclusion and cross-reference tags a reader may want next. A tag absent
# from an older release's header is emitted as '.'.
INFO_TAGS = (
    "ALLELEID", "CLNSIG", "CLNREVSTAT", "CLNSIGCONF", "CLNSIGSCV", "CLNSIGINCL",
    "CLNDN", "CLNDISDB", "CLNDNINCL", "CLNHGVS", "CLNVC", "CLNVCSO", "CLNVI",
    "GENEINFO", "MC", "ORIGIN", "RS", "ONC", "ONCREVSTAT", "ONCCONF", "SCI", "SCIREVSTAT",
)
_SITE = ("CHROM", "POS", "ID", "REF", "ALT")

# Review status → stars, per https://www.ncbi.nlm.nih.gov/clinvar/docs/review_status/,
# in the VCF's spelling (every value observed in the 2026-09-05 release is covered).
# Spellings from before the 2024 "interpretation" → "classification" rename are kept
# so an older pinned release still gets stars.
STARS: dict[str, int] = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_conflicting_classifications": 1,
    "criteria_provided,_conflicting_interpretations": 1,
    "criteria_provided,_single_submitter": 1,
    "no_assertion_criteria_provided": 0,
    "no_classification_provided": 0,
    "no_assertion_provided": 0,
    "no_classification_for_the_single_variant": 0,
    "no_interpretation_for_the_single_variant": 0,
    "no_classifications_from_unflagged_records": 0,
}

PATHOGENICITY_CLASSES = (
    "Pathogenic", "Likely_pathogenic", "Uncertain_significance", "Likely_benign", "Benign",
    "Conflicting_classifications_of_pathogenicity",
)
"""The members that make a CLNSIG term a germline pathogenicity classification; a
``VUS-<tier>`` member counts too."""
_CLASS_ALIASES = {"Conflicting_interpretations_of_pathogenicity": "Conflicting_classifications_of_pathogenicity"}

STREAM_THRESHOLD = 100_000
"""Above this many positions, stream the whole file (``-T``) instead of index seeks
(``-R``): one sequential pass beats a seek per position."""


def stars(revstat: str) -> str:
    """``CLNREVSTAT`` → star count as a string, or '' for an unknown/absent status."""
    n = STARS.get(revstat)
    return "" if n is None else str(n)


def pathogenicity(clnsig: str) -> str:
    """The germline pathogenicity term of a ``CLNSIG`` value, verbatim, or ''.

    CLNSIG joins distinct aggregate classifications with ``|``
    (``Pathogenic|drug_response``); inside one term, ``/`` joins the members of a
    combined category (``Pathogenic/Likely_pathogenic``) and ``,_`` hangs a qualifier
    (``Pathogenic,_low_penetrance``). The term with a pathogenicity class or a VUS tier
    among its members is returned as written: a combined or qualified category is a
    different claim from its strongest member, so it is never collapsed. Terms that
    are not a pathogenicity classification (``drug_response``, risk alleles,
    ``not_provided``, ``.``) yield ''. The pre-2024 "interpretations" spelling of the
    conflicting term is normalised to the current one.
    """
    for term in clnsig.split("|"):
        term = _CLASS_ALIASES.get(term, term)
        members = [m.split(",")[0] for m in term.split("/")]
        if any(m in PATHOGENICITY_CLASSES or m.startswith("VUS-") for m in members):
            return term
    return ""


# ---------------------------------------------------------------- release download

@dataclass(frozen=True)
class ClinvarRelease:
    """A pinned ClinVar VCF release: where to fetch it and what its bytes must hash to."""

    date: str
    """``YYYYMMDD`` as in the filename."""
    url: str
    md5: str
    bytes: int
    tbi_bytes: int | None = None
    fallback_dirs: tuple[str, ...] = ("weekly/", "archive_2.0/{year}/")
    """Sibling directories of ``url`` that keep the same file after the top-level
    dated file has rotated; ``{year}`` is filled from ``date``."""

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> "ClinvarRelease":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ClinVar release pin not found: {path} (configs/clinvar.yaml in the source tree)")
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        return cls(
            date=str(raw["date"]), url=raw["url"], md5=raw["md5"], bytes=int(raw["bytes"]),
            tbi_bytes=int(raw["tbi_bytes"]) if raw.get("tbi_bytes") else None,
            fallback_dirs=tuple(raw.get("fallback_dirs") or ()),
        )

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[1]

    def candidate_urls(self) -> list[str]:
        base = self.url.rsplit("/", 1)[0] + "/"
        return [self.url] + [base + d.format(year=self.date[:4]) + self.filename for d in self.fallback_dirs]

    def resolve_url(self) -> str:
        """First candidate URL that answers a HEAD with 200 — the pinned file or, once
        the top-level file has rotated, its copy in weekly/ or the archive."""
        return self._resolve()[0]

    def _resolve(self) -> tuple[str, str]:
        """(url, Last-Modified) of the first candidate that is there."""
        for url in self.candidate_urls():
            last_modified = _head(url)
            if last_modified is not None:
                return url, last_modified
        raise FileNotFoundError(f"{self.filename} not found at any of {self.candidate_urls()}")

    def download(self, dest_dir: Path) -> Path:
        """Fetch the VCF, its .tbi, NCBI's .md5 sidecar and the ``.retrieved.json``
        sidecar into ``dest_dir``; return the VCF path.

        Skipped when the VCF is already there with the pinned md5 and size and the .tbi
        has the pinned size (a missing ``.retrieved.json`` beside such a copy is rebuilt
        from one HEAD). A download whose md5 or size disagrees with the pin is deleted
        and raises — the pin is the contract. The small .tbi is fetched first, so a
        candidate directory that lacks it is skipped before the big transfer.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        vcf = dest_dir / self.filename
        tbi = dest_dir / (self.filename + ".tbi")
        sidecar = dest_dir / (self.filename + RETRIEVED_SUFFIX)
        if self._verified(vcf, tbi):
            log.info("clinvar release %s already present and md5-verified", self.date)
            if not sidecar.exists():
                url, last_modified = self._resolve()
                _write_retrieved(sidecar, url=url, last_modified=last_modified, md5=self.md5)
            return vcf
        for url in self.candidate_urls():
            try:
                log.info("clinvar release %s: downloading %s", self.date, url)
                _stream(url + ".tbi", tbi, expect_md5=None, expect_bytes=self.tbi_bytes)
                last_modified = _stream(url, vcf, expect_md5=self.md5, expect_bytes=self.bytes)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue
                raise
            except ValueError:
                tbi.unlink(missing_ok=True)  # the pin failed: leave no half-release behind
                raise
            self._check_sidecar(url + ".md5", dest_dir / (self.filename + ".md5"))
            _write_retrieved(sidecar, url=url, last_modified=last_modified, md5=self.md5)
            return vcf
        raise FileNotFoundError(f"{self.filename} not found at any of {self.candidate_urls()}")

    def _verified(self, vcf: Path, tbi: Path) -> bool:
        if not (vcf.exists() and tbi.exists()):
            return False
        tbi_size = tbi.stat().st_size
        tbi_ok = tbi_size == self.tbi_bytes if self.tbi_bytes is not None else tbi_size > 0
        return tbi_ok and vcf.stat().st_size == self.bytes and _md5_file(vcf) == self.md5

    def _check_sidecar(self, url: str, dest: Path) -> None:
        """NCBI's own .md5 must agree with the pin; a 404 for the sidecar is tolerated."""
        try:
            _stream(url, dest, expect_md5=None, expect_bytes=None)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return
            raise
        observed = dest.read_text().split()[0]
        if observed != self.md5:
            raise ValueError(f"pinned md5 {self.md5} differs from NCBI sidecar {observed} at {url}")

    @classmethod
    def latest_filename(cls, http: Http, base_url: str = BASE_URL) -> str:
        """Newest ``clinvar_YYYYMMDD.vcf.gz`` in the release directory listing, for
        re-pinning — always looked up live, since a cached listing would name the
        newest release of the day it was cached forever."""
        resp = http.get(base_url, cache_ok=False)
        found = re.findall(r'href="(clinvar_(\d{8})\.vcf\.gz)"', resp.text)
        if not found:
            raise RuntimeError(f"no clinvar_YYYYMMDD.vcf.gz in the listing at {base_url}")
        return max(found, key=lambda t: t[1])[0]


def _request(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, method=method, headers={"User-Agent": DEFAULT_USER_AGENT})


def _head(url: str) -> str | None:
    """HEAD ``url``: its ``Last-Modified`` header ('' when the server sends none), or
    ``None`` when the file is not there (404)."""
    try:
        with urllib.request.urlopen(_request(url, "HEAD"), timeout=60) as r:
            return r.headers.get("Last-Modified") or ""
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _stream(url: str, dest: Path, *, expect_md5: str | None, expect_bytes: int | None) -> str:
    """Stream ``url`` to ``dest`` via a ``.part`` file, verifying before the rename.
    Returns the response's ``Last-Modified`` header ('' when absent)."""
    part = dest.with_name(dest.name + ".part")
    h = hashlib.md5()
    n = 0
    with urllib.request.urlopen(_request(url), timeout=120) as r, open(part, "wb") as out:
        last_modified = r.headers.get("Last-Modified") or ""
        while chunk := r.read(1 << 20):
            out.write(chunk)
            h.update(chunk)
            n += len(chunk)
    if expect_bytes is not None and n != expect_bytes:
        part.unlink()
        raise ValueError(f"{url}: got {n} bytes, pinned {expect_bytes}")
    if expect_md5 is not None and h.hexdigest() != expect_md5:
        part.unlink()
        raise ValueError(f"{url}: md5 {h.hexdigest()} differs from pinned {expect_md5}")
    part.replace(dest)
    return last_modified


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _iso_utc(http_date: str) -> str:
    return parsedate_to_datetime(http_date).astimezone(timezone.utc).isoformat(timespec="seconds")


def _write_retrieved(sidecar: Path, *, url: str, last_modified: str | None, md5: str) -> None:
    """Record what the download observed, so ``retrieved_at`` survives any copy of the
    file. Without a ``Last-Modified`` from the server the recording instant is used —
    still observed once and persisted, so still stable."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {
        "retrieved_at": _iso_utc(last_modified) if last_modified else now,
        "last_modified": last_modified or "",
        "recorded_at": now,
        "url": url,
        "md5": md5,
    }
    sidecar.write_text(json.dumps(meta, sort_keys=True, indent=1) + "\n")


# ---------------------------------------------------------------- retriever

@dataclass(frozen=True)
class _Header:
    contigs: tuple[str, ...]
    info: frozenset[str]
    file_date: str | None
    reference: str | None


def _read_header(vcf: str, cwd: str | None) -> _Header:
    out = subprocess.run(["bcftools", "view", "-h", vcf], capture_output=True, text=True, check=False, cwd=cwd)
    if out.returncode != 0:
        raise RuntimeError(f"bcftools view -h failed ({out.returncode}): {out.stderr.strip()[:300]}")
    contigs: list[str] = []
    info: set[str] = set()
    file_date = reference = None
    for line in out.stdout.splitlines():
        if line.startswith("##contig=<ID="):
            contigs.append(line[13:].split(",", 1)[0].rstrip(">"))
        elif line.startswith("##INFO=<ID="):
            info.add(line[11:].split(",", 1)[0])
        elif line.startswith("##fileDate="):
            file_date = line[11:].strip()
        elif line.startswith("##reference="):
            reference = line[12:].strip()
    return _Header(tuple(contigs), frozenset(info), file_date, reference)


def _release_date(filename: str, file_date: str | None) -> str:
    """``YYYY-MM-DD`` from the dated filename, cross-checked against ``##fileDate``."""
    m = re.search(r"clinvar_(\d{4})(\d{2})(\d{2})", filename)
    from_name = f"{m[1]}-{m[2]}-{m[3]}" if m else None
    if from_name and file_date and from_name != file_date:
        raise ValueError(f"{filename} is named for {from_name} but its header says fileDate={file_date}")
    date = from_name or file_date
    if not date:
        raise ValueError(f"cannot determine the ClinVar release date of {filename} (no dated name, no ##fileDate)")
    return date


def _retrieved_at(vcf: str, remote: bool) -> str:
    """See the module docstring. Remote: the HEAD ``Last-Modified``. Local: the
    ``.retrieved.json`` sidecar, whose md5 the file must still hash to; a copy without
    a sidecar falls back to its mtime, which a plain ``cp`` changes — logged."""
    if remote:
        last_modified = _head(vcf)
        if not last_modified:
            raise RuntimeError(f"{vcf}: no Last-Modified header (or not found); cannot date the release copy")
        return _iso_utc(last_modified)
    sidecar = Path(vcf + RETRIEVED_SUFFIX)
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        observed = _md5_file(Path(vcf))
        if observed != meta.get("md5"):
            raise ValueError(f"{Path(vcf).name}: md5 {observed} differs from {meta.get('md5')} recorded at download "
                             f"in {sidecar.name}; the file is not the release that was downloaded")
        return str(meta["retrieved_at"])
    log.warning("%s: no %s sidecar (not fetched by ClinvarRelease.download?); retrieved_at falls back to the "
                "file mtime, which any copy changes", Path(vcf).name, RETRIEVED_SUFFIX)
    return datetime.fromtimestamp(Path(vcf).stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


class ClinvarRetriever:
    """Exact-allele ClinVar lookups against one release-pinned GRCh38 VCF.

    ``vcf_path`` is a local, indexed ``clinvar_YYYYMMDD.vcf.gz`` (see
    :meth:`ClinvarRelease.download`); an ``https://`` URL also works through htslib's
    range requests, for live checks — its index copy goes to a private temporary
    directory, not the working directory.
    """

    source = "clinvar"
    columns = (
        "clinvar_vcv", "clinvar_clnsig", "clinvar_revstat", "clinvar_stars", "clinvar_conditions",
        "clinvar_alleleid", "clinvar_hgvs", "clinvar_pathogenicity", "clinvar_release",
    )

    def __init__(self, vcf_path: str | Path):
        self.vcf = str(vcf_path)
        self.remote = "://" in self.vcf
        self.filename = self.vcf.rsplit("/", 1)[-1]
        self._cwd: str | None = None
        if self.remote:
            # htslib saves a remote file's index into the working directory.
            self._index_dir = tempfile.TemporaryDirectory(prefix="clinvar-remote-index-")
            self._cwd = self._index_dir.name
        else:
            p = Path(self.vcf)
            if not p.exists():
                raise FileNotFoundError(f"ClinVar VCF not found: {p}")
            if not any(p.with_name(p.name + ext).exists() for ext in (".tbi", ".csi")):
                raise FileNotFoundError(f"ClinVar VCF is not indexed (no .tbi/.csi beside it): {p}")
        hdr = _read_header(self.vcf, self._cwd)
        if not hdr.reference or "grch38" not in hdr.reference.lower():
            raise ValueError(f"{self.filename}: ##reference={hdr.reference!r}, expected a GRCh38 ClinVar VCF")
        self.release_date = _release_date(self.filename, hdr.file_date)
        # canonical name → the file's own spelling, from its header or (the real
        # release declares none) the ##contig lines htslib synthesises from the index.
        self._spelling: dict[str, str] = {}
        for c in hdr.contigs:
            canon = canonical(c)
            if canon is not None:
                self._spelling.setdefault(canon, c)
        if not self._spelling:
            raise ValueError(f"{self.filename}: no primary chromosome in the header or index")
        self._tags_present = frozenset(t for t in INFO_TAGS if t in hdr.info)
        self.retrieved_at = _retrieved_at(self.vcf, self.remote)

    def version(self) -> str:
        return f"ClinVar {self.release_date} GRCh38 VCF"

    # -- retrieve

    def retrieve(self, keys: list[VariantKey]) -> dict[VariantKey, list[EvidenceRecord]]:
        out: dict[VariantKey, list[EvidenceRecord]] = {k: [] for k in keys}
        wanted: dict[VariantKey, list[VariantKey]] = {}
        positions: set[tuple[str, int]] = set()
        for k in dict.fromkeys(keys):  # a key given twice is one lookup and one record
            spelled = self._spelling.get(k[0])
            if spelled is None:
                continue  # no rows on that chromosome in this release: absent
            wanted.setdefault((k[0], k[1], k[2].upper(), k[3].upper()), []).append(k)
            positions.add((spelled, k[1]))
        if not positions:
            return out
        rows = self._query(sorted(positions))
        n_match = 0
        for row in rows:
            if "," in row["ALT"]:
                raise ValueError(f"ClinVar row VCV{row['ID']} is multi-allelic (ALT {row['ALT']}); a release "
                                 "writes one ALT per VariationID, so this file is not one")
            chrom = canonical(row["CHROM"])
            if chrom is None:
                continue
            for k in wanted.get((chrom, int(row["POS"]), row["REF"].upper(), row["ALT"].upper()), ()):
                out[k].append(self._record(k, row))
                n_match += 1
        for recs in out.values():
            recs.sort(key=lambda r: r.record_id)
        log.info("clinvar: %d positions in one bcftools batch → %d rows in range, %d exact-allele matches",
                 len(positions), len(rows), n_match)
        return out

    def _query(self, positions: list[tuple[str, int]]) -> list[dict[str, str]]:
        fmt = "\t".join(f"%{c}" for c in _SITE)
        fmt += "\t" + "\t".join(f"%INFO/{t}" if t in self._tags_present else "." for t in INFO_TAGS) + "\n"
        mode = "-T" if len(positions) > STREAM_THRESHOLD else "-R"
        # Positions go in on stdin ('-'); two tab columns are read as 1-based CHROM POS.
        cmd = ["bcftools", "query", mode, "-", "-f", fmt, self.vcf]
        p = subprocess.run(cmd, input="".join(f"{c}\t{pos}\n" for c, pos in positions),
                           capture_output=True, text=True, check=False, cwd=self._cwd)
        if p.returncode != 0:
            raise RuntimeError(f"bcftools query failed ({p.returncode}): {p.stderr.strip()[:300]}")
        return [dict(zip(_SITE + INFO_TAGS, line.split("\t"))) for line in p.stdout.splitlines()]

    def _record(self, k: VariantKey, row: dict[str, str]) -> EvidenceRecord:
        vid = row["ID"]
        if not vid.isdigit():
            raise ValueError(f"ClinVar row without a numeric VariationID: {vid!r}")
        payload = dict(row)
        payload["clinvar_release"] = self.release_date
        return EvidenceRecord(
            record_id=f"clinvar:VCV{int(vid):09d}",
            source=self.source,
            source_version=self.version(),
            query={"variant": key_str(k), "chrom": k[0], "pos": k[1], "ref": k[2], "alt": k[3],
                   "vcf": self.filename},
            url=VARIATION_URL.format(id=vid),
            retrieved_at=self.retrieved_at,
            payload=payload,
        )

    # -- extract

    def extract(self, records: list[EvidenceRecord]) -> dict[str, str]:
        """Project the first (lowest VCV) record; a second exact match at one allele has
        never been observed in a release and would still be visible in the store."""
        if not records:
            return {c: "" for c in self.columns}
        p = records[0].payload

        def tag(t: str) -> str:
            v = p.get(t, ".")
            return "" if v == "." else v

        return {
            "clinvar_vcv": records[0].record_id.split(":", 1)[1],
            "clinvar_clnsig": tag("CLNSIG"),
            "clinvar_revstat": tag("CLNREVSTAT"),
            "clinvar_stars": stars(tag("CLNREVSTAT")),
            "clinvar_conditions": tag("CLNDN"),
            "clinvar_alleleid": tag("ALLELEID"),
            "clinvar_hgvs": tag("CLNHGVS"),
            "clinvar_pathogenicity": pathogenicity(tag("CLNSIG")),
            "clinvar_release": p.get("clinvar_release", ""),
        }
