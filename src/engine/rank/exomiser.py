"""Exomiser 14.0.0 in Docker: the pin, the per-run configuration, the invocation, and
the parsers for what it writes.

Why Exomiser at all: it is the one ranker here that sees the phenotype and *not* the
gene panel — a blind second opinion on the stage-3 shortlist. So the analysis file it
gets carries the case's HPO terms and nothing else about the case; the template in
``configs/exomiser.yaml`` (inheritance cut-offs, frequency and pathogenicity sources,
filter steps) is fixed and echoed into the manifest.

Why the image is run by digest: the tag ``14.0.0`` could be re-pushed; the manifest-list
digest cannot change under us, and ``docker run repo@digest`` resolves it for whatever
platform the daemon runs (verified: the locally pulled ``14.0.0`` answers to the digest
without a second pull). The observed digest is recorded at run time regardless of what
``--image`` named.

How the container is configured (all verified against the image, see the probe): the
image is distroless (``java -cp @/app/jib-classpath-file …Main``, uid 65532, no shell)
and ships **no** ``application.properties`` — without ``exomiser.data-directory`` and
the two ``*.data-version`` properties it fails at start-up. The engine writes those into
``<config dir>/application.properties`` and passes ``--spring.config.location`` as the
last argument (the documented rule). The heap is set with ``JAVA_TOOL_OPTIONS=-Xmx…``.

Memory, for a whole genome: the Exomiser docs quote 8 GB of RAM for a 4.4M-variant
genome; Docker Desktop on this 8 GiB Mac grants the daemon 3.8 GiB and the container's
default heap is ~1 GB. ``heap: auto`` sizes ``-Xmx`` from what the daemon has *left*:
its total memory minus what the containers already running use (``docker stats``) minus
a 1 GiB reserve — a heap the daemon cannot back starts fine and is OOM-killed mid-run
(exit 137), which is what happened here with a dozen resident database containers when
the heap was sized from the total alone. A second exomiser-cli container on the same
daemon is refused unless ``--heap`` is explicit: its JVM grows into the memory this one
would be sized from. To run a WGS here: raise Docker Desktop → Resources → Memory to at
least 6 GiB (realistically the ceiling on an 8 GiB machine), stop other containers, pass
``--heap 5g``, keep ``analysisMode: PASS_ONLY`` with ``hiPhivePrioritiser`` followed by
``priorityScoreFilter`` (the template does), and if it still dies, ingest with a regions
BED and rank that VCF, or split the VCF by chromosome — gene scores are per-gene and
combinable, but ``#RANK`` and ``P-VALUE`` are run-relative. Reading the ~30 GB H2 variant
store through a bind mount is slow; a genome run is hours, not minutes.

The mitochondrion: Exomiser 14.0.0 resolves contigs through svart, which knows ``MT``
and ``chrM`` but not bare ``M`` — verified live with two VCFs identical but for that
id: with ``MT`` the MT-ND1 m.3243A>G record was ranked, with ``M`` it was silently
dropped (no warning in the log). A VCF that spells it ``M`` is therefore not mounted as
is: :func:`rename_contigs` writes a copy with ``M`` → ``MT`` (``bcftools annotate
--rename-chrs``, header and records, ``--no-version`` so the bytes are reproducible)
into the run directory, the container sees that copy, and both files' sha256 plus the
rename map go into ``run.json`` and the manifest. The copy is removed after the run.

Regions (``--regions <bed>``): a whole genome does not fit a small Docker daemon, and a
gene-blind ranking of the regions a case can act on (the funnel BED) is still gene-blind.
:func:`subset_regions` writes ``04_rank/input.regions.vcf.gz`` — the records overlapping
the BED's intervals (``bcftools view -R``, ``--regions-overlap 1`` so an indel that
starts before an interval and reaches into it is kept, ``--no-version`` so the same
inputs give the same bytes) — renames ``M`` → ``MT`` *in that subset* when the header
spells it so (the genome is never copied), indexes it (``bcftools index -t``) and counts
its records from the index (``bcftools index -n``). The subset and its ``.tbi`` are kept
beside the raw outputs as the file Exomiser saw; the BED's sha256, the record counts
before and after, the rename map and the bcftools lines go into ``run.json`` and the
manifest, with a note that the ranking is restricted, not genome-wide.

Output parsing is by header name (the docs call one column ``CLINVAR_VARIANT_ID``; the
14.0.0 writer spells it ``CLINVAR_VARIATION_ID``). Both TSV writers emit unquoted fields
(``CSVFormat.newFormat('\\t').withQuote(null)``), so the reader here applies no quoting
either — a disease name that starts with a double quote is data, not a quoted field.
MOI is spelled differently per format — ``AD``/``AR``/… in the TSVs,
``AUTOSOMAL_DOMINANT``/… in the JSON — and is normalised to the TSV abbreviation here.
Frequencies in every Exomiser output are percentages. The JSON is parsed with the
shapes the 14.0.0 classes actually serialise (``frequencyData.frequencies[]``,
``pathogenicityData.pathogenicityScores[]``, one ``geneScores`` entry per inheritance
mode, most without contributing variants), not the stale test fixture on the tag.

The JSON is read once (``load_json_genes``) and both maps are derived from the same
object: with ``numGenes: 0`` and ``outputContributingVariantsOnly: false`` a genome's
JSON carries every variant evaluation and is large.

The ``P-VALUE`` column is *not* deterministic: Exomiser estimates it by sampling, and
two runs on identical inputs differ in the JSON's ``pValue`` and can differ in the
TSV's 4-decimal rounding. It is parsed (``GeneRow.p_value``) so a judge can read it
from the raw files, but it is excluded from everything derived — ``ranking.tsv``,
``joined.json`` and the evidence records — so those stay byte-identical across runs.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from engine.contigs import canonical
from engine.retrieve.store import VariantKey, key_str

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "exomiser.yaml"
ANALYSIS_FILENAME = "analysis.yml"
PROPERTIES_FILENAME = "application.properties"

MOI_ABBREV = {
    "AUTOSOMAL_DOMINANT": "AD",
    "AUTOSOMAL_RECESSIVE": "AR",
    "X_DOMINANT": "XD",
    "X_RECESSIVE": "XR",
    "MITOCHONDRIAL": "MT",
    "ANY": "ANY",
}
MOI_ORDER = {m: i for i, m in enumerate(("AR", "AD", "XR", "XD", "MT", "ANY"))}
"""Tie-break when a gene scores the same under two MOIs: recessive first, because the
engine's stage-3 models are biallelic first."""

OUTPUT_FORMATS = ("HTML", "VCF", "TSV_GENE", "TSV_VARIANT", "JSON")

_VERSION_RE = re.compile(r"Prioriti[sz]e Exome Variants\s+v(\d+\.\d+\.\d+)")


# ---------------------------------------------------------------- the pin

@dataclass(frozen=True)
class Bundle:
    name: str
    filename: str
    bytes: int
    md5: str
    sha256: str | None
    crc32c: str | None
    extracted_dir: str
    required_files: tuple[str, ...]
    url: str
    member_sha256: dict[str, str] = field(default_factory=dict)
    """Pinned sha256 per extracted file (relative to ``extracted_dir``), when the
    release ships one — 2406_hg38 carries ``2406_hg38.sha256`` for its six members."""
    checksum_list: str | None = None
    """Name of the ``sha256sum``-style list inside the bundle, verified after extraction."""

    def extracted_path(self, data_dir: Path) -> Path:
        return Path(data_dir) / self.extracted_dir

    def missing_files(self, data_dir: Path) -> list[str]:
        """Required files not present (or empty) under the extracted directory; the
        directory itself counts as a missing file when it is not there."""
        d = self.extracted_path(data_dir)
        if not d.is_dir():
            return [self.extracted_dir + "/"]
        return [f for f in self.required_files if not (d / f).is_file() or (d / f).stat().st_size == 0]


@dataclass(frozen=True)
class ExomiserConfig:
    image_repository: str
    image_tag: str
    image_digest: str
    image_version: str
    platform_digests: dict[str, str]
    data_version: str
    data_base_url: str
    data_last_modified: str
    data_citation: str
    bundles: dict[str, Bundle]
    docker: dict[str, Any]
    analysis: dict[str, Any]
    output_options: dict[str, Any]
    join: dict[str, Any]
    path: Path
    sha256: str

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> "ExomiserConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Exomiser pin not found: {path} (configs/exomiser.yaml in the source tree)")
        text = path.read_text(encoding="utf-8")
        raw: dict[str, Any] = yaml.safe_load(text) or {}
        img, data = raw["image"], raw["data"]
        base = data["base_url"]
        bundles = {
            name: Bundle(
                name=name, filename=b["filename"], bytes=int(b["bytes"]), md5=str(b["md5"]),
                sha256=(str(b["sha256"]) if b.get("sha256") else None), crc32c=b.get("crc32c"),
                extracted_dir=b["extracted_dir"], required_files=tuple(b.get("required_files") or ()),
                url=b.get("url") or (base + b["filename"]),
                member_sha256={str(k): str(v) for k, v in (b.get("member_sha256") or {}).items()},
                checksum_list=b.get("checksum_list") or None,
            )
            for name, b in data["bundles"].items()
        }
        return cls(
            image_repository=img["repository"], image_tag=str(img["tag"]), image_digest=img["digest"],
            image_version=str(img["version"]), platform_digests=dict(img.get("platform_digests") or {}),
            data_version=str(data["version"]), data_base_url=base, data_last_modified=str(data.get("last_modified", "")),
            data_citation=str(data.get("citation", "")).strip(), bundles=bundles,
            docker=dict(raw.get("docker") or {}), analysis=dict(raw["analysis"]),
            output_options=dict(raw.get("outputOptions") or {}), join=dict(raw.get("join") or {}),
            path=path, sha256=hashlib.sha256(text.encode()).hexdigest(),
        )

    @property
    def image_ref(self) -> str:
        """``repo@sha256:…`` — the reference the engine runs by default."""
        return f"{self.image_repository}@{self.image_digest}"

    @property
    def output_filename(self) -> str:
        return str(self.docker.get("output_filename", "exomiser"))

    @property
    def output_formats(self) -> tuple[str, ...]:
        return tuple(str(f).upper() for f in self.output_options.get("outputFormats") or ("JSON",))

    def output_paths(self, results_dir: Path) -> dict[str, Path]:
        """Where the three raw outputs land, keyed ``genes``/``variants``/``json``."""
        base = Path(results_dir) / self.output_filename
        return {
            "genes": base.with_name(base.name + ".genes.tsv"),
            "variants": base.with_name(base.name + ".variants.tsv"),
            "json": base.with_name(base.name + ".json"),
        }

    def missing_data(self, data_dir: Path) -> dict[str, list[str]]:
        """Bundle name → required files missing under ``data_dir``; empty when complete."""
        out = {name: b.missing_files(data_dir) for name, b in self.bundles.items()}
        return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------- per-run files

def render_analysis(
    cfg: ExomiserConfig,
    hpo_ids: list[str],
    vcf_container_path: str,
    results_mount: str,
    output_formats: tuple[str, ...] | None = None,
) -> str:
    """The analysis YAML Exomiser reads. Only ``hpoIds`` and ``vcf`` come from the case;
    every other key is the template verbatim, in template order, so the same case gives
    the same bytes. No gene list, no seed genes, no candidate gene."""
    if not hpo_ids:
        raise ValueError("Exomiser needs at least one HPO term; case.yaml has none")
    formats = tuple(output_formats or cfg.output_formats)
    bad = [f for f in formats if f not in OUTPUT_FORMATS]
    if bad:
        raise ValueError(f"unknown Exomiser output formats {bad}; allowed: {list(OUTPUT_FORMATS)}")
    if "TSV_GENE" not in formats:
        raise ValueError("TSV_GENE must be among the output formats: ranking.tsv is built from it")
    check_template(cfg.analysis)
    analysis: dict[str, Any] = {
        "genomeAssembly": cfg.analysis["genomeAssembly"],
        "vcf": vcf_container_path,
        "ped": None,
        "proband": None,
        "hpoIds": list(hpo_ids),
    }
    for k, v in cfg.analysis.items():
        if k != "genomeAssembly":
            analysis[k] = v
    output = dict(cfg.output_options)
    output["outputDirectory"] = results_mount
    output["outputFileName"] = cfg.output_filename
    output["outputFormats"] = list(formats)
    doc = {"analysis": analysis, "outputOptions": output}
    text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=120)
    # The documented single-sample form is an empty value (``ped:``), which parses to the
    # same null; emit exactly that rather than an explicit ``null``.
    text = text.replace("\n  ped: null\n", "\n  ped:\n").replace("\n  proband: null\n", "\n  proband:\n")
    return "---\n" + text


CASE_KEYS = ("vcf", "ped", "proband", "hpoIds")
"""Analysis keys that describe the case; they come from ``case.yaml`` and may not be in the template."""
GENE_STEERING_STEPS = ("genePanelFilter", "exomeWalkerPrioritiser")
"""Steps that hand Exomiser a gene list or seed genes — the blind ranking forbids them."""


def check_template(analysis: dict[str, Any]) -> None:
    """Refuse a template that would steer the ranking or override the case: the docstring
    of :func:`render_analysis` is enforced here, not assumed."""
    present = [k for k in CASE_KEYS if k in analysis]
    if present:
        raise ValueError(f"analysis template must not carry case keys {present}: they come from case.yaml only")
    for step in analysis.get("steps") or []:
        names = list(step) if isinstance(step, dict) else [str(step)]
        bad = [n for n in names if n in GENE_STEERING_STEPS]
        if bad:
            raise ValueError(f"analysis template step {bad} would steer the ranking with a gene list; not allowed")


def render_properties(cfg: ExomiserConfig) -> str:
    """``application.properties`` for the container: data directory and the two data
    versions — the image ships none. Only hg38 is loaded (each assembly costs ~1 GB)."""
    assembly = str(cfg.analysis["genomeAssembly"]).lower()
    return (
        f"exomiser.data-directory={cfg.docker.get('data_mount', '/exomiser-data')}\n"
        f"exomiser.{assembly}.data-version={cfg.data_version}\n"
        f"exomiser.phenotype.data-version={cfg.data_version}\n"
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------- the VCF, header only

def vcf_header_contigs(vcf: Path) -> tuple[list[str], str | None]:
    """Contig ids declared in the VCF header, and — only when the header declares none —
    the CHROM of the first record. Nothing else of the file is read."""
    opener = gzip.open if vcf.suffix.lower() in (".gz", ".bgz") else open
    contigs: list[str] = []
    with opener(vcf, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        for line in f:
            if line.startswith("##contig=<ID="):
                contigs.append(line[13:].split(",", 1)[0].rstrip(">\n"))
            elif line.startswith("#CHROM"):
                if contigs:
                    return contigs, None
                first = f.readline()
                return contigs, (first.split("\t", 1)[0] if first else None)
            elif not line.startswith("#"):
                break
    return contigs, None


def mito_contig(contigs: list[str]) -> str | None:
    """How the VCF spells the mitochondrion, or ``None`` when it declares none."""
    for c in contigs:
        if canonical(c) == "MT":
            return c
    return None


MITO_RENAME = {"M": "MT"}
"""Contig ids Exomiser cannot resolve → the spelling it can (see the module docstring)."""


def contig_renames(contigs: list[str]) -> dict[str, str]:
    """The renames the VCF needs before Exomiser sees it; empty when none."""
    return {c: MITO_RENAME[c] for c in contigs if c in MITO_RENAME}


def rename_contigs(vcf: Path, renames: dict[str, str], dest: Path, run=subprocess.run, *,
                   mapping: Path | None = None) -> Path:
    """Write ``dest`` (bgzipped) as ``vcf`` with the contig ids in ``renames`` replaced in
    the header and the records. ``bcftools annotate --rename-chrs`` with ``--no-version``,
    so the same input gives the same bytes. The map file stays beside ``dest`` (or at
    ``mapping``)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    mapping = Path(mapping) if mapping else dest.with_name(dest.name + ".rename.tsv")
    mapping.write_text("".join(f"{old}\t{new}\n" for old, new in sorted(renames.items())), encoding="utf-8")
    cmd = ["bcftools", "annotate", "--no-version", "--rename-chrs", str(mapping), "-Oz", "-o", str(dest), str(vcf)]
    out = run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"bcftools annotate --rename-chrs failed (exit {out.returncode}): {(out.stderr or '').strip()[:300]}")
    return dest


# ---------------------------------------------------------------- the VCF, restricted to regions

REGIONS_VCF_FILENAME = "input.regions.vcf.gz"
"""The ``--regions`` subset of the case VCF, in ``04_rank/``; its ``.tbi`` and the rename
map (when one applied) sit beside it. Kept after the run: it is the file Exomiser saw."""

REGIONS_OVERLAP = 1
"""``bcftools --regions-overlap``: a record overlapping an interval counts, not only one
whose POS lies inside it (stage 1 uses the same rule for its regions BED)."""


@dataclass(frozen=True)
class RegionsSubset:
    """What :func:`subset_regions` wrote: the subset, its index, how many records it holds,
    the contig renames applied to it (empty when none) and the bcftools lines that made it."""
    path: Path
    index: Path
    records: int
    renames: dict[str, str]
    commands: list[list[str]]


def subset_regions(vcf: Path, bed: Path, dest: Path, renames: dict[str, str] | None = None, *,
                   regions_overlap: int = REGIONS_OVERLAP, run=subprocess.run) -> RegionsSubset:
    """Write ``dest`` (bgzipped, ``.tbi`` beside it) as the records of ``vcf`` overlapping
    the intervals of ``bed`` — ``bcftools view -R`` over the index beside ``vcf``, with
    ``--no-version`` so the same inputs give the same bytes — then, when ``renames`` is
    non-empty, rewrite the subset with those contig ids replaced (:func:`rename_contigs`,
    on the subset only), index it and read its record count from the index. Nothing of
    ``dest`` survives a failure."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    index = dest.with_name(dest.name + ".tbi")
    mapping = dest.with_name(dest.name + ".rename.tsv")
    renamed = dest.with_name(dest.name + ".renamed.tmp")
    for stale in (dest, index, mapping, renamed):
        stale.unlink(missing_ok=True)
    commands: list[list[str]] = []
    try:
        view = ["bcftools", "view", "--no-version", "-R", str(bed), "--regions-overlap", str(regions_overlap),
                "-Oz", "-o", str(dest), str(vcf)]
        commands.append(view)
        out = run(view, capture_output=True, text=True, check=False)
        if out.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError(f"bcftools view -R failed (exit {out.returncode}): {(out.stderr or '').strip()[:300]}")
        if renames:
            commands.append(["bcftools", "annotate", "--no-version", "--rename-chrs", str(mapping), "-Oz", "-o", str(renamed),
                             str(dest)])
            rename_contigs(dest, renames, renamed, run, mapping=mapping)
            renamed.replace(dest)  # the renamed subset takes the subset's name; the map stays beside it
        idx = ["bcftools", "index", "-t", "-f", str(dest)]
        commands.append(idx)
        out = run(idx, capture_output=True, text=True, check=False)
        if out.returncode != 0 or not index.exists():
            raise RuntimeError(f"bcftools index -t failed on the regions subset (exit {out.returncode}): "
                               f"{(out.stderr or '').strip()[:300]}")
        commands.append(["bcftools", "index", "-n", str(dest)])
        n = vcf_record_count(dest, run)
        if n is None:
            raise RuntimeError(f"bcftools index -n could not count the records of {dest}")
    except Exception:
        for p in (dest, index, mapping, renamed):
            p.unlink(missing_ok=True)
        raise
    return RegionsSubset(path=dest, index=index, records=n, renames=dict(renames or {}), commands=commands)


def vcf_record_count(vcf: Path, run=subprocess.run, *, index: Path | None = None) -> int | None:
    """Records in ``vcf`` as its index counts them (``bcftools index -n``; ``index`` names
    one that is not beside the file). ``None`` when the index carries no counts — a
    provider's tabix index may not — or cannot be read; nothing of the file is read."""
    target = f"{vcf}##idx##{index}" if index else str(vcf)
    out = run(["bcftools", "index", "-n", target], capture_output=True, text=True, check=False)
    text = (out.stdout or "").strip()
    if out.returncode != 0 or not text.isdigit():
        return None
    return int(text)


def count_records(vcf: Path, run=subprocess.run) -> int:
    """Records in ``vcf`` by one pass over it — ``bcftools query -f '\\n'`` prints an empty
    line per record, so a genome costs minutes of reading and a few megabytes, never the
    records themselves. For a VCF whose index carries no counts."""
    out = run(["bcftools", "query", "-f", "\\n", str(vcf)], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"bcftools query failed while counting records (exit {out.returncode}): "
                           f"{(out.stderr or '').strip()[:300]}")
    return (out.stdout or "").count("\n")


def bed_interval_count(bed: Path) -> int:
    """Data lines of a BED — blank and ``#`` lines excluded, as bcftools reads it (a
    ``track`` line is not a comment to bcftools either; it fails on one)."""
    n = 0
    with open(bed, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                n += 1
    return n


# ---------------------------------------------------------------- docker

def docker_info_memory_bytes(run=subprocess.run) -> int | None:
    """Total memory the Docker daemon may use, or ``None`` if the daemon is unreachable."""
    try:
        out = run(["docker", "info", "--format", "{{.MemTotal}}"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    if out.returncode != 0:
        return None
    try:
        n = int(out.stdout.strip())
    except ValueError:
        return None
    return n or None


@dataclass(frozen=True)
class DaemonLoad:
    """What the daemon's running containers use right now (``docker stats``)."""
    in_use_bytes: int | None
    """Sum of the running containers' memory, or ``None`` when it could not be read."""
    containers: tuple[dict[str, Any], ...]
    """``{name, image, mem_bytes}`` per running container (``docker ps`` + ``docker stats``)."""

    @property
    def exomiser_containers(self) -> list[str]:
        """Names of running containers whose image is an exomiser-cli."""
        return [c["name"] for c in self.containers if "exomiser" in str(c.get("image", "")).lower()]


_MEM_UNITS = {"b": 1, "kib": 1 << 10, "mib": 1 << 20, "gib": 1 << 30, "tib": 1 << 40,
              "kb": 10 ** 3, "mb": 10 ** 6, "gb": 10 ** 9, "tb": 10 ** 12}


def parse_mem(text: str) -> int | None:
    """``163MiB`` / ``3.827GiB`` / ``512kB`` (the ``docker stats`` spelling) → bytes."""
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)?\s*$", text)
    if not m:
        return None
    unit = (m.group(2) or "B").lower()
    if unit not in _MEM_UNITS:
        return None
    return int(float(m.group(1)) * _MEM_UNITS[unit])


def docker_load(run=subprocess.run) -> DaemonLoad:
    """Memory already committed on the daemon by other containers, per container. The
    daemon's total (``docker info``) says nothing about this; a dozen resident database
    containers were what killed the first auto-sized runs here."""
    try:
        ps = run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}"], capture_output=True, text=True, check=False)
        stats = run(["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}"],
                    capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return DaemonLoad(None, ())
    if ps.returncode != 0 or stats.returncode != 0:
        return DaemonLoad(None, ())
    images: dict[str, str] = {}
    for line in ps.stdout.splitlines():
        name, _, image = line.partition("\t")
        if name.strip():
            images[name.strip()] = image.strip()
    containers: list[dict[str, Any]] = []
    total = 0
    for line in stats.stdout.splitlines():
        name, _, usage = line.partition("\t")
        if not name.strip():
            continue
        used = parse_mem(usage.split("/", 1)[0])
        containers.append({"name": name.strip(), "image": images.get(name.strip(), ""), "mem_bytes": used})
        total += used or 0
    containers.sort(key=lambda c: c["name"])
    return DaemonLoad(total, tuple(containers))


def heap_bytes(heap: str) -> int | None:
    """A ``-Xmx`` value (``2816m``, ``5g``, ``512k``, bare bytes) → bytes; ``None`` if unparsable."""
    m = re.match(r"^\s*([0-9]+)\s*([kKmMgGtT]?)\s*$", heap)
    if not m:
        return None
    return int(m.group(1)) * {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}[m.group(2).lower()]


def heap_available_bytes(cfg: ExomiserConfig, daemon_bytes: int | None, in_use_bytes: int | None) -> int | None:
    """What the daemon can back for one more JVM: total minus what is in use minus the reserve."""
    if daemon_bytes is None:
        return None
    reserve = int(cfg.docker.get("heap_reserve_mib", 1024)) << 20
    return daemon_bytes - (in_use_bytes or 0) - reserve


def heap_for(cfg: ExomiserConfig, daemon_bytes: int | None, in_use_bytes: int | None = None) -> str:
    """``-Xmx`` value: the daemon's free memory (total minus the running containers'
    usage minus the reserve), floored at ``heap_min_mib``, rounded down to 256 MiB."""
    setting = str(cfg.docker.get("heap", "auto"))
    if setting != "auto":
        return setting
    available = heap_available_bytes(cfg, daemon_bytes, in_use_bytes)
    if available is None:
        return str(cfg.docker.get("heap_fallback", "2g"))
    floor = int(cfg.docker.get("heap_min_mib", 1024))
    mib = max(floor, available // (1 << 20))
    mib -= mib % 256
    return f"{mib}m"


def image_digest(image: str, run=subprocess.run) -> dict[str, Any]:
    """What the daemon has for ``image``: repo digests, image id, platform. Raises when
    the image is not present locally — pulling is left to the caller."""
    out = run(["docker", "image", "inspect", image, "--format",
               '{"repo_digests": {{json .RepoDigests}}, "id": {{json .Id}}, "os": {{json .Os}}, "arch": {{json .Architecture}}}'],
              capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"docker image inspect {image} failed: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout.strip().splitlines()[0])


def docker_command(
    cfg: ExomiserConfig,
    *,
    image: str,
    data_dir: Path,
    config_dir: Path,
    results_dir: Path,
    vcf: Path,
    vcf_container_name: str,
    heap: str,
) -> list[str]:
    """The ``docker run`` line. Data, config and the VCF are mounted read-only; the VCF
    (and its index, when beside it) is bind-mounted as a single file so nothing else in
    its directory is visible to the container."""
    d = cfg.docker
    data_mount = d.get("data_mount", "/exomiser-data")
    config_mount = d.get("config_mount", "/exomiser")
    results_mount = d.get("results_mount", "/results")
    vcf_mount = d.get("vcf_mount", "/vcf")
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{Path(data_dir).resolve()}:{data_mount}:ro",
        "-v", f"{Path(config_dir).resolve()}:{config_mount}:ro",
        "-v", f"{Path(results_dir).resolve()}:{results_mount}",
        "-v", f"{Path(vcf).resolve()}:{vcf_mount}/{vcf_container_name}:ro",
    ]
    for ext in (".tbi", ".csi"):
        idx = vcf.with_name(vcf.name + ext)
        if idx.exists():
            cmd += ["-v", f"{idx.resolve()}:{vcf_mount}/{vcf_container_name}{ext}:ro"]
            break
    cmd += [
        "-e", f"JAVA_TOOL_OPTIONS=-Xmx{heap}",
        image,
        "--analysis", f"{config_mount}/{ANALYSIS_FILENAME}",
        "--spring.config.location=" + f"{config_mount}/{PROPERTIES_FILENAME}",  # must be last
    ]
    return cmd


def container_vcf_name(vcf: Path) -> str:
    """A fixed name inside the container: the proband's file name never appears in the
    analysis file. The compression suffix is kept so htsjdk picks the right reader."""
    name = vcf.name.lower()
    if name.endswith(".vcf.gz") or name.endswith(".vcf.bgz"):
        return "case.vcf.gz"
    return "case.vcf"


def version_from_log(text: str) -> str | None:
    """``v14.0.0`` from the banner Exomiser prints on every real run."""
    m = _VERSION_RE.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------- output parsing

@dataclass(frozen=True)
class GeneRow:
    rank: int
    gene_symbol: str
    entrez_id: str
    moi: str
    combined_score: float
    phenotype_score: float
    variant_score: float
    p_value: float | None
    raw: dict[str, str]


@dataclass(frozen=True)
class VariantRow:
    key: VariantKey
    gene_symbol: str
    moi: str
    contributing: bool
    variant_score: float | None
    genotype: str
    raw: dict[str, str]


def _rows(path: Path) -> Iterator[dict[str, str]]:
    """Rows of an Exomiser TSV as dicts keyed by header name (the header starts ``#RANK``)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path}: empty (no header)")
        if not header or header[0] != "#RANK":
            raise ValueError(f"{path}: not an Exomiser TSV (header starts {header[:1]!r}, expected '#RANK')")
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"{path}: row with {len(row)} fields, header has {len(header)} — truncated output?")
            yield dict(zip(header, row))


def _float(s: str | None) -> float | None:
    if s in (None, "", "NA", "."):
        return None
    return float(s)


def parse_genes_tsv(path: Path) -> list[GeneRow]:
    out = []
    for r in _rows(path):
        out.append(GeneRow(
            rank=int(r["#RANK"]), gene_symbol=r["GENE_SYMBOL"], entrez_id=r.get("ENTREZ_GENE_ID", ""),
            moi=r["MOI"], combined_score=float(r["EXOMISER_GENE_COMBINED_SCORE"]),
            phenotype_score=float(r["EXOMISER_GENE_PHENO_SCORE"]), variant_score=float(r["EXOMISER_GENE_VARIANT_SCORE"]),
            p_value=_float(r.get("P-VALUE")), raw=r,
        ))
    return out


def parse_variants_tsv(path: Path) -> list[VariantRow]:
    out = []
    for r in _rows(path):
        chrom = canonical(r["CONTIG"])
        if chrom is None:
            continue  # an unplaced contig; nothing on it is in the engine's tables
        key: VariantKey = (chrom, int(r["START"]), r["REF"], r["ALT"])
        out.append(VariantRow(
            key=key, gene_symbol=r["GENE_SYMBOL"], moi=r["MOI"], contributing=r.get("CONTRIBUTING_VARIANT") == "1",
            variant_score=_float(r.get("EXOMISER_VARIANT_SCORE")), genotype=r.get("GENOTYPE", ""), raw=r,
        ))
    return out


def _moi_abbrev(moi: str) -> str:
    return MOI_ABBREV.get(moi, moi)


JsonGenes = list[dict[str, Any]]
"""The top-level array of Exomiser's JSON output: one object per gene."""


def load_json_genes(path: Path) -> JsonGenes:
    """Parse the JSON output once; the callers below take the parsed list (or a path,
    for convenience in tests) so a genome's JSON is not decoded twice."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, list):
        raise ValueError(f"{path}: not an Exomiser JSON output (expected a top-level array of genes)")
    return doc


def _genes(source: Path | JsonGenes) -> JsonGenes:
    return source if isinstance(source, list) else load_json_genes(source)


def contributing_from_json(source: Path | JsonGenes) -> dict[tuple[str, str], list[VariantKey]]:
    """``(gene_symbol, MOI abbrev) → contributing variant keys`` from the JSON output —
    the fallback when the variants TSV is absent (the 14.0.0 TSV_VARIANT writer can die
    on a variant whose only frequencies are 0.0, issue #565)."""
    genes = _genes(source)
    out: dict[tuple[str, str], list[VariantKey]] = {}
    for g in genes:
        for gs in g.get("geneScores", []):
            moi = _moi_abbrev(gs.get("modeOfInheritance", ""))
            keys = []
            for v in gs.get("contributingVariants", []):
                chrom = canonical(str(v.get("contigName", "")))
                if chrom is None:
                    continue
                keys.append((chrom, int(v["start"]), v["ref"], v["alt"]))
            if not keys:
                continue  # 14.0.0 lists every MOI; one without contributing variants says nothing
            keys.sort(key=lambda k: (_chrom_order(k[0]), k[1], k[2], k[3]))
            out[(g["geneSymbol"], moi)] = keys
    return out


def gene_identifiers_from_json(source: Path | JsonGenes) -> dict[str, dict[str, str]]:
    """``gene_symbol → {ensembl_id, entrez_id, hgnc_id}`` from the JSON output, so a
    stage-3 candidate can be matched on its Ensembl id when the symbols disagree
    (Exomiser 2406 carries Ensembl 111 symbols; VEP in stage 2 is newer)."""
    genes = _genes(source)
    out: dict[str, dict[str, str]] = {}
    for g in genes:
        gi = g.get("geneIdentifier") or {}
        out[g["geneSymbol"]] = {
            "ensembl_id": str(gi.get("ensemblId") or ""),
            "entrez_id": str(gi.get("entrezId") or ""),
            "hgnc_id": str(gi.get("hgncId") or ""),
        }
    return out


@dataclass(frozen=True)
class GeneRanking:
    """One gene, best MOI first, with every MOI's scores kept."""
    rank: int
    gene_symbol: str
    entrez_id: str
    moi: str
    combined_score: float
    phenotype_score: float
    variant_score: float
    variants: tuple[str, ...]
    """Contributing variants under the best MOI, as ``chrom:pos:ref:alt``."""
    by_moi: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank, "gene_symbol": self.gene_symbol, "entrez_id": self.entrez_id, "moi": self.moi,
            "exomiser_score": self.combined_score, "phenotype_score": self.phenotype_score,
            "variant_score": self.variant_score, "n_variants": len(self.variants), "variants": list(self.variants),
            "by_moi": self.by_moi,
        }

    def all_variants(self) -> set[str]:
        """Contributing variants under any MOI."""
        return {v for d in self.by_moi.values() for v in d["variants"]}


def build_ranking(genes: list[GeneRow], contributing: dict[tuple[str, str], list[VariantKey]]) -> list[GeneRanking]:
    """Collapse the per-gene-per-MOI rows to one entry per gene: the best-scoring MOI
    (ties: recessive first) carries the gene's rank; the others are kept in ``by_moi``.
    Output order is Exomiser's — by rank, then symbol for the ties."""
    by_gene: dict[str, list[GeneRow]] = {}
    for g in genes:
        by_gene.setdefault(g.gene_symbol, []).append(g)
    out = []
    for symbol, rows in by_gene.items():
        rows = sorted(rows, key=lambda r: (-r.combined_score, r.rank, MOI_ORDER.get(r.moi, 99), r.moi))
        best = rows[0]
        # No p_value here: it is sampled per run (see the module docstring) and would
        # make joined.json and the evidence records differ between identical runs.
        by_moi = {
            r.moi: {
                "rank": r.rank, "exomiser_score": r.combined_score, "phenotype_score": r.phenotype_score,
                "variant_score": r.variant_score,
                "variants": [key_str(k) for k in contributing.get((symbol, r.moi), [])],
            }
            for r in rows
        }
        out.append(GeneRanking(
            rank=best.rank, gene_symbol=symbol, entrez_id=best.entrez_id, moi=best.moi,
            combined_score=best.combined_score, phenotype_score=best.phenotype_score, variant_score=best.variant_score,
            variants=tuple(key_str(k) for k in contributing.get((symbol, best.moi), [])), by_moi=by_moi,
        ))
    out.sort(key=lambda g: (g.rank, -g.combined_score, g.gene_symbol))
    return out


def contributing_from_tsv(variants: list[VariantRow]) -> dict[tuple[str, str], list[VariantKey]]:
    out: dict[tuple[str, str], list[VariantKey]] = {}
    for v in variants:
        if v.contributing:
            out.setdefault((v.gene_symbol, v.moi), []).append(v.key)
    for keys in out.values():
        keys.sort(key=lambda k: (_chrom_order(k[0]), k[1], k[2], k[3]))
    return out


def _chrom_order(c: str) -> tuple[int, str]:
    return (int(c), "") if c.isdigit() else (100, c)
