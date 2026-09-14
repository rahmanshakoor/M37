"""Stage 3 rules — pure functions over annotated row dicts.

Why pure and why in one place: the shortlist is the last point where a variant can
vanish without anyone reading a reason, so every rule is a function of one row (or
one gene's rows) and the config, returns the *reason* it fired as a string, and
touches nothing else. ``run.py`` only streams rows through these and writes what
comes back; a test can call any rule on a hand-made dict.

Column names are not typed here from memory: each one the rules read is checked at
import time against the class that writes it (``VepRetriever.columns``,
``ClinvarRetriever.columns``, ``GnomadRetriever.columns``, ``engine.ingest.COLUMNS``),
so a renamed column upstream breaks this module at import instead of silently
reading ``''`` for every row.

Order of the per-row rules (evaluated lazily; the first that drops is the one
recorded, and nothing after it is evaluated)::

    duplicate → genotype → sex → consequence → clinvar_benign → rarity → (quality caveats)

then the gene-level models — ``hom``, ``hemi``, ``mito``, ``comphet`` with phasing
from PID/PGT, ``het_single`` — and a deterministic priority over the candidates.
``duplicate`` and ``genotype`` precede the contract's rule 1: a repeated variant key
and a non-carrier call (the ``0/0`` a multi-allelic split leaves behind) cannot
support any model, and dropping them first keeps ``comphet``'s "distinct
heterozygous rows" and the downstream counts honest. ``sex`` (:mod:`engine.sex`)
follows: in a male a heterozygous call on X outside the pseudoautosomal regions is
a genotype a haploid chromosome cannot carry (``sex:x_het_in_male``; a ClinVar P/LP
row is kept as ``hemi`` with the caveat ``het_call_on_haploid_x``), and in a female
any Y call is dropped (``sex:y_call_in_female``). With sex unknown the rule is
silent and two such X hets can still form a comphet — the manifest says so.

ClinVar P/LP rescues (each a recorded hit, each refusal recorded too): from the
consequence rule (contract rule 1, no star bound); from the rarity ceiling and the
``hom`` model's homozygote ceiling — the config says "regardless of consequence/AF",
and CFTR p.Phe508del sits at gnomAD v4 AF 0.0119 with 58 homozygotes — but only
with ``clinvar.rescue_min_stars`` review stars and at or under
``rarity.rescue_max_af``, so an unreviewed "Pathogenic" at 30% is not a candidate.

A het whose only gene partners are in cis is one allele with them and cannot form
a comphet; the contract drops it as ``phase:cis``. Because such a row may still
have qualified under the dominant model on its own, that case is recorded (hit
``model:het_single_eligible`` before the drop) and counted, so a dominant P/LP
allele that vanished behind a phased neighbour is visible in the manifest.

``drop_terms`` (rule 1) vetoes a kept impact only when the reported transcript is
the one that earned it — its own ``impact`` ranks at least ``impact_any_coding``.
Stage 2 reports the MANE transcript of the most severe gene whatever that
transcript's own term is, so a synonymous MANE call with a stop_gained on another
coding transcript is ``impact_any_coding`` HIGH and is kept, as rule 1 says.

Privacy: no error message carries a variant key. A malformed cell names its column
and the table row number (``screen_rows`` / the model step add the row), so a
traceback pasted into a bug report says where, not what.

ClinVar terms: ``clinvar_pathogenicity`` is the term verbatim, so a combined category
(``Pathogenic/Likely_pathogenic``) or a qualified one (``Pathogenic,_low_penetrance``)
matches a class list when every ``/``-member's class (qualifier stripped) is in it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterable, Iterator

from engine.filter.config import IMPACT_RANK, FilterConfig
from engine.ingest import COLUMNS as INGEST_COLUMNS
from engine.retrieve.clinvar import ClinvarRetriever
from engine.retrieve.gnomad import GnomadRetriever
from engine.retrieve.store import VariantKey, key_str
from engine.retrieve.vep import VepRetriever
from engine.sex import in_par

MODEL_ORDER: tuple[str, ...] = ("hom", "comphet", "hemi", "mito", "het_single")
MODEL_RANK = {m: i for i, m in enumerate(MODEL_ORDER)}

BENIGN_CLASSES = frozenset({"Benign", "Likely_benign"})
SEX_CHROMS = frozenset({"X", "Y"})
MITO_CHROM = "MT"
PHASE_NOT_APPLICABLE = "not_applicable"


# ---------------------------------------------------------------- column registry

def _col(owner: object, name: str) -> str:
    """``name``, provided ``owner`` (a ``columns`` tuple or a class carrying one) writes it."""
    cols = owner if isinstance(owner, tuple) else getattr(owner, "columns", None)
    if cols is None or name not in cols:
        raise ImportError(f"column {name!r} is not written by {owner!r}; stage 3 cannot read it")
    return name


# ingest (stage 1)
CHROM = _col(INGEST_COLUMNS, "chrom")
POS = _col(INGEST_COLUMNS, "pos")
REF = _col(INGEST_COLUMNS, "ref")
ALT = _col(INGEST_COLUMNS, "alt")
GT = _col(INGEST_COLUMNS, "gt")
AD = _col(INGEST_COLUMNS, "ad")
DP = _col(INGEST_COLUMNS, "dp")
GQ = _col(INGEST_COLUMNS, "gq")
PGT = _col(INGEST_COLUMNS, "pgt")
PID = _col(INGEST_COLUMNS, "pid")
QUALITY_FLAG = _col(INGEST_COLUMNS, "quality_flag")
# VEP
GENE_SYMBOL = _col(VepRetriever, "gene_symbol")
GENE_ID = _col(VepRetriever, "gene_id")
TRANSCRIPT_ID = _col(VepRetriever, "transcript_id")
MANE = _col(VepRetriever, "mane")
CONSEQUENCE = _col(VepRetriever, "consequence")
IMPACT = _col(VepRetriever, "impact")
IMPACT_ANY_CODING = _col(VepRetriever, "impact_any_coding")
HGVSC = _col(VepRetriever, "hgvsc")
HGVSP = _col(VepRetriever, "hgvsp")
SPLICEAI_DS_MAX = _col(VepRetriever, "spliceai_ds_max")
SIFT_PRED = _col(VepRetriever, "sift_pred")
POLYPHEN_PRED = _col(VepRetriever, "polyphen_pred")
# ClinVar
CLINVAR_VCV = _col(ClinvarRetriever, "clinvar_vcv")
CLINVAR_PATHOGENICITY = _col(ClinvarRetriever, "clinvar_pathogenicity")
CLINVAR_STARS = _col(ClinvarRetriever, "clinvar_stars")
# gnomAD
GNOMAD_NHOM = _col(GnomadRetriever, "gnomad_nhom")
# stage 2 adds these two outside any retriever
FUNNEL_REGION = "funnel_region"
EVIDENCE_IDS = "evidence_ids"

ANNOTATED_COLUMNS: tuple[str, ...] = (
    tuple(INGEST_COLUMNS) + (FUNNEL_REGION,)
    + tuple(VepRetriever.columns) + tuple(ClinvarRetriever.columns) + tuple(GnomadRetriever.columns)
    + (EVIDENCE_IDS,)
)
"""The stage-2 table header when every source ran, in order."""

REQUIRED_COLUMNS: tuple[str, ...] = (CHROM, POS, REF, ALT, GT, AD, DP, GQ, PGT, PID, QUALITY_FLAG)
"""Stage-1 columns the rules and views read. A table without one of them is not the
contract's table and is refused, rather than filtered with '' in every cell (a
missing ``gt`` would drop every row as a non-carrier and call that a result)."""

DECISION_COLUMNS: tuple[str, ...] = (
    CHROM, POS, REF, ALT, GENE_SYMBOL, "kept", "rule", "model", "caveats", "af_used", "af_source", "rule_hits",
)
"""The contract's columns, then ``rule_hits``: every rule that touched the row (kept
rows included), not only the one that dropped it. ``caveats`` and ``rule_hits`` are
``;``-joined lists; no caveat or hit contains ``;`` (a multi-filter ``quality_flag``
becomes one ``flagged:<code>`` caveat per code), so a cell splits back into its items."""

SHORTLIST_EXTRA_COLUMNS: tuple[str, ...] = (
    "model", "partner_keys", "phase", "caveats", "af_used", "af_source", "candidate_id",
)

AF_ABSENT = "absent"
"""``af_source`` when no column of ``af_source_order`` has a value (AF counts as 0)."""


# ---------------------------------------------------------------- small parsers

def val(row: dict[str, str], col: str) -> str:
    """Cell as a string with bcftools' ``.`` (missing) read as ``''``."""
    v = row.get(col, "")
    return "" if v is None or v == "." else v


def as_float(text: str, col: str) -> float | None:
    """``''`` → None; anything else must parse — a malformed number in an annotated
    column is corrupt input, not absence. The message names the column and the
    cell, never the variant."""
    if text == "":
        return None
    try:
        return float(text)
    except ValueError as e:
        raise ValueError(f"{col}: not a number: {text!r}") from e


def as_int(text: str, col: str) -> int | None:
    f = as_float(text, col)
    return None if f is None else int(f)


def as_pos(text: str) -> int:
    """A 1-based position. The cell is a coordinate of the variant, so on failure
    the message names the column only and the original error is not chained."""
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{POS}: not an integer") from None


def row_key(row: dict[str, str]) -> VariantKey:
    return (row[CHROM], as_pos(row[POS]), row[REF], row[ALT])


def gene_label(row: dict[str, str]) -> str:
    """Grouping label: ``gene_symbol``, else ``gene_id``, else the variant key so a
    gene-less row never pairs with another gene-less row."""
    return val(row, GENE_SYMBOL) or val(row, GENE_ID) or key_str(row_key(row))


_GT_SEP = re.compile(r"[/|]")
ALT_INDEX = "1"
"""After stage 1's multi-allelic split every row carries exactly one ALT, so allele
index 1 is the row's variant; any other index is another allele of the site."""


def zygosity(gt: str) -> str:
    """``hom`` (every allele is this ALT), ``het`` (some are, some are not), ``hap``
    (a haploid call of this ALT), or ``none`` (this ALT not called, or no call)."""
    alleles = [a for a in _GT_SEP.split(gt) if a != ""]
    n_alt = sum(1 for a in alleles if a == ALT_INDEX)
    if n_alt == 0:
        return "none"
    if len(alleles) == 1:
        return "hap"
    return "hom" if n_alt == len(alleles) else "het"


def genotype_model(chrom: str, gt: str, sex: str = "unknown") -> str:
    """The model a carrier genotype can support before any partner is known:
    ``mito`` on MT, ``hemi`` for a homozygous-looking or haploid call on X/Y (``hom``
    on X when the sample is female), ``hom`` for one on an autosome, ``het``
    otherwise; ``''`` for a non-carrier."""
    z = zygosity(gt)
    if z == "none":
        return ""
    if chrom == MITO_CHROM:
        return "mito"
    if z == "het":
        return "het"
    if chrom == "X" and sex == "female":
        return "hom"
    return "hemi" if chrom in SEX_CHROMS else "hom"


def clinvar_members(term: str) -> list[str]:
    """Class members of a ClinVar term: ``Pathogenic/Likely_pathogenic`` → both;
    ``Pathogenic,_low_penetrance`` → ``Pathogenic``; ``''`` → none."""
    if term == "":
        return []
    return [m.split(",")[0] for m in term.split("/")]


def clinvar_in(term: str, classes: Iterable[str]) -> bool:
    """True when the term has members and every member's class is in ``classes``."""
    members = clinvar_members(term)
    allowed = set(classes)
    return bool(members) and all(m in allowed for m in members)


def always_keep(row: dict[str, str], cfg: FilterConfig) -> bool:
    return clinvar_in(val(row, CLINVAR_PATHOGENICITY), cfg.clinvar.always_keep)


def first_term(consequence: str) -> str:
    """The most severe SO term of the reported transcript (VEP writes them sorted)."""
    return consequence.split(",", 1)[0] if consequence else ""


# ---------------------------------------------------------------- per-row rules

@dataclass(frozen=True)
class Hit:
    """One rule's verdict on one row: ``rule:detail``, and whether it dropped the row."""

    rule: str
    detail: str = ""
    drop: bool = False

    def __str__(self) -> str:
        return f"{self.rule}:{self.detail}" if self.detail else self.rule


def rule_family(rule: str) -> str:
    """The histogram key for a rule string: the leading ``:``-segments up to the first
    that carries a value (``k=v``), so ``rarity:af=0.3227`` counts as ``rarity``,
    ``model:hom:nhom=12`` as ``model:hom``, while ``consequence:LOW`` and ``phase:cis``
    keep their qualifier because the contract names them that way."""
    kept: list[str] = []
    for part in rule.split(":"):
        if "=" in part:
            break
        kept.append(part)
    return ":".join(kept)


def rescue_refusal(row: dict[str, str], af: float, cfg: FilterConfig) -> str:
    """Why a ClinVar always-keep row may *not* be rescued from a frequency ceiling:
    ``''`` when it may, else ``stars=<n><<min>`` or ``af><max>``. Unknown stars
    count as 0 — an entry without a review status has not earned a rescue."""
    stars = as_int(val(row, CLINVAR_STARS), CLINVAR_STARS) or 0
    if stars < cfg.clinvar.rescue_min_stars:
        return f"stars={stars}<{cfg.clinvar.rescue_min_stars}"
    if af > cfg.rarity.rescue_max_af:
        return f"af>{cfg.rarity.rescue_max_af}"
    return ""


def rescue_hit(row: dict[str, str], ceiling: str) -> Hit:
    """``clinvar:always_keep=<term>`` for the rarity ceiling, ``clinvar:always_keep_<ceiling>=<term>``
    for any other, so two rescues on one row leave two different traces."""
    name = "always_keep" if ceiling == "af" else f"always_keep_{ceiling}"
    return Hit("clinvar", f"{name}={val(row, CLINVAR_PATHOGENICITY)}")


def refusal_hit(row: dict[str, str], why: str) -> Hit:
    return Hit("clinvar", f"rescue_refused={val(row, CLINVAR_PATHOGENICITY)},{why}")


def genotype_rule(row: dict[str, str]) -> list[Hit]:
    """A non-carrier call supports no model. Silent for carriers."""
    gt = val(row, GT)
    if genotype_model(row[CHROM], gt) == "":
        return [Hit("genotype", f"gt={gt or 'missing'}", drop=True)]
    return []


HET_ON_HAPLOID_X = "het_call_on_haploid_x"
"""Caveat on a ClinVar P/LP row kept through ``sex:x_het_in_male``: the call is
heterozygous where a male is haploid, so the genotype itself is in doubt."""


def sex_rule(row: dict[str, str], sex: str, cfg: FilterConfig) -> list[Hit]:
    """A genotype the sample's karyotype cannot carry. Silent when sex is unknown, on
    autosomes, on MT and inside the pseudoautosomal regions."""
    chrom = row[CHROM]
    if sex not in ("male", "female") or chrom not in SEX_CHROMS or in_par(chrom, as_pos(row[POS])):
        return []
    if sex == "female" and chrom == "Y":
        return [Hit("sex", "y_call_in_female", drop=True)]
    if sex == "male" and chrom == "X" and zygosity(val(row, GT)) == "het":
        if always_keep(row, cfg):
            return [Hit("sex", f"x_het_in_male_kept={val(row, CLINVAR_PATHOGENICITY)}")]
        return [Hit("sex", "x_het_in_male", drop=True)]
    return []


def drop_term(row: dict[str, str], cfg: FilterConfig) -> str:
    """The reported transcript's most severe term when it is in ``drop_terms`` *and*
    that transcript is the one that earned ``impact_any_coding`` (its own impact
    ranks at least as high); ``''`` otherwise. A drop term on the reported
    transcript never vetoes an impact another coding transcript earned."""
    term = first_term(val(row, CONSEQUENCE))
    if term not in cfg.consequence.drop_terms:
        return ""
    reported = IMPACT_RANK.get(val(row, IMPACT), -1)
    return term if reported >= IMPACT_RANK.get(val(row, IMPACT_ANY_CODING), -1) else ""


def consequence_rule(row: dict[str, str], cfg: FilterConfig) -> list[Hit]:
    """Rule 1: keep on ``impact_any_coding`` (unless a drop term vetoes it, see
    :func:`drop_term`), on SpliceAI, or on a ClinVar always-keep class; else drop
    as ``consequence:<impact>`` — or ``consequence:<term>`` when only the drop
    term stood in the way, so the reason is the term and not a kept impact."""
    c = cfg.consequence
    impact = val(row, IMPACT_ANY_CODING)
    vetoed = drop_term(row, cfg) if impact in c.keep_impacts else ""
    if impact in c.keep_impacts and not vetoed:
        return [Hit("consequence", impact)]
    ds_text = val(row, SPLICEAI_DS_MAX)
    ds = as_float(ds_text, SPLICEAI_DS_MAX)
    if ds is not None and ds >= c.keep_splice_min_ds:
        return [Hit("consequence", f"splice={ds_text}")]
    if always_keep(row, cfg):
        return [Hit("consequence", f"clinvar={val(row, CLINVAR_PATHOGENICITY)}")]
    if vetoed:
        return [Hit("consequence", vetoed, drop=True)]
    return [Hit("consequence", impact or "none", drop=True)]


def clinvar_benign_rule(row: dict[str, str], cfg: FilterConfig) -> list[Hit]:
    """Rule 2: Benign/Likely_benign with enough review stars is dropped. Silent when
    ClinVar says nothing benign; a benign call under the star threshold is recorded
    as a hit that did not drop."""
    path = val(row, CLINVAR_PATHOGENICITY)
    if not clinvar_in(path, BENIGN_CLASSES):
        return []
    stars_text = val(row, CLINVAR_STARS)
    stars = as_int(stars_text, CLINVAR_STARS) or 0
    detail = f"{path},stars={stars_text or '?'}"
    return [Hit("clinvar_benign", detail, drop=stars >= cfg.clinvar.drop_if_benign_min_stars)]


@dataclass(frozen=True)
class AlleleFrequency:
    """The AF a row is judged on: the cell text verbatim (so it matches the table),
    the column it came from, its value, and whether another present column of the
    fallback tier was above the recessive ceiling while the picked one was not
    (only possible under ``af_fallback_pick: first``)."""

    text: str
    source: str
    value: float
    masked: bool = False


def allele_frequency(row: dict[str, str], cfg: FilterConfig) -> AlleleFrequency:
    """The first column of ``af_source_order`` when it has a value (the citable
    record); else the fallback tier — the remaining columns — read by
    ``af_fallback_pick``: ``max`` (the largest present value; ties go to the earlier
    column) or ``first`` (the first non-empty). ``('0', 'absent', 0.0)`` when none."""
    order = cfg.rarity.af_source_order
    present = [(col, text, float(as_float(text, col) or 0.0))
               for col in order if (text := val(row, col)) != ""]
    if not present:
        return AlleleFrequency("0", AF_ABSENT, 0.0)
    col, text, value = present[0]
    if col == order[0] or len(present) == 1:
        return AlleleFrequency(text, col, value)
    largest = max(present, key=lambda p: p[2])
    if cfg.rarity.af_fallback_pick == "max":
        return AlleleFrequency(largest[1], largest[0], largest[2])
    ceiling = cfg.rarity.recessive_max_af
    return AlleleFrequency(text, col, value, masked=value <= ceiling < largest[2])


def rarity_rule(row: dict[str, str], af_used: str, af: float, cfg: FilterConfig) -> list[Hit]:
    """Rule 3: drop above the recessive ceiling unless a reviewed ClinVar always-keep
    class rescues (dominant candidates are re-tested at ``dominant_max_af`` in the
    model step). A rescue and a refused rescue are both recorded."""
    hit = Hit("rarity", f"af={af_used}")
    if af <= cfg.rarity.recessive_max_af:
        return [hit]
    if always_keep(row, cfg):
        why = rescue_refusal(row, af, cfg)
        if not why:
            return [hit, rescue_hit(row, "af")]
        return [refusal_hit(row, why), Hit(hit.rule, hit.detail, drop=True)]
    return [Hit(hit.rule, hit.detail, drop=True)]


def quality_caveats(row: dict[str, str], cfg: FilterConfig) -> list[str]:
    """Rule 4: caveats only — never a drop. Unknown depth or GQ is not low depth. A
    caller FILTER string is ``;``-separated (VCF forbids ``;`` inside a code), so
    each code becomes its own ``flagged:<code>`` and the ``;``-joined TSV cell
    still splits back into caveats."""
    q = cfg.quality
    out: list[str] = []
    dp = as_int(val(row, DP), DP)
    if dp is not None and dp < q.caveat_min_dp:
        out.append(f"dp<{q.caveat_min_dp}")
    gq = as_int(val(row, GQ), GQ)
    if gq is not None and gq < q.caveat_min_gq:
        out.append(f"gq<{q.caveat_min_gq}")
    if q.caveat_flagged_filters:
        out += [f"flagged:{code}" for code in val(row, QUALITY_FLAG).split(";") if code]
    return out


@dataclass
class Decision:
    """What stage 3 decided about one input row. Small enough to hold for every row."""

    index: int
    key: VariantKey
    gene_symbol: str
    gene: str
    kept: bool = False
    rule: str = ""
    model: str = ""
    caveats: list[str] = field(default_factory=list)
    af_used: str = "0"
    af_source: str = AF_ABSENT
    af: float = 0.0
    af_masked: bool = False
    hits: list[str] = field(default_factory=list)
    partner_keys: list[VariantKey] = field(default_factory=list)
    phase: str = ""
    candidate_id: str = ""

    def drop(self, hit: Hit) -> None:
        self.hits.append(str(hit))
        self.rule = str(hit)
        self.kept = False

    def to_row(self) -> list[str]:
        return [
            self.key[0], str(self.key[1]), self.key[2], self.key[3], self.gene_symbol,
            "1" if self.kept else "0", self.rule, self.model, ";".join(self.caveats),
            self.af_used, self.af_source, ";".join(self.hits),
        ]


def screen_row(index: int, row: dict[str, str], cfg: FilterConfig, *, duplicate: bool = False,
               sex: str = "unknown") -> Decision:
    """Rules 0–4 on one row. A decision with ``rule == ''`` survived to the model
    step (``kept`` is decided there); a decision with a rule was dropped here.
    The AF and caveats are filled for every row, dropped or not, so the decisions
    table is complete; the dropping rules run lazily, so a cell a later rule would
    have read is never parsed once an earlier rule has dropped the row."""
    d = Decision(index=index, key=row_key(row), gene_symbol=val(row, GENE_SYMBOL), gene=gene_label(row))
    af = allele_frequency(row, cfg)
    d.af_used, d.af_source, d.af, d.af_masked = af.text, af.source, af.value, af.masked
    d.caveats = quality_caveats(row, cfg)
    rules: list[Callable[[], list[Hit]]] = [lambda: [Hit("duplicate", drop=True)]] if duplicate else [
        lambda: genotype_rule(row),
        lambda: sex_rule(row, sex, cfg),
        lambda: consequence_rule(row, cfg),
        lambda: clinvar_benign_rule(row, cfg),
        lambda: rarity_rule(row, d.af_used, d.af, cfg),
    ]
    for rule in rules:
        for hit in rule():
            if hit.drop:
                d.drop(hit)
                return d
            d.hits.append(str(hit))
            if hit.rule == "sex":
                d.caveats.append(HET_ON_HAPLOID_X)
    return d


def screen_rows(rows: Iterable[dict[str, str]], cfg: FilterConfig, *, sex: str = "unknown",
                ) -> Iterator[tuple[Decision, dict[str, str]]]:
    """Stream rows through the per-row rules, yielding ``(decision, row)``. The row is
    the caller's to keep (survivors) or forget (dropped). A malformed cell is
    reported by table row number (0-based, header excluded) and column only."""
    seen: set[VariantKey] = set()
    for i, row in enumerate(rows):
        try:
            k = row_key(row)
            d = screen_row(i, row, cfg, duplicate=k in seen, sex=sex)
        except ValueError as e:
            raise ValueError(f"table row {i}: {e}") from e
        seen.add(k)
        yield d, row


# ---------------------------------------------------------------- phase

def distance_text(a: dict[str, str], b: dict[str, str]) -> str:
    bp = abs(as_pos(a[POS]) - as_pos(b[POS]))
    return f"{bp / 1000:.1f} kb apart" if bp >= 1000 else f"{bp} bp apart"


def alt_haplotype(pgt: str) -> int | None:
    """Which haplotype (0 or 1) of a phased PGT carries the row's ALT, or None when
    the PGT is unphased, missing, or does not carry exactly one ALT allele."""
    if "|" not in pgt:
        return None
    alleles = pgt.split("|")
    if len(alleles) != 2 or sum(1 for a in alleles if a == ALT_INDEX) != 1:
        return None
    return alleles.index(ALT_INDEX)


def pair_phase(a: dict[str, str], b: dict[str, str], *, trust_pgt: bool = False) -> tuple[str, str]:
    """``(status, evidence)`` for two heterozygous rows of one gene.

    A shared non-empty PID puts both in the caller's physical-phasing group and is
    read as the same haplotype → ``cis`` (the contract's rule). With ``trust_pgt``
    (a config choice, off by default) the phased genotypes say which haplotype each
    ALT sits on instead: different haplotypes → ``trans``, the same → ``cis``; when
    either PGT is missing or unphased the PID alone decides → ``cis``. Different or
    missing PIDs say nothing → ``unknown``. The PGT values are quoted in the
    evidence when the caller wrote them, so a reader can see what the caller said.
    """
    pid_a, pid_b = val(a, PID), val(b, PID)
    dist = distance_text(a, b)
    if pid_a and pid_a == pid_b:
        pgt_a, pgt_b = val(a, PGT), val(b, PGT)
        quoted = f" (PGT {pgt_a}, {pgt_b})" if pgt_a or pgt_b else ""
        hap_a, hap_b = alt_haplotype(pgt_a), alt_haplotype(pgt_b)
        if trust_pgt and hap_a is not None and hap_b is not None:
            if hap_a != hap_b:
                return "trans", f"shared PID {pid_a} with opposite PGT{quoted}; {dist}"
            return "cis", f"shared PID {pid_a} with the same PGT{quoted}; {dist}"
        return "cis", f"shared PID {pid_a}{quoted}; {dist}"
    if pid_a and pid_b:
        return "unknown", f"different PIDs ({pid_a}, {pid_b}); {dist}"
    return "unknown", f"no shared PID; {dist}"


def opposite_pgt(a: dict[str, str], b: dict[str, str]) -> bool:
    """True when the caller phased both rows in one PID group with their ALTs on
    opposite haplotypes (``0|1`` vs ``1|0``) — the caller's statement that they are
    in trans. Counted in every run so the ``phase.trust_pgt`` choice is visible."""
    pid_a, pid_b = val(a, PID), val(b, PID)
    hap_a, hap_b = alt_haplotype(val(a, PGT)), alt_haplotype(val(b, PGT))
    return bool(pid_a) and pid_a == pid_b and hap_a is not None and hap_b is not None and hap_a != hap_b


# ---------------------------------------------------------------- gene models

Survivor = tuple[Decision, dict[str, str]]


@dataclass
class Candidate:
    candidate_id: str
    gene_symbol: str
    gene_id: str
    model: str
    members: list[Survivor]
    phase_status: str
    phase_evidence: str
    priority: int = 0

    def sort_key(self, cfg: FilterConfig) -> tuple:
        """Contract order: ClinVar P/LP first, then model rank, best impact, AF, gene.
        The AF of a multi-variant candidate is its *commonest* allele — the one that
        limits how rare the genotype is."""
        plp = any(always_keep(r, cfg) for _, r in self.members)
        best_impact = max(IMPACT_RANK.get(val(r, IMPACT_ANY_CODING), -1) for _, r in self.members)
        af = max(d.af for d, _ in self.members)
        return (0 if plp else 1, MODEL_RANK[self.model], 1 if is_clustered(self) else 0, -best_impact, af,
                self.gene_symbol or self.gene_id, self.candidate_id)


def _uniq(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _make(gene: str, model: str, members: list[Survivor], status: str, evidence: str) -> Candidate:
    d0, r0 = members[0]
    cid = f"{gene}:{model}"
    for d, _ in members:
        d.kept, d.model, d.candidate_id = True, model, cid
        d.phase = d.phase or status  # comphet members already carry their own pair status
        d.hits.append(f"model:{model}")
    return Candidate(cid, val(r0, GENE_SYMBOL), val(r0, GENE_ID), model, members, status, evidence)


def _hom_candidates(gene: str, homs: list[Survivor], cfg: FilterConfig) -> list[Candidate]:
    """Homozygous rows under the gnomAD homozygote ceiling (unknown count passes:
    absence from gnomAD is not evidence of homozygotes). A reviewed ClinVar P/LP
    allele over the ceiling is kept, with the exceeded count and the rescue both
    on record."""
    ok: list[Survivor] = []
    for d, r in homs:
        nhom_text = val(r, GNOMAD_NHOM)
        try:
            nhom = as_int(nhom_text, GNOMAD_NHOM)
        except ValueError as e:
            raise ValueError(f"table row {d.index}: {e}") from e
        if nhom is not None and nhom > cfg.rarity.homozygote_max_nhom:
            exceeded = Hit("model", f"hom:nhom={nhom_text}")
            if always_keep(r, cfg):
                why = rescue_refusal(r, d.af, cfg)
                if not why:
                    d.hits += [str(exceeded), str(rescue_hit(r, "nhom"))]
                    ok.append((d, r))
                    continue
                d.hits.append(str(refusal_hit(r, why)))
            d.drop(Hit(exceeded.rule, exceeded.detail, drop=True))
            continue
        ok.append((d, r))
    if not ok:
        return []
    n = len(ok)
    return [_make(gene, "hom", ok, PHASE_NOT_APPLICABLE,
                  f"homozygous call{'s' if n > 1 else ''}; both alleles are the variant")]


def _simple_candidates(gene: str, model: str, members: list[Survivor], evidence: str,
                       status: str = PHASE_NOT_APPLICABLE) -> list[Candidate]:
    return [_make(gene, model, members, status, evidence)] if members else []


def het_single_ok(d: Decision, row: dict[str, str], cfg: FilterConfig) -> bool:
    """A lone het is a (low-priority, dominant) candidate only when it is HIGH impact
    or ClinVar P/LP, *and* at or under the dominant AF ceiling."""
    severe = val(row, IMPACT_ANY_CODING) == "HIGH" or always_keep(row, cfg)
    return severe and d.af <= cfg.rarity.dominant_max_af


HET_SINGLE_EVIDENCE = "single heterozygous allele; a dominant model needs no partner"


def _het_single_candidates(gene: str, hets: list[Survivor], cfg: FilterConfig) -> list[Candidate]:
    ok: list[Survivor] = []
    for d, r in hets:
        if het_single_ok(d, r, cfg):
            ok.append((d, r))
        else:
            d.drop(Hit("model", "het_single", drop=True))
    return _simple_candidates(gene, "het_single", ok, HET_SINGLE_EVIDENCE)


HET_SINGLE_ELIGIBLE = "model:het_single_eligible"
"""Hit on a row dropped as ``phase:cis`` that would have qualified as ``het_single``
on its own — a dominant candidate hidden by a phased neighbour, for the manifest."""


def _comphet_candidates(gene: str, hets: list[Survivor], cfg: FilterConfig) -> list[Candidate]:
    """≥2 distinct hets: pair every two; a het with at least one non-cis partner is a
    comphet member. A het whose partners are all cis shares their haplotype and is
    one allele with them: it is dropped as ``phase:cis`` (contract rule 5), with
    :data:`HET_SINGLE_ELIGIBLE` recorded first when it would have stood alone."""
    pairs: dict[tuple[VariantKey, VariantKey], tuple[str, str]] = {}
    for (da, ra), (db, rb) in combinations(hets, 2):
        pairs[(da.key, db.key)] = pair_phase(ra, rb, trust_pgt=cfg.phase.trust_pgt)

    def partners_of(k: VariantKey, *, cis: bool) -> list[VariantKey]:
        return [b if a == k else a for (a, b), (s, _) in pairs.items() if k in (a, b) and (s == "cis") == cis]

    def has_trans_partner(k: VariantKey) -> bool:
        return any(s == "trans" for (a, b), (s, _) in pairs.items() if k in (a, b))

    members: list[Survivor] = []
    cis_only: list[Survivor] = []
    for d, r in hets:
        d.partner_keys = partners_of(d.key, cis=False)
        if d.partner_keys:
            d.phase = "trans" if has_trans_partner(d.key) else "unknown"
            members.append((d, r))
        else:
            d.partner_keys = partners_of(d.key, cis=True)
            d.phase = "cis"
            cis_only.append((d, r))

    out: list[Candidate] = []
    if members:
        member_keys = {d.key for d, _ in members}
        member_pairs = [(k, v) for k, v in pairs.items() if k[0] in member_keys and k[1] in member_keys]
        status = "trans" if any(s == "trans" for _, (s, _) in member_pairs) else "unknown"
        if len(member_pairs) == 1:
            evidence = member_pairs[0][1][1]
        else:
            evidence = " | ".join(f"{key_str(a)}~{key_str(b)}: {s}, {e}" for (a, b), (s, e) in member_pairs)
        out.append(_make(gene, "comphet", members, status, evidence))

    for d, r in cis_only:
        if het_single_ok(d, r, cfg):
            d.hits.append(HET_SINGLE_ELIGIBLE)
        d.drop(Hit("phase", "cis", drop=True))
    return out


DENSE_CLUSTER = "dense_cluster"
"""Caveat prefix: ``dense_cluster:<n>in<window>bp`` — this surviving row sits among
``n`` surviving rows of its gene inside one window (``quality.cluster_min_rows`` /
``quality.cluster_window_bp``), the signature of mis-mapped reads rather than of a
genotype. A caveat, never a drop; candidates made only of such rows sort last."""


def mark_dense_clusters(members: list[Survivor], cfg: FilterConfig) -> int:
    """Attach :data:`DENSE_CLUSTER` to every surviving row of a gene that has at least
    ``cluster_min_rows`` survivors within ``cluster_window_bp`` of it (itself
    included). Returns how many rows were marked."""
    q = cfg.quality
    positions = sorted(as_pos(r[POS]) for _, r in members)
    marked = 0
    for d, r in members:
        pos = as_pos(r[POS])
        n = sum(1 for p in positions if abs(p - pos) <= q.cluster_window_bp)
        if n >= q.cluster_min_rows:
            d.caveats.append(f"{DENSE_CLUSTER}:{n}in{q.cluster_window_bp}bp")
            marked += 1
    return marked


def is_clustered(c: "Candidate") -> bool:
    """Every allele of the candidate carries the dense-cluster caveat."""
    return all(any(cv.startswith(DENSE_CLUSTER + ":") for cv in d.caveats) for d, _ in c.members)


def resolve_gene(gene: str, members: list[Survivor], cfg: FilterConfig, sex: str = "unknown") -> list[Candidate]:
    """Step 5 for one gene: assign models, drop what fits none, return the candidates.
    Marks every decision in ``members`` (kept/model/rule/phase) as a side effect.
    A male's X het kept through the sex rule is a doubtful hemizygote, never half
    of a compound heterozygote."""
    mark_dense_clusters(members, cfg)
    by_model: dict[str, list[Survivor]] = {"hom": [], "hemi": [], "mito": [], "het": []}
    for d, r in members:
        model = genotype_model(r[CHROM], val(r, GT), sex)
        if model == "het" and HET_ON_HAPLOID_X in d.caveats:
            model = "hemi"
        by_model[model].append((d, r))
    out: list[Candidate] = []
    out += _hom_candidates(gene, by_model["hom"], cfg)
    out += _simple_candidates(gene, "hemi", by_model["hemi"], "homozygous-looking call on a sex chromosome; hemizygous in a male")
    out += _simple_candidates(gene, "mito", by_model["mito"], "mitochondrial variant; heteroplasmy level is in AD")
    hets = by_model["het"]
    if len({d.key for d, _ in hets}) >= 2:
        out += _comphet_candidates(gene, hets, cfg)
    elif hets:
        out += _het_single_candidates(gene, hets, cfg)
    return out


def group_by_gene(survivors: Iterable[Survivor]) -> dict[str, list[Survivor]]:
    """Gene label → its surviving rows, genes in order of first appearance."""
    groups: dict[str, list[Survivor]] = {}
    for d, r in survivors:
        groups.setdefault(d.gene, []).append((d, r))
    return groups


def resolve_models(survivors: list[Survivor], cfg: FilterConfig, sex: str = "unknown") -> list[Candidate]:
    """Steps 5–6 over every gene: candidates sorted by the contract's priority, with
    ``priority`` numbered from 1."""
    cands: list[Candidate] = []
    for gene, members in group_by_gene(survivors).items():
        cands.extend(resolve_gene(gene, members, cfg, sex))
    cands.sort(key=lambda c: c.sort_key(cfg))
    for i, c in enumerate(cands, 1):
        c.priority = i
    return cands


# ---------------------------------------------------------------- output shapes

def variant_view(d: Decision, row: dict[str, str]) -> dict[str, object]:
    """One variant of a candidate, field for field as CONTRACTS.md draws it."""
    return {
        "key": key_str(d.key),
        "consequence": val(row, CONSEQUENCE),
        "impact": val(row, IMPACT),
        "hgvsc": val(row, HGVSC),
        "hgvsp": val(row, HGVSP),
        "transcript_id": val(row, TRANSCRIPT_ID),
        "mane": val(row, MANE),
        "gt": val(row, GT),
        "ad": val(row, AD),
        "dp": val(row, DP),
        "gq": val(row, GQ),
        "quality_flag": val(row, QUALITY_FLAG),
        "af_used": d.af_used,
        "af_source": d.af_source,
        "gnomad_nhom": val(row, GNOMAD_NHOM),
        "clinvar_vcv": val(row, CLINVAR_VCV),
        "clinvar_pathogenicity": val(row, CLINVAR_PATHOGENICITY),
        "clinvar_stars": val(row, CLINVAR_STARS),
        "spliceai_ds_max": val(row, SPLICEAI_DS_MAX),
        "sift_pred": val(row, SIFT_PRED),
        "polyphen_pred": val(row, POLYPHEN_PRED),
        "caveats": list(d.caveats),
        "evidence_ids": [e for e in val(row, EVIDENCE_IDS).split(";") if e],
    }


def candidate_view(c: Candidate) -> dict[str, object]:
    return {
        "candidate_id": c.candidate_id,
        "gene_symbol": c.gene_symbol,
        "gene_id": c.gene_id,
        "model": c.model,
        "priority": c.priority,
        "variants": [variant_view(d, r) for d, r in c.members],
        "phase": {"status": c.phase_status, "evidence": c.phase_evidence},
        "rule_hits": _uniq(h for d, _ in c.members for h in d.hits),
        "caveats": _uniq(cv for d, _ in c.members for cv in d.caveats),
    }


def phase_counts(cands: list[Candidate], survivors: list[Survivor]) -> dict[str, int]:
    """How the het pairs resolved, for the manifest: comphet candidates by status,
    rows dropped as cis (and how many of those would have stood alone as
    ``het_single``), and shared-PID pairs whose PGT says trans (see
    :func:`opposite_pgt`) whatever ``trust_pgt`` did with them."""
    c: Counter[str] = Counter()
    for cand in cands:
        if cand.model == "comphet":
            c[cand.phase_status] += 1
    cis = [d for d, _ in survivors if d.rule == "phase:cis"]
    c["cis_rows_dropped"] = len(cis)
    c["cis_rows_dropped_het_single_eligible"] = sum(1 for d in cis if HET_SINGLE_ELIGIBLE in d.hits)
    c["shared_pid_opposite_pgt_pairs"] = 0
    for members in group_by_gene(survivors).values():
        hets = [(d, r) for d, r in members if genotype_model(r[CHROM], val(r, GT)) == "het"]
        c["shared_pid_opposite_pgt_pairs"] += sum(1 for (_, a), (_, b) in combinations(hets, 2) if opposite_pgt(a, b))
    return dict(sorted(c.items()))
