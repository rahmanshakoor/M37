"""The dossier's own checks, run on the model's answer before the shared validator.

Four rules, in this order, each leaving a :class:`~engine.agents.validator.Rejection`
or a note with the model's own path:

1. ``variant_positions`` keys are pinned to the chain's: a foreign key (the API's enum
   should have refused it, but the check does not trust that) or a second entry for a
   key is dropped; a chain variant the model left out is noted — and gets an
   engine-filled entry after validation (:func:`fill_positions`), so the dossier
   always holds one position per chain variant.
2. ``protein_position`` and ``region`` are the engine's: whatever the model wrote is
   replaced by the residue the ``vep:`` record's HGVS protein change names and the
   UniProt features covering it (computed once, in the bundle, so the model saw the
   same values it is now held to); a differing model value is noted and counted.
3. Bare accessions in prose resolve or are redacted — stage 5's rule
   (:class:`engine.reason.checks.AccessionResolver`) extended by UniProt accessions,
   Pfam and InterPro ids, which resolve through the ``uniprot:`` record's id and
   payload (the entry lists its own cross-references). A dossier that names a domain
   by a Pfam id the entry does not carry is naming a domain from memory.
4. A ``protein`` claim that cites no ``uniprot:`` record is dropped: the protein's
   name, length and organisation are what the entry says, and nothing else in the
   store can say them.

Then :func:`engine.agents.validator.validate` applies the shared rules (unknown ids
drop the claim; inline citations that do not resolve are redacted; ``literature`` is
pruned), and :func:`fill_positions` restores every chain variant's engine fields.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from engine.agents.redaction import Redactions, unwrap
from engine.agents.terms import check_terms
from engine.agents.validator import EvidenceIndex, Rejection, ValidationReport, canonical_id, canonical_key, validate
from engine.dossier.schema import GeneDossier
from engine.reason import checks as reason_checks
from engine.reason.checks import AccessionResolver
from engine.retrieve.store import EvidenceRecord

PROSE_LISTS = ("protein", "mechanism_of_disease", "region_knowledge", "genotype_patterns", "functional_test")
"""The claim lists whose ``statement`` is prose."""
RULES = {
    "keys": "variant_positions keys pinned to the chain's; a foreign or duplicate key is dropped, an omitted chain "
            "variant gets an engine-filled entry",
    "engine_fields": "protein_position and region are set by the engine from the vep: record's hgvsp and the uniprot: "
                     "record's features; a differing model value is noted",
    "bare_accessions_in_prose": "redacted unless carried by a citable record (stage 5's rule plus UniProt, Pfam and "
                                "InterPro ids resolving through the uniprot: record)",
    "protein_claims": "a protein claim citing no uniprot: record is dropped",
}
"""The four rules as the manifest records them (``params.stage_checks``)."""

_ACCESSION = re.compile(
    reason_checks._ACCESSION.pattern
    + r"|(?<![\w:/.-])(?-i:(?P<uniprot>[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}))\b"
    + r"|(?<![\w:/.])(?P<pfam>PF\d{5})\b"
    + r"|(?<![\w:/.])(?P<interpro>IPR\d{6})\b",
    re.IGNORECASE,
)
"""Stage 5's accession forms plus a UniProt accession (case-sensitive: the grammar
is upper-case, and a six-letter word must not match), a Pfam id and an InterPro id."""
_TRAIL = ".,;:"
_LIST_PATH = re.compile(r"^(protein|variant_positions)\[(\d+)\]")


class DossierResolver(AccessionResolver):
    """Stage 5's resolver over the dossier's citable records, answering for the three
    extra forms too: a UniProt accession stands when ``uniprot:<acc>`` is a citable
    record (or the accession appears in a record); a Pfam or InterPro id when a
    record's payload carries it."""

    def resolves(self, m: re.Match[str]) -> bool:
        acc = m.groupdict().get("uniprot")
        if acc:
            return f"uniprot:{acc}" in self.ids or acc.lower() in self.corpus
        token = m.groupdict().get("pfam") or m.groupdict().get("interpro")
        if token:
            return token.lower() in self.corpus
        return super().resolves(m)


@dataclass
class StageChecks:
    """What the checks did: the dossier as it goes on to the validator, the entries and
    claims dropped, the accessions redacted, the notes, and the counts."""
    data: dict[str, Any]
    dropped: list[Rejection] = field(default_factory=list)
    redacted: list[Rejection] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    positions_replaced: int = 0
    positions: dict[str, list[int]] = field(default_factory=dict)
    """For ``protein`` and ``variant_positions``: the model's own index of each surviving item."""


def stage_checks(data: dict[str, Any], bundle: Any, resolver: AccessionResolver,
                 redactions: Redactions | None = None) -> StageChecks:
    """The four rules over the model's answer (a dict). ``bundle`` is the
    :class:`~engine.dossier.run.DossierBundle`: its ``variants`` carry the engine's
    residue and region per key. ``redactions`` is the dossier's marker counter;
    without one the count continues from the markers the answer already carries."""
    redactions = redactions or Redactions.continuing(data)
    out = StageChecks(copy.deepcopy(data), redacted=redactions.rejections)
    d = out.data
    engine = {v["key"]: v for v in bundle.variants}
    keys = [v["key"] for v in bundle.variants]
    by_canonical = {canonical_key(k) or k: k for k in keys}

    # 1 + 2: keys pinned, engine fields set
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    positions: list[int] = []
    for i, entry in enumerate(_dicts(d.get("variant_positions"))):
        path = f"variant_positions[{i}]"
        raw = str(entry.get("key", ""))
        match = by_canonical.get(canonical_key(raw) or raw)
        if match is None:
            out.dropped.append(Rejection(path, f"key {raw!r} is not a variant of this candidate ({', '.join(keys)}); the entry was dropped"))
            continue
        if match in seen:
            out.dropped.append(Rejection(path, f"a second entry for {match!r}; the first is kept"))
            continue
        if match != raw:
            out.notes.append(f"{path}.key: the model wrote {raw!r}; matched to the chain's {match!r}")
            entry["key"] = match
        seen.add(match)
        _fill(entry, engine[match], path, out)
        positions.append(i)
        kept.append(entry)
    for k in keys:
        if k not in seen:
            out.notes.append(f"variant_positions: the model returned no entry for {k!r}; an engine-filled entry "
                             "(residue and region, no consequence) is added after validation")
    pairs = sorted(zip(positions, kept), key=lambda pe: keys.index(pe[1]["key"]))  # the chain's order, positions alongside
    positions = [p for p, _ in pairs]
    d["variant_positions"] = [e for _, e in pairs]
    out.positions["variant_positions"] = positions

    # 3: bare accessions in prose
    for name in PROSE_LISTS:
        for i, claim in enumerate(_dicts(d.get(name))):
            claim["statement"] = _redact(claim.get("statement"), f"{name}[{i}].statement", resolver, redactions)
    for i, entry in enumerate(d["variant_positions"]):
        entry["consequence"] = _redact(entry.get("consequence"), f"variant_positions[{positions[i]}].consequence", resolver, redactions)
    d["limits"] = [_redact(x, f"limits[{k}]", resolver, redactions) for k, x in enumerate(d.get("limits") or [])]

    # 4: protein claims rest on the entry
    kept_protein: list[Any] = []
    protein_positions: list[int] = []
    for i, claim in enumerate(d.get("protein") if isinstance(d.get("protein"), list) else []):
        ids = [canonical_id(str(x)) for x in (claim.get("evidence_ids") or [] if isinstance(claim, dict) else [])]
        if isinstance(claim, dict) and not any(x.startswith("uniprot:") for x in ids):
            out.dropped.append(Rejection(f"protein[{i}]", "protein claim cites no UniProt record"))
            continue
        protein_positions.append(i)
        kept_protein.append(claim)
    d["protein"] = kept_protein
    out.positions["protein"] = protein_positions
    return out


def check_dossier(output: Any, bundle: Any, records: list[EvidenceRecord], scoped: EvidenceIndex) -> tuple[GeneDossier, ValidationReport]:
    """Everything between the model's answer and the dossier on disk: identity pinned
    to the bundle, :func:`stage_checks`, the HPO term check
    (:func:`~engine.agents.terms.check_terms`: the dossier's phenotype prose may name
    ``HP:`` ids, so an id no ``hpo:`` record in scope carries becomes a footnote
    marker and a label that is not the record's is disputed in place), the shared
    validator over the candidate's scoped index (paths mapped back to the model's own
    positions), then :func:`fill_positions`. One report carries all of it."""
    data: dict[str, Any] = output.model_dump() if hasattr(output, "model_dump") else copy.deepcopy(dict(output))
    notes: list[str] = []
    for key, value in (("candidate_id", bundle.candidate_id), ("gene_symbol", bundle.gene_symbol), ("uniprot_accession", bundle.accession)):
        if data.get(key) != value:
            notes.append(f"{key}: the model wrote {data.get(key)!r}; replaced by the bundle's {value!r}")
            data[key] = value
    redactions = Redactions.continuing(data)
    checks = stage_checks(data, bundle, DossierResolver(records), redactions)
    staged = len(redactions.rejections)  # the stage's own redactions already name the model's positions
    terms = check_terms(checks.data, scoped, redact=redactions.redact)
    for r in redactions.rejections[staged:] + terms.disputes:  # the term check saw the lists after the drops
        r.path = _original_path(r.path, checks.positions)
    pre = list(redactions.rejections)  # the stage's and the term check's redactions, each with its marker
    cleaned, report = validate(checks.data, scoped, model=GeneDossier, redactions=redactions)
    for r in report.rejections + report.disputes:  # the validator saw the lists after the stage's drops
        r.path = _original_path(r.path, checks.positions)
    report.rejections[:0] = checks.dropped + pre
    report.disputes.extend(terms.disputes)
    report.counts["items_dropped"] += len(checks.dropped)
    for name in ("redactions", "ids_checked", "ids_unknown"):
        report.counts[name] += len(pre)
    report.counts.update(terms.counts)
    report.counts["positions_replaced"] = checks.positions_replaced
    filled, fill_notes = fill_positions(cleaned, bundle)
    report.notes = notes + checks.notes + terms.notes + fill_notes + report.notes
    return filled, report


def fill_positions(dossier: GeneDossier, bundle: Any) -> tuple[GeneDossier, list[str]]:
    """Every chain variant with its engine fields, in the chain's order: an entry the
    validator dropped (it cited nothing, or an unknown id) or the model omitted comes
    back with residue and region and no consequence — noted, so the omission is on
    record and the dossier still says where every variant falls."""
    data = dossier.model_dump()
    present = {e["key"]: e for e in data["variant_positions"]}
    notes: list[str] = []
    entries = []
    for v in bundle.variants:
        entry = present.get(v["key"])
        if entry is None:
            notes.append(f"variant_positions: no validated entry for {v['key']!r}; an engine-filled entry was added")
            entry = {"key": v["key"], "consequence": "", "evidence_ids": []}
        entry["protein_position"] = v["residue"]
        entry["region"] = list(v["region"])
        entries.append(entry)
    data["variant_positions"] = entries
    return GeneDossier.model_validate(data), notes


def _fill(entry: dict[str, Any], engine: dict[str, Any], path: str, out: StageChecks) -> None:
    written = entry.get("protein_position")
    if written is not None and written != engine["residue"]:
        out.notes.append(f"{path}.protein_position: the model wrote {written}; the vep: record's hgvsp gives "
                         f"{engine['residue'] if engine['residue'] is not None else 'no residue'}")
        out.positions_replaced += 1
    if entry.get("region") and list(entry["region"]) != list(engine["region"]):
        out.notes.append(f"{path}.region: the model's {len(entry['region'])} entr{'y' if len(entry['region']) == 1 else 'ies'} "
                         "replaced by the features the uniprot: record carries")
        out.positions_replaced += 1
    entry["protein_position"] = engine["residue"]
    entry["region"] = list(engine["region"])


def accession_tokens(text: str) -> list[str]:
    """Every bare accession in ``text`` as written, in order — for tests and audits."""
    return [m.group(0) for m in _ACCESSION.finditer(text)]


def _redact(text: Any, path: str, resolver: AccessionResolver, redactions: Redactions) -> str:
    def repl(m: re.Match[str]) -> str:
        if resolver.resolves(m):
            return m.group(0)
        marker = redactions.redact(path, f"bare accession not carried by any citable record: {m.group(0).rstrip(_TRAIL)}")
        return marker + m.group(0)[len(m.group(0).rstrip(_TRAIL)):]  # keep the sentence's punctuation
    return unwrap(_ACCESSION.sub(repl, "" if text is None else str(text)))  # an accession the model bracketed on its own


def _original_path(path: str, positions: dict[str, list[int]]) -> str:
    """A validator path over a list the stage checks shortened → the model's own position."""
    m = _LIST_PATH.match(path)
    if not m:
        return path
    name, i = m.group(1), int(m.group(2))
    pos = positions.get(name) or []
    return f"{name}[{pos[i] if i < len(pos) else i}]" + path[m.end():]


def _dicts(items: Any) -> list[dict[str, Any]]:
    return [x for x in items if isinstance(x, dict)] if isinstance(items, list) else []
