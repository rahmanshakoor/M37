"""The six tools the medicine agent may call, and what they leave behind.

Three of them are stage 5's (:class:`~engine.reason.tools.ReasonTools`): ``get_record``
reads a citable record, ``search_literature`` and ``get_paper`` retrieve papers and
write them as ``pmid:`` records — here into ``06_medicine/evidence/``. The three
others are how a drug enters the report at all:

* ``drugs_for_gene(gene)`` asks the three gene-centric sources in turn — Open Targets
  (the target profile, the top :data:`DISEASES_PER_GENE` target–disease associations
  by score, and every drug or clinical candidate acting on it), DGIdb (every
  drug–gene interaction with its sources) and ChEMBL (curated mechanisms, the
  molecule, its indications and warnings) — writes every record into the stage store
  before the model sees it, and returns one compact projection per record with its
  ``record_id``. The citation vocabulary grows only through the store, exactly as
  papers do in stage 5; a drug the model remembers but no source returned has no id
  and cannot survive the validator.
* ``search_trials(condition, intervention, term)`` asks ClinicalTrials.gov and returns
  study records ``nct:<id>`` plus the search's own ``nct-search:`` record (what was
  asked, the registry's total, the ids in order) — the reason a trial was on the table.
* ``search_chembl(mechanism_text, indication_text)`` asks ChEMBL's curated mechanism
  table and its drug-indication table by *text* — a mechanism phrase, a disease name —
  so a class of intervention that acts on the consequence of the defect rather than on
  the gene product (a read-through compound, a drug for the disease's own indication)
  can be found without a gene symbol. Rows and their molecules become the same
  ``chembl:`` records the gene route makes; the search's own ``chembl-search:`` record
  (what was asked, the server's total, the ids returned) says where the engine looked.

Scope and absence. ``get_record`` serves the candidate's own list (the stage-5 bundle,
the ids its chain cites) and whatever the tools returned in this conversation; any
other id is a ``KeyError`` to the model, whether or not the run holds it. A source that
does not know the gene, or knows it and lists nothing, is reported as such with the
record that says so (the DGIdb gene record, the Open Targets target counts) — absence
is a result the report can cite. The number of distinct genes one conversation may
look up is capped (``max_genes``: the candidate gene plus the pathway nodes the ladder
names), because every symbol goes to three public APIs.

A trial reached by a search and again by a later search yields records that differ
only in ``retrieved_at``; the first one in any store of the run is the one served and
cited, so a rerun on a warm cache is byte-identical (``EvidenceStore.put`` would
otherwise let the last writer win). The same rule serves a drug record an earlier run
of this stage already holds.

Two kinds of refusal, as :mod:`engine.agents.client` expects. A bad argument — a
symbol that is not HGNC-shaped, a query that spells a genomic coordinate, a
``max_results`` out of range, the gene cap — raises ``ValueError``/``KeyError`` and goes
back to the model as an error result. A failure of a service (``OpenTargetsError``,
``DgidbError``, ``ChemblError``, ``CtgovError``, an HTTP error, an offline cache miss)
propagates and aborts the run: a report written around a silently missing drug list is
exactly what this engine refuses to produce.

Privacy: no genomic coordinate is ever sent. The gene-centric sources receive a gene
symbol or an Ensembl gene id; the registry receives condition, intervention and
free-text terms that :func:`engine.reason.tools.check_query` has refused to let carry a
position.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from engine.agents.client import ToolSpec
from engine.agents.validator import EvidenceIndex
from engine.medicine.chembl import DEFAULT_CHEMBL_SEARCH, MAX_CHEMBL_SEARCH, SEARCH_SOURCE, drug_rows, phase_label
from engine.medicine.opentargets import extract_association, extract_drug, extract_target, normalise_id
from engine.reason.tools import DEFAULT_MAX_RESULTS, MAX_RESULTS_CAP, ReasonTools, ToolLog, check_query
from engine.retrieve.store import EvidenceRecord, EvidenceStore

MAX_GENES = 8
"""Distinct gene symbols one conversation may send to the drug sources: the candidate
gene plus the pathway nodes the ladder's intervention classes act through (a
checkpoint gene's partners, the stressed pathway's effectors); not a screen."""
DEFAULT_TRIALS = 10
MAX_TRIALS = 25
"""ClinicalTrials.gov returns the most recently updated first; a report needs the few
that matter, and every study is a ~7 KB record."""
FIELD_CHARS = 240
"""Long projected fields (conditions, indications, outcomes) are cut in the tool's
answer at this length; ``get_record`` has the whole payload."""
DISEASES_PER_GENE = 5
"""Open Targets target–disease associations returned per gene: the top ones by
overall score (page 0 of the server's ordering), each a citable ``opentargets:
association:`` record whose query slice carries the server's total. Enough to say
which diseases the evidence ties the gene to — the mechanism half of the report —
without turning a drug lookup into a disease screen."""
DRUG_SOURCES = ("opentargets", "dgidb", "chembl")

_SYMBOL = re.compile(r"^[A-Za-z][A-Za-z0-9.@-]*$")
"""HGNC shape: a letter first (``CFTR``, ``C10orf10``, ``MT-ND1``, ``IGH@``); a string that
starts with a digit is a position, never a symbol, and one with an underscore is an
accession (``NM_000492.4``), which the sources do not take."""
_ENSG = re.compile(r"^ENSG\d{11}$")

OPENTARGETS_TARGET_FIELDS = ("symbol", "name", "biotype", "tractability", "tractable_modalities", "pathways",
                             "n_pathways", "n_drug_rows", "n_associated_diseases")
OPENTARGETS_DISEASE_FIELDS = ("disease_id", "disease_name", "score", "datatype_scores")
OPENTARGETS_DRUG_FIELDS = ("chembl_id", "name", "drug_type", "max_clinical_stage", "drug_max_clinical_stage",
                           "mechanism_of_action", "action_type", "moa_targets", "n_mechanisms_other_targets",
                           "approved_indications", "conditions", "n_reports", "report_statuses", "trial_ids")
DGIDB_FIELDS = ("dgidb_drug", "dgidb_drug_concept_id", "dgidb_unnamed_compound", "dgidb_approved", "dgidb_anti_neoplastic",
                "dgidb_interaction_types", "dgidb_directionality", "dgidb_evidence_score", "dgidb_interaction_score",
                "dgidb_sources", "dgidb_pmids")
CHEMBL_FIELDS = ("chembl_id", "name", "molecule_type", "max_phase", "first_approval", "withdrawn", "black_box_warning",
                 "warnings", "target_chembl_id", "target_name", "action_type", "mechanism_of_action", "variant_mutation",
                 "indications")
TRIAL_FIELDS = ("nct_id", "title", "status", "why_stopped", "study_type", "phases", "conditions", "interventions",
                "start_date", "primary_completion_date", "enrollment", "enrollment_type", "sponsor", "primary_outcomes",
                "sex", "min_age", "max_age", "has_results", "last_update_post_date")
CHEMBL_MECHANISM_SEARCH_FIELDS = ("record_id", "chembl_id", "name", "action_type", "mechanism_of_action", "max_phase",
                                  "first_approval", "target_chembl_id", "target_name")
CHEMBL_INDICATION_SEARCH_FIELDS = ("record_id", "chembl_id", "name", "mesh_heading", "efo_term", "max_phase_for_ind",
                                   "first_approval")


@dataclass
class MedicineRetrievers:
    """The services behind the tools. Duck-typed on purpose: the stage passes the real
    retrievers, tests pass fakes with the same methods. ``None`` in a dry run."""

    opentargets: Any
    """``resolve_symbol(symbol) -> ENSG | None``, ``target(ensg)``, ``known_drugs(ensg)``,
    ``associated_diseases(ensg, size=)``."""
    dgidb: Any
    """``fetch(symbol) -> GeneResult``."""
    chembl: Any
    """``drugs_for_target_symbol(symbol) -> list[EvidenceRecord]``, ``search_mechanisms(text, limit=)``
    and ``search_indications(text, limit=)`` ``-> (records, total_count)``."""
    trials: Any
    """``find(condition, intervention, max_results, term=) -> SearchResult``, ``extract(record)``."""
    literature: Any
    """A :class:`~engine.retrieve.literature.LiteratureRetriever` for the stage-5 tools."""


@dataclass
class MedicineLog(ToolLog):
    """Stage 5's log plus what the two drug tools returned — every id here is citable."""

    genes: list[str] = field(default_factory=list)
    """Symbols ``drugs_for_gene`` was asked, as sent (upper-cased), first appearance only."""
    drug_records: list[str] = field(default_factory=list)
    """Open Targets, DGIdb and ChEMBL record ids returned, first appearance only."""
    trial_searches: list[str] = field(default_factory=list)
    """``nct-search:`` record ids, in call order."""
    trials: list[str] = field(default_factory=list)
    """``nct:`` record ids returned, first appearance only."""
    chembl_searches: list[str] = field(default_factory=list)
    """``chembl-search:`` record ids, in call order."""

    def ids(self) -> list[str]:
        return list(dict.fromkeys([*super().ids(), *self.drug_records, *self.trial_searches, *self.trials,
                                   *self.chembl_searches]))

    def as_dict(self) -> dict[str, Any]:
        return {**super().as_dict(), "genes": list(self.genes), "drug_records": list(self.drug_records),
                "trial_searches": list(self.trial_searches), "trials": list(self.trials),
                "chembl_searches": list(self.chembl_searches)}


class MedicineTools(ReasonTools):
    """Handlers bound to one run's evidence index, the stage store that receives new
    records, the retrievers (``None`` in a dry run, where no tool is ever called), the
    candidate's gene and the candidate's citable list."""

    def __init__(self, index: EvidenceIndex, store: EvidenceStore, retrievers: MedicineRetrievers | None, *,
                 gene_symbol: str, gene_id: str | None = None, citable: Iterable[str] | None = None,
                 default_max_results: int = DEFAULT_MAX_RESULTS, max_results_cap: int = MAX_RESULTS_CAP,
                 max_genes: int = MAX_GENES, default_trials: int = DEFAULT_TRIALS, max_trials: int = MAX_TRIALS,
                 default_chembl_search: int = DEFAULT_CHEMBL_SEARCH, max_chembl_search: int = MAX_CHEMBL_SEARCH):
        super().__init__(index, store, retrievers.literature if retrievers is not None else None, citable=citable,
                         default_max_results=default_max_results, max_results_cap=max_results_cap)
        self.retrievers = retrievers
        self.gene_symbol = gene_symbol.strip().upper()
        self.gene_id = gene_id.strip().upper() if gene_id and _ENSG.match(gene_id.strip().upper()) else None
        self.max_genes = max_genes
        self.default_trials = default_trials
        self.max_trials = max_trials
        self.default_chembl_search = default_chembl_search
        self.max_chembl_search = max_chembl_search
        self.log = MedicineLog()

    # ---- handlers

    def drugs_for_gene(self, inputs: dict[str, Any]) -> dict[str, Any]:
        symbol = _symbol(inputs.get("gene"))
        if symbol not in self.log.genes and len(self.log.genes) >= self.max_genes:
            raise ValueError(f"at most {self.max_genes} distinct genes may be looked up in one report "
                             f"(already asked: {', '.join(self.log.genes)}); work with what was returned")
        r = self._retrievers()
        ensg = self._ensembl_id(symbol, r)
        out = {
            "gene": symbol,
            "ensembl_id": ensg,
            "opentargets": self._opentargets(r, ensg),
            "dgidb": self._dgidb(r, symbol),
            "chembl": self._chembl(r, symbol),
            "note": ("Every record_id above is now citable. Approval status is what the records say (ChEMBL max_phase 4 "
                     "and first_approval; Open Targets max_clinical_stage APPROVAL; DGIdb dgidb_approved). A trial id "
                     "listed inside a record is not citable until search_trials returns it as nct:<id>; a PMID listed by "
                     "DGIdb is not citable until get_paper returns it. Read a record with get_record before quoting a "
                     "detail from it."),
        }
        if symbol not in self.log.genes:
            self.log.genes.append(symbol)
        return out

    def search_trials(self, inputs: dict[str, Any]) -> dict[str, Any]:
        condition, intervention, term = (_text(inputs.get(k)) for k in ("condition", "intervention", "term"))
        if not (condition or intervention or term):
            raise ValueError("a trial search needs a condition, an intervention or a free-text term")
        for text in (condition, intervention, term):
            if text:
                check_query(text)
        n = inputs.get("max_results")
        n = self.default_trials if n is None else int(n)
        if not 1 <= n <= self.max_trials:
            raise ValueError(f"max_results must be between 1 and {self.max_trials}, got {n}")
        trials = self._retrievers().trials
        result = trials.find(condition, intervention, n, term=term)
        self._put(result.record)
        self.log.trial_searches.append(result.record.record_id)
        studies = [self._keep_trial(s) for s in result.studies]
        return {
            "condition": condition,
            "intervention": intervention,
            "term": term,
            "total_count": result.total_count,
            "returned": len(studies),
            "search_record": result.record.record_id,
            "trials": [self._trial_summary(s, trials) for s in studies],
            "note": ("Cite a trial by its record_id in trial_ids. total_count is how many trials matched in the registry; "
                     "the ones returned are the most recently updated. Read a trial with get_record before relying on "
                     "its eligibility criteria or outcome."),
        }

    def search_chembl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        mechanism_text, indication_text = (_text(inputs.get(k)) for k in ("mechanism_text", "indication_text"))
        if not (mechanism_text or indication_text):
            raise ValueError("a ChEMBL search needs a mechanism_text (words of a curated mechanism of action) "
                             "and/or an indication_text (a disease or condition name)")
        n = inputs.get("max_results")
        n = self.default_chembl_search if n is None else int(n)
        if not 1 <= n <= self.max_chembl_search:
            raise ValueError(f"max_results must be between 1 and {self.max_chembl_search}, got {n}")
        chembl = self._retrievers().chembl
        out: dict[str, Any] = {"mechanism_text": mechanism_text, "indication_text": indication_text, "search_records": [],
                               "total_mechanisms": None, "total_indications": None, "mechanisms": [], "indications": []}
        if mechanism_text:
            records, total = chembl.search_mechanisms(mechanism_text, limit=n)
            search, rows, molecules = self._keep_search(records)
            out["search_records"].append(search.record_id)
            out["total_mechanisms"] = total
            out["mechanisms"] = [_mechanism_hit(r, molecules) for r in rows]
        if indication_text:
            records, total = chembl.search_indications(indication_text, limit=n)
            search, rows, molecules = self._keep_search(records)
            out["search_records"].append(search.record_id)
            out["total_indications"] = total
            out["indications"] = [_indication_hit(r, molecules) for r in rows]
        out["note"] = ("Every record_id above is now citable, the search_records included (list them in a class's "
                       "`searched`). Cite the molecule record (chembl:<CHEMBL id>) for a drug; a mechanism or "
                       "indication row alone does not name it. total_* is how many rows matched in ChEMBL; the ones "
                       "returned are the first by id. Approval status is the molecule's max_phase and first_approval; "
                       "an indication row's max_phase_for_ind says only how far that pairing was documented.")
        return out

    # ---- specs

    def specs(self) -> list[ToolSpec]:
        base = {t.name: t for t in super().specs()}
        get_record = dataclasses.replace(
            base["get_record"],
            description=("Return one citable evidence record by its id — the candidate's variant records (e.g. "
                         "vep:7:117559590:ATCT:A, gnomad:7-117559590-ATCT-A, clinvar:VCV000007105), the stage-5 chain's "
                         "papers, and every record drugs_for_gene, search_trials, search_chembl, search_literature or get_paper returned "
                         "in this conversation (e.g. chembl:CHEMBL2010601, opentargets:drug:ENSG00000001626:CHEMBL2010601, "
                         "dgidb:CFTR:rxcui:1243041, nct:NCT01807923): source, version, the query that produced it, its "
                         "URL and the full raw payload. Use it to quote a detail exactly (a mechanism row's variant, a "
                         "trial's eligibility criteria, a warning's text). Errors for any other id."),
        )
        return [
            get_record,
            base["search_literature"],
            base["get_paper"],
            ToolSpec(
                "drugs_for_gene",
                "Retrieve everything three public sources know about drugs acting on a gene product, by HGNC symbol "
                "(e.g. CFTR): Open Targets (target profile with tractability and pathways; the top "
                f"{DISEASES_PER_GENE} target-disease associations by score; every drug or clinical "
                "candidate with its mechanism, clinical stage, approved indications and trial ids), DGIdb (every "
                "drug-gene interaction with type, directionality, sources, scores and PMIDs) and ChEMBL (curated "
                "mechanism rows joined to the molecule: phase, first approval, withdrawal/black-box flags, indications, "
                "the protein variant the mechanism was shown on). Every record returned becomes citable by its record_id; "
                "a source that does not know the gene says so. At most "
                f"{MAX_GENES} distinct genes per report — the candidate gene plus the pathway nodes your intervention "
                "classes act through; list them in the class's `targets`.",
                {"type": "object", "properties": {"gene": {"type": "string", "description": "HGNC gene symbol, e.g. CFTR."}}},
                self.drugs_for_gene,
            ),
            ToolSpec(
                "search_trials",
                "Search ClinicalTrials.gov (API v2) for registered trials: condition (disease name, e.g. cystic "
                "fibrosis), intervention (drug name, e.g. ivacaftor) and/or a free-text term (a gene symbol, a "
                "compound code); null for any part not used. Returns trial records nct:<id> — most recently updated "
                "first, with status, phases, conditions, interventions, dates, enrolment, sponsor, age range, "
                "primary outcomes and whether results were posted — each citable in trial_ids. Never put a genomic "
                f"coordinate in a query. max_results 1–{MAX_TRIALS}; null for the default of {DEFAULT_TRIALS}.",
                {"type": "object", "properties": {
                    "condition": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Disease or condition name, or null."},
                    "intervention": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Drug or intervention name, or null."},
                    "term": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Free-text term (e.g. a gene symbol), or null."},
                    "max_results": {"anyOf": [{"type": "integer"}, {"type": "null"}],
                                    "description": f"Trials to return, 1–{MAX_TRIALS}; null for the default of {DEFAULT_TRIALS}."},
                }},
                self.search_trials,
            ),
            ToolSpec(
                "search_chembl",
                "Search ChEMBL by text, without a gene: mechanism_text is matched against the curated mechanism-of-"
                "action table (e.g. 'read-through', 'proteasome inhibitor', 'conductance regulator'; a gene symbol "
                "finds nothing there) and indication_text against the drug-indication table's MeSH headings (a "
                "disease name, e.g. 'cystic fibrosis'). Returns the matching rows as chembl:mechanism:<id> / "
                "chembl:indication:<id> records, each molecule as chembl:<CHEMBL id> with max_phase and "
                "first_approval, and a chembl-search:<sha256> record per table that says what was asked and how "
                "many rows matched — cite it in a class's `searched`. One page of max_results rows per table "
                f"(1–{MAX_CHEMBL_SEARCH}; null for the default of {DEFAULT_CHEMBL_SEARCH}), ordered by id. Null for a "
                "text not used; at least one is required. Never put a genomic coordinate in a query.",
                {"type": "object", "properties": {
                    "mechanism_text": {"anyOf": [{"type": "string"}, {"type": "null"}],
                                       "description": "Words of a curated mechanism of action, or null."},
                    "indication_text": {"anyOf": [{"type": "string"}, {"type": "null"}],
                                        "description": "A disease or condition name (MeSH heading words), or null."},
                    "max_results": {"anyOf": [{"type": "integer"}, {"type": "null"}],
                                    "description": f"Rows to return per table, 1–{MAX_CHEMBL_SEARCH}; null for the default of {DEFAULT_CHEMBL_SEARCH}."},
                }},
                self.search_chembl,
            ),
        ]

    # ---- sources

    def _retrievers(self) -> MedicineRetrievers:
        if self.retrievers is None:
            raise RuntimeError("no drug or trial service is configured (dry run)")
        return self.retrievers

    def _ensembl_id(self, symbol: str, r: MedicineRetrievers) -> str | None:
        """The candidate gene's own Ensembl id (from stage 3) when that is the symbol
        asked; otherwise Open Targets' mapping of the symbol, ``None`` when it has none."""
        if symbol == self.gene_symbol and self.gene_id:
            return self.gene_id
        ensg = r.opentargets.resolve_symbol(symbol)
        return normalise_id(ensg) if ensg else None

    def _opentargets(self, r: MedicineRetrievers, ensg: str | None) -> dict[str, Any]:
        if ensg is None:
            return {"known": False, "note": "Open Targets maps no target to this symbol", "target": None, "diseases": [], "drugs": []}
        target = r.opentargets.target(ensg)
        if target is None:
            return {"known": False, "note": f"Open Targets has no target {ensg}", "target": None, "diseases": [], "drugs": []}
        target = self._keep_drug_record(target)
        # the top associations by score: the target's n_associated_diseases says how many there are in all
        diseases = [self._keep_drug_record(a) for a in r.opentargets.associated_diseases(ensg, size=DISEASES_PER_GENE)]
        drugs = [self._keep_drug_record(d) for d in r.opentargets.known_drugs(ensg)]
        return {
            "known": True,
            "version": target.source_version,
            "target": {"record_id": target.record_id, "url": target.url,
                       **_pick(extract_target(target), OPENTARGETS_TARGET_FIELDS)},
            "diseases": [{"record_id": a.record_id, "url": a.url, **_pick(extract_association(a), OPENTARGETS_DISEASE_FIELDS)}
                         for a in diseases],
            "drugs": [{"record_id": d.record_id, **_pick(extract_drug(d), OPENTARGETS_DRUG_FIELDS)} for d in drugs],
        }

    def _dgidb(self, r: MedicineRetrievers, symbol: str) -> dict[str, Any]:
        result = r.dgidb.fetch(symbol)
        if result.gene is None:
            return {"known": False, "note": "DGIdb does not know this symbol", "gene_record": None, "interactions": []}
        gene = self._keep_drug_record(result.gene)
        interactions = [self._keep_drug_record(i) for i in result.interactions]
        return {
            "known": True,
            "version": gene.source_version,
            "gene_record": gene.record_id,
            "gene_name": gene.payload["gene"].get("name"),
            "resolved_via": result.resolved_via,
            "n_interactions": len(interactions),
            "interactions": [{"record_id": i.record_id, **_pick(r.dgidb.extract(i), DGIDB_FIELDS)} for i in interactions],
        }

    def _chembl(self, r: MedicineRetrievers, symbol: str) -> dict[str, Any]:
        records = [self._keep_drug_record(rec) for rec in r.chembl.drugs_for_target_symbol(symbol)]
        targets = [rec for rec in records if rec.query.get("endpoint") == "target"]
        rows = drug_rows(records)
        return {
            "known": bool(targets),
            "note": None if targets else "ChEMBL lists no human single-protein target for this symbol",
            "version": records[0].source_version if records else None,
            "target_records": [t.record_id for t in targets],
            "n_records": len(records),
            "drugs": [{**_pick(row, CHEMBL_FIELDS), "evidence_ids": row["evidence_ids"].split(";")} for row in rows],
        }

    # ---- records

    def _put(self, rec: EvidenceRecord) -> None:
        """Write a record into the stage store — unless a store of the run already holds
        it (a search record stage 5 made for the same query, say): the first copy is the
        one the index serves, and a second would only differ in ``retrieved_at``."""
        if self.index.get(rec.record_id) is None:
            super()._put(rec)

    def _keep_drug_record(self, rec: EvidenceRecord) -> EvidenceRecord:
        """A drug-source record the model was shown: in the stage store unless a store of
        the run already holds it (then that one is served), and on the citable list."""
        held = self.index.get(rec.record_id)
        if held is None:
            self._put(rec)
        if rec.record_id not in self.log.drug_records:
            self.log.drug_records.append(rec.record_id)
        return held or rec

    def _keep_trial(self, rec: EvidenceRecord) -> EvidenceRecord:
        held = self.index.get(rec.record_id)
        if held is None:
            self._put(rec)
        if rec.record_id not in self.log.trials:
            self.log.trials.append(rec.record_id)
        return held or rec

    def _keep_search(self, records: list[EvidenceRecord]) -> tuple[EvidenceRecord, list[EvidenceRecord], dict[str, EvidenceRecord]]:
        """A ChEMBL text search's records into the store and onto the citable list:
        the ``chembl-search:`` record (logged as a search), the rows, and the molecules
        keyed by ChEMBL id."""
        search = next(r for r in records if r.source == SEARCH_SOURCE)
        self._put(search)
        self.log.chembl_searches.append(search.record_id)
        rows, molecules = [], {}
        for rec in records:
            if rec.source == SEARCH_SOURCE:
                continue
            kept = self._keep_drug_record(rec)
            if kept.query.get("endpoint") == "molecule":
                molecules[str(kept.payload.get("molecule_chembl_id"))] = kept
            else:
                rows.append(kept)
        return search, rows, molecules

    @staticmethod
    def _trial_summary(rec: EvidenceRecord, trials: Any) -> dict[str, Any]:
        return {"record_id": rec.record_id, "url": rec.url, **_pick(trials.extract(rec), TRIAL_FIELDS)}


def _pick(cols: dict[str, Any], names: Iterable[str]) -> dict[str, Any]:
    """The named columns, long strings cut at :data:`FIELD_CHARS` — except id lists
    (``trial_ids``, ``dgidb_pmids``), which are never cut."""
    out: dict[str, Any] = {}
    for name in names:
        v = cols.get(name, "")
        if isinstance(v, str) and len(v) > FIELD_CHARS and not name.endswith(("_ids", "_pmids")):
            v = v[: FIELD_CHARS - 1].rstrip() + "…"
        out[name] = v
    return out


def _mechanism_hit(row: EvidenceRecord, molecules: dict[str, EvidenceRecord]) -> dict[str, Any]:
    """One mechanism row of a text search joined with its molecule's name and phase."""
    m = row.payload
    mol = molecules.get(str(m.get("molecule_chembl_id")))
    p = mol.payload if mol is not None else {}
    return _pick({
        "record_id": row.record_id,
        "chembl_id": str(m.get("molecule_chembl_id") or ""),
        "name": str(p.get("pref_name") or ""),
        "action_type": str(m.get("action_type") or ""),
        "mechanism_of_action": str(m.get("mechanism_of_action") or ""),
        "max_phase": phase_label(p["max_phase"] if p.get("max_phase") is not None else m.get("max_phase")),
        "first_approval": "" if p.get("first_approval") is None else str(p["first_approval"]),
        "target_chembl_id": str(m.get("target_chembl_id") or ""),
        "target_name": str(m.get("target_name") or ""),
    }, CHEMBL_MECHANISM_SEARCH_FIELDS)


def _indication_hit(row: EvidenceRecord, molecules: dict[str, EvidenceRecord]) -> dict[str, Any]:
    """One indication row of a text search joined with its molecule's name and approval year."""
    ind = row.payload
    mol = molecules.get(str(ind.get("molecule_chembl_id")))
    p = mol.payload if mol is not None else {}
    return _pick({
        "record_id": row.record_id,
        "chembl_id": str(ind.get("molecule_chembl_id") or ""),
        "name": str(p.get("pref_name") or ""),
        "mesh_heading": str(ind.get("mesh_heading") or ""),
        "efo_term": str(ind.get("efo_term") or ""),
        "max_phase_for_ind": phase_label(ind.get("max_phase_for_ind")),
        "first_approval": "" if p.get("first_approval") is None else str(p["first_approval"]),
    }, CHEMBL_INDICATION_SEARCH_FIELDS)


def _symbol(value: Any) -> str:
    """An HGNC-shaped symbol, upper-cased — or ``ValueError`` before any request. The
    shape alone admits ``chr7-117559590`` and ``g.117559590``, so the string is also held
    to :func:`~engine.reason.tools.check_query`: a symbol never spells a position."""
    s = str(value if value is not None else "").strip()
    if not s or not _SYMBOL.match(s) or len(s) > 40:
        raise ValueError(f"not a gene symbol: {value!r} (expected an HGNC symbol such as CFTR)")
    try:
        check_query(s)
    except ValueError as e:
        raise ValueError(f"not a gene symbol: {value!r} (a gene symbol never carries a genomic position; "
                         "expected an HGNC symbol such as CFTR)") from e
    return s.upper()


def _text(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None
