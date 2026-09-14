# agents fixtures

`evidence/` is a stage-2-format evidence store (`<source>/<safe_name>.json` +
`index.json`) holding **public data only**: the three public textbook variants

- MTHFR `1:11796321:G:A` (rs1801133, common)
- CFTR `7:117559590:ATCT:A` (p.Phe508del, pathogenic)
- TP53 `17:7675088:C:T` (p.Arg175His)

with their gnomAD v4, Ensembl VEP and ClinVar records copied from the recorded
fixtures in `../gnomad`, `../vep` and `../clinvar`; the Frosst 1995 MTHFR paper
(`pmid:7647779`, from `../literature/fetch_7647779.json`); the Van Goor 2011 VX-809
(lumacaftor) paper `pmid:21976485` (PMC3219147, doi 10.1073/pnas.1105787108 — a paper
that *has* a PMC id, for the PMC/DOI citation checks), built by
`engine.retrieve.literature.LiteratureRetriever.fetch("21976485")` against the live
Europe PMC REST API (version 6.9) on 2026-09-13; and one public trial,
`nct:NCT01807923` (TRAFFIC, lumacaftor/ivacaftor in F508del-homozygous CF), built by
`engine.medicine.ctgov.TrialsRetriever.study("NCT01807923")` against the live
ClinicalTrials.gov v2 API on 2026-09-12 (API 2.0.5, data snapshot 2026-09-11 — the
observed `source_version`, as the retriever records it).

The tests never use any other real variant: `MT:1:A:G` and `1:5:N:A` are deliberately
impossible spellings that exercise "gnomAD cannot be asked" paths.

`chembl_CHEMBL2103870.json` (beside the store, not in it) is the ChEMBL molecule
record `chembl:CHEMBL2103870` (LUMACAFTOR, max phase 4, first approval 2015), built by
`engine.medicine.chembl.ChemblRetriever.molecule("CHEMBL2103870")` against the live
ChEMBL API (ChEMBL_37, 2026-05-01) on 2026-09-13. It is added to an index in memory by
the test that needs a `chembl:` record; it stays out of `evidence/` because the
stage-5/6 tests copy that directory as their stage-2 store and stage 6 retrieves this
molecule itself (a record already held is served, not duplicated, which their counts
pin).

Built by a one-off script (variant/paper records) plus the retriever calls above,
written with `EvidenceStore.put` and `EvidenceStore.write_index` (so `index.json`
lists every record with its sha256); regenerate by re-running against the same
fixtures and sources. Nothing here derives from a patient.
