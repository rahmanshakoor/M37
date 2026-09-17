# medicine fixtures

Stage-5 output for `tests/test_medicine.py`, over **public data only**. The run
directory the tests assemble is the one `tests/test_reason.py` uses (the
`../agents/evidence` store, `../reason/candidates.json` and `joined.json`) plus these
files under `05_reason/chains/`:

- `chain_CFTR_hom.json` — the validated chain for `CFTR:hom` (p.Phe508del homozygous):
  classification `likely_pathogenic` computed by the engine from PS3 (citing
  `pmid:42616613`, the first result of the recorded Europe PMC search
  `TITLE_ABS:"CFTR" AND TITLE_ABS:"F508del"` in `../literature/search_page1.json`),
  PM4, PP5, PP4, PP3 and a PM2 not met; a misfolding loss-of-function mechanism
  hypothesis.
- `chain_TP53_het_single.json` — the validated chain for `TP53:het_single`
  (p.Arg175His): `vus`.

Both were produced by running the real stage 5 (`engine.reason.run.run_reason`) over
that run directory with a scripted `FakeClient` (one `search_literature` turn served
from the recorded fixture) — so they have exactly the shape and key order stage 5
writes. The tests rebuild the stage-5 evidence store (`05_reason/evidence`: the five
papers and the search record) the same way, from `../literature/search_page1.json`.

The drug, interaction, mechanism and trial responses come from the recorded fixtures
in `../opentargets`, `../dgidb`, `../chembl` and `../ctgov` (CFTR, ivacaftor and the
other CFTR modulators; public trials), served by a stub `Http` keyed on the exact
request. Nothing here derives from a patient.

## The scripted answer (2026-09-17, task B)

`tests/test_medicine.py::cftr_answer` is now a walk of the stage-6 ladder with one
break per gate, so every check has a test:

- `mechanism` (rung 1) and `consequence` (rung 2, citing the public disease record
  `opentargets:disease:MONDO_0009061` and a case term's `hpo:` record);
- `intervention_classes` (rung 3): one `candidates_proposed` class backed by the three
  search records the scripted conversation makes (the recorded Europe PMC,
  ClinicalTrials.gov and ChEMBL-mechanism searches), one `considered_and_rejected`
  class with its reason, one with an empty `searched`, one naming a search no tool
  returned, and one whose verdict carries no `rejection_reason` — the last three are
  dropped by the stage;
- `candidates` (rung 4): one good candidate, then one with a single counter-argument,
  one with no `paediatric_safety`, one with no `approved_indication`, one in a class
  the stage dropped, one with a fabricated trial, one naming a drug no record carries,
  and one whose ChEMBL id and bare accessions are cleared or redacted;
- `considered_and_rejected`: one entry on a record, one citing nothing (the validator
  drops it), one in a dropped class;
- `surveillance` (rung 5) on the disease record, and three `follow_up_experiments`, of
  which one cites nothing and is dropped.

The patient context and the secondary findings are never in the answer: the engine
fills them from the records (the case terms' `hpo:` records, seeded into
`05_reason/evidence` by the test's `run_dir` fixture from the labels the public JAX
fixtures carry, and the disease record the stub serves from
`../opentargets/disease_MONDO_0009061.json`). The two ChEMBL text searches are served
from `../chembl/search_mechanism_conductance_regulator_limit3.json` and
`../chembl/search_indication_cystic_fibrosis_limit3.json` (recorded live 2026-09-17,
ChEMBL_37) with the molecules their rows name.
