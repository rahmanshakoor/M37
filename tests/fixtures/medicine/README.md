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
