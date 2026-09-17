# dossier fixtures

Europe PMC searches recorded live on **2026-09-17** (REST 6.9) and the scripted model
answer for `tests/test_dossier.py`, over **public data only** (the CFTR public case).

The four searches are exactly what `engine.dossier.tools.fixed_queries` sends for CFTR
when the UniProt entry names Cystic fibrosis and Congenital bilateral absence of the
vas deferens (`tests/fixtures/uniprot/entry_P13569.json`), at `pageSize` 8:

- `search_CFTR_cystic_fibrosis.json` — `TITLE_ABS:"CFTR" AND TITLE_ABS:"Cystic fibrosis"` (11,336 hits)
- `search_CFTR_cbavd.json` — the same for `Congenital bilateral absence of the vas deferens` (178 hits)
- `search_CFTR_mechanism.json` — `TITLE_ABS:"CFTR" AND (TITLE_ABS:"loss of function" OR TITLE_ABS:"mechanism" OR TITLE_ABS:"pathogenic variants")` (1,657)
- `search_CFTR_genotype.json` — `TITLE_ABS:"CFTR" AND (TITLE_ABS:"genotype" OR TITLE_ABS:"functional assay" OR TITLE_ABS:"functional characterization")` (955)

Format: the `tests/fixtures/literature` format — `request` (method, url, params as the
retriever sends them, body), `status`, `headers`, `retrieved_at`, `body`. Served by the
`StubHttp` in `tests/test_dossier.py` keyed on the exact request. No query carries a
coordinate or an HGVS string; the test asserts that for every request the stub saw.

`answer_CFTR_comphet.json` is the scripted `FakeClient` answer: a `GeneDossier` (before
the engine's fields) that breaks one rule per gate — a `protein_position` of 999 for
`7:117559590:ATCT:A` (the engine sets 508 from the `vep:` record's `p.Phe508del`), a
remembered `region`, a bare `O60566` and `PF99999` no citable record carries (redacted)
beside a bare `P13569` that the record does (kept), `pmid:99999999` (the claim is
dropped and the id pruned from `literature`), the TP53 candidate's variant key
(dropped), and a protein claim citing no `uniprot:` record (dropped). Nothing here
derives from a patient.
