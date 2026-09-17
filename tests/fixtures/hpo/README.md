# hpo fixtures

Recorded answers of the JAX ontology API (`GET https://ontology.jax.org/api/hp/terms/<id>`)
for **public HPO ids only** — the five cystic-fibrosis terms of the public demo case
(`tests/fixtures/public_case/case.yaml`) and one well-formed id the ontology does not
hold. Recorded on 2026-09-17 with the real `engine.retrieve.http.Http` over a
throw-away cache: each file is that cache entry in the `tests/fixtures/literature`
shape (`{"request": {method, url, params, body}, "status", "headers", "retrieved_at",
"body" | "text"}`), served by the stub `Http` classes in `tests/test_hpo.py` and
`tests/test_reason.py` for the exact request `HpoRetriever.term` makes. Nothing here
derives from a person.

## `requests/`

| file | id | status | what it shows |
|---|---|---|---|
| `term_HP_0012236.json` | HP:0012236 Elevated sweat chloride | 200 | the term object verbatim (`id`, `name`, `definition`, `comment`, `descendantCount`, `synonyms`, `xrefs`, `publicationReferences`, `translations`); no version header |
| `term_HP_0001738.json` | HP:0001738 Exocrine pancreatic insufficiency | 200 | |
| `term_HP_0002205.json` | HP:0002205 Recurrent respiratory infections | 200 | five synonyms |
| `term_HP_0006528.json` | HP:0006528 Chronic lung disease | 200 | |
| `term_HP_0002110.json` | HP:0002110 Bronchiectasis | 200 | |
| `term_HP_9999999.json` | HP:9999999 | 404 | `text/html`, **empty** body — the only response shape the retriever reads as "no such term" |

## `records/hpo/`

The five `EvidenceRecord`s `HpoRetriever.term` wrote from the answers above
(`hpo:HP:<id>`, `source: hpo`, payload the term object verbatim, `url`
`https://hpo.jax.org/browse/term/<id>`), in `EvidenceStore.put` layout, so
`shutil.copytree(FIXTURES / "hpo" / "records", run / "05_reason" / "evidence", dirs_exist_ok=True)`
seeds a run directory with every case term and stage 5 fetches nothing. The same
five records with an `index.json` live under
`tests/fixtures/public_case/05_reason/evidence/` for the public-case tests.

Regenerate with the real `Http` over a throw-away cache (the module docstring of
`engine.retrieve.hpo` names the verified response shapes), then convert each cache
entry: `body` parsed for a 200, `text` kept for the 404.
