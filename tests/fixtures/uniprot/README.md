# uniprot fixtures

UniProtKB REST responses recorded live on **2026-09-17**, release **2026_03**
(`x-uniprot-release-date: 02-September-2026`), over **public entries only**. Format:
the `tests/fixtures/literature` format — `request` (method, url, params, body) as the
retriever sends it, `status`, `headers` (the release headers and `x-total-results`,
which `engine.retrieve.uniprot` reads), `retrieved_at`, and `body` (parsed JSON) or
`text`. Served by the `StubHttp` in `tests/test_uniprot.py` keyed on the exact request.

- `probe_release.json` — the version probe: `search?query=(reviewed:true) AND
  (organism_id:9606)&fields=accession&size=1`. Names no gene; only its headers are read.
- `search_CFTR.json` — `(gene_exact:CFTR) AND (organism_id:9606) AND (reviewed:true)`
  → one result, `P13569` / `CFTR_HUMAN`, 1480 aa (`x-total-results: 1`).
- `search_BUB1B.json` — the same for BUB1B → `O60566` / `BUB1B_HUMAN`, 1050 aa.
- `search_NOTAGENE1.json` — a symbol UniProt does not know → `{"results": []}`
  (`x-total-results: 0`): absence, not failure.
- `entry_P13569.json` — `P13569.json?fields=<FIELDS>` (139 KB; 357 KB without
  `fields`). Verified facts the tests pin: 4 Domains, 3 Regions, 1 Motif, 13
  Topological domains, 12 Transmembrane and 6 Binding site features (39 in the map),
  209 Natural variants, entry version 286; residue 508 lies in `Domain: ABC
  transporter 1 (423–646)` and `Topological domain: Cytoplasmic (359–858)` and carries
  `VAR_000171` (deletion, "in CF and CBAVD; …", 20 evidences) and `VAR_000172` (F→C);
  residue 542 lies in the same two features with `VAR_080305` "in CF"; the DISEASE
  comments are Cystic fibrosis and Congenital bilateral absence of the vas deferens.
- `entry_Q00000_404.json` — the only 404 shape accepted as "no such entry":
  `{"url": …, "messages": ["Resource not found"]}`. `Q00000` is well-formed and
  unassigned; a deleted accession (`A9A9A9`) answers 200 with `entryType: Inactive`,
  which the retriever refuses too.
- `entry_notanaccession_400.json` — a malformed accession: 400 with
  `messages[0]` "The 'accession' value has invalid format …". The retriever refuses
  such a string before any request; the fixture records what the API would say.

Regenerate by replaying the same requests (the retriever's own `params`) and keeping
the headers listed above. Nothing here derives from a patient.
