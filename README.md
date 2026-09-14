# engine

A reproducible rare-disease variant engine. Given one proband's genome and coded
phenotype, it produces an auditable, coded evidence chain a clinical geneticist can
check line by line — and it can be replayed on cases with known answers to measure
how often it recovers them.

The one rule that governs the design: **the model never supplies facts — it reasons
over retrieved ones.** Every factual claim downstream carries the database record it
came from.

## Stages

| # | Command | Stage | Status |
|---|---|---|---|
| 1 | `engine ingest` | Normalize the VCF into a flat, checksummed variant table | ✅ built |
| — | `engine regions` | Gene panel → merged, padded BED (optional restriction for stage 1) | ✅ built |
| 2 | `engine retrieve` | Fetch citable records: Ensembl VEP, gnomAD, ClinVar (Europe PMC literature is fetched by stage 5's tools) | ✅ built |
| 3 | `engine filter` | Rule-based biallelic shortlist with a recorded reason per variant | ✅ built |
| 4 | `engine rank` | Blind phenotype ranking (Exomiser in Docker), no gene prior | ✅ built |
| 5 | `engine reason` | ACMG/AMP evidence chain, every criterion tied to a record id | ✅ built |
| 6 | `engine medicine` | Mechanism → pathway → drug candidates with counter-arguments | ✅ built |
| — | `engine report` | One self-contained HTML document over whatever stages a run holds | ✅ built |
| — | `engine ui` | Live mode: a local web app that browses runs and launches stages | ✅ built |
| 7 | `engine bench` | Replay on known-answer cases; recovery@k and F-max | planned (P7) |
| — | `engine submit` | Track 1 CSV in the scorer's conventions, dry-run through the challenge's own `evaluation.py` | ✅ built |

Each stage reads the previous stage's run directory and writes `manifest.json`:
input checksums, tool versions, parameters, counts in/out, wall time. A rerun by
someone else should reproduce the manifest, not merely the answer.

## Patient data never enters this repository

`.gitignore` blocks sequencing files, every tabular output, the case file, clinical
documents, and the `work/`, `cache/`, `ref/` and `data/` directories — and
`tests/test_gitignore.py` asserts it using git's own matcher. Keep the case file and
run directories *outside* the checkout anyway:

```
project/
├── case.yaml            # private: proband id, VCF path, HPO terms  (never committed)
├── data/                # private: the VCF                            (never committed)
├── work/<run>/          # private: stage outputs                      (never committed)
└── engine/              # this repository
```

## Quick start

Requires Python ≥ 3.12, [uv](https://docs.astral.sh/uv/), and bcftools/htslib on PATH
(`brew install bcftools samtools htslib`).

```bash
cd engine
uv sync
cp case.example.yaml ../case.yaml          # then edit: vcf path, HPO terms

# Stage 1 — whole genome (about ten seconds for a 5M-record WGS VCF)
uv run engine ingest --case ../case.yaml --out ../work/proband --threads 4

# Stage 1 — restricted to a gene panel
uv run engine regions --panel configs/panel.yaml --coords configs/panel_coords.grch38.json \
                      --out configs/panel.grch38.bed
uv run engine ingest --case ../case.yaml --out ../work/proband-panel \
                     --regions configs/panel.grch38.bed
```

Output: `work/<run>/01_ingest/variants.tsv.gz` (one row per allele, canonical
chromosome names, caller quality carried as a flag rather than a filter) plus
`manifest.json` and `ingest.log`.

## Try it

The public demo case (`tests/fixtures/public_case/`: twelve public variants, among
them the two classic CFTR alleles, with public CF phenotype terms) runs every stage
without any patient data. One script does it; the report and the live UI then read
the run directory it wrote.

```bash
cd engine
uv sync

# 1. Stages 1–6 on the demo case. Stage 2 is live (Ensembl VEP, gnomAD; the pinned
#    ClinVar VCF from `engine clinvar-download --ref-dir ../ref`); stage 4 runs only
#    when the verified Exomiser bundle and Docker are present (PUBLIC_SKIP_RANK=1
#    skips it); stages 5 and 6 run dry — bundles and the exact prompts, no model —
#    and the recorded public chain is copied in so stage 6 has something to read.
bash scripts/run_public_case.sh                # writes ${TMPDIR:-/tmp}/engine-public-case/run
DEMO="${TMPDIR:-/tmp}/engine-public-case/run"  # PUBLIC_RUN_DIR / PUBLIC_CACHE move it

# 2. The report: one self-contained HTML file, every evidence id a link to its record.
uv run engine report --run "$DEMO"             # → $DEMO/report.html  (--out to put it elsewhere)

# 3. Live mode: the web app. No options needed — it finds case files, ref/, cache/
#    and work/ in the directory above the checkout and shows what it found.
uv run engine ui                               # then open http://127.0.0.1:8765/
#    New run → pick a case file → "Create and run": ingest → retrieve → filter →
#    rank → reason → medicine as one job, each stage with its defaults, the log and
#    the agent's tool calls streaming in. Live mode (the model in stages 5–6) is on
#    by default when a key is present. A stage whose outputs are fresh is skipped;
#    rank is skipped (not fatal) if Docker cannot give Exomiser its memory.

# 4. Stage 5 for real (a model call; see "Where the key lives" below):
uv run engine reason --run ../demo/public-demo --top 1 --provider openrouter \
                     --case tests/fixtures/public_case/case.yaml
# or from the UI: job panel → reason → switch "Live mode" on → Start.
# PUBLIC_LIVE_MODEL=1 bash scripts/run_public_case.sh does the same inside the demo script.
```

`engine reason` and `engine medicine` take `--provider anthropic|openrouter|fake`.
`fake` answers an empty document with no key and no network — the pipeline's wiring
end to end. The default is `openrouter` when `OPENROUTER_API_KEY` is set, else
`anthropic` when `ANTHROPIC_API_KEY` is set.

**Where the key lives.** Either in the environment, or in a `.env` file in the
directory *above* the checkout — `project/.env` beside `case.yaml` in the layout
above (`ENGINE_ENV=/path/to/file` names another location; `ENGINE_ENV=""` reads none).
The engine refuses a dotenv inside the repository, never overrides a variable the
shell already exports, never prints a key, and a `pytest` session never reads the
file at all. `GET /api/providers` in the UI says only whether a key is present.

**Privacy.** The UI binds to `127.0.0.1` only — there is no option to bind elsewhere
and no authentication, because loopback is the whole access control; do not put it
behind a port forward or a reverse proxy. Everything it serves and everything
`engine report` writes is quoted from the run directory, so **the report of a real
case contains patient data** — variants, genotypes, phenotype terms, file paths — and
must be treated like the VCF itself: kept outside the repository, never shared, never
attached to an issue. The public demo run is the only run directory that may be
shared. Model calls carry the candidate's variants to the provider: the OpenRouter
client pins the upstream and sets `data_collection: deny` on every request, and the
disclosure line in each manifest records exactly which provider and model answered.

## Chromosome naming

The engine stores chromosomes as `1`…`22`, `X`, `Y`, `MT` and converts only at the
edges: `engine.contigs.to_annotation` for Ensembl, `engine.contigs.to_submission`
for the `chr`-prefixed form the challenge scorer compares as an exact string. Nothing
outside the primary assembly has a canonical name; those records are counted and
excluded at ingest.

## Development

```bash
uv run pytest            # ~30 s, network-free: stubbed HTTP and recorded public fixtures;
                         # ingest tests build a synthetic VCF with bgzip/tabix
ENGINE_LIVE_TESTS=1 uv run pytest -m live   # the tests that hit real APIs (public data only)
```

## Provenance of bundled configuration

- `configs/panel.yaml` — 120 public gene symbols in nine differential groups.
- `configs/panel_coords.grch38.json` — GRCh38 gene spans from the Ensembl REST lookup
  endpoint. Two symbols have since been renamed by HGNC (`CENPJ` → `CPAP`,
  `SLC9A3R1` → `NHERF1`); the panel keeps the older symbols and the coordinates were
  fetched by stable Ensembl gene id.
