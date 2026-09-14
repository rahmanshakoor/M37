# Stage contracts

What each stage reads and writes. Stages communicate only through the run directory;
a stage may read any earlier stage's outputs and must not modify them. Every stage
writes `manifest.json` via `engine.manifest.Manifest`. Chromosomes are canonical
(`1`…`22`, `X`, `Y`, `MT`) everywhere; a variant key is `(chrom, pos, ref, alt)` and
its string form is `engine.retrieve.store.key_str` → `7:117559590:ATCT:A`.

The one rule that governs everything downstream of stage 2: **a claim may cite only a
record id that exists in the evidence store** (`02_retrieve/evidence/index.json`, plus
records a later stage adds to its own `evidence/` directory in the same format). The
validator in `engine.agents.validator` enforces it; anything it rejects is logged and
never rendered.

```
work/<run>/
├── 01_ingest/    variants.tsv.gz · input.csi · manifest.json · ingest.log
├── 02_retrieve/  variants.annotated.tsv.gz · evidence/<source>/*.json · evidence/index.json · manifest.json
├── 03_filter/    decisions.tsv.gz · shortlist.tsv.gz · candidates.json · manifest.json
├── 04_rank/      exomiser/ (raw) · ranking.tsv · joined.json · evidence/ · manifest.json
├── 05_reason/    bundles/ · prompts/ · transcripts/ · validation/ · chains/<candidate_id>.json · evidence/ · evidence_chain.md · manifest.json
└── 06_medicine/  bundles/ · prompts/ · transcripts/ · validation/ · evidence/ · report.json · report.md · manifest.json
```

Stages 4–6 add to that: `04_rank/evidence/` holds one `exomiser:<gene>` record per ranked
gene so a later stage can cite the rank; stages 5 and 6 keep, per candidate, the bundle the
model saw (`bundles/`), the exact request (`prompts/<id>.json` + `.md`), every turn and tool
call (`transcripts/`; `<id>.failed.json` when the model or a tool failed) and the validator's
rejections and redactions (`validation/`). A rerun replaces those and the reports; `evidence/`
is a store and is kept.

---

## Stage 3 — `engine filter --run <dir> [--config configs/filter.yaml]`

Package `engine.filter`. Pure rules; no network; deterministic.

**Reads** `02_retrieve/variants.annotated.tsv.gz` (columns: the stage-1 columns, then
`funnel_region`, then every retriever's `columns` in order vep → clinvar → gnomad, then
`evidence_ids`). Column names come from `VepRetriever.columns`, `ClinvarRetriever.columns`,
`GnomadRetriever.columns` — read them from the classes, never hardcode the list.

**Config** `configs/filter.yaml` (all thresholds live here and are echoed into the manifest):

```yaml
rarity:
  recessive_max_af: 0.01        # per-allele AF ceiling for a candidate in a biallelic model
  dominant_max_af: 0.0001       # for a single heterozygous HIGH/ClinVar-P variant kept as a low-priority candidate
  homozygote_max_nhom: 5        # gnomAD homozygotes allowed for a recessive candidate allele
  rescue_max_af: 0.05           # a reviewed ClinVar P/LP allele is rescued from the two ceilings above only at or under this AF
  af_source_order: [gnomad_af, vep_gnomade_af, vep_gnomadg_af]   # first non-empty wins; absent everywhere = 0
  af_fallback_pick: max         # how the columns after the first are read when the first is empty: first | max (max since 2026-09-14: "first" let 4,219 common rows read as rare on the first full case)
consequence:
  keep_impacts: [HIGH, MODERATE]
  keep_splice_min_ds: 0.2       # any impact kept if spliceai_ds_max >= this
  drop_terms: [synonymous_variant]   # unless ClinVar says otherwise
clinvar:
  always_keep: [Pathogenic, Likely_pathogenic]            # regardless of consequence/AF
  drop_if_benign_min_stars: 2                             # Benign/Likely_benign with ≥2 stars → dropped
  rescue_min_stars: 1                                     # review stars a P/LP entry needs to rescue a row from the AF/nhom ceilings
quality:
  cluster_min_rows: 5           # ≥ this many surviving rows of one gene within cluster_window_bp → caveat dense_cluster (a read pile)
  cluster_window_bp: 200
  caveat_min_dp: 10
  caveat_min_gq: 20
  caveat_flagged_filters: true  # a non-empty quality_flag is a caveat, never a drop
phase:
  trust_pgt: false              # true: a shared PID with opposite PGT (0|1 vs 1|0) is read as trans, not cis
```

Every key above is required (`engine.filter.config.FilterConfig` is strict: no defaults, no
unknown keys), and this block is the shipped `configs/filter.yaml` — a test loads it from this
file. `rescue_max_af`, `rescue_min_stars`, `af_fallback_pick` and `phase.trust_pgt` were added
during the build: without a bounded rescue the "regardless of AF" clause of `always_keep` would
let an unreviewed "Pathogenic" at 30% through, and without it CFTR p.Phe508del (gnomAD v4 AF
0.0119, 58 homozygotes, 4 stars) would fail rule 3; the two switches keep the contract's literal
readings as the defaults and make the alternatives (`max`, `trust_pgt: true`) explicit choices
echoed in the manifest.

**Rules, in order, per row** (the first rule that drops wins and is recorded). Two
housekeeping rules run before rule 1 (built): `duplicate` — a repeated variant key — and
`genotype` — a non-carrier call (the `0/0` a multi-allelic split leaves behind) — since neither
can support any model and counting them would make "distinct heterozygous rows" untrue.

1. `consequence`: keep if `impact_any_coding ∈ keep_impacts`, or `spliceai_ds_max ≥ keep_splice_min_ds`,
   or `clinvar_pathogenicity ∈ always_keep`. Otherwise drop with rule `consequence:<impact_any_coding>`.
2. `clinvar_benign`: drop if `clinvar_pathogenicity ∈ {Benign, Likely_benign}` and `clinvar_stars ≥ drop_if_benign_min_stars`.
3. `rarity`: AF = first non-empty of `af_source_order`, else 0 (`af_fallback_pick: max` reads the
   largest of the fallback columns instead; rows whose reading would differ are counted either way).
   Drop if AF > `recessive_max_af`, rule `rarity:af=<value>` — unless `clinvar_pathogenicity ∈
   always_keep` with ≥ `rescue_min_stars` stars and AF ≤ `rescue_max_af`, in which case the row is
   rescued and the rescue (or its refusal) is a recorded hit (built: the "regardless of AF" clause
   of `always_keep`, bounded). (Single-het dominant candidates apply `dominant_max_af` in step 5.)
4. Quality caveats are attached, never dropped on: `dp<10`, `gq<20`, `flagged:<filter>`; and, per
   gene over the surviving rows, `dense_cluster:<n>in<window>bp` when `quality.cluster_min_rows`
   (5) or more survivors fall within `quality.cluster_window_bp` (200) of a row — the read pile of
   a paralog, pseudogene or divergent haplotype (SERPINA1, the HLA genes), not a genotype. A
   candidate whose every allele carries it sorts after clean candidates of its band (priority key
   `dense_cluster asc`) and the submit stage orders it with the artefact families (built 2026-09-14).
   **Sex** (built 2026-09-14, `engine.sex`): stage 1 records the sex stated in the case file
   (`sex: male|female|unknown`) and the sex inferred from the calls (X non-PAR heterozygous
   fraction, Y non-PAR carrier calls; GRCh38 PARs); stated wins, inferred fills in, and a
   disagreement is a manifest note. Stage 3 applies it after the genotype rule: in a male a
   heterozygous call on X outside the PARs is a genotype a haploid chromosome cannot carry —
   dropped as `sex:x_het_in_male` (a ClinVar P/LP row is kept as `hemi` with the caveat
   `het_call_on_haploid_x`, never as half of a comphet); in a female any Y call is dropped as
   `sex:y_call_in_female`; a female's X homozygote is `hom`. Sex unknown: the rule is off and the
   manifest says so.
5. **Group by gene** (`gene_symbol`, fall back to `gene_id`). Models:
   - `hom`: GT is homozygous alt (`1/1`, `1|1`) and `gnomad_nhom ≤ homozygote_max_nhom` (the same
     bounded ClinVar rescue applies to the homozygote ceiling, with a trace).
   - `hemi`: on `X`/`Y`, GT homozygous alt (a male's hemizygous call looks homozygous). `MT`: `mito`.
   - `comphet`: ≥2 distinct heterozygous rows in the gene. **Phase**: if two rows share a
     non-empty `pid` → same haplotype → `cis` → they do not form a pair (record it). Rows with
     different `pid`s or no `pid` → `unknown`. With `phase.trust_pgt: true` a shared `pid` whose
     PGTs are opposite (`0|1` vs `1|0`) is `trans` instead. A gene whose only het pairs are all
     `cis` is not a comphet candidate (dropped rows get rule `phase:cis`) — even when one of them
     would have qualified as `het_single` on its own: that row is still dropped, but carries the hit
     `model:het_single_eligible`, is counted, and the manifest notes it (built: a dominant candidate
     must not vanish silently behind a phased neighbour).
   - `het_single`: one surviving het in the gene → candidate only if `impact_any_coding == HIGH`
     or `clinvar_pathogenicity ∈ always_keep`, **and** AF ≤ `dominant_max_af`; else dropped with
     rule `model:het_single`.
6. Candidate priority (deterministic sort key, best first):
   `(clinvar P/LP present desc, model rank [hom, comphet, hemi, mito, het_single], best impact desc, AF asc, gene)`.

**Writes**

- `decisions.tsv.gz` — every input row: `chrom pos ref alt gene_symbol kept rule model caveats af_used af_source rule_hits`
  (`rule_hits`, built: every rule that touched the row, `;`-joined — kept rows included — not only
  the one that dropped it).
- `shortlist.tsv.gz` — kept rows: all annotated columns + `model partner_keys phase caveats af_used af_source candidate_id`.
- `candidates.json` — `{"candidates": [Candidate…], "config": {...}, "counts": {...}}` where

```json
{
  "candidate_id": "CFTR:comphet",
  "gene_symbol": "…", "gene_id": "ENSG…", "model": "comphet",
  "priority": 3,
  "variants": [
    {"key": "7:117587778:G:T", "consequence": "stop_gained", "impact": "HIGH",
     "hgvsc": "…", "hgvsp": "…", "transcript_id": "…", "mane": "…",
     "gt": "0/1", "ad": "21,25", "dp": "46", "gq": "99", "quality_flag": "",
     "af_used": "0.00036", "af_source": "gnomad_af", "gnomad_nhom": "0",
     "clinvar_vcv": "", "clinvar_pathogenicity": "", "clinvar_stars": "",
     "spliceai_ds_max": "0", "sift_pred": "", "polyphen_pred": "",
     "caveats": [], "evidence_ids": ["vep:…", "gnomad:…"]}
  ],
  "phase": {"status": "unknown", "evidence": "no shared PID; 28.2 kb apart"},
  "rule_hits": ["consequence:HIGH", "rarity:af=0.00036", …],
  "caveats": []
}
```

Candidates sorted by `priority`. `counts`: rows_in, kept, dropped_by_rule{…}, candidates_by_model{…}.

---

## Stage 4 — `engine rank --run <dir> --case <case.yaml> --exomiser-data <dir> [--image …]`

Package `engine.rank`. Runs Exomiser in Docker with the case VCF and HPO terms only — no
gene list, no prior — then joins the ranking onto stage-3 candidates.

- `configs/exomiser.yaml` pins the Docker image (tag **and** digest), the data bundle
  version, URLs, sizes and hashes, the analysis template (inheritance modes, frequency
  sources, pathogenicity sources, frequency cutoff). Provenance of the hashes (built):
  Monarch publishes no checksum sidecar for the 2406 zips, so `bytes` and `md5` are the
  server's own (ETag / `x-goog-hash`, re-checked by HEAD) and the zip `sha256` values were
  computed by `engine exomiser-download` on transfers whose md5 matched; only the hg38
  `member_sha256` values come from a checksum list the release ships (`2406_hg38.sha256`).
- `engine exomiser-download --ref-dir <dir>` fetches and verifies the data bundle.
- Writes `04_rank/exomiser/` (raw Exomiser outputs: genes TSV, variants TSV, JSON),
  `ranking.tsv` (`rank gene_symbol exomiser_score phenotype_score variant_score moi n_variants variants`),
  `joined.json` (every stage-3 candidate with `exomiser_rank`, `exomiser_score`,
  `phenotype_score`, or `null` when Exomiser did not rank the gene, plus the top-N Exomiser
  genes not in the shortlist so disagreement is visible), `evidence/` (one `exomiser:<gene>`
  record per ranked gene, citable downstream), `manifest.json` (image digest, Exomiser
  version, data version, HPO terms, analysis YAML sha256, wall time). `--join-only` rebuilds
  the ranking and the join from the raw outputs of an earlier run without Docker.
- Contig naming: Exomiser accepts either style; record which the VCF used. The case VCF is
  the original file from `case.yaml`, never a copy inside the repo.

---

## Agents core — `engine.agents`

Shared by stages 5 and 6. Built against the Anthropic Python SDK 1.x (already a dependency);
API shapes come from the `claude-api` skill files, not memory.

- `client.py` — `class ModelClient(Protocol)`: `run(request: AgentRequest) -> AgentResult`.
  `AgentRequest`: `system`, `user` (text), `tools` (list of `ToolSpec{name, description,
  input_schema, handler}`), `output_model` (a pydantic model the final answer must validate
  against), `max_turns`, `model`, `effort`. `AgentResult`: `output` (the validated pydantic
  object), `transcript` (list of turns: tool calls + results), `usage` (tokens), `model`,
  `stop_reason`. Two implementations: `AnthropicClient` (real: `anthropic.Anthropic()` zero-arg,
  model `claude-opus-5`, adaptive thinking, `output_config.effort`, a manual tool loop with
  `strict: true` tools, structured final output via `output_config.format` / `messages.parse`
  as the skill documents; streaming for the final call) and `FakeClient` (scripted responses
  for tests — including one that returns a fabricated evidence id).
- `validator.py` — `validate(obj, store: EvidenceIndex) -> ValidationReport`: walks any
  pydantic output, checks every `evidence_ids` entry exists in the index, every PMID cited
  exists as `pmid:<n>`, recomputes the ACMG frequency criteria (PM2/BA1/BS1) from the gnomAD
  record and flags disagreement; returns the object with rejected items removed and a list of
  `Rejection{path, reason}`. Never raises on a rejection; raises only on a malformed object.
- `bundle.py` — `build_bundle(candidate, run_dir) -> Bundle`: every evidence record for the
  candidate's variants (full payloads), the extracted columns, the stage-4 rank, the case
  HPO terms, and a compact text rendering for the prompt. Bundles are written to
  `05_reason/bundles/<candidate_id>.json` so a judge can see exactly what the model saw.
- `schema.py` — the pydantic output models below. The **classification is computed by the
  engine** from the criteria, never asked of the model — by the ClinGen SVI point system
  (Tavtigian et al. 2020: supporting 1, moderate 2, strong 4, very strong 8, benign negative;
  P ≥ 10, LP 6–9, VUS 0–5, LB −1 to −6, B ≤ −7; BA1 stand-alone), with the 2015 Table 5
  verdict recorded beside it (`classification_richards_2015`) and the point total (`points`).
  PP5/BP6 are retired (Biesecker & Harrison 2018): accepted from the model, marked
  `[RETIRED — …]`, set not met, never counted. PP3/BP4 are recomputed from the `vep:` record
  at the ClinGen SVI calibration (Pejaver et al. 2022: REVEL 0.644/0.773/0.932 for PP3
  supporting/moderate/strong, 0.290/0.183/0.016 for BP4; CADD 25.3/22.7 without REVEL;
  SpliceAI 0.2/0.1 for splicing) — a claimed strength above what the scores support is
  lowered, a claim below every threshold is disputed to not met, a variant without a record
  is unverified. AlphaMissense is carried for the reader and never counted. (Built
  2026-09-14, replacing the 2015 Table 5 as the verdict.)

```python
class Criterion(BaseModel):
    code: str                     # PVS1, PS1..PS4, PM1..PM6, PP1..PP5, BA1, BS1..BS4, BP1..BP7
    strength: Literal["very_strong","strong","moderate","supporting","stand_alone"]
    met: bool
    justification: str
    evidence_ids: list[str]       # must resolve; a criterion with none is rejected unless code in {PP4, PM3, PS2, PS4, PP1, PM6, BS4, BP2, BP5} which may cite the case (PM6, BP2, BP5 added in the build: de novo without confirmation, cis/trans observation and an alternate cause are case observations too)
class VariantChain(BaseModel):
    key: str
    criteria: list[Criterion]
    classification: Literal["pathogenic","likely_pathogenic","vus","likely_benign","benign"] | None = None  # filled by the engine
    points: int | None = None                       # filled by the engine: the SVI point total
    classification_richards_2015: Literal[...] | None = None  # filled by the engine: the 2015 Table 5 verdict
    summary: str
class EvidenceChain(BaseModel):
    candidate_id: str
    variants: list[VariantChain]
    phase_statement: str          # what the data can and cannot show about cis/trans
    mechanism_hypothesis: str     # LoF vs hypomorphic etc., with evidence_ids in text as [id]
    limits: list[str]
    what_would_change_the_call: list[str]
    literature: list[str]         # pmid:<n> ids used
class MechanismClaim(BaseModel):
    statement: str
    evidence_ids: list[str]
class DrugCandidate(BaseModel):
    name: str
    chembl_id: str | None
    mechanism_of_action: str
    approval_status: str
    rationale: str
    counter_arguments: list[str]  # required, non-empty
    evidence_ids: list[str]
    trial_ids: list[str]          # nct:<id> ids
class MedicineReport(BaseModel):
    candidate_id: str
    gene_symbol: str
    mechanism: list[MechanismClaim]
    pathway_targets: list[MechanismClaim]
    candidates: list[DrugCandidate]
    follow_up_experiments: list[str]
    limits: list[str]
    literature: list[str]
```

---

## Stage 5 — `engine reason --run <dir> [--top N] [--model …] [--effort …] [--dry-run] [--case case.yaml] [--cache <dir>] [--offline]`

Package `engine.reason`. For each of the top-N stage-3 candidates: build the bundle, run the
first agent with tools `get_record(record_id)`, `search_literature(query)` (Europe PMC via
`engine.retrieve.literature`; every paper fetched becomes a `pmid:` record in
`05_reason/evidence/`), `get_paper(pmid)`; validate; compute the classification; render.
`--dry-run` writes the bundle and the exact prompt without calling a model. HPO terms come
from `--case`, else stage 4's `joined.json`. Writes `bundles/`, `prompts/`, `transcripts/`,
`validation/`, `chains/<candidate_id>.json`, `evidence_chain.md`, manifest (model, effort,
usage, provider disclosure line, validator rejections).

## Stage 6 — `engine medicine --run <dir> [--candidate <id>] [--dry-run] [--case case.yaml] [--cache <dir>] [--offline]`

Package `engine.medicine`. Retrievers (each an `EvidenceRecord` source with the P2 cache):
`opentargets.py` (target–disease associations, pathways, tractability, known drugs),
`dgidb.py` (drug–gene interactions), `chembl.py` (compound, mechanism, max phase / approval),
`ctgov.py` (ClinicalTrials.gov v2 studies). Then the second agent with tools `get_record`,
`search_literature`, `get_paper`, `drugs_for_gene(gene)` and `search_trials(condition,
intervention, term, max_results)` (built: the registry's own three query parts, each nullable,
rather than one free-text query), producing a `MedicineReport`; validate; render `report.md`
in the rubric's order (mechanism → candidates with counter-arguments → follow-up → limits).
`drugs_for_gene` writes every record into `06_medicine/evidence/` before the model sees it:
the Open Targets target profile, its top five target–disease associations by score
(`opentargets:association:<ENSG>:<disease>`) and every known drug (`opentargets:drug:…`);
the DGIdb gene record and interactions; ChEMBL's target, mechanism, molecule, indication and
warning rows. Absence is a result the report can cite (the target's counts, the DGIdb gene
record, an empty ChEMBL answer). One candidate per run: `--candidate` defaults to the
stage-5 chain of the best stage-3 priority.

---

## Public demo case — `tests/fixtures/public_case/`

A hand-written, bgzipped VCF of 12 public variants (two CFTR pathogenic alleles, decoys,
common benign SNPs) plus `case.yaml` with public CF HPO terms. Exercises stages 1–6 end to
end without any patient data; it is also the first benchmark case for stage 7.
`scripts/run_public_case.sh` runs it (stage 2 live, stage 4 when the verified Exomiser
bundle and Docker are present, stage 5 dry unless `PUBLIC_LIVE_MODEL=1`, stage 6 dry). The
fixture also keeps the recorded stage-2 output (`02_retrieve/`) and the stage-5 chain the
real stage 5 wrote over it with a scripted client (`05_reason/chain_CFTR_comphet.json`), so
stages 3–6 run offline in the tests and the script can give stage 6 a chain without a model.

---

## Model providers — `engine.agents.providers`

Stages 5 and 6 take `--provider anthropic|openrouter|fake` (default: `openrouter` when
`OPENROUTER_API_KEY` is set, else `anthropic` when `ANTHROPIC_API_KEY` is set, else an
error naming both). Keys come from the environment or from a dotenv file **outside the
repository**: `$ENGINE_ENV` if set, else `<repo>/../.env` (the directory above the
checkout, which is where the case file lives). Never from a file inside the repo.

- `providers.py` — `load_env_file(path=None) -> dict` (no override of real env vars),
  `select_client(provider, model, effort) -> ModelClient`, `default_model(provider)`,
  `disclosure(provider, model, effort) -> str` (the provider-disclosure line the manifests
  carry; for OpenRouter it names the upstream provider and the data-collection setting).
- `openrouter.py` — `OpenRouterClient(api_key, *, model, base_url="https://openrouter.ai/api/v1",
  effort, data_collection="deny", app_title="engine", referer=None)` implementing
  `ModelClient` exactly like `AnthropicClient`: the same `AgentRequest` in, the same
  `AgentResult` out (validated pydantic `output`, transcript of tool calls, usage, model,
  stop reason, disclosure). Built on OpenRouter's OpenAI-compatible `/chat/completions`
  with function tools and a JSON-schema final answer, using **only** the request shapes
  a live probe verified for the chosen Claude model; every request carries
  `provider: {"data_collection": "deny"}` so no upstream may train on case data, and the
  response's `provider`/`model` echo is recorded. Transport is `engine.retrieve.http.Http`
  with `cache_ok=False` (model calls are never cached) and typed errors
  (`AgentError` with status, request id). Stdlib only.

## Report — `engine report --run <dir> [--out <file>]`

Package `engine.report`. `render_run(run_dir) -> str` returns one self-contained HTML
file (inline CSS/JS, no build step; Google Fonts allowed with real fallbacks) that reads
whatever stages exist in the run directory and degrades gracefully when later stages have
not run. Sections, in order: run header (proband id, case HPO terms, which stages ran with
version + wall time), candidates (stage 3, joined with stage-4 rank and scores; every rule
hit and caveat visible; blind-ranker agreement/disagreement marked), blind ranking (stage 4
top N with the shortlist overlap), evidence chain per candidate (stage 5: criteria table
with strength, met, justification, and each evidence id as a link to the record's `url`
from the evidence index; the engine-computed classification; phase statement; limits;
validator rejections shown, not hidden), medicine report (stage 6, rubric order), and a
provenance panel (every manifest: inputs with sha256, tools, params, counts, notes).
Design: a clinical document, not a dashboard — typographic hierarchy, tabular numerals,
light/dark via tokens, no emoji, no decorative cards. `engine.report.views` exposes the
JSON view functions the UI reuses: `run_summary`, `candidates_view`, `ranking_view`,
`chain_view`, `medicine_view`, `provenance_view`.

## UI — `engine ui --work <dir> [--port 8765] [--repo <dir>]`

Package `engine.ui`. A local, single-user web app served by the standard library
(`ThreadingHTTPServer`, bound to `127.0.0.1` only, no auth) with static files in
`engine/ui/static/` (`index.html`, `app.js`, `app.css`; vanilla JS). It is the
**live mode** of the engine: browse runs, read every stage's output through the report
views, and launch stages as jobs with their output streaming in.

- `GET /` app shell · `GET /api/runs` (run directories under `--work`, each with the
  stages present, their `finished_at` and counts) · `GET /api/runs/<run>/summary|candidates|
  ranking|chain/<candidate_id>|medicine|provenance` (the report views) ·
  `GET /api/runs/<run>/report.html` (the rendered report) · `GET /api/providers`
  (which keys are present — never the keys — and the default model per provider).
- `POST /api/runs/<run>/jobs` with `{"stage": "ingest|retrieve|filter|rank|reason|medicine",
  "args": {...}}` starts `uv run engine <stage> …` as a subprocess (arguments whitelisted
  per stage — e.g. reason/medicine take provider, model, effort, top, dry_run, candidate)
  and returns a job id; `GET /api/jobs` · `GET /api/jobs/<id>` · `GET /api/jobs/<id>/events`
  is a Server-Sent Events stream of stdout/stderr lines, and for stages 5–6 also the
  agent's tool calls as they appear in `05_reason/transcripts/` / `06_medicine/transcripts/`,
  ending with a `done` event carrying the exit code and the manifest path. One job per run
  at a time. `POST /api/jobs/<id>/cancel`.
- `POST /api/runs` with `{"name", "case_path", "regions"?}` creates a run directory (the
  case file path is used as given; nothing is copied into the repo).
- The page: run list · stage strip with status · tabs Candidates / Ranking / Evidence
  chain / Medicine / Provenance · a job panel with a live log · a "Live mode" switch that
  sets `dry_run=false` and shows provider, model and effort before a stage-5/6 launch.
- Tests: the API with a stub subprocess runner over the public demo run; the SSE stream;
  argument whitelisting (an unknown argument is rejected, never passed through); binding
  to loopback only.

---

## UI — one-click pipeline (live mode that just works)

The UI must run a case end to end with **no paths typed and no settings known**.

**Discovery.** `engine ui` works with no options: `--work` defaults to `<repo>/../work`,
`--cache` to `<repo>/../cache`, `--clinvar-vcf` to the newest `clinvar_*.vcf.gz` in
`<repo>/../ref`, `--exomiser-data` to `<repo>/../ref/exomiser` when it holds a verified
bundle, and the funnel BED to the newest `funnel*.bed` in `<repo>/../ref`. Every
discovered path is shown on the page (a "Setup" panel with a green/grey state per item and
the command that would create a missing one: `engine clinvar-download --ref-dir ../ref`,
`engine funnel --ref-dir ../ref`, `engine exomiser-download --ref-dir ../ref`) and each
can be started from the panel as a job. `GET /api/setup` returns it.

**Cases.** "New run" offers a picker of case files found in `<repo>/..` (`case.yaml`,
`case.*.yaml`) and shows each file's proband id, VCF name and HPO count; a free path field
stays available. The run name defaults to `<proband_id>-<yyyymmdd-hhmm>`.

**Stage defaults carried automatically.** retrieve: the discovered funnel BED and
ClinVar VCF, 6 VEP workers; rank: the discovered Exomiser data, a heap sized from
`docker info` minus 1 GiB, and `regions` = the funnel BED (see rank below) so the full
genome never goes into Exomiser on a small Docker daemon; reason/medicine: provider =
the one whose key is present (openrouter first), its default model, effort `high`, top 3,
and the first candidate id for medicine. Live mode is **on by default when a key is
present**; the switch shows provider · model · effort and the estimated cost band.

**Run pipeline.** One button per run runs ingest → retrieve → filter → rank → reason →
medicine as a single *pipeline job*: the runner executes the stages in order, streams each
stage's log and tool calls into the same job panel with a per-stage status strip, skips a
stage whose outputs already exist and are newer than its inputs (unless "Re-run all" is
ticked), and stops at the first failure — except `rank`, which on exit code 137
(Docker out of memory) is marked *skipped: Docker memory* and the pipeline continues,
because stages 5–6 do not require it. `POST /api/runs/<run>/pipeline` with
`{"live": bool, "rerun": bool, "top": int}`; the SSE stream carries `stage` events
(`started|done|skipped|failed`, stage, seconds) in addition to `line`/`transcript`/`done`.
Cancel stops the current stage and the pipeline.

**Rank — `--regions <bed>`.** `engine rank` gains `--regions <bed>`: it subsets the case
VCF with `bcftools view -R <bed>` into `04_rank/input.regions.vcf.gz` (+ `.tbi`), runs
Exomiser on that, and records in the manifest the BED (sha256), the record counts before
and after, and that the ranking is gene-blind but restricted to the BED's regions. The
mitochondrion is renamed `M` → `MT` in that subset so Exomiser does not drop it (recorded).

**Home page.** Opening `http://127.0.0.1:8765/` with an empty work directory shows the
Setup panel and the New run form; with runs present it shows the run list with each
run's stage strip, and selecting a run shows the Report tab first.


---

## Stage 8 — `engine submit --run <dir> [--proband-id PROBAND01] [--also-pair KEY1,KEY2]…`

Package `engine.submit`. Writes `08_submit/track1_submission.csv` from `03_filter/candidates.json`,
using `04_rank/joined.json` and `05_reason/chains/` when present, and dry-runs it through the
challenge's `evaluation.py` (default `<repo>/../space/evaluation.py`) against three hypothetical
keys — the lead row, only its first allele, and an absent key — recording the scores in the manifest.
Rules: chromosomes via `contigs.to_submission`; a compound-het candidate is one row with its two
most consequential alleles; `--also-pair` rows follow the lead (alternative partners the funnel may
not carry); backups are ordered by blind-ranker rank, then stage-3 priority, with known artefact gene
families (HLA, MUC, KIR, …) last; candidates a stage-5 chain classified benign are skipped, as
are pairs whose every allele ClinVar calls benign/likely benign and compound heterozygotes on X
when stage 1 recorded a male (a guard for runs older than the sex rule); `finding_type` follows
the template's meaning — `secondary` is an incidental finding: a dominant single heterozygote the
stage-5 chain classified P/LP; an unclassified lone heterozygote is a weak `primary`; EPCRs 0.95,
0.90, … unique and strictly decreasing; at most ten rows; notes never contain a comma.
