#!/usr/bin/env bash
# Run the public demo case (tests/fixtures/public_case) through every stage.
#
# Why a script and not a test: stages 2 and 4 need the network and Docker, and the
# whole run is the thing a judge wants to reproduce with one command. The case holds
# public alleles only (see tests/fixtures/public_case/README.md), so its run
# directory, its cache and this transcript may be shared freely.
#
#   ingest -> retrieve (live: Ensembl VEP, local ClinVar, gnomAD; no funnel)
#          -> filter -> rank (only when the Exomiser bundle is present and verified)
#          -> reason (dry run, or live with PUBLIC_LIVE_MODEL=1)
#          -> dossier --dry-run (the gene's UniProt entry and the fixed searches, cache-backed)
#          -> medicine --dry-run
#
# Stage 5 runs dry by default (bundles and the exact prompts, no model), which writes
# no chain — and stage 6 reports on a chain. So that stage 6 is still exercised, the
# default path copies the recorded chain in tests/fixtures/public_case/05_reason/
# (the real stage 5 over this same recorded evidence with a scripted client; README.md)
# into 05_reason/chains/ and says so; the stage-5 manifest still records a dry run.
# PUBLIC_LIVE_MODEL=1 calls the model instead (needs OPENROUTER_API_KEY or
# ANTHROPIC_API_KEY, in the environment or in the .env above the checkout): stage 5
# for real, then stage 6's dry run on the chain the model produced.
#
# Environment (every one optional):
#   PUBLIC_RUN_DIR   run directory            (default: ${TMPDIR:-/tmp}/engine-public-case/run)
#   PUBLIC_CACHE     HTTP cache directory     (default: ${TMPDIR:-/tmp}/engine-public-case/cache)
#   CLINVAR_VCF      release-pinned ClinVar   (default: <M37>/ref/clinvar_20260905.vcf.gz)
#   EXOMISER_DATA    extracted Exomiser data  (default: <M37>/ref/exomiser)
#   EXOMISER_HEAP    JVM -Xmx for Exomiser    (default: 2g — auto sizing was OOM-killed on a 4 GiB daemon)
#   REASON_TOP       stage-5 candidates       (default: 3; the case has 2)
#   PUBLIC_LIVE_MODEL=1 run stage 5 against a model instead of dry: OpenRouter when OPENROUTER_API_KEY
#                       is set, else the Anthropic API with ANTHROPIC_API_KEY — from the environment or
#                       from the .env above the checkout ($ENGINE_ENV overrides), as `engine reason` does
#   PUBLIC_SKIP_RANK=1  skip stage 4 even when the bundle is verified (no Docker)
#   PUBLIC_FRESH=1      remove the stage directories of an earlier run in PUBLIC_RUN_DIR first
#   PUBLIC_OFFLINE=1    serve stage 2 from the cache only (fails on a cache miss)
#
# Neither directory may lie under <M37>/work or <M37>/cache, which hold the real case;
# the script refuses before writing anything.
#
# Prints counts and file paths only; no variant reaches the terminal.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M37="$(dirname "$REPO")"
PUBLIC_TMP="${TMPDIR:-/tmp}/engine-public-case"

RUN_DIR="${PUBLIC_RUN_DIR:-$PUBLIC_TMP/run}"
CACHE="${PUBLIC_CACHE:-$PUBLIC_TMP/cache}"
CLINVAR_VCF="${CLINVAR_VCF:-$M37/ref/clinvar_20260905.vcf.gz}"
EXOMISER_DATA="${EXOMISER_DATA:-$M37/ref/exomiser}"
EXOMISER_HEAP="${EXOMISER_HEAP:-2g}"
REASON_TOP="${REASON_TOP:-3}"
LIVE_MODEL="${PUBLIC_LIVE_MODEL:-0}"
CASE="$REPO/tests/fixtures/public_case/case.yaml"
CHAIN_FIXTURE="$REPO/tests/fixtures/public_case/05_reason/chain_CFTR_comphet.json"

cd "$REPO"
unset VIRTUAL_ENV

# <M37>/work and <M37>/cache hold the real case; the public run never goes there.
uv run python - "$M37" "$RUN_DIR" "$CACHE" <<'PY'
import os
import sys

m37, *dirs = (os.path.realpath(p) for p in sys.argv[1:])
for d in dirs:
    for name in ("work", "cache"):
        private = os.path.join(m37, name)
        if d == private or d.startswith(private + os.sep):
            sys.exit(f"refusing to write the public case under {d}: {private} holds the real case; "
                     "set PUBLIC_RUN_DIR / PUBLIC_CACHE elsewhere")
PY
# Checked here, not after four stages have run: a key is read on the first model call.
# The engine's own lookup (engine.agents.providers): the environment wins, else the
# dotenv above the checkout; the key itself is never printed.
live_key_check() {
  uv run python - <<'PY'
import os
import sys
from engine.agents import providers

providers.load_env_file()
if not (providers.key_present("openrouter") or providers.key_present("anthropic") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
    sys.exit("PUBLIC_LIVE_MODEL=1: " + providers.no_key_message())
print("live model: provider " + providers.default_provider(strict=False))
PY
}
if [ "$LIVE_MODEL" = "1" ] && ! live_key_check; then
  exit 2
fi
mkdir -p "$RUN_DIR" "$CACHE"

# Stage 1 indexes the VCF beside its output and refuses to overwrite that index, so
# an earlier run in the same directory must be removed first — only on request.
if [ -e "$RUN_DIR/01_ingest" ]; then
  if [ "${PUBLIC_FRESH:-0}" = "1" ]; then
    echo "replacing the earlier run in $RUN_DIR (PUBLIC_FRESH=1)"
    rm -rf "$RUN_DIR"/0[1-6]_*
  else
    echo "$RUN_DIR already holds a run; set PUBLIC_FRESH=1 to replace it or PUBLIC_RUN_DIR to run elsewhere" >&2
    exit 2
  fi
fi

ST_ingest= ST_retrieve= ST_filter= ST_rank= ST_reason= ST_dossier= ST_medicine=   # one line per stage for the summary (bash 3.2: no assoc arrays)
say() { printf '\n== %s\n' "$*"; }
run() { printf '$ %s\n' "$*"; "$@"; }

# ------------------------------------------------------------------ 1 ingest
say "stage 1 · ingest"
run uv run engine ingest --case "$CASE" --out "$RUN_DIR"
ST_ingest="ok"

# ------------------------------------------------------------------ 2 retrieve
say "stage 2 · retrieve (live, no funnel)"
[ -f "$CLINVAR_VCF" ] || { echo "ClinVar VCF not found: $CLINVAR_VCF (see: engine clinvar-download)" >&2; exit 1; }
offline=""
[ "${PUBLIC_OFFLINE:-0}" = "1" ] && offline="--offline"
# shellcheck disable=SC2086  # $offline is empty or one flag
run uv run engine retrieve --run "$RUN_DIR" --cache "$CACHE" --clinvar-vcf "$CLINVAR_VCF" $offline
ST_retrieve="ok"

# ------------------------------------------------------------------ 3 filter
say "stage 3 · filter"
run uv run engine filter --run "$RUN_DIR"
ST_filter="ok"

# ------------------------------------------------------------------ 4 rank
say "stage 4 · rank"
# "Verified" means: every required file of both bundles is present and the
# .verified.json sidecar the downloader wrote carries the pinned sha256 (and, for
# hg38, every pinned member matched). The check reads the sidecars, not 30 GB.
bundle_check() {
  uv run python - "$EXOMISER_DATA" <<'PY'
import sys
from pathlib import Path
from engine.rank.download import members_status, read_verified
from engine.rank.exomiser import DEFAULT_CONFIG, ExomiserConfig

data = Path(sys.argv[1])
cfg = ExomiserConfig.load(DEFAULT_CONFIG)
problems = []
missing = cfg.missing_data(data)
for name, files in missing.items():
    problems.append(f"{name}: missing {files}")
for name, b in cfg.bundles.items():
    meta = read_verified(data / b.filename)
    if meta is None:
        problems.append(f"{name}: no {b.filename}.verified.json sidecar")
        continue
    if b.sha256 and meta.get("sha256") != b.sha256:
        problems.append(f"{name}: sidecar sha256 {meta.get('sha256')} != pin {b.sha256}")
    if b.member_sha256 and not members_status(b, data)["all_pinned_verified"]:
        problems.append(f"{name}: pinned members not all verified (see engine exomiser-download --verify-extracted)")
if problems:
    print("exomiser data not verified under " + str(data) + ": " + "; ".join(problems))
    sys.exit(3)
print(f"exomiser data verified under {data}: release {cfg.data_version}, bundles {', '.join(cfg.bundles)}")
PY
}
if [ "${PUBLIC_SKIP_RANK:-0}" = "1" ]; then
  echo "skipped: PUBLIC_SKIP_RANK=1"
  ST_rank="skipped (PUBLIC_SKIP_RANK=1)"
elif msg="$(bundle_check)"; then
  echo "$msg"
  run uv run engine rank --run "$RUN_DIR" --case "$CASE" --exomiser-data "$EXOMISER_DATA" --heap "$EXOMISER_HEAP"
  ST_rank="ok"
else
  echo "$msg"
  echo "skipped: stage 4 needs a verified Exomiser data bundle (engine exomiser-download --ref-dir $EXOMISER_DATA)"
  ST_rank="skipped (no verified bundle)"
fi

# ------------------------------------------------------------------ 5 reason
if [ "$LIVE_MODEL" = "1" ]; then
  say "stage 5 · reason (live model, PUBLIC_LIVE_MODEL=1)"
  run uv run engine reason --run "$RUN_DIR" --top "$REASON_TOP" --cache "$CACHE" --case "$CASE"
  ST_reason="ok (live model: chains written by stage 5)"
  CHAIN_SOURCE="live"
else
  say "stage 5 · reason --dry-run"
  run uv run engine reason --run "$RUN_DIR" --dry-run --top "$REASON_TOP" --cache "$CACHE" --case "$CASE"
  # A dry run writes prompts, not chains. Stage 6 needs one, so the recorded chain
  # stands in: it is what the real stage 5 wrote over this same recorded stage-2
  # evidence with a scripted client (tests/fixtures/public_case/README.md) — every
  # id it cites is a record stage 2 wrote here — not what a model said about this
  # run. The stage-5 manifest keeps saying dry run / chains_written 0; this is the
  # only file in 05_reason/ that manifest does not claim.
  [ -f "$CHAIN_FIXTURE" ] || { echo "recorded chain not found: $CHAIN_FIXTURE" >&2; exit 1; }
  mkdir -p "$RUN_DIR/05_reason/chains"
  cp "$CHAIN_FIXTURE" "$RUN_DIR/05_reason/chains/CFTR:comphet.json"
  echo "copied the recorded stage-5 chain into $RUN_DIR/05_reason/chains/CFTR:comphet.json"
  echo "  (from tests/fixtures/public_case/05_reason/chain_CFTR_comphet.json; no model was called — set PUBLIC_LIVE_MODEL=1 for a real chain)"
  ST_reason="ok (dry run: bundles and prompts, no model; recorded chain copied in for stage 6)"
  CHAIN_SOURCE="recorded"
fi

# ------------------------------------------------------- 5b dossier (dry run)
# A step inside stage 5, on the chain's candidate. Its retrieval is the engine's, so
# a dry run still fetches (UniProt and Europe PMC, through $CACHE) and writes the
# bundle and the prompt; only the model is skipped, so no dossier is produced.
say "stage 5 · dossier --dry-run"
set +e
run uv run engine dossier --run "$RUN_DIR" --dry-run --cache "$CACHE"
dos_rc=$?
set -e
if [ "$dos_rc" -ne 0 ]; then
  echo "the dossier step failed (exit $dos_rc); it reads a stage-5 chain in $RUN_DIR/05_reason/chains/" >&2
  ST_dossier="FAILED (exit $dos_rc)"
else
  ST_dossier="ok (dry run: UniProt entry, fixed searches, bundle and prompt; no model)"
fi

# ------------------------------------------------------------------ 6 medicine (dry run)
say "stage 6 · medicine --dry-run"
set +e
run uv run engine medicine --run "$RUN_DIR" --dry-run --cache "$CACHE" --case "$CASE"
med_rc=$?
set -e
if [ "$med_rc" -ne 0 ]; then
  rmdir "$RUN_DIR/06_medicine" 2>/dev/null || true   # a refusal before writing leaves an empty directory
  echo "stage 6 failed (exit $med_rc); it reports on a stage-5 chain in $RUN_DIR/05_reason/chains/" >&2
  exit "$med_rc"
fi
ST_medicine="ok (dry run: bundle and prompt on the $CHAIN_SOURCE chain, no model)"

# ------------------------------------------------------------------ summary
say "summary · $RUN_DIR"
for s in ingest retrieve filter rank reason dossier medicine; do eval "v=\$ST_$s"; printf '  %-9s %s\n' "$s" "$v"; done
uv run python - "$RUN_DIR" "$CHAIN_SOURCE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
chain_source = sys.argv[2]

def manifest(stage):
    p = run / stage / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None

m = manifest("01_ingest")["counts"]
print(f"  01_ingest   rows {m['rows_out']} · sites {m['sites_in']} · flagged {m['flagged_rows']} · per contig {m['rows_per_contig']}")
m = manifest("02_retrieve")
c = m["counts"]
print(f"  02_retrieve variants {c['unique_variants']} · records {c['evidence_records']} "
      f"(vep {c.get('vep_records', 0)}, clinvar {c.get('clinvar_records', 0)}, gnomad {c.get('gnomad_records', 0)}) · "
      f"live requests {c['http']['live_requests']} · cache hits {c['http']['hits']} · versions {m['params'].get('source_versions')}")
m = manifest("03_filter")["counts"]
cands = json.loads((run / "03_filter" / "candidates.json").read_text())["candidates"]
first = cands[0]["candidate_id"] if cands else None
print(f"  03_filter   kept {m['kept']} · dropped {m['dropped']} {m.get('dropped_by_rule')} · candidates {m['candidates']} {m.get('candidates_by_model')}")
print(f"              candidate #1 is CFTR:comphet: {'YES' if first == 'CFTR:comphet' else 'NO (' + str(first) + ')'} "
      f"· order: {[c['candidate_id'] for c in cands]}")
m = manifest("04_rank")
if m:
    c, p = m["counts"], m["params"]
    joined = json.loads((run / "04_rank" / "joined.json").read_text())
    ranks = {c_['candidate_id']: c_.get('exomiser_rank') for c_ in joined['candidates']}
    print(f"  04_rank     genes ranked {c['genes_ranked']} · candidates ranked {c['candidates_ranked']}/{c['candidates']} · "
          f"exomiser-only {c['exomiser_only']} · exomiser {m['tools'].get('exomiser')} data {p.get('data_version')} · "
          f"exit {p.get('exomiser_exit_code')} · wall {p.get('wall_time_s')}s · ranks {ranks}")
else:
    print("  04_rank     not run")
m = manifest("05_reason")
c, p = m["counts"], m["params"]
prompts = sorted((run / "05_reason" / "prompts").glob("*.json"))
print(f"  05_reason   candidates selected {c['candidates_selected']}/{c['candidates_total']} · dry run {p['dry_run']} · "
      f"chains written by stage 5 {c['chains_written']} · prompt version {p['prompt_version']} · "
      f"hpo {p.get('hpo')} ({p.get('hpo_source')}) · prompts {len(prompts)}")
for path in prompts:
    d = json.loads(path.read_text())
    sys_sections = [l[3:] for l in d["system"].splitlines() if l.startswith("## ")]
    user_lines = d["user"].splitlines()
    n_variants = sum(1 for l in user_lines if l.startswith("## Variant "))
    tools = [t["name"] for t in d["tool_definitions"]]
    print(f"              {d['candidate_id']}: system {len(d['system'])} chars, {len(sys_sections)} sections {sys_sections}")
    print(f"              {' ' * len(d['candidate_id'])}  user {len(d['user'])} chars, {n_variants} variant section(s), "
          f"tools {tools}, schema keys {sorted(d['output_schema'].get('properties', {}))}")
    for line in user_lines:
        if line.startswith("case HPO:") or line.startswith("HPO definitions:"):
            print(f"              {' ' * len(d['candidate_id'])}  {line[:160]}")
t = p.get("hpo_terms") or {}
print(f"              hpo terms: served {len(t.get('served') or [])} · fetched {len(t.get('fetched') or [])} · "
      f"missing {t.get('missing')} · store {t.get('store')} · records {c.get('hpo_records')}")
if not p["dry_run"]:
    print(f"              model {p['model']} · effort {p['effort']} · usage {c.get('usage')} · rejections {c.get('rejections')}")
for path in sorted((run / "05_reason" / "chains").glob("*.json")):
    d = json.loads(path.read_text())
    cls = {v["key"]: v["classification"] for v in d["variants"]}
    n_crit = sum(len(v["criteria"]) for v in d["variants"])
    n_met = sum(1 for v in d["variants"] for cr in v["criteria"] if cr["met"])
    print(f"              chain {d['candidate_id']} ({chain_source}, sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:12]}…): "
          f"{n_crit} criteria, {n_met} met · classification {cls} · literature {len(d['literature'])}")
m = manifest("05_reason/dossier")
if m is None:
    print("  dossier     not run")
else:
    c, p = m["counts"], m["params"]
    [d_bundle] = (run / "05_reason" / "dossier" / "bundles").glob("*.json")
    [d_prompt] = (run / "05_reason" / "dossier" / "prompts").glob("*.json")
    b = json.loads(d_bundle.read_text())
    pr = json.loads(d_prompt.read_text())
    print(f"  dossier     candidate {p['candidate']} ({b['gene_symbol']}) · uniprot {b['accession']} · dry run {p['dry_run']} · "
          f"features {len(b['feature_map'])} · papers {len(b['papers'])} · citable record ids {len(b['record_ids'])}")
    print(f"              prompt: system {len(pr['system'])} chars, user {len(pr['user'])} chars, "
          f"schema keys {sorted(pr['output_schema'].get('properties', {}))}")
    for v in b["variants"]:
        print(f"              {v['key']}: residue {v['residue']} · region {(v['region'] or ['none'])[0][:90]}")

m = manifest("06_medicine")
c, p = m["counts"], m["params"]
[bundle_path] = (run / "06_medicine" / "bundles").glob("*.json")   # stage 6 reports on one candidate
[prompt_path] = (run / "06_medicine" / "prompts").glob("*.json")
bundle = json.loads(bundle_path.read_text())
prompt = json.loads(prompt_path.read_text())
tools = [t["name"] for t in prompt["tool_definitions"]]
print(f"  06_medicine candidate {p['candidate']} ({bundle['gene_symbol']}) · dry run {p['dry_run']} · chains available {c['chains_available']} · "
      f"reports written {c['reports_written']} · chain input {Path(m['inputs']['chain']['path']).name} sha256 {m['inputs']['chain']['sha256'][:12]}…")
print(f"              bundle: {len(bundle['record_ids'])} citable record ids · {len(bundle['missing_ids'])} cited id(s) missing from the run · "
      f"chain classifications {{{', '.join(v['key'] + ': ' + str(v['classification']) for v in bundle['chain']['variants'])}}}")
print(f"              prompt: system {len(prompt['system'])} chars, user {len(prompt['user'])} chars, tools {tools}, "
      f"schema keys {sorted(prompt['output_schema'].get('properties', {}))}")
for line in prompt["user"].splitlines():
    if line.startswith("case HPO:") or line.startswith("HPO definitions:"):
        print(f"              {line[:160]}")
print(f"              dossier in the bundle: {'yes' if bundle.get('dossier') else 'no (the step ran dry: no validated dossier to read)'}"
      f" · hpo terms served {len((p.get('hpo_terms') or {}).get('served') or [])} fetched "
      f"{len((p.get('hpo_terms') or {}).get('fetched') or [])}")
PY
