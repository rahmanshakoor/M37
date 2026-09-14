"""Stage 6 command — registered by ``engine.cli`` from ``commands``.

Prints counts and the manifest path. No gene symbol, drug name or variant reaches the
terminal unless ``--verbose`` asks for the candidate id, so a pasted shell transcript
carries no patient data.

The provider is chosen exactly as stage 5 does it (``engine.agents.providers.resolve``):
dotenv outside the repository first (never under a pytest session), then ``--provider``
and ``--model`` defaults; a failure with a defaulted provider whose key is absent adds
one line naming both keys.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from engine.agents import providers
from engine.agents.client import DEFAULT_EFFORT, EFFORTS


@click.command()
@click.option("--run", "run_dir", required=True, type=click.Path(file_okay=False, exists=True, path_type=Path),
              help="Run directory that already holds 05_reason/chains/ (and the earlier stages).")
@click.option("--candidate", "candidate_id", default=None,
              help="Candidate id to report on (default: the stage-5 chain of best stage-3 priority).")
@click.option("--provider", type=click.Choice(providers.PROVIDERS), default=None,
              help="Who answers: anthropic (the SDK), openrouter, or fake (scripted empty answer, no key). Default: "
                   "openrouter when OPENROUTER_API_KEY is set, else anthropic. Keys come from the environment or "
                   "from $ENGINE_ENV / <repo>/../.env — never a file inside the repository.")
@click.option("--model", default=None,
              help="Model id for the provider. Default: the provider's (claude-opus-5 for anthropic, "
                   "anthropic/claude-opus-5 for openrouter). OpenRouter ids are author/slug; the bare "
                   "claude-opus-5 means its OpenRouter default.")
@click.option("--effort", type=click.Choice(EFFORTS), default=DEFAULT_EFFORT, show_default=True,
              help="Effort for every call (output_config.effort / reasoning.effort).")
@click.option("--dry-run", is_flag=True,
              help="Write the bundle and the exact prompt, call no model and no drug source, exit 0. Like any run it "
                   "replaces the stage's earlier outputs (the report included); the evidence store is kept.")
@click.option("--max-turns", type=click.IntRange(1), default=None,
              help="Tool-calling turns allowed (default: 10).")
@click.option("--cache", "cache_root", type=click.Path(file_okay=False, path_type=Path),
              help="HTTP cache for Open Targets, DGIdb, ChEMBL, ClinicalTrials.gov and Europe PMC (outside the repo). "
                   "Default: $ENGINE_CACHE or ../cache.")
@click.option("--offline", is_flag=True, help="Serve every source from the cache only; fail on any cache miss.")
@click.option("--case", "case_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="case.yaml, for the HPO terms when stage 4 has not run (default: 04_rank/joined.json).")
@click.option("-v", "--verbose", is_flag=True, help="Name the candidate (gene symbol) on the terminal.")
def medicine(run_dir: Path, candidate_id: str | None, provider: str | None, model: str | None, effort: str,
             dry_run: bool, max_turns: int | None, cache_root: Path | None, offline: bool, case_path: Path | None,
             verbose: bool) -> None:
    """Stage 6: a medicine report for one candidate — mechanism, drug hypotheses with counter-arguments, follow-up."""
    from engine.agents.client import AgentError
    from engine.medicine.run import DEFAULT_MAX_TURNS, run_medicine

    max_turns = max_turns or DEFAULT_MAX_TURNS
    defaulted = provider is None
    try:
        provider, model = providers.resolve(provider, model)
    except (FileNotFoundError, ValueError, providers.ProviderError) as e:
        click.echo(f"  FAILED: {e}", err=True)
        sys.exit(1)
    click.echo(f"medicine · {run_dir} · {'candidate given' if candidate_id else 'best-priority candidate'} · {provider} · "
               f"{model} · effort {effort}{' · OFFLINE' if offline else ''}{' · DRY RUN' if dry_run else ''}")
    try:
        manifest = run_medicine(run_dir, candidate_id, None, model, effort, dry_run, max_turns=max_turns,
                                cache_root=cache_root, offline=offline, case_path=case_path, provider=provider,
                                progress=lambda msg: click.echo("  " + msg))
    except AgentError as e:
        click.echo(f"  FAILED: {e}", err=True)
        if defaulted and not providers.key_present(provider):
            click.echo(f"  {providers.no_key_message()}", err=True)
        sys.exit(1)
    except (FileNotFoundError, KeyError, ValueError) as e:
        click.echo(f"  FAILED: {e}", err=True)
        sys.exit(1)
    m = json.loads(manifest.read_text())
    c, p = m["counts"], m["params"]
    label = p["candidate"] if verbose else "candidate"
    if dry_run:
        click.echo(f"  {label}: bundle and prompt written · chains available: {c['chains_available']:,} · no model called")
    else:
        u = c.get("usage", {})
        click.echo(f"  {label}: mechanism claims: {c['mechanism_claims']:,} · drug candidates: {c['drug_candidates']:,} "
                   f"(with trials: {c['drug_candidates_with_trials']:,}) · follow-up: {c['follow_up_experiments']:,} · "
                   f"rejections: {c['rejections']:,} · records added: {c['evidence_records_added']:,}")
        click.echo(f"  tokens in/out: {u.get('input_tokens', 0):,}/{u.get('output_tokens', 0):,} · "
                   f"api calls: {u.get('api_calls', 0):,} · tool calls: {u.get('tool_calls', 0):,}")
    click.echo(f"  manifest: {manifest}")


commands: list[click.Command] = [medicine]
