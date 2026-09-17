"""The dossier command — registered by ``engine.cli`` from ``commands``.

Prints counts and the manifest path. No gene symbol, accession or variant reaches the
terminal unless ``--verbose`` asks for the candidate id, so a pasted shell transcript
carries no patient data. The provider is chosen exactly as stages 5 and 6 do it
(``engine.agents.providers.resolve``).
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
              help="Candidate id to write the dossier for (default: the stage-5 chain of best stage-3 priority).")
@click.option("--provider", type=click.Choice(providers.PROVIDERS), default=None,
              help="Who answers: anthropic (the SDK), openrouter, or fake (scripted empty answer, no key). Default: "
                   "openrouter when OPENROUTER_API_KEY is set, else anthropic. Keys come from the environment or "
                   "from $ENGINE_ENV / <repo>/../.env — never a file inside the repository.")
@click.option("--model", default=None,
              help="Model id for the provider. Default: the provider's (claude-opus-5 for anthropic, "
                   "anthropic/claude-opus-5 for openrouter).")
@click.option("--effort", type=click.Choice(EFFORTS), default=DEFAULT_EFFORT, show_default=True,
              help="Effort for every call (output_config.effort / reasoning.effort).")
@click.option("--dry-run", is_flag=True,
              help="Fetch the UniProt entry and the fixed searches (cache-backed), write the bundle and the exact "
                   "prompt, call no model, exit 0. Replaces the step's earlier outputs for the candidate; the "
                   "evidence store is kept.")
@click.option("--max-turns", type=click.IntRange(1), default=None, help="Tool-calling turns allowed (default: 8).")
@click.option("--cache", "cache_root", type=click.Path(file_okay=False, path_type=Path),
              help="HTTP cache for UniProt and Europe PMC (outside the repo). Default: $ENGINE_CACHE or ../cache.")
@click.option("--offline", is_flag=True, help="Serve UniProt and Europe PMC from the cache only; fail on any cache miss.")
@click.option("-v", "--verbose", is_flag=True, help="Name the candidate (gene symbol) on the terminal.")
def dossier(run_dir: Path, candidate_id: str | None, provider: str | None, model: str | None, effort: str,
            dry_run: bool, max_turns: int | None, cache_root: Path | None, offline: bool, verbose: bool) -> None:
    """Stage 5, second step: a gene dossier (UniProt + literature) for one candidate, validated like the chain."""
    from engine.agents.client import AgentError
    from engine.dossier.run import DEFAULT_MAX_TURNS, run_dossier
    from engine.retrieve.literature import LiteratureError
    from engine.retrieve.uniprot import UniprotError

    max_turns = max_turns or DEFAULT_MAX_TURNS
    defaulted = provider is None
    try:
        provider, model = providers.resolve(provider, model)
    except (FileNotFoundError, ValueError, providers.ProviderError) as e:
        click.echo(f"  FAILED: {e}", err=True)
        sys.exit(1)
    click.echo(f"dossier · {run_dir} · {'candidate given' if candidate_id else 'best-priority candidate'} · {provider} · "
               f"{model} · effort {effort}{' · OFFLINE' if offline else ''}{' · DRY RUN' if dry_run else ''}")
    try:
        manifest = run_dossier(run_dir, candidate_id, None, model, effort, dry_run, max_turns=max_turns,
                               cache_root=cache_root, offline=offline, provider=provider,
                               progress=lambda msg: click.echo("  " + msg))
    except AgentError as e:
        click.echo(f"  FAILED: {e}", err=True)
        if defaulted and not providers.key_present(provider):
            click.echo(f"  {providers.no_key_message()}", err=True)
        sys.exit(1)
    except (FileNotFoundError, KeyError, ValueError, UniprotError, LiteratureError) as e:
        # A source that answered with a shape the engine cannot cite, a run directory
        # without stage 5, a candidate with no chain. Anything else (an offline cache
        # miss, a record on the citable list that no store holds) keeps its traceback:
        # it is a fault in the run, not an answer about the gene.
        click.echo(f"  FAILED: {e}", err=True)
        sys.exit(1)
    m = json.loads(manifest.read_text())
    c, p = m["counts"], m["params"]
    label = p["candidate"] if verbose else "candidate"
    if p.get("accession") is None:
        click.echo(f"  {label}: UniProt has no reviewed human entry for the gene · no dossier (an absence, recorded)")
    elif dry_run:
        click.echo(f"  {label}: bundle and prompt written · papers retrieved: {c['papers_retrieved']:,} · "
                   f"features in map: {c['features_in_map']:,} · no model called")
    else:
        u = c.get("usage", {})
        click.echo(f"  {label}: claims kept: {c['claims_kept']:,} · variant positions: {c['variant_positions']:,} "
                   f"(replaced: {c['positions_replaced']:,}) · rejections: {c['rejections']:,} · "
                   f"records added: {c['evidence_records_added']:,}")
        click.echo(f"  tokens in/out: {u.get('input_tokens', 0):,}/{u.get('output_tokens', 0):,} · "
                   f"api calls: {u.get('api_calls', 0):,} · tool calls: {u.get('tool_calls', 0):,}")
        if p.get("evidence_chain_md"):
            click.echo("  evidence_chain.md: dossier section written")
    click.echo(f"  manifest: {manifest}")


commands: list[click.Command] = [dossier]
