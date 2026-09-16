"""Stage 5 command — registered by ``engine.cli`` from ``commands``.

Prints counts, classifications and the manifest path. No variant, genotype or gene
symbol reaches the terminal unless ``--verbose`` asks for candidate ids, so a pasted
shell transcript carries no patient data.

The provider is chosen here (``--provider``; ``engine.agents.providers.resolve``): the
dotenv outside the repository is loaded first — never overriding the real environment,
never echoed, and never at all under a pytest session, so a scripted test beside the
operator's key cannot become a paid call — then the provider defaults to whichever key
is present and ``--model`` to that provider's model. A failure with a defaulted
provider whose key is absent adds one line naming both keys.
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
              help="Run directory that already holds 02_retrieve/evidence and 03_filter/candidates.json.")
@click.option("--top", "top_n", type=click.IntRange(1), default=None,
              help="How many stage-3 candidates to reason about, in priority order (default: 3).")
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
              help="Write the bundles and the exact prompts, call no model, exit 0. Like any run it replaces the "
                   "stage's earlier outputs (chains included); the evidence store is kept.")
@click.option("--max-turns", type=click.IntRange(1), default=None,
              help="Tool-calling turns allowed per candidate (default: 10).")
@click.option("--cache", "cache_root", type=click.Path(file_okay=False, path_type=Path),
              help="HTTP cache for Europe PMC (outside the repo). Default: $ENGINE_CACHE or ../cache.")
@click.option("--offline", is_flag=True, help="Serve literature from the cache only; fail on any cache miss.")
@click.option("--case", "case_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="case.yaml, for the HPO terms when stage 4 has not run (default: 04_rank/joined.json).")
@click.option("-v", "--verbose", is_flag=True, help="Name the candidates (gene symbols) on the terminal.")
@click.option("--revalidate", is_flag=True,
              help="Call no model: replay the recorded transcripts (same tool calls, same final answers) through the "
                   "current validator and rewrite the chains. For re-checking a recorded answer after a rule change.")
def reason(run_dir: Path, top_n: int | None, provider: str | None, model: str | None, effort: str, dry_run: bool,
           max_turns: int | None, cache_root: Path | None, offline: bool, case_path: Path | None, verbose: bool,
           revalidate: bool) -> None:
    """Stage 5: an ACMG/AMP evidence chain per candidate, every criterion tied to a record id."""
    from engine.agents.client import AgentError
    from engine.reason.run import DEFAULT_MAX_TURNS, DEFAULT_TOP_N, STAGE_DIR, ReplayClient, run_reason

    top_n = top_n or DEFAULT_TOP_N
    max_turns = max_turns or DEFAULT_MAX_TURNS
    defaulted = provider is None
    client = None
    if revalidate:
        try:
            client = ReplayClient(run_dir / STAGE_DIR / "transcripts")
        except FileNotFoundError as e:
            click.echo(f"  FAILED: {e}", err=True)
            sys.exit(1)
        first = next(iter(client.transcripts.values()))
        provider, model, effort = "replay", first.get("model") or "replay", first.get("effort") or effort
    else:
        try:
            provider, model = providers.resolve(provider, model)
        except (FileNotFoundError, ValueError, providers.ProviderError) as e:
            click.echo(f"  FAILED: {e}", err=True)
            sys.exit(1)
    click.echo(f"reason · {run_dir} · top {top_n} · {provider} · {model} · effort {effort}"
               f"{' · OFFLINE' if offline else ''}{' · DRY RUN' if dry_run else ''}{' · REVALIDATE (replayed transcripts)' if revalidate else ''}")
    try:
        manifest = run_reason(run_dir, top_n, client, model, effort, dry_run, max_turns=max_turns, cache_root=cache_root,
                              offline=offline, case_path=case_path, provider=None if revalidate else provider,
                              progress=lambda msg: click.echo("  " + msg))
    except AgentError as e:
        click.echo(f"  FAILED: {e}", err=True)
        if defaulted and not providers.key_present(provider):
            click.echo(f"  {providers.no_key_message()}", err=True)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"  FAILED: {e}", err=True)
        sys.exit(1)
    m = json.loads(manifest.read_text())
    c, p = m["counts"], m["params"]
    if dry_run:
        click.echo(f"  candidates: {c['candidates_selected']:,} of {c['candidates_total']:,} · bundles and prompts written · no model called")
    else:
        u = c.get("usage", {})
        click.echo(f"  candidates: {c['candidates_reasoned']:,} of {c['candidates_total']:,} · chains: {c['chains_written']:,} · "
                   f"rejections: {c.get('rejections', 0):,} · records added: {c['evidence_records_added']:,} · "
                   f"tokens in/out: {u.get('input_tokens', 0):,}/{u.get('output_tokens', 0):,} · api calls: {u.get('api_calls', 0):,}")
        for i, cid in enumerate(p["candidates"], 1):
            cls = c.get("classifications", {}).get(cid, {})
            label = cid if verbose else f"candidate {i}"
            click.echo(f"  {label}: " + (", ".join(v.replace("_", " ") for v in cls.values()) or "no chain"))
    click.echo(f"  manifest: {manifest}")


commands: list[click.Command] = [reason]
