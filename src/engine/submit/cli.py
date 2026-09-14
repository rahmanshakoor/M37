"""``engine submit`` — write the Track 1 CSV and dry-run it through the scorer."""

from __future__ import annotations

import json
from pathlib import Path

import click


def _pair(_ctx, _param, values):
    out = []
    for v in values:
        parts = [p.strip() for p in v.split(",")]
        if len(parts) != 2 or any(p.count(":") != 3 for p in parts):
            raise click.BadParameter("expected two keys like 7:117587778:G:T,7:117530975:G:A")
        out.append((parts[0], parts[1]))
    return out


@click.command(name="submit")
@click.option("--run", "run_dir", required=True, type=click.Path(file_okay=False, exists=True, path_type=Path),
              help="Run directory holding 03_filter/ (and, if run, 04_rank/ and 05_reason/).")
@click.option("--proband-id", default="PROBAND01", show_default=True, help="Value of the proband_id column.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path),
              help="CSV path. Default: <run>/08_submit/track1_submission.csv")
@click.option("--scorer", type=click.Path(dir_okay=False, path_type=Path),
              help="The challenge's evaluation.py for the local dry run. Default: <repo>/../space/evaluation.py if present.")
@click.option("--also-pair", multiple=True, callback=_pair,
              help="Extra pair row after the lead: KEY1,KEY2 in canonical spelling (repeatable).")
def submit(run_dir: Path, proband_id: str, out_path: Path | None, scorer: Path | None, also_pair) -> None:
    """Stage 8: the Track 1 CSV in the scorer's conventions, dry-run locally."""
    from engine.submit.run import STAGE_DIR, run_submit

    if scorer is None:
        default = Path(__file__).resolve().parents[4] / "space" / "evaluation.py"
        scorer = default if default.exists() else None
    csv_path = run_submit(run_dir, proband_id=proband_id, out=out_path, scorer=scorer, also_pairs=list(also_pair))
    m = json.loads((run_dir / STAGE_DIR / "manifest.json").read_text())
    click.echo(f"submit · {csv_path}")
    for r in m["params"]["rows"]:
        click.echo(f"  {r['epcr']:.2f}  {r['finding_type']:<9} {r['candidate_id']:<24} {'pair' if r['n_variants'] == 2 else 'single'}")
    for s in m["params"]["skipped"]:
        click.echo(f"  skipped {s['candidate_id']}: {s['why']}")
    d = m["params"].get("dry_run") or {}
    if d:
        a = d["if_lead_is_the_key"]
        click.echo(f"  dry run · if the lead row is the answer: rank points {a['rank_points']:.0f} · F-max {a['f_max']:.2f}"
                   f" (full match at rank {a['full_match_rank']})")
        if "if_only_first_allele_is_right" in d:
            b = d["if_only_first_allele_is_right"]
            click.echo(f"  dry run · if only the first allele is right: rank points {b['rank_points']:.0f} · F-max {b['f_max']:.2f}")
        z = d["if_key_is_absent"]
        click.echo(f"  dry run · if the answer is not in the file: rank points {z['rank_points']:.0f} · F-max {z['f_max']:.2f}")
    else:
        click.echo("  (no scorer found: pass --scorer path/to/evaluation.py for the dry run)")


commands = [submit]
