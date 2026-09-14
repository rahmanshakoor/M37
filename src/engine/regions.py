"""Gene panels → BED regions for a restricted ingest.

A panel is a YAML mapping of group name → gene symbols. Coordinates come from a JSON
file of ``symbol: [chrom, start, end]`` in 1-based inclusive GRCh38 coordinates (as
returned by the Ensembl REST lookup endpoint). The BED is 0-based half-open, padded,
sorted, and merged so no variant is emitted twice when regions overlap.

Gene attribution is deliberately *not* done here: a variant's gene comes from the
annotation stage, where the transcript model decides it.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from engine.contigs import canonical


def load_panel(path: Path) -> dict[str, list[str]]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    panel = {k: list(v) for k, v in raw.items() if k != "source"}
    return panel


def panel_genes(panel: dict[str, list[str]]) -> list[str]:
    return sorted({g for genes in panel.values() for g in genes})


def load_coords(path: Path) -> dict[str, tuple[str, int, int]]:
    raw = json.loads(Path(path).read_text())
    return {g: (str(c), int(s), int(e)) for g, (c, s, e) in raw.items()}


def _sort_key(chrom: str) -> tuple[int, str]:
    c = canonical(chrom) or chrom
    return (int(c), "") if c.isdigit() else (100, c)


def merge_intervals(intervals: list[tuple[str, int, int, str]]) -> list[tuple[str, int, int, str]]:
    """Merge overlapping/adjacent (chrom, start, end, name) intervals; names joined by ','."""
    out: list[tuple[str, int, int, str]] = []
    for chrom, s, e, name in sorted(intervals, key=lambda t: (_sort_key(t[0]), t[1], t[2])):
        if out and out[-1][0] == chrom and s <= out[-1][2]:
            pc, ps, pe, pn = out[-1]
            out[-1] = (pc, ps, max(pe, e), pn + "," + name)
        else:
            out.append((chrom, s, e, name))
    return out


def write_bed(
    genes: list[str],
    coords: dict[str, tuple[str, int, int]],
    out: Path,
    *,
    pad: int = 200,
    chrom_style: str = "ensembl",
) -> tuple[int, list[str]]:
    """Write a merged, padded BED for ``genes``. Returns (intervals written, genes lacking coords)."""
    missing = [g for g in genes if g not in coords]
    intervals = []
    for g in genes:
        if g in missing:
            continue
        chrom, start, end = coords[g]
        c = canonical(chrom)
        if c is None:
            missing.append(g)
            continue
        name = f"chr{'M' if c == 'MT' else c}" if chrom_style == "ucsc" else c
        intervals.append((name, max(0, start - 1 - pad), end + pad, g))
    merged = merge_intervals(intervals)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for chrom, s, e, name in merged:
            f.write(f"{chrom}\t{s}\t{e}\t{name}\n")
    return len(merged), missing
