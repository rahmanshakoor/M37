"""Markdown and HTML for a validated gene dossier.

The rendering rule mirrors the validation rule (as in :mod:`engine.agents.render`):
every claim shows the record it rests on as ``[record_id]`` where it is made, and the
document ends with References mapping every cited id to a URL. The engine-filled
residue and region print in a table beside the model's cited consequence, so a reader
sees at once which column is the record's and which the model's.

Footnotes: a redaction leaves a marker ``[^k]`` in the prose (task D's convention;
until it lands no marker appears and nothing here changes). Each section collects
the texts it printed and appends, before its blank line, one ``[^k]: citation removed
by the validator: <reason>`` per marker found — the reason from the rejection whose
``marker`` is ``k``, or the fixed sentence when no rejection is given.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable

from engine.agents.render import cited_ids, paper_label
from engine.agents.validator import EvidenceIndex, citation_tokens
from engine.dossier.schema import GeneDossier

MARKER = re.compile(r"\[\^(\d+)\]")
FOOTNOTE = "citation removed by the validator: {reason}"
FOOTNOTE_UNKNOWN = "citation removed by the validator (reason in the validation record)"
SECTIONS = (
    ("protein", "Protein"),
    ("mechanism_of_disease", "Mechanism of disease"),
    ("variant_positions", "Where the variants fall"),
    ("region_knowledge", "What is known about those regions"),
    ("genotype_patterns", "Published genotype patterns"),
    ("functional_test", "What a functional test would show"),
    ("limits", "Limits"),
)
"""Field → heading, in document order (then Literature and References)."""
_WS = re.compile(r"\s+")


# ------------------------------------------------------------------------ markdown

def render_dossier(dossier: GeneDossier, index: EvidenceIndex | None = None, *,
                   disclosure: str | None = None, rejections: Iterable[Any] = ()) -> str:
    rejections = list(rejections)
    out = [f"# Gene dossier — {dossier.gene_symbol} ({dossier.candidate_id})", ""]
    out.append(f"UniProt entry [uniprot:{dossier.uniprot_accession}] · residue and region per variant filled by the engine "
               "from the vep: and uniprot: records; every other statement is the model's, cited to a record.")
    out.append("")
    for name, title in SECTIONS:
        value = getattr(dossier, name)
        if name == "variant_positions":
            out.extend(_positions_section(title, value, rejections))
        elif name == "limits":
            out.extend(_section(title, [(_para(x), []) for x in value], rejections))
        else:
            out.extend(_section(title, [(_para(c.statement), list(c.evidence_ids)) for c in value], rejections))
    out.extend(_literature(dossier.literature, index))
    out.extend(_references(cited_ids(dossier, index), index))
    if disclosure:
        out.extend(["---", "", f"_{disclosure}_", ""])
    return "\n".join(out).rstrip("\n") + "\n"


def _section(title: str, items: list[tuple[str, list[str]]], rejections: list[Any]) -> list[str]:
    out = [f"## {title}", ""]
    if not items:
        out.append("- none")
    for text, ids in items:
        out.append(f"- {text} {_cite(ids, after=text)}".rstrip())
    out.extend(footnotes([t for t, _ in items], rejections))
    out.append("")
    return out


def _positions_section(title: str, positions: list[Any], rejections: list[Any]) -> list[str]:
    out = [f"## {title}", ""]
    if not positions:
        out.append("- none")
    else:
        out.extend(["| variant | residue | region | consequence (cited) |", "|---|---|---|---|"])
        for p in positions:
            residue = str(p.protein_position) if p.protein_position is not None else "–"
            region = "; ".join(p.region) if p.region else "–"
            text = _para(p.consequence) if p.consequence.strip() else "(no validated statement)"
            cites = _cite(p.evidence_ids, after=text)
            out.append(f"| {_cell(p.key)} | {residue} | {_cell(region)} | {_cell(text + (' ' + cites if cites else ''))} |")
    out.extend(footnotes([p.consequence for p in positions] + [r for p in positions for r in p.region], rejections))
    out.append("")
    return out


def footnotes(texts: Iterable[str], rejections: Iterable[Any]) -> list[str]:
    """One ``[^k]: …`` line per marker found in ``texts``, in order of first
    appearance, the reason from the rejection carrying that marker (objects or dicts)."""
    by_marker: dict[int, str] = {}
    for r in rejections:
        marker = r.get("marker") if isinstance(r, dict) else getattr(r, "marker", None)
        reason = r.get("reason") if isinstance(r, dict) else getattr(r, "reason", None)
        if isinstance(marker, int) and marker not in by_marker:
            by_marker[marker] = str(reason or "")
    found: list[int] = []
    for text in texts:
        for m in MARKER.finditer(str(text)):
            k = int(m.group(1))
            if k not in found:
                found.append(k)
    return [f"[^{k}]: " + (FOOTNOTE.format(reason=by_marker[k]) if k in by_marker else FOOTNOTE_UNKNOWN) for k in found]


def _literature(ids: list[str], index: EvidenceIndex | None) -> list[str]:
    out = ["## Literature", ""]
    if not ids:
        out.append("- none")
    for rid in ids:
        label = paper_label(index.get(rid)) if index is not None else ""
        out.append(f"- [{rid}]" + (f" {label}" if label else ""))
    out.append("")
    return out


def _references(ids: Iterable[str], index: EvidenceIndex | None) -> list[str]:
    out = ["## References", ""]
    ids = list(ids)
    if not ids:
        out.append("- none")
    for rid in ids:
        url = index.url(rid) if index is not None else None
        out.append(f"- [{rid}] — {url}" if url else f"- [{rid}] — (no URL in the evidence store)")
    out.append("")
    return out


def demote(markdown: str) -> str:
    """The dossier's headings one level down and its title line dropped, for splicing
    under a ``##`` heading of another document."""
    out = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            continue
        out.append("#" + line if line.startswith("#") else line)
    return "\n".join(out).strip("\n") + "\n"


def _cite(ids: list[str], after: str = "") -> str:
    inline = set(citation_tokens(after)) if after else set()
    return " ".join(f"[{i}]" for i in ids if i not in inline)


def _para(text: Any) -> str:
    return _WS.sub(" ", str(text if text is not None else "")).strip()


def _cell(text: str) -> str:
    return _para(text).replace("|", "\\|")


# ---------------------------------------------------------------------------- html

def render_dossier_html(view: dict[str, Any], prose: Callable[[Any], str], link: Callable[[dict[str, Any]], str]) -> str:
    """``<article class="dossier">`` for the report: the same sections as the Markdown,
    as ``<h4>`` headings. ``prose`` links the citations inside a text (the report's
    linker), ``link`` renders one resolved id (``{id, url, source, stage}``) as a
    link. Absence is said, never hidden."""
    out = ["<article class=\"dossier\"><h3>Gene dossier</h3>"]
    if not view.get("present"):
        out.append("<p class=\"muted\">No dossier step in this run (<span class=\"id\">05_reason/dossier/</span> absent).</p></article>")
        return "".join(out)
    d = view.get("dossier")
    head = f"Candidate <span class=\"id\">{_esc(view.get('candidate_id'))}</span> ({_esc(view.get('gene_symbol'))})"
    if view.get("uniprot"):
        head += f" · UniProt {link(view['uniprot'])}"
    out.append(f"<p>{head}. Residue and region per variant are the engine's, from the records; every other statement is the model's, cited.</p>")
    if view.get("dry_run"):
        out.append("<p class=\"notice warn\">The dossier step ran dry: the UniProt entry and the fixed searches were retrieved, the bundle and the prompt written; no model was called and no dossier was produced.</p>")
    elif view.get("manifest"):
        out.append(f"<p class=\"muted small\">model {_esc(view.get('model') or '–')}, effort {_esc(view.get('effort') or '–')}"
                   + (f" · {_esc(view['disclosure'])}" if view.get("disclosure") else "") + "</p>")
    for n in view.get("notes") or []:
        out.append(f"<p class=\"notice\">{_esc(n)}</p>")
    if d is None:
        if not view.get("dry_run"):
            out.append("<p class=\"muted\">No validated dossier for this candidate in the run directory.</p>")
        out.append("</article>")
        return "".join(out)
    for name, title in SECTIONS:
        out.append(f"<h4>{_esc(title)}</h4>")
        items = d.get(name) or []
        if name == "variant_positions":
            if not items:
                out.append("<p class=\"muted\">none</p>")
                continue
            rows = []
            for p in items:
                region = "".join(f"<li>{_esc(r)}</li>" for r in p.get("region") or []) or "<li class=\"muted\">–</li>"
                cites = "".join(f" {link(e)}" for e in p.get("evidence") or [])
                text = prose(p.get("consequence")) if str(p.get("consequence") or "").strip() else "<span class=\"muted\">(no validated statement)</span>"
                rows.append(f"<tr><td class=\"id\">{_esc(p.get('key'))}</td><td class=\"num\">{_esc(p.get('protein_position') if p.get('protein_position') is not None else '–')}</td>"
                            f"<td><ul class=\"region\">{region}</ul></td><td>{text}{cites}</td></tr>")
            out.append("<div class=\"tbl\"><table class=\"positions\"><thead><tr><th>variant</th><th>residue</th><th>region</th>"
                       "<th>consequence (cited)</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
        elif name == "limits":
            out.append(_ul([prose(x) for x in items]))
        else:
            out.append(_ul([prose(c.get("statement")) + "".join(f" {link(e)}" for e in c.get("evidence") or []) for c in items]))
    out.append("<h4>Literature</h4>" + _ul([link(e) for e in d.get("literature") or []]))
    out.append("<h4>References</h4>" + _ul([link(e) for e in d.get("references") or []]))
    out.append("</article>")
    return "".join(out)


def _ul(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>" if items else "<p class=\"muted\">none</p>"


def _esc(value: Any) -> str:
    return (str(value if value is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))
