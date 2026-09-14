"""The run report as one HTML document — a clinical document, not a dashboard.

Why one file: a report is handed over, attached, archived. It carries its own CSS and
its one small script (the theme switch), links Google Fonts with real fallbacks, and
loads nothing else, so it reads the same on a laptop without the engine as it does in
the UI. Why a document: a reader checks it line by line — a candidate, then the rule
hits that put it there, then each ACMG criterion with the record it rests on — so the
page is typographic hierarchy and tables, numbered only where the order is real (the
stages are a sequence, the candidates are ranked), with every evidence id a link to
the record's own URL.

Everything on the page comes from :mod:`engine.report.views`; this module only
formats. It never computes a figure (a long allele frequency is shortened for the eye
with the full text in the ``title``), never invents a timestamp (the document is a
pure function of the run directory, so the same run renders to identical bytes), and
shows what the validator rejected rather than dropping it.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Iterable

from engine import __version__
from engine.agents import validator
from engine.agents.validator import KNOWN_SOURCES, canonical_id, citation_tokens
from engine.report import views

FONTS_URL = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
             "&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400"
             "&family=Spectral:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap")

_CITATION = validator._CITATION
"""The validator's own citation tokenizer (the regex behind ``citation_tokens``), so the
links the page draws and the citations the validator counted are the same tokens."""


# ---------------------------------------------------------------------- public

def render_run(run_dir: Path, *, top_n: int = views.DEFAULT_TOP_N) -> str:
    """The complete report for ``run_dir`` as an HTML string."""
    run_dir = Path(run_dir)
    summary = views.run_summary(run_dir)
    candidates = views.candidates_view(run_dir)
    ranking = views.ranking_view(run_dir, top_n=top_n)
    chains = views.chain_view(run_dir)
    medicine = views.medicine_view(run_dir)
    provenance = views.provenance_view(run_dir)
    index = views.evidence_index(run_dir)
    sources = set(KNOWN_SOURCES) | {rid.split(":", 1)[0] for rid in index}
    linker = _Linker(index, sources)

    title = f"Run report · {summary.get('sample') or summary['run_name']}"
    body = [
        _header(summary),
        _nav(),
        _candidates_section(candidates),
        _ranking_section(ranking),
        _chains_section(chains, linker),
        _medicine_section(medicine, linker),
        _provenance_section(provenance),
        _footer(summary),
    ]
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{esc(title)}</title>\n"
        "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">\n"
        "<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>\n"
        f"<link rel=\"stylesheet\" href=\"{FONTS_URL}\">\n"
        f"<style>\n{CSS}\n</style>\n</head>\n<body>\n<div class=\"page\">\n"
        + "\n".join(body)
        + f"\n</div>\n<script>\n{JS}\n</script>\n</body>\n</html>\n"
    )


def write_report(run_dir: Path, out: Path | None = None) -> Path:
    """Render ``run_dir`` to ``out`` (default ``<run>/report.html``); returns the path."""
    run_dir = Path(run_dir)
    out = Path(out) if out is not None else run_dir / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_run(run_dir), encoding="utf-8")
    return out


# ------------------------------------------------------------------- utilities

def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _dash(value: Any) -> str:
    """A cell: the value escaped, or an en dash for nothing."""
    return esc(value) if value not in (None, "", [], {}) else "–"


def _num(value: Any) -> str:
    """A figure as the file wrote it, in the numeric cell style."""
    return f"<td class=\"num\">{_dash(value)}</td>"


def _af(value: Any) -> str:
    """An allele frequency shortened for the eye (three significant figures), with the
    text as written in the ``title`` — a presentation of the file's number, not a new one."""
    text = "" if value is None else str(value)
    if not text:
        return "–"
    try:
        x = float(text)
    except ValueError:
        return esc(text)
    if len(text) <= 8:
        return esc(text)
    return f"<span title=\"{esc(text)}\">{esc(f'{x:.3g}')}</span>"


def _time(value: Any) -> str:
    """A manifest timestamp for reading: ``2026-09-13T17:23:50+00:00`` → ``2026-09-13 17:23:50``
    (the header says UTC); anything else is shown as written."""
    text = "" if value is None else str(value)
    if not text:
        return "–"
    if text.endswith("+00:00") and "T" in text:
        text = text[:-6].replace("T", " ", 1)
    return esc(text)


def _mark(text: str, kind: str = "") -> str:
    cls = f"mark {kind}".strip()
    return f"<span class=\"{cls}\">{esc(text)}</span>"


def _label(value: Any) -> str:
    """``likely_pathogenic`` → ``likely pathogenic``."""
    return str(value).replace("_", " ") if value is not None else "not computed"


def _classification_mark(value: Any) -> str:
    kinds = {"pathogenic": "crit", "likely_pathogenic": "crit", "vus": "warn", "likely_benign": "good", "benign": "good"}
    return _mark(_label(value), kinds.get(str(value), ""))


def _link(entry: dict[str, Any]) -> str:
    """An evidence id as a link to its record URL, the id as the link text; an id the
    stores do not hold is shown unresolved, never dropped."""
    rid = esc(entry.get("id"))
    url = entry.get("url")
    if url:
        return f"<a class=\"id\" href=\"{esc(url)}\" rel=\"noopener\">{rid}</a>"
    return f"<span class=\"id unresolved\" title=\"not in the evidence store of this run\">{rid}</span>"


def _links(entries: Iterable[dict[str, Any]]) -> str:
    out = [_link(e) for e in entries]
    return " ".join(out) if out else "–"


def _ids(values: Iterable[Any], sep: str = ", ") -> str:
    """Identifiers (variant keys, candidate ids, HPO terms) in the monospace style."""
    out = [f"<span class=\"id\">{esc(v)}</span>" for v in values]
    return sep.join(out)


def _list(items: Iterable[Any], cls: str = "") -> str:
    items = list(items)
    if not items:
        return "<p class=\"muted\">none</p>"
    attr = f" class=\"{cls}\"" if cls else ""
    return f"<ul{attr}>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def _pre(doc: Any) -> str:
    return "<pre>" + esc(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False, default=str)) + "</pre>"


def _table(headers: list[str], rows: list[str], cls: str = "") -> str:
    attr = f" class=\"{cls}\"" if cls else ""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    return f"<div class=\"tbl\"><table{attr}><thead><tr>{head}</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def _dl(pairs: Iterable[tuple[str, str]]) -> str:
    return "<dl class=\"facts\">" + "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in pairs) + "</dl>"


class _Linker:
    """Turns the citations inside prose into links — every token the validator's
    tokenizer (:func:`engine.agents.validator.citation_tokens`) counts as a citation,
    and no other: a bracket group ``[vep:…, pmid:…]``, a bare ``pmid:12345``,
    ``PMID 12345``, ``VCV…`` or ``NCT…``. A token whose prefix is not a record source
    (``[HP:0002205]``, ``chr7:117559590``) is text. The validator rewrites a resolving
    bare id to its canonical spelling without adding brackets, so bare ids are what
    validated prose carries; linking only bracketed ones would lose them. Runs on the
    escaped text, so the markup it adds is the only markup."""

    def __init__(self, index: dict[str, dict[str, Any]], sources: set[str]):
        self.index = index
        self.sources = {s.lower() for s in sources}

    def prose(self, text: Any) -> str:
        return self.linked(text)[0]

    def linked(self, text: Any) -> tuple[str, list[str]]:
        """The prose with its citations linked, and the record ids linked, in order —
        the same set ``citation_tokens`` finds, so a list after a sentence can leave
        out exactly what the sentence already links."""
        ids: list[str] = []

        def link(rid: str) -> str:
            if rid not in ids:
                ids.append(rid)
            return _link(views.resolve([rid], self.index)[0])

        def repl(m: re.Match[str]) -> str:
            if m.group("group") is not None:
                tokens = validator._group_tokens(m.group("group"))
                if tokens is None:  # ``[see pmid:1]``, ``[1]``: the bare rules apply inside the brackets
                    return "[" + _CITATION.sub(repl, m.group("group")) + "]"
                return "[" + ", ".join(link(canonical_id(t)) if self._cites(t) else t for t in tokens) + "]"
            if m.group("src") is not None:
                return link(canonical_id(m.group(0))) if self._cites(m.group(0)) else m.group(0)
            found = citation_tokens(m.group(0), self.sources)  # PMID n / VCV… / NCT… name a record; PMC, DOI, rs do not
            return link(found[0]) if found else m.group(0)

        return _CITATION.sub(repl, esc(text)), ids

    def _cites(self, token: str) -> bool:
        return token.split(":", 1)[0].lower() in self.sources


# ----------------------------------------------------------------------- header

def _header(s: dict[str, Any]) -> str:
    vcf = s.get("vcf") or {}
    hpo = _ids(s.get("hpo") or []) or "<span class=\"muted\">not recorded in this run</span>"
    facts = [
        ("Sample", f"<span class=\"id\">{_dash(s.get('sample'))}</span>"),
        ("Run directory", f"<span class=\"id\">{esc(s['run_dir'])}</span>"),
        ("VCF", f"<span class=\"id\">{_dash(vcf.get('path'))}</span>"),
        ("VCF sha256", f"<span class=\"id\">{_dash(vcf.get('sha256'))}</span>"),
        ("Case HPO terms", hpo + (f" <span class=\"muted\">({esc(s['hpo_source'])})</span>" if s.get("hpo_source") else "")),
    ]
    rows = []
    for st in s["stages"]:
        n = st["dir"][:2]
        if not st["present"]:
            status = _mark("not run", "")
        elif st.get("failed"):
            status = _mark("failed", "crit")
        elif not st["manifest"]:
            status = _mark("no manifest", "warn")
        elif st.get("dry_run"):
            status = _mark("dry run", "warn")
        else:
            status = _mark("ran", "good")
        headline = " · ".join(f"{esc(k)} {esc(v)}" for k, v in st.get("headline") or [] if v is not None)
        rows.append(
            f"<tr><td class=\"num\">{esc(n)}</td><td>{esc(st['stage'])}</td><td>{status}</td>"
            f"<td class=\"num\">{_dash(st.get('engine_version'))}</td>"
            f"<td class=\"num\">{_time(st.get('started_at'))}</td><td class=\"num\">{_time(st.get('finished_at'))}</td>"
            f"<td class=\"num\">{_dash(st.get('duration_s'))}</td><td>{headline or '–'}</td></tr>")
    return (
        "<header class=\"doc-head\">"
        "<p class=\"kicker\">Reproducible rare-disease variant engine</p>"
        f"<h1>Run report <span class=\"id\">{esc(s.get('sample') or s['run_name'])}</span></h1>"
        "<div class=\"head-tools\"><button type=\"button\" id=\"theme\" class=\"btn\" aria-live=\"polite\">Theme: system</button></div>"
        + _dl(facts)
        + "<h2 class=\"sub\">Stages</h2>"
        + _table(["#", "stage", "status", "engine", "started (UTC)", "finished (UTC)", "duration s", "counts from the manifest"], rows, "stages")
        + "<p class=\"muted small\">Duration is the span between each manifest's own started and finished timestamps.</p>"
        "</header>"
    )


def _nav() -> str:
    items = [("candidates", "Candidates"), ("ranking", "Blind ranking"), ("chains", "Evidence chains"),
             ("medicine", "Medicine"), ("provenance", "Provenance")]
    return "<nav class=\"toc\" aria-label=\"Sections\">" + "".join(f"<a href=\"#{a}\">{esc(t)}</a>" for a, t in items) + "</nav>"


# ------------------------------------------------------------------- candidates

def _candidates_section(c: dict[str, Any]) -> str:
    out = ["<section id=\"candidates\"><h2>Candidates</h2>"]
    if not c["present"]:
        out.append("<p class=\"muted\">Stage 3 has not run: no <span class=\"id\">03_filter/candidates.json</span> in this run.</p></section>")
        return "".join(out)
    counts = c.get("counts") or {}
    by_model = ", ".join(f"{esc(k)} {esc(v)}" for k, v in (counts.get("candidates_by_model") or {}).items()) or "none"
    dropped = ", ".join(f"{esc(k)} {esc(v)}" for k, v in (counts.get("dropped_by_rule") or {}).items()) or "none"
    out.append(f"<p>Stage 3 read {_dash(counts.get('rows_in'))} rows, kept {_dash(counts.get('kept'))} and dropped "
               f"{_dash(counts.get('dropped'))} ({dropped}); {_dash(counts.get('candidates'))} candidate(s): {by_model}. "
               "Candidates are listed in the shortlist's priority order"
               + (", with the blind ranker's rank beside each (stage 4)." if c["rank_present"] else "; stage 4 has not run, so no blind rank is shown.")
               + "</p>")
    if c["rank_present"] and c["candidates"]:
        top = "is also the blind ranker's rank 1" if c.get("top_agreement") else "is not the blind ranker's rank 1"
        order = ("the two orderings of the ranked candidates agree" if c.get("order_agreement")
                 else "the two orderings of the ranked candidates differ" if c.get("order_agreement") is not None
                 else "Exomiser ranked none of the joined candidates")
        out.append(f"<p>The first candidate {top}; {order}.</p>")
    out.append(_stale_join_notice(c.get("join_check")))
    rows = []
    for cand in c["candidates"]:
        ex = cand.get("exomiser")
        agreement = {"agrees": _mark("agrees", "good"), "disagrees": _mark("disagrees", "warn"), "unranked": _mark("unranked", "warn"),
                     "not_joined": _mark("not in the stage-4 join", "warn")}
        rows.append(
            f"<tr><td class=\"num\">{_dash(cand.get('priority'))}</td>"
            f"<td><a href=\"#cand-{esc(views.candidate_file_name(str(cand['candidate_id'])))}\" class=\"id\">{esc(cand['candidate_id'])}</a></td>"
            f"<td>{_dash(cand.get('gene_symbol'))}</td><td>{_dash(cand.get('model'))}</td>"
            f"<td class=\"num\">{len(cand.get('variants') or [])}</td>"
            f"<td>{'yes' if cand.get('clinvar_plp') else 'no'}</td>"
            f"<td>{_dash((cand.get('phase') or {}).get('status'))}</td>"
            + (f"{_num((ex or {}).get('rank'))}{_num((ex or {}).get('score'))}{_num((ex or {}).get('phenotype_score'))}"
               f"{_num((ex or {}).get('variant_score'))}<td>{_dash((ex or {}).get('moi'))}</td>"
               f"<td>{agreement.get(cand.get('agreement'), '–')}</td>" if c["rank_present"] else "")
            + "</tr>")
    headers = ["priority", "candidate", "gene", "model", "alleles", "ClinVar P/LP", "phase"]
    if c["rank_present"]:
        headers += ["Exomiser rank", "score", "phenotype", "variant", "MOI", "blind ranker"]
    out.append(_table(headers, rows, "candidates"))
    for cand in c["candidates"]:
        out.append(_candidate_block(cand, c["rank_present"]))
    out.append("</section>")
    return "".join(out)


def _candidate_block(cand: dict[str, Any], rank_present: bool) -> str:
    cid = str(cand["candidate_id"])
    anchor = views.candidate_file_name(cid)
    phase = cand.get("phase") or {}
    out = [f"<article class=\"candidate\" id=\"cand-{esc(anchor)}\">"
           f"<h3><span class=\"num\">{_dash(cand.get('priority'))}</span> <span class=\"id\">{esc(cid)}</span> "
           f"<span class=\"muted\">{_dash(cand.get('gene_symbol'))} · {_dash(cand.get('gene_id'))} · {_dash(cand.get('model'))}</span></h3>"]
    rows = []
    for v in cand.get("variants") or []:
        clinvar = esc(v.get("clinvar_pathogenicity"))
        if v.get("clinvar_stars"):
            clinvar += f" <span class=\"muted\">{esc(v['clinvar_stars'])} stars</span>"
        rows.append(
            f"<tr><td class=\"id\">{esc(v.get('key'))}</td><td>{_dash(v.get('consequence'))} <span class=\"muted\">{_dash(v.get('impact'))}</span></td>"
            f"<td class=\"id\">{_dash(v.get('hgvsc'))}<br>{_dash(v.get('hgvsp'))}</td>"
            f"<td class=\"id\">{_dash(v.get('transcript_id'))}<br>{_dash(v.get('mane'))}</td>"
            f"<td class=\"num\">{_dash(v.get('gt'))}</td><td class=\"num\">{_dash(v.get('ad'))}</td>"
            f"{_num(v.get('dp'))}{_num(v.get('gq'))}"
            f"<td class=\"num\">{_af(v.get('af_used'))}<br><span class=\"muted small\">{_dash(v.get('af_source'))}</span></td>"
            f"{_num(v.get('gnomad_nhom'))}<td>{clinvar or '–'}<br><span class=\"id small\">{_dash(v.get('clinvar_vcv'))}</span></td>"
            f"{_num(v.get('spliceai_ds_max'))}<td>{_dash(v.get('sift_pred'))} / {_dash(v.get('polyphen_pred'))}</td>"
            f"<td>{', '.join(esc(x) for x in v.get('caveats') or []) or '–'}</td>"
            f"<td class=\"ev\">{_links(v.get('evidence') or [])}</td></tr>")
    out.append(_table(["variant", "consequence", "HGVS", "transcript · MANE", "GT", "AD", "DP", "GQ", "AF used", "nhom",
                       "ClinVar", "SpliceAI", "SIFT / PolyPhen", "caveats", "evidence"], rows, "variants"))
    facts = [
        ("Phase", f"{_dash(phase.get('status'))} <span class=\"muted\">{_dash(phase.get('evidence'))}</span>"),
        ("Rule hits", " ".join(f"<span class=\"id chip\">{esc(h)}</span>" for h in cand.get("rule_hits") or []) or "–"),
        ("Caveats", ", ".join(esc(x) for x in cand.get("caveats") or []) or "none"),
    ]
    ex = cand.get("exomiser")
    if rank_present:
        if ex:
            by_moi = "; ".join(f"{esc(moi)} rank {esc(d.get('rank'))} score {esc(d.get('exomiser_score'))}"
                               for moi, d in sorted((ex.get("by_moi") or {}).items()))
            facts.append(("Blind ranker", f"rank {esc(ex.get('rank'))} · score {esc(ex.get('score'))} · phenotype {esc(ex.get('phenotype_score'))} · "
                                          f"variant {esc(ex.get('variant_score'))} · MOI {_dash(ex.get('moi'))} · matched by {_dash(ex.get('match'))} · "
                                          f"contributing {_ids(ex.get('variants_matched') or []) or 'none of the shortlist alleles'}"
                                          + (f" · <span class=\"muted\">{by_moi}</span>" if by_moi else "")
                                          + (f" · {_link(ex['evidence'])}" if ex.get("evidence") else "")))
        elif cand.get("join") == "missing":
            facts.append(("Blind ranker", "not in the stage-4 join: <span class=\"id\">04_rank/joined.json</span> does not hold this candidate "
                                          "(stage 3 wrote the shortlist after stage 4 joined it); whether Exomiser ranked the gene is in the "
                                          "<a href=\"#ranking\">blind ranking</a> below, not here. Rerun <span class=\"id\">engine rank --join-only</span>."))
        else:
            facts.append(("Blind ranker", "Exomiser did not rank this gene"))
    out.append(_dl(facts))
    out.append("</article>")
    return "".join(out)


def _stale_join_notice(check: dict[str, Any] | None) -> str:
    """A warning when stage 4's join predates the shortlist it joins (see
    :func:`views.join_status`); nothing when the join is current or stage 4 is absent."""
    if not check or not check.get("stale"):
        return ""
    why = []
    if check.get("missing"):
        why.append(f"shortlist candidate(s) {_ids(check['missing'])} are not in <span class=\"id\">04_rank/joined.json</span>")
    if check.get("dropped"):
        why.append(f"joined candidate(s) {_ids(check['dropped'])} are no longer on the shortlist")
    rec, cur = check.get("candidates_sha256_recorded"), check.get("candidates_sha256_current")
    if rec and cur and rec != cur:
        why.append(f"<span class=\"id\">03_filter/candidates.json</span> has changed since stage 4 read it "
                   f"(sha256 recorded <span class=\"id\">{esc(rec[:12])}…</span>, now <span class=\"id\">{esc(cur[:12])}…</span>)")
    return ("<p class=\"notice warn\">The stage-4 join is stale — stage 3 wrote the current shortlist after stage 4 joined it: " + "; ".join(why) +
            ". Stage-4 figures beside a candidate describe the shortlist as it was then; rerun "
            "<span class=\"id\">engine rank --join-only</span> to join the current one.</p>")


# ---------------------------------------------------------------------- ranking

def _ranking_section(r: dict[str, Any]) -> str:
    out = ["<section id=\"ranking\"><h2>Blind ranking</h2>"]
    if not r["present"]:
        out.append("<p class=\"muted\">Stage 4 has not run: no <span class=\"id\">04_rank/ranking.tsv</span> or <span class=\"id\">joined.json</span> in this run.</p></section>")
        return "".join(out)
    counts = r.get("counts") or {}
    out.append(f"<p>Exomiser {_dash(r.get('exomiser_version'))} (data {_dash(r.get('data_version'))}) saw the VCF and "
               f"{len(r.get('hpo') or [])} HPO term(s) only — no gene list, no shortlist. It ranked "
               f"{_dash(counts.get('candidates_ranked'))} of {_dash(counts.get('candidates'))} shortlist candidate(s); "
               f"{_dash(counts.get('exomiser_genes_not_in_shortlist'))} ranked gene(s) are not on the shortlist. "
               f"Showing the top {esc(r['top_n'])} of {esc(r['rows_total'])} ranked gene(s) plus every shortlist gene; scores as written in "
               "<span class=\"id\">ranking.tsv</span>.</p>")
    out.append(_stale_join_notice(r.get("join_check")))
    rows = []
    for row in r["rows"]:
        if not row.get("in_shortlist"):
            on = _mark("not on shortlist", "warn")
        else:
            on = f"<a class=\"id\" href=\"#cand-{esc(views.candidate_file_name(str(row['candidate_id'])))}\">{esc(row['candidate_id'])}</a>"
            if row.get("joined") is False:
                on += " " + _mark("not in the stage-4 join", "warn")
        rows.append(
            f"<tr>{_num(row.get('rank'))}<td>{_dash(row.get('gene_symbol'))}</td>{_num(row.get('exomiser_score'))}"
            f"{_num(row.get('phenotype_score'))}{_num(row.get('variant_score'))}<td>{_dash(row.get('moi'))}</td>"
            f"{_num(row.get('n_variants'))}<td>{_ids(row.get('variants') or [], ' ') or '–'}</td>"
            f"<td>{on}</td><td>{_link(row['evidence'])}</td></tr>")
    out.append(_table(["rank", "gene", "score", "phenotype", "variant", "MOI", "n", "contributing variants", "shortlist", "record"], rows, "ranking"))
    only = r.get("exomiser_only") or []
    if only:
        out.append("<h3>Ranked genes not on the shortlist</h3>")
        rows = [f"<tr>{_num(g.get('rank'))}<td>{_dash(g.get('gene_symbol'))}</td>{_num(g.get('exomiser_score'))}{_num(g.get('phenotype_score'))}"
                f"{_num(g.get('variant_score'))}<td>{_dash(g.get('moi'))}</td>{_num(g.get('n_variants'))}<td>{_link(g['evidence'])}</td></tr>" for g in only]
        out.append(_table(["rank", "gene", "score", "phenotype", "variant", "MOI", "n", "record"], rows, "ranking"))
    else:
        out.append("<p class=\"muted\">Every ranked gene recorded in <span class=\"id\">joined.json</span> is on the shortlist.</p>")
    out.append("</section>")
    return "".join(out)


# ----------------------------------------------------------------------- chains

def _chains_section(ch: dict[str, Any], linker: _Linker) -> str:
    out = ["<section id=\"chains\"><h2>Evidence chains</h2>"]
    if not ch["present"]:
        out.append("<p class=\"muted\">Stage 5 has not run: no <span class=\"id\">05_reason/</span> in this run.</p></section>")
        return "".join(out)
    if ch.get("manifest"):
        mode = "a dry run: bundles and prompts were written and no model was called" if ch.get("dry_run") else \
            f"model {_dash(ch.get('model'))}, effort {_dash(ch.get('effort'))}"
        out.append(f"<p>Stage 5 was {mode}; candidates selected: "
                   f"{_ids(ch.get('candidates_selected') or []) or 'none'}; "
                   f"chains written by the stage: {_dash(ch.get('chains_written'))}. The classification of every variant is computed by the "
                   "engine from the criteria that survived validation (ACMG/AMP 2015 combining rules), never asserted by the model.</p>")
        if ch.get("disclosure"):
            out.append(f"<p class=\"muted small\">{esc(ch['disclosure'])}</p>")
    else:
        out.append("<p class=\"muted\">Stage 5 wrote no manifest.</p>")
    out.append(_failures_notice(ch.get("failures") or [], 5, "chain"))
    if not ch["chains"]:
        out.append("<p class=\"muted\">No chain in <span class=\"id\">05_reason/chains/</span>.</p></section>")
        return "".join(out)
    for chain in ch["chains"]:
        out.append(_chain_block(chain, linker))
    out.append("</section>")
    return "".join(out)


def _chain_block(chain: dict[str, Any], linker: _Linker) -> str:
    cid = str(chain["candidate_id"])
    anchor = views.candidate_file_name(cid)
    out = [f"<article class=\"chain\" id=\"chain-{esc(anchor)}\">"
           f"<h3>Chain <span class=\"id\">{esc(cid)}</span> <span class=\"muted\">{_dash(chain.get('gene_symbol'))} · {_dash(chain.get('model'))}</span></h3>"]
    if not chain.get("claimed_by_manifest"):
        out.append(f"<p class=\"notice warn\">Not claimed by the stage-5 manifest: {esc(chain.get('manifest_note'))} "
                   f"(<span class=\"id\">{esc(chain.get('file'))}</span>).</p>")
    verdicts = " · ".join(f"<span class=\"id\">{esc(v.get('key'))}</span> {_classification_mark(v.get('classification'))}" for v in chain["variants"])
    out.append(f"<p class=\"verdict\"><strong>Classification (engine-computed):</strong> {verdicts or '–'}</p>")
    for v in chain["variants"]:
        out.append(f"<h4>Variant <span class=\"id\">{esc(v.get('key'))}</span> {_classification_mark(v.get('classification'))}</h4>")
        if str(v.get("summary") or "").strip():
            out.append(f"<p>{linker.prose(v['summary'])}</p>")
        rows = []
        for c in v.get("criteria") or []:
            met = _mark("met", "good") if c.get("met") else _mark("not met", "")
            evidence = _links(c.get("evidence") or []) if c.get("evidence") else "<span class=\"muted\">case-level; cites no record</span>"
            rows.append(f"<tr><td class=\"id\">{esc(c.get('code'))}</td><td>{_label(c.get('strength'))}</td><td>{met}</td>"
                        f"<td class=\"prose\">{linker.prose(c.get('justification'))}</td><td class=\"ev\">{evidence}</td></tr>")
        if rows:
            out.append(_table(["code", "strength", "met", "justification", "evidence"], rows, "criteria"))
        else:
            out.append("<p class=\"muted\">No criterion survived validation for this variant.</p>")
    out.append(f"<h4>Phase</h4><p>{linker.prose(chain.get('phase_statement'))}</p>")
    out.append(f"<h4>Mechanism hypothesis</h4><p>{linker.prose(chain.get('mechanism_hypothesis'))}</p>")
    out.append("<h4>Limits</h4>" + _list(linker.prose(x) for x in chain.get("limits") or []))
    out.append("<h4>What would change the call</h4>" + _list(linker.prose(x) for x in chain.get("what_would_change_the_call") or []))
    out.append("<h4>Literature</h4>" + _list(_link(e) for e in chain.get("literature") or []))
    out.append("<h4>References</h4>" + _references(chain.get("references") or []))
    out.append(_validation_block(chain.get("validation"), "chain"))
    out.append("</article>")
    return "".join(out)


def _failures_notice(failures: list[dict[str, Any]], stage_n: int, product: str) -> str:
    """One warning per ``transcripts/<candidate>.failed.json``: a live stage that failed
    on the model is a fact about the run, stated with its error and request id — not
    a stage that never ran."""
    out = []
    for f in failures:
        rid = f" (request id <span class=\"id\">{esc(f['request_id'])}</span>)" if f.get("request_id") else ""
        out.append(f"<p class=\"notice warn\">Stage {stage_n} failed on <span class=\"id\">{esc(f['candidate_id'])}</span>: "
                   f"{esc(f['error'])}{rid}; no {product} was written for it and the stage wrote no manifest. "
                   f"{esc(f.get('turns') or 0)} turn(s) completed before the failure are in "
                   f"<span class=\"id\">{esc(f['file'])}</span>.</p>")
    return "".join(out)


def _references(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "<p class=\"muted\">none</p>"
    rows = [f"<tr><td>{_link(e)}</td><td>{_dash(e.get('source'))}</td><td class=\"id small\">{_dash(e.get('stage'))}</td>"
            f"<td class=\"id small url\">{_dash(e.get('url'))}</td></tr>" for e in entries]
    return _table(["record", "source", "store", "url"], rows, "references")


def _validation_block(v: dict[str, Any] | None, what: str) -> str:
    out = ["<h4>Validator</h4>"]
    if v is None:
        out.append(f"<p class=\"muted\">No validation record for this {what} in the run directory.</p>")
        return "".join(out)
    counts = v.get("counts") or {}
    shown = [(k, counts[k]) for k in ("ids_checked", "ids_unknown", "items_dropped", "duplicates_dropped", "literature_removed",
                                       "redactions", "identifiers_unverified", "strength_capped", "chembl_cleared",
                                       "frequency_recomputed", "frequency_disputed", "frequency_unverified", "classification_replaced")
             if k in counts]
    out.append("<p>" + " · ".join(f"{esc(k.replace('_', ' '))} <span class=\"num\">{esc(n)}</span>" for k, n in shown) + "</p>")
    rej = v.get("rejections") or []
    out.append(f"<h5>Rejections ({len(rej)})</h5>")
    if rej:
        out.append(_table(["path", "reason"], [f"<tr><td class=\"id\">{esc(r.get('path'))}</td><td>{esc(r.get('reason'))}</td></tr>" for r in rej], "rejections"))
    else:
        out.append("<p class=\"muted\">nothing rejected</p>")
    dis = v.get("disputes") or []
    out.append(f"<h5>Frequency disputes ({len(dis)})</h5>")
    if dis:
        rows = [f"<tr><td class=\"id\">{esc(d.get('path'))}</td><td class=\"id\">{esc(d.get('code'))}</td>"
                f"<td>{'met' if d.get('claimed_met') else 'not met'}</td><td>{'met' if d.get('recomputed_met') else 'not met'}</td>"
                f"{_num(d.get('af'))}<td class=\"id\">{_dash(d.get('record_id'))}</td><td>{esc(d.get('reason'))}</td></tr>" for d in dis]
        out.append(_table(["path", "code", "claimed", "recomputed", "af", "record", "reason"], rows, "disputes"))
    else:
        out.append("<p class=\"muted\">no dispute</p>")
    notes = v.get("notes") or []
    if notes:
        out.append("<h5>Notes</h5>" + _list(esc(n) for n in notes))
    rules = v.get("rules") or {}
    if rules:
        out.append("<details><summary>Rules the validator applied</summary>" + _dl((k, esc(val)) for k, val in rules.items()) + "</details>")
    return "".join(out)


# --------------------------------------------------------------------- medicine

def _medicine_section(m: dict[str, Any], linker: _Linker) -> str:
    out = ["<section id=\"medicine\"><h2>Medicine</h2>"]
    if not m["present"]:
        out.append("<p class=\"muted\">Stage 6 has not run: no <span class=\"id\">06_medicine/</span> in this run.</p></section>")
        return "".join(out)
    verdicts = " · ".join(f"<span class=\"id\">{esc(v.get('key'))}</span> {_classification_mark(v.get('classification'))}" for v in m.get("stage5_verdicts") or [])
    out.append(f"<p>Candidate <span class=\"id\">{_dash(m.get('candidate_id'))}</span> ({_dash(m.get('gene_symbol'))})"
               + (f"; stage-5 classification as the bundle carried it: {verdicts}" if verdicts else "") + ". "
               "Hypotheses for follow-up argued from retrieved records only — never a treatment recommendation.</p>")
    if m.get("dry_run"):
        out.append("<p class=\"notice warn\">Stage 6 ran dry: the bundle and the exact prompt were written; no model was called and no report was produced.</p>")
    elif m.get("manifest"):
        out.append(f"<p class=\"muted small\">model {_dash(m.get('model'))}, effort {_dash(m.get('effort'))}"
                   + (f" · {esc(m['disclosure'])}" if m.get("disclosure") else "") + "</p>")
    out.append(_failures_notice(m.get("failures") or [], 6, "report"))
    r = m.get("report")
    if r is None:
        if not m.get("dry_run") and not m.get("failures"):
            out.append("<p class=\"muted\">No <span class=\"id\">06_medicine/report.json</span> in this run.</p>")
        out.append(_validation_block(m.get("validation"), "report") if m.get("validation") else "")
        out.append("</section>")
        return "".join(out)
    out.append("<h3>Mechanism</h3>" + _claims(r.get("mechanism") or [], linker))
    out.append("<h3>Pathway targets</h3>" + _claims(r.get("pathway_targets") or [], linker))
    out.append("<h3>Drug candidates</h3>")
    drugs = r.get("candidates") or []
    if not drugs:
        out.append("<p class=\"muted\">None survived validation.</p>")
    for d in drugs:
        head = f"<span class=\"num\">{esc(d['n'])}</span> {esc(d.get('name'))}"
        if d.get("chembl"):
            head += f" <span class=\"muted\">({_link(d['chembl'])})</span>"
        elif d.get("chembl_id"):
            head += f" <span class=\"muted id\">({esc(d['chembl_id'])})</span>"
        out.append(f"<article class=\"drug\"><h4>{head}</h4>" + _dl([
            ("Mechanism of action", linker.prose(d.get("mechanism_of_action"))),
            ("Approval status", linker.prose(d.get("approval_status"))),
            ("Rationale", linker.prose(d.get("rationale"))),
            ("Evidence", _links(d.get("evidence") or [])),
            ("Trials", _links(d.get("trials") or [])),
            ("Counter-arguments", _list(linker.prose(a) for a in d.get("counter_arguments") or [])),
        ]) + "</article>")
    out.append("<h3>Follow-up experiments</h3>" + _list(linker.prose(x) for x in r.get("follow_up_experiments") or []))
    out.append("<h3>Limits</h3>" + _list(linker.prose(x) for x in r.get("limits") or []))
    out.append("<h3>Literature</h3>" + _list(_link(e) for e in r.get("literature") or []))
    out.append("<h3>References</h3>" + _references(r.get("references") or []))
    out.append(_validation_block(m.get("validation"), "report"))
    out.append("</section>")
    return "".join(out)


def _claims(claims: list[dict[str, Any]], linker: _Linker) -> str:
    """One bullet per claim: the statement with its inline citations linked, then the
    ``evidence_ids`` the sentence did not already cite."""
    if not claims:
        return "<p class=\"muted\">none</p>"
    items = []
    for c in claims:
        prose, inline = linker.linked(c.get("statement"))
        rest = [e for e in c.get("evidence") or [] if e["id"] not in inline]
        items.append(prose + (f" <span class=\"cites\">{_links(rest)}</span>" if rest else ""))
    return _list(items)


# ------------------------------------------------------------------- provenance

def _provenance_section(p: dict[str, Any]) -> str:
    out = ["<section id=\"provenance\"><h2>Provenance</h2>",
           "<p>Every manifest of the run: what went in (path, size, sha256), which tools touched it, the parameters, "
           "what came out, and the stage's notes — as written.</p>"]
    for st in p["stages"]:
        m = st.get("manifest")
        out.append(f"<article class=\"manifest\"><h3><span class=\"num\">{esc(st['dir'][:2])}</span> {esc(st['stage'])}</h3>")
        if m is None:
            out.append("<p class=\"muted\">" + ("directory present, no manifest" if st.get("present") else "not run") + "</p></article>")
            continue
        out.append(_dl([
            ("Engine", f"<span class=\"num\">{_dash(m.get('engine_version'))}</span> · {_dash(m.get('platform'))}"),
            ("Started (UTC)", f"<span class=\"num\">{_time(m.get('started_at'))}</span>"),
            ("Finished (UTC)", f"<span class=\"num\">{_time(m.get('finished_at'))}</span>"),
        ]))
        out.append("<h4>Inputs</h4>" + _files_table(m.get("inputs") or []))
        tools = m.get("tools") or {}
        out.append("<h4>Tools</h4>" + _kv((k, esc(v)) for k, v in tools.items()))
        out.append("<h4>Counts</h4>" + _counts(m.get("counts") or {}))
        out.append("<h4>Outputs</h4>" + _files_table(m.get("outputs") or []))
        notes = m.get("notes") or []
        out.append("<h4>Notes</h4>" + (_list(esc(n) for n in notes) if notes else "<p class=\"muted\">none</p>"))
        out.append("<details><summary>Parameters</summary>" + _pre(m.get("params") or {}) + "</details>")
        out.append("</article>")
    out.append("</section>")
    return "".join(out)


def _files_table(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "<p class=\"muted\">none recorded</p>"
    rows = [f"<tr><td class=\"id\">{esc(e.get('name'))}</td><td class=\"id path\">{_dash(e.get('path'))}</td>{_num(e.get('bytes'))}"
            f"<td class=\"id sha\">{_dash(e.get('sha256'))}</td></tr>" for e in entries]
    return _table(["name", "path", "bytes", "sha256"], rows, "files")


def _counts(counts: dict[str, Any]) -> str:
    return _kv((k, esc(json.dumps(v, sort_keys=True, ensure_ascii=False)) if isinstance(v, (dict, list)) else _dash(v))
               for k, v in counts.items())


def _kv(pairs: Iterable[tuple[str, str]]) -> str:
    """A compact key–value grid for a manifest's counts and tools."""
    items = list(pairs)
    if not items:
        return "<p class=\"muted\">none recorded</p>"
    return "<dl class=\"kv\">" + "".join(f"<div><dt class=\"id\">{esc(k)}</dt><dd class=\"num\">{v}</dd></div>" for k, v in items) + "</dl>"


def _footer(s: dict[str, Any]) -> str:
    versions = ", ".join(s.get("engine_versions") or []) or __version__
    return ("<footer class=\"doc-foot\"><p>Rendered by <span class=\"id\">engine report</span> from the files of "
            f"<span class=\"id\">{esc(s['run_dir'])}</span> (engine {esc(versions)}). Every figure is quoted from a file of the run; "
            "every evidence id links to the record the run stored. Hypotheses only — not a diagnosis, not a treatment recommendation.</p></footer>")


# --------------------------------------------------------------------------- CSS

CSS = """
:root {
  --bg: #FBFAF7; --bg-2: #F2F0EA; --ink: #1E1E1B; --ink-2: #57554F; --ink-3: #8B8880;
  --line: #D9D5CC; --line-2: #E9E6DF; --accent: #2A5D7C;
  --good: #2E6A3A; --good-bg: #E7F0E8; --warn: #85560A; --warn-bg: #F6EDD9; --crit: #963030; --crit-bg: #F5E3E3;
  --serif: "Spectral", Georgia, "Times New Roman", serif;
  --sans: "IBM Plex Sans", -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #171819; --bg-2: #1F2123; --ink: #E7E5E0; --ink-2: #B5B2AA; --ink-3: #807D76;
    --line: #3B3E42; --line-2: #2B2E32; --accent: #83B4D2;
    --good: #93CC9F; --good-bg: #1C2E20; --warn: #E4B96A; --warn-bg: #33290F; --crit: #E5A0A0; --crit-bg: #3B1F1F;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #171819; --bg-2: #1F2123; --ink: #E7E5E0; --ink-2: #B5B2AA; --ink-3: #807D76;
  --line: #3B3E42; --line-2: #2B2E32; --accent: #83B4D2;
  --good: #93CC9F; --good-bg: #1C2E20; --warn: #E4B96A; --warn-bg: #33290F; --crit: #E5A0A0; --crit-bg: #3B1F1F;
  color-scheme: dark;
}
* { box-sizing: border-box; }
html { background: var(--bg); }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 15px; line-height: 1.55;
       -webkit-font-smoothing: antialiased; }
.page { max-width: 78rem; margin: 0 auto; padding-block: 2rem 4rem; padding-inline: clamp(16px, 4vw, 48px); }
h1, h2, h3 { font-family: var(--serif); font-weight: 500; line-height: 1.2; letter-spacing: -0.005em; }
h1 { font-size: 2rem; margin: 0.25rem 0 1rem; }
h2 { font-size: 1.5rem; margin: 3rem 0 0.75rem; padding-top: 1rem; border-top: 1px solid var(--line); }
h2.sub { font-size: 1.15rem; margin-top: 1.5rem; border-top: 0; padding-top: 0; }
h3 { font-size: 1.2rem; margin: 2rem 0 0.5rem; }
h4 { font-family: var(--sans); font-size: 0.95rem; font-weight: 600; margin: 1.5rem 0 0.4rem; }
h5 { font-family: var(--sans); font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
     color: var(--ink-2); margin: 1rem 0 0.3rem; }
p { margin: 0.4rem 0 0.8rem; max-width: 72ch; }
p.prose, td.prose { max-width: none; }
a { color: var(--accent); text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 2px; }
a:hover { text-decoration-thickness: 2px; }
.kicker { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-2); margin: 0; }
.doc-head { position: relative; }
.head-tools { position: absolute; top: 0; right: 0; }
.btn { font: inherit; font-size: 0.8rem; color: var(--ink-2); background: transparent; border: 1px solid var(--line);
       padding: 0.2rem 0.6rem; cursor: pointer; }
.btn:hover { color: var(--ink); border-color: var(--ink-3); }
.toc + section h2 { border-top: 0; padding-top: 0; margin-top: 2.5rem; }
.toc { display: flex; flex-wrap: wrap; gap: 0.25rem 1.25rem; margin: 1.5rem 0 0; padding: 0.6rem 0; border-top: 1px solid var(--line);
       border-bottom: 1px solid var(--line); font-size: 0.9rem; }
.id { font-family: var(--mono); font-size: 0.86em; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
td .id, td.id { overflow-wrap: normal; white-space: nowrap; }
td.ev a, td.ev span.id { display: block; }
td .id.chip { white-space: normal; }
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 0.9em; }
td.num, th.num { text-align: right; white-space: nowrap; }
td.num.left { text-align: left; }
.small { font-size: 0.8rem; }
.muted { color: var(--ink-2); }
.unresolved { color: var(--crit); border-bottom: 1px dashed var(--crit); }
.mark { display: inline-block; font-family: var(--sans); font-size: 0.75rem; font-weight: 500; line-height: 1.5; padding: 0 0.45em;
        border: 1px solid var(--line); color: var(--ink-2); white-space: nowrap; vertical-align: middle; }
.mark.good { color: var(--good); border-color: var(--good); background: var(--good-bg); }
.mark.warn { color: var(--warn); border-color: var(--warn); background: var(--warn-bg); }
.mark.crit { color: var(--crit); border-color: var(--crit); background: var(--crit-bg); }
.chip { display: inline-block; padding: 0 0.35em; border: 1px solid var(--line-2); margin: 0.1rem 0.15rem 0.1rem 0; white-space: nowrap; }
.notice { padding: 0.5rem 0.75rem; border-left: 3px solid var(--line); background: var(--bg-2); max-width: none; }
.notice.warn { border-left-color: var(--warn); }
.facts { display: grid; grid-template-columns: max-content 1fr; gap: 0.3rem 1.25rem; margin: 0.75rem 0 1rem; font-size: 0.92rem; }
.facts dt { color: var(--ink-2); font-weight: 500; }
.facts dd { margin: 0; overflow-wrap: anywhere; }
@media (max-width: 560px) { .facts { grid-template-columns: 1fr; gap: 0.1rem 0; } .facts dt { margin-top: 0.5rem; } }
.tbl { overflow-x: auto; margin: 0.5rem 0 1rem; max-width: 100%; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
th { text-align: left; font-weight: 600; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--ink-2);
     padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--line); white-space: nowrap; vertical-align: bottom; }
td { padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--line-2); vertical-align: top; }
tbody tr:last-child td { border-bottom: 1px solid var(--line); }
table.candidates td:nth-child(2), table.stages td:nth-child(2) { font-weight: 500; }
table.criteria td.prose { min-width: 24rem; }
table.references td.url, table.files td.path, table.files td.sha, table.counts td.small { max-width: 34rem; overflow-wrap: anywhere; white-space: normal; }
.kv { display: grid; grid-template-columns: repeat(auto-fill, minmax(21rem, 1fr)); gap: 0.15rem 2rem; margin: 0.5rem 0 1rem; font-size: 0.88rem; }
.kv div { display: flex; justify-content: space-between; gap: 1rem; border-bottom: 1px solid var(--line-2); padding: 0.2rem 0; }
.kv dt { color: var(--ink-2); white-space: nowrap; }
.kv dd { margin: 0; text-align: right; overflow-wrap: anywhere; }
.verdict { max-width: none; }
.verdict { font-size: 1rem; }
ul { margin: 0.3rem 0 0.8rem; padding-left: 1.25rem; }
li { margin: 0.2rem 0; max-width: 80ch; }
.cites { white-space: nowrap; }
details { margin: 0.75rem 0; border: 1px solid var(--line-2); }
summary { cursor: pointer; padding: 0.4rem 0.75rem; font-size: 0.85rem; color: var(--ink-2); background: var(--bg-2); }
details > :not(summary) { padding: 0 0.75rem; }
pre { font-family: var(--mono); font-size: 0.78rem; line-height: 1.45; overflow-x: auto; margin: 0.5rem 0; white-space: pre; }
article.candidate, article.chain, article.manifest, article.drug { margin: 1rem 0 2rem; }
.doc-foot { margin-top: 4rem; padding-top: 1rem; border-top: 1px solid var(--line); font-size: 0.82rem; color: var(--ink-2); }
@media print { .head-tools, .toc { display: none; } body { font-size: 11pt; } a { color: inherit; } .page { padding-inline: 0; } }
"""

JS = """
(function () {
  var root = document.documentElement, btn = document.getElementById('theme');
  var order = ['system', 'light', 'dark'], current = 'system';
  function read() { try { return localStorage.getItem('engine-report-theme') || 'system'; } catch (e) { return 'system'; } }
  function save(v) { try { localStorage.setItem('engine-report-theme', v); } catch (e) {} }
  function apply(v) {
    current = order.indexOf(v) >= 0 ? v : 'system';
    if (current === 'system') { root.removeAttribute('data-theme'); } else { root.setAttribute('data-theme', current); }
    if (btn) { btn.textContent = 'Theme: ' + current; }
  }
  apply(read());
  if (btn) { btn.addEventListener('click', function () { apply(order[(order.indexOf(current) + 1) % order.length]); save(current); }); }
})();
"""
