"""UniProtKB — the reviewed human entry for a gene product, as one evidence record.

Why UniProt in stage 5: the evidence chain and the medicine report both want to say
where a variant falls in the protein — which domain, whether a truncation removes a
nucleotide-binding fold, which natural variants sit at the same residue — and nothing
the stage-2 sources hold carries that. A VEP record gives the residue number
(``p.Phe508del``); the reviewed UniProt entry gives the domain map, the FUNCTION and
DISEASE texts, and the curated natural variants. Putting the entry in the store makes
every such sentence a citation of ``uniprot:<accession>`` a judge can open.

Two requests, both verified live on 2026-09-17 (release ``2026_03``):

* Symbol → accession: ``GET search?query=(gene_exact:<SYM>) AND (organism_id:9606) AND
  (reviewed:true)&format=json&fields=…``. One reviewed human entry per symbol is the
  normal answer; ``{"results": []}`` is absence (``None``); two reviewed entries whose
  primary gene is the symbol asked is an ambiguity this module refuses to resolve by
  picking the first (``UniprotError``), and a result whose primary gene is *not* the
  symbol (a synonym match) is not the gene asked for.
* Entry: ``GET <accession>.json?fields=<FIELDS>`` — a projection of the entry (the
  CFTR body is 139 KB with ``fields``, 357 KB without). The payload is kept verbatim.
  ``{"messages": ["Resource not found"]}`` on a 404 is the only shape accepted as "no
  such entry", and since the accession came from the search it is raised, not
  returned: an entry that vanished between two requests is a failure to report. A
  200 whose ``entryType`` is ``Inactive`` (a deleted or demerged accession) or whose
  ``primaryAccession`` is not the one asked (a secondary accession redirected to its
  primary) is refused for the same reason — the record id must name the entry.

Version. Every response carries ``x-uniprot-release`` and ``x-uniprot-release-date``;
:meth:`UniprotRetriever.version` reads them from a fixed probe request (a search that
names no gene), and — as :mod:`engine.medicine.chembl` does — checks the cached
observation against a live one once per instance when online, before the first data
request, so a cache spanning a release is refused rather than mixed. Every data
response's headers are compared with the pinned release too. The shared
:class:`~engine.retrieve.http.Http` keeps only the headers named in its ``_KEEP``
tuple; those two (and ``x-total-results``) must be on it for a live run.

Privacy: a symbol and an accession are all that is ever sent — never a coordinate,
never an HGVS string. Error text names counts and shapes, not the symbol asked.

Rate: UniProt publishes no quota; the limiter keeps ``rest.uniprot.org`` at 3
requests/second unless the orchestrator set a rate already.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

from engine.retrieve.http import Http, Response
from engine.retrieve.store import EvidenceRecord

SOURCE = "uniprot"
API_URL = "https://rest.uniprot.org/uniprotkb/"
ENTRY_URL = "https://www.uniprot.org/uniprotkb/{acc}/entry"
UNIPROT_PER_SECOND = 3.0
HUMAN = 9606
FIELDS = ("accession,id,gene_primary,protein_name,length,cc_function,cc_domain,cc_disease,cc_subunit,"
          "cc_subcellular_location,cc_ptm,ft_chain,ft_domain,ft_region,ft_motif,ft_topo_dom,ft_transmem,ft_binding,"
          "ft_act_site,ft_site,ft_variant,ft_mutagen,xref_pfam,xref_interpro,xref_ensembl,xref_mim,xref_hgnc,keyword,"
          "version,date_modified,sequence")
"""The entry projection: names, the six comment types a dossier reads, the features of
the domain map plus natural variants and mutagenesis, the cross-references that name
domains elsewhere (Pfam, InterPro), the audit fields, and the canonical sequence — so
a deletion's original residue can be shown and a transcript's reference amino acid
can be checked against the isoform the features are numbered on."""
SEARCH_FIELDS = "accession,id,gene_primary,gene_names,protein_name,length,organism_id"
PROBE_QUERY = f"(reviewed:true) AND (organism_id:{HUMAN})"
"""The version probe: a search that names no gene and asks for one accession."""
MAP_FEATURES = ("Domain", "Region", "Motif", "Topological domain", "Transmembrane", "Binding site", "Active site", "Site")
"""Feature types that make up the domain map, in the order ties are broken."""
NATURAL_VARIANT = "Natural variant"
DESCRIPTION_CHARS = 120
"""A natural variant's description is cut here: the record has the rest."""
COLUMNS = ("accession", "entry_name", "gene", "protein_name", "length", "release", "entry_version", "n_features",
           "n_natural_variants", "function", "diseases")
RELEASE_HEADERS = ("x-uniprot-release", "x-uniprot-release-date")
"""The response headers the release is read from — both must be present."""

ACCESSION = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")
"""UniProtKB's accession grammar (six or ten characters); anything else is refused
before a request — the API answers such a string with a 400."""
_SYMBOL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]*$")
_RESIDUE = re.compile(r"p\.\(?([A-Z][a-z]{2}|\*)(\d+)")
"""``p.Phe508del`` → (Phe, 508); ``p.(Gly542Ter)`` too; ``p.Met1?`` → (Met, 1)."""
_PUBMED_REF = re.compile(r"\s*\((?:PubMed:\d+(?:,\s*)?)+\)")
"""``(PubMed:26823428, PubMed:1712898)`` inside a UniProt text: a paper the store may
not hold, so it is stripped from anything shown to the model as text."""

ONE_LETTER = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H",
    "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Sec": "U", "Pyl": "O", "Ter": "*", "Xaa": "X",
}


class UniprotError(RuntimeError):
    """The API answered, but not with something this module can cite: an ambiguous
    symbol, a result for another gene, an inactive or redirected entry, a 404 of an
    unexpected shape, a response without a release header, or a release boundary."""


class UniprotRetriever:
    """The reviewed human UniProtKB entry for a gene symbol, as an evidence record.
    Not a variant retriever: nothing genomic is ever sent."""

    source = SOURCE
    columns = COLUMNS

    def __init__(self, http: Http, *, base_url: str = API_URL, fields: str = FIELDS):
        self.http = http
        self.base_url = base_url.rstrip("/") + "/"
        self.fields = fields
        self._version: str | None = None
        _pin_rate(http, _host(self.base_url), UNIPROT_PER_SECOND)

    # ------------------------------------------------------------------ version

    def version(self) -> str:
        """``UniProt release 2026_03 (02-September-2026)`` from the probe response's
        headers — observed, cached on the instance; when online the cached observation
        must agree with a live one. Every data request calls this first, so a release
        boundary is caught before an entry served under the new release can be
        recorded under the old one."""
        if self._version is None:
            cached = self._observe(cache_ok=True)
            if not getattr(self.http, "offline", False):
                live = self._observe(cache_ok=False)
                if live != cached:
                    raise UniprotError(
                        f"the HTTP cache was filled under {cached!r} but the server now reports {live!r}: "
                        "delete the HTTP cache rather than mixing releases")
            self._version = cached
        return self._version

    def _observe(self, *, cache_ok: bool) -> str:
        resp = self.http.get(self.base_url + "search", params={"query": PROBE_QUERY, "fields": "accession",
                                                               "format": "json", "size": 1}, cache_ok=cache_ok)
        self._json(resp)
        return release_of(resp)

    def _check_release(self, resp: Response) -> None:
        seen = release_of(resp)
        if seen != self.version():
            raise UniprotError(f"a response carries {seen!r} but this run pinned {self._version!r}: "
                               "the release changed mid-run; rerun with a fresh HTTP cache")

    @property
    def params(self) -> dict[str, Any]:
        """Every setting that shapes the output, for the stage manifest's ``params``."""
        limiter = getattr(self.http, "limiter", None)
        per_second = (limiter.per_second.get(_host(self.base_url), UNIPROT_PER_SECOND)
                      if limiter is not None else UNIPROT_PER_SECOND)
        return {
            "api_url": self.base_url,
            "fields": self.fields,
            "search_fields": SEARCH_FIELDS,
            "search_query": f"(gene_exact:<SYMBOL>) AND (organism_id:{HUMAN}) AND (reviewed:true)",
            "probe_query": PROBE_QUERY,
            "map_features": list(MAP_FEATURES),
            "description_chars": DESCRIPTION_CHARS,
            "per_second": per_second,
            "release_headers": list(RELEASE_HEADERS),
        }

    # -------------------------------------------------------------------- lookup

    def accession_for_symbol(self, symbol: str) -> str | None:
        """The primary accession of the one reviewed human entry whose primary gene
        name is ``symbol``; ``None`` when the search returns nothing. A result for
        another gene, or more than one entry for the symbol, is a :class:`UniprotError`
        — never a silent first pick."""
        sym = valid_symbol(symbol)
        self.version()
        resp = self.http.get(self.base_url + "search", params={
            "query": f"(gene_exact:{sym}) AND (organism_id:{HUMAN}) AND (reviewed:true)",
            "format": "json", "fields": SEARCH_FIELDS,
        })
        body = self._json(resp)
        self._check_release(resp)
        results = body.get("results")
        if not isinstance(results, list):
            raise UniprotError(f"the search answered without a results list: {resp.text[:200]!r}")
        if not results:
            return None
        matches = [r for r in results if _primary_gene(r) == sym]
        if not matches:
            raise UniprotError(f"the search returned {len(results)} reviewed human entr{'y' if len(results) == 1 else 'ies'}, "
                               "none with the symbol asked as its primary gene name (a synonym match is not the gene asked)")
        if len(matches) > 1:
            raise UniprotError(f"the symbol names {len(matches)} reviewed human entries "
                               f"({', '.join(str(r.get('primaryAccession')) for r in matches)}); refusing to pick one")
        entry = matches[0]
        acc = str(entry.get("primaryAccession") or "")
        if not ACCESSION.match(acc):
            raise UniprotError(f"the search returned an accession this module cannot spell: {acc!r}")
        if (entry.get("organism") or {}).get("taxonId") != HUMAN:
            raise UniprotError(f"the search returned an entry of organism {(entry.get('organism') or {}).get('taxonId')!r}, not human")
        return acc

    def entry(self, accession: str) -> EvidenceRecord:
        """The entry projected to :attr:`fields`, verbatim, as ``uniprot:<accession>``."""
        acc = valid_accession(accession)
        self.version()
        resp = self.http.get(self.base_url + f"{acc}.json", params={"fields": self.fields}, cache_404=True)
        if resp.status == 404:
            if _messages(resp) == ["Resource not found"]:
                raise UniprotError("UniProt has no entry for the accession the search returned "
                                   "(404 Resource not found): the entry vanished between two requests")
            raise UniprotError(f"a 404 whose body is not UniProt's 'Resource not found' shape ({resp.text[:120]!r}): "
                               "the path or the accession is wrong")
        body = self._json(resp)
        self._check_release(resp)
        if str(body.get("entryType") or "").lower().startswith("inactive"):
            reason = (body.get("inactiveReason") or {}).get("inactiveReasonType")
            raise UniprotError(f"the accession is inactive ({reason or 'no reason given'}); an inactive entry has no "
                               "features to cite")
        if body.get("primaryAccession") != acc:
            raise UniprotError(f"asked for one accession, answered for {body.get('primaryAccession')!r} (a secondary "
                               "accession redirected to its primary entry); ask for the primary")
        return EvidenceRecord(
            record_id=f"{SOURCE}:{acc}",
            source=SOURCE,
            source_version=self.version(),
            query={"accession": acc, "api": self._entry_api(acc), "fields": self.fields},
            url=ENTRY_URL.format(acc=acc),
            retrieved_at=resp.retrieved_at,
            payload=body,
        )

    def _entry_api(self, acc: str) -> str:
        return f"{self.base_url}{acc}.json?" + urllib.parse.urlencode({"fields": self.fields})

    def _json(self, resp: Response) -> dict[str, Any]:
        if resp.status != 200:
            raise UniprotError(f"UniProt: HTTP {resp.status}: {resp.text[:200]!r}")
        try:
            body = resp.json()
        except ValueError as e:
            raise UniprotError(f"UniProt: non-JSON body: {resp.text[:200]!r}") from e
        if not isinstance(body, dict):
            raise UniprotError(f"UniProt: expected an object, got {type(body).__name__}")
        return body

    # ---------------------------------------------------------------- projection

    def extract(self, record: EvidenceRecord | None) -> dict[str, str]:
        """:data:`COLUMNS` as strings; ``''`` throughout for ``None``."""
        if record is None:
            return {c: "" for c in COLUMNS}
        p = _payload(record)
        audit = p.get("entryAudit") or {}
        return {
            "accession": str(p.get("primaryAccession") or ""),
            "entry_name": str(p.get("uniProtkbId") or ""),
            "gene": gene_of(record),
            "protein_name": protein_name(record),
            "length": str(length_of(record) or ""),
            "release": str(getattr(record, "source_version", "") or ""),
            "entry_version": str(audit.get("entryVersion") or ""),
            "n_features": str(len(feature_map(record))),
            "n_natural_variants": str(sum(1 for f in _features(p) if f.get("type") == NATURAL_VARIANT)),
            "function": function_text(record),
            "diseases": ";".join(disease_names(record)),
        }


# ------------------------------------------------------------------------ helpers

def release_of(resp: Response) -> str:
    """``UniProt release 2026_03 (02-September-2026)`` from a response's headers, or
    :class:`UniprotError` when either header is missing — a transport that dropped
    them (``engine.retrieve.http._KEEP``) cannot observe the release."""
    headers = {str(k).lower(): str(v) for k, v in (getattr(resp, "headers", None) or {}).items()}
    release, date = (headers.get(h) for h in RELEASE_HEADERS)
    if not release or not date:
        raise UniprotError(f"the response carries no {' / '.join(RELEASE_HEADERS)} header (kept headers: "
                           f"{sorted(headers) or 'none'}): the release cannot be observed, so nothing can be cited")
    return f"UniProt release {release} ({date})"


def valid_accession(accession: str) -> str:
    acc = str(accession or "").strip().upper()
    if not ACCESSION.match(acc):
        raise ValueError(f"not a UniProtKB accession: {accession!r} (expected e.g. P13569 or A0A0C5B5G6)")
    return acc


def valid_symbol(symbol: str) -> str:
    sym = str(symbol or "").strip().upper()
    if not sym or not _SYMBOL.match(sym):
        raise ValueError("not a gene symbol (letters, digits, '.', '_', '-', '@')")
    return sym


def feature_map(record: Any) -> list[dict[str, Any]]:
    """Every :data:`MAP_FEATURES` feature as ``{type, start, end, description}``, sorted
    by position — the domain map. A feature without exact integer ends is left out."""
    out = []
    for f in _features(_payload(record)):
        kind = f.get("type")
        if kind not in MAP_FEATURES:
            continue
        span = _span(f)
        if span is None:
            continue
        out.append({"type": kind, "start": span[0], "end": span[1], "description": str(f.get("description") or "")})
    return sorted(out, key=lambda d: (d["start"], d["end"], MAP_FEATURES.index(d["type"]), d["description"]))


def regions_at(record: Any, position: int) -> list[str]:
    """``Domain: ABC transporter 1 (423–646)`` for every map feature whose span covers
    the residue; ``[]`` when none does."""
    out = []
    for f in feature_map(record):
        if f["start"] <= int(position) <= f["end"]:
            label = f"{f['type']}: {f['description']}" if f["description"] else f["type"]
            out.append(f"{label} ({f['start']}–{f['end']})")
    return out


def natural_variants_at(record: Any, position: int) -> list[str]:
    """``VAR_000171 F→del: in CF and CBAVD; … (20 UniProt evidence references)`` for
    every natural variant starting at the residue. An empty ``alternativeSequence`` is
    a deletion; the original residue then comes from the canonical sequence. The
    description is cut at :data:`DESCRIPTION_CHARS` and carries no PubMed number —
    the references are counted, and the record holds them."""
    p = _payload(record)
    seq = str((p.get("sequence") or {}).get("value") or "")
    out = []
    for f in _features(p):
        if f.get("type") != NATURAL_VARIANT:
            continue
        span = _span(f)
        if span is None or span[0] != int(position):
            continue
        alt = f.get("alternativeSequence") or {}
        original = str(alt.get("originalSequence") or "")
        if not original and span[0] == span[1] and 0 < span[0] <= len(seq):
            original = seq[span[0] - 1]
        if not original and span[0] < span[1]:  # a multi-residue deletion: its ends, not the whole tail
            first, last = residue_at(p, span[0]), residue_at(p, span[1])
            original = f"{first or '?'}{span[0]}–{last or '?'}{span[1]}"
        changed = ",".join(str(x) for x in alt.get("alternativeSequences") or []) or "del"
        change = f"{original or '?'}→{changed}"
        desc = _strip_pubmed(str(f.get("description") or "")).strip() or "no description"
        if len(desc) > DESCRIPTION_CHARS:
            desc = desc[:DESCRIPTION_CHARS - 1].rstrip() + "…"
        n = len(f.get("evidences") or [])
        out.append(f"{f.get('featureId') or 'variant'} {change}: {desc} ({n} UniProt evidence reference{'s' if n != 1 else ''})")
    return sorted(out)


def function_text(record: Any) -> str:
    """The FUNCTION comment's texts, joined; PubMed references stripped."""
    return " ".join(_strip_pubmed(t) for t in _comment_texts(record, "FUNCTION")).strip()


def disease_texts(record: Any) -> list[str]:
    """``<diseaseId>: <description>`` per DISEASE comment, PubMed references stripped."""
    out = []
    for c in _comments(_payload(record), "DISEASE"):
        d = c.get("disease") or {}
        name = str(d.get("diseaseId") or "").strip()
        text = _strip_pubmed(str(d.get("description") or "")).strip()
        if name or text:
            out.append(f"{name}: {text}" if name and text else name or text)
    return out


def disease_names(record: Any) -> list[str]:
    """The DISEASE comments' ``diseaseId`` (``Cystic fibrosis``), in order, each once."""
    names = []
    for c in _comments(_payload(record), "DISEASE"):
        name = str((c.get("disease") or {}).get("diseaseId") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def protein_name(record: Any) -> str:
    p = _payload(record)
    rec = (p.get("proteinDescription") or {}).get("recommendedName") or {}
    name = (rec.get("fullName") or {}).get("value")
    if not name:
        subs = (p.get("proteinDescription") or {}).get("submissionNames") or []
        name = ((subs[0].get("fullName") or {}).get("value") if subs else None)
    return str(name or "")


def gene_of(record: Any) -> str:
    genes = _payload(record).get("genes") or []
    return str(((genes[0].get("geneName") or {}).get("value") if genes else "") or "")


def length_of(record: Any) -> int | None:
    n = (_payload(record).get("sequence") or {}).get("length")
    return int(n) if isinstance(n, int) else None


def residue_at(record: Any, position: int) -> str | None:
    """The one-letter residue of the canonical sequence at ``position`` (1-based), or
    ``None`` when the record carries no sequence or the position is off its end."""
    seq = str((_payload(record).get("sequence") or {}).get("value") or "")
    return seq[position - 1] if 0 < position <= len(seq) else None


def residue_of(hgvsp: Any) -> int | None:
    """``ENSP00000003084.6:p.Phe508del`` → 508; ``p.Gly542Ter`` → 542; ``p.Met1?`` → 1;
    ``None`` for anything without a ``p.<Aaa><n>`` (``p.?``, ``-``, an empty cell)."""
    m = _RESIDUE.search(str(hgvsp or ""))
    return int(m.group(2)) if m else None


def reference_residue(hgvsp: Any) -> str | None:
    """The one-letter reference amino acid the HGVS protein change names (``Phe`` →
    ``F``), or ``None``."""
    m = _RESIDUE.search(str(hgvsp or ""))
    return ONE_LETTER.get(m.group(1)) if m else None


def _payload(record: Any) -> dict[str, Any]:
    p = record.get("payload") if isinstance(record, dict) and "payload" in record else getattr(record, "payload", record)
    return p if isinstance(p, dict) else {}


def _features(p: dict[str, Any]) -> list[dict[str, Any]]:
    return [f for f in p.get("features") or [] if isinstance(f, dict)]


def _comments(p: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [c for c in p.get("comments") or [] if isinstance(c, dict) and c.get("commentType") == kind]


def _comment_texts(record: Any, kind: str) -> list[str]:
    return [str(t.get("value") or "") for c in _comments(_payload(record), kind) for t in c.get("texts") or [] if isinstance(t, dict)]


def _span(f: dict[str, Any]) -> tuple[int, int] | None:
    loc = f.get("location") or {}
    start, end = (loc.get("start") or {}).get("value"), (loc.get("end") or {}).get("value")
    if isinstance(start, int) and isinstance(end, int) and not isinstance(start, bool):
        return start, end
    return None


def _primary_gene(result: dict[str, Any]) -> str:
    genes = result.get("genes") or []
    return str(((genes[0].get("geneName") or {}).get("value") if genes and isinstance(genes[0], dict) else "") or "").upper()


def _messages(resp: Response) -> Any:
    try:
        body = resp.json()
    except ValueError:
        return None
    return body.get("messages") if isinstance(body, dict) else None


def _strip_pubmed(text: str) -> str:
    return _PUBMED_REF.sub("", text)


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc


def _pin_rate(http: Any, host: str, per_second: float) -> None:
    """Etiquette for ``host`` on the shared limiter — unless a rate is already set,
    since the orchestrator's table may be stricter for runs that share an IP."""
    limiter = getattr(http, "limiter", None)
    if limiter is not None:
        limiter.per_second.setdefault(host, per_second)
