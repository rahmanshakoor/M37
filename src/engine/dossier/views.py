"""The dossier as the report reads it — every id resolved to its URL.

Same contract as :mod:`engine.report.views`: a view never raises on a file a step did
not finish writing; absence is a field, not an exception. ``dossier_view`` finds the
candidate the manifest names (or the one asked for), reads the validated dossier and
its validation record, and resolves every cited id through the run's evidence
indexes, so the report's renderer only draws.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from engine.report.views import candidate_file_name, cited_in, evidence_index, manifest, read_json, resolve

STAGE_DIR = "05_reason/dossier"


def dossier_view(run_dir: Path, candidate_id: str | None = None) -> dict[str, Any]:
    """``present`` (the step's directory exists), ``manifest``, ``dry_run``, ``model``,
    ``effort``, ``disclosure``, ``candidate_id``, ``gene_symbol``, ``uniprot`` (the
    entry resolved), ``notes``, ``validation``, ``file`` (the dossier's path) and
    ``dossier`` — the validated object with every claim's ids resolved, plus
    ``references`` — or ``None`` when no dossier was written."""
    run_dir = Path(run_dir)
    stage = run_dir / STAGE_DIR
    if not stage.is_dir():
        return {"present": False, "dossier": None}
    m = manifest(run_dir, STAGE_DIR)
    params = (m or {}).get("params") or {}
    cid = candidate_id or params.get("candidate")
    if cid is None:
        found = sorted(p for p in stage.glob("*.json") if p.name != "manifest.json")
        cid = (read_json(found[0]) or {}).get("candidate_id") if found else None
    stem = candidate_file_name(str(cid)) if cid else None
    doc = read_json(stage / f"{stem}.json") if stem else None
    index = evidence_index(run_dir)
    acc = params.get("accession") or (doc or {}).get("uniprot_accession")
    validation = read_json(stage / "validation" / f"{stem}.json") if stem else None
    out: dict[str, Any] = {
        "present": True,
        "manifest": m is not None,
        "dry_run": params.get("dry_run"),
        "model": params.get("model"),
        "effort": params.get("effort"),
        "disclosure": params.get("disclosure"),
        "candidate_id": cid,
        "gene_symbol": (doc or {}).get("gene_symbol") or _gene_from_bundle(stage, stem),
        "uniprot": resolve([f"uniprot:{acc}"], index)[0] if acc else None,
        "notes": list((m or {}).get("notes") or []),
        "validation": _validation(validation),
        "file": str(stage / f"{stem}.json") if doc is not None else None,
        "dossier": None,
    }
    if not isinstance(doc, dict):
        return out
    texts: list[str] = []

    def claims(items: Any) -> list[dict[str, Any]]:
        out_ = [{"statement": c.get("statement", ""), "evidence": resolve(list(c.get("evidence_ids") or []), index)}
                for c in items or [] if isinstance(c, dict)]
        texts.extend(c["statement"] for c in out_)
        return out_

    dossier = {
        "candidate_id": doc.get("candidate_id"),
        "gene_symbol": doc.get("gene_symbol"),
        "uniprot_accession": doc.get("uniprot_accession"),
        "protein": claims(doc.get("protein")),
        "mechanism_of_disease": claims(doc.get("mechanism_of_disease")),
        "variant_positions": [{
            "key": p.get("key"), "protein_position": p.get("protein_position"), "region": list(p.get("region") or []),
            "consequence": p.get("consequence", ""), "evidence": resolve(list(p.get("evidence_ids") or []), index),
        } for p in doc.get("variant_positions") or [] if isinstance(p, dict)],
        "region_knowledge": claims(doc.get("region_knowledge")),
        "genotype_patterns": claims(doc.get("genotype_patterns")),
        "functional_test": claims(doc.get("functional_test")),
        "limits": [str(x) for x in doc.get("limits") or []],
        "literature": resolve(list(doc.get("literature") or []), index),
    }
    texts.extend(p["consequence"] for p in dossier["variant_positions"])
    texts.extend(dossier["limits"])
    cited = cited_in(texts, index)
    for group in ("protein", "mechanism_of_disease", "region_knowledge", "genotype_patterns", "functional_test"):
        for c in dossier[group]:
            cited += [e["id"] for e in c["evidence"] if e["id"] not in cited]
    for p in dossier["variant_positions"]:
        cited += [e["id"] for e in p["evidence"] if e["id"] not in cited]
    cited += [e["id"] for e in dossier["literature"] if e["id"] not in cited]
    if acc and f"uniprot:{acc}" not in cited:
        cited.append(f"uniprot:{acc}")
    dossier["references"] = resolve(sorted(dict.fromkeys(cited)), index)
    out["dossier"] = dossier
    return out


def _gene_from_bundle(stage: Path, stem: str | None) -> str | None:
    bundle = read_json(stage / "bundles" / f"{stem}.json") if stem else None
    return bundle.get("gene_symbol") if isinstance(bundle, dict) else None


def _validation(doc: Any) -> dict[str, Any] | None:
    if not isinstance(doc, dict):
        return None
    return {"counts": doc.get("counts") or {}, "rejections": doc.get("rejections") or [], "disputes": doc.get("disputes") or [],
            "notes": doc.get("notes") or [], "rules": doc.get("rules") or {}}
