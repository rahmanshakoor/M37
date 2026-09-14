"""Exomiser data bundles: fetch, verify, extract — resumably.

The two 2406 bundles are 21.9 GiB and 5.8 GiB. A transfer that size will be
interrupted at some point, so the download goes to a ``.part`` file and every attempt
resumes from its current length with an HTTP ``Range`` header (the server answers
``accept-ranges: bytes`` and ``206``; a ``200`` means it ignored the range and the part
is started over). Hashes are computed in one pass: the existing prefix is hashed
before resuming, then the stream is hashed as it arrives, so the final verification
costs no second read of a 22 GB file.

What is verified, and why not sha256 alone: Monarch publishes no checksum sidecar for
2406 (every ``*.sha256``/``*.md5``/``SHA256SUMS`` probe is a 404; the last sidecars are
for 2109). The server does publish each object's MD5 — in the ETag and the
``x-goog-hash: md5=`` header — and ``configs/exomiser.yaml`` pins that MD5 and the byte
count. Before a transfer starts, a HEAD compares the server's current size and MD5 with
the pin, so a silently re-published bundle is refused before a 22 GB download rather
than after. After the transfer, size, MD5 and (when pinned) sha256 are checked; the
sha256 computed on the way is written to ``<file>.verified.json`` so the pin can be
tightened to sha256 once a verified copy exists (both 2406 pins now carry one). A
complete zip that arrived by other means (``curl``) is hashed in place rather than
fetched again.

The resume budget counts *stalls*, not drops: a failure after which the file has grown
resets the counter, so a 22 GB transfer over a link that drops every few hundred MB
still completes; only ``RETRIES`` consecutive failures with no progress give up.

Extraction uses the standard ``zipfile`` (ZIP64-aware) into a temporary directory that
is renamed into place only when every member has been written and CRC-checked, so a
half-extracted bundle never looks like a finished one. Each member is sha256-hashed as
it is written and compared with the pinned ``member_sha256`` and with the checksum list
the bundle itself ships (``2406_hg38/2406_hg38.sha256``); the result goes into the
sidecar and, from there, into the stage manifest. ``2406_hg38.zip`` extracts to about
32 GB (``2406_hg38_variants.mv.db`` alone is 29.7 GiB), so the free space is checked
against the members' sizes before a byte is written; with the zips kept the whole set
peaks near 70 GB, which is why ``delete_zip`` exists — the sidecar keeps the hashes that
were verified. A pin is compared with the sidecar's *recorded* member hashes whenever it
is read (``members_status``), so a member pinned after its extraction — the phenotype
files were — counts as verified without re-hashing; an extraction that has neither zip
nor sidecar (unzipped by hand) is hashed against the pin rather than fetched again.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from engine.rank.exomiser import Bundle, ExomiserConfig
from engine.retrieve.http import DEFAULT_USER_AGENT

VERIFIED_SUFFIX = ".verified.json"
CHUNK = 1 << 20
RETRIES = 12
"""Consecutive resume attempts that make no progress before a bundle is given up.
Each attempt resumes from the bytes already on disk, so a retry never repeats a
transfer; the sleep between attempts grows to a minute."""
EXTRACT_MARGIN = 2 << 30
"""Free space required beyond the members' total size before extraction starts."""

Progress = Callable[[str], None]
Opener = Callable[..., Any]


class PinMismatch(ValueError):
    """The bytes (or the server's description of them) disagree with the pin."""


@dataclass(frozen=True)
class RemoteInfo:
    bytes: int | None
    md5: str | None
    """Hex MD5 from ``x-goog-hash`` (base64 on the wire) or from a bare-hex ETag."""
    etag: str
    last_modified: str
    accept_ranges: bool


@dataclass(frozen=True)
class Hashes:
    bytes: int
    md5: str
    sha256: str


def _request(url: str, method: str = "GET", headers: dict[str, str] | None = None) -> urllib.request.Request:
    h = {"User-Agent": DEFAULT_USER_AGENT}
    h.update(headers or {})
    return urllib.request.Request(url, method=method, headers=h)


def head(url: str, opener: Opener = urllib.request.urlopen) -> RemoteInfo:
    with opener(_request(url, "HEAD"), timeout=60) as r:
        h = r.headers
        md5: str | None = None
        for v in h.get_all("x-goog-hash") or []:
            if v.startswith("md5="):
                md5 = base64.b64decode(v[4:]).hex()
        etag = (h.get("ETag") or "").strip('"')
        if md5 is None and len(etag) == 32 and all(c in "0123456789abcdef" for c in etag):
            md5 = etag
        length = h.get("Content-Length")
        return RemoteInfo(
            bytes=int(length) if length else None, md5=md5, etag=etag,
            last_modified=h.get("Last-Modified") or "", accept_ranges=(h.get("Accept-Ranges") or "").lower() == "bytes",
        )


def check_remote(bundle: Bundle, info: RemoteInfo) -> None:
    """Refuse before transferring: the server must describe the pinned bytes."""
    if info.bytes is not None and info.bytes != bundle.bytes:
        raise PinMismatch(f"{bundle.url}: server reports {info.bytes} bytes, pinned {bundle.bytes}")
    if info.md5 is not None and info.md5 != bundle.md5:
        raise PinMismatch(f"{bundle.url}: server reports md5 {info.md5}, pinned {bundle.md5}")


class _Stream:
    """Hash state that moves in lock-step with the bytes on disk."""

    def __init__(self) -> None:
        self.md5 = hashlib.md5()
        self.sha = hashlib.sha256()
        self.have = 0

    def hash_prefix(self, path: Path) -> None:
        with open(path, "rb") as f:
            while chunk := f.read(CHUNK):
                self.update(chunk)

    def update(self, chunk: bytes) -> None:
        self.md5.update(chunk)
        self.sha.update(chunk)
        self.have += len(chunk)

    def hashes(self) -> Hashes:
        return Hashes(bytes=self.have, md5=self.md5.hexdigest(), sha256=self.sha.hexdigest())


def _state_for(part: Path) -> _Stream:
    st = _Stream()
    if part.exists():
        st.hash_prefix(part)
    return st


def download(
    bundle: Bundle,
    dest_dir: Path,
    *,
    progress: Progress = lambda s: None,
    opener: Opener = urllib.request.urlopen,
    retries: int = RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Path, Hashes, RemoteInfo]:
    """Fetch ``bundle`` into ``dest_dir/<filename>`` (resuming ``<filename>.part``),
    verify it, and write the ``.verified.json`` sidecar. Returns the path, the observed
    hashes and what the HEAD said. Raises :class:`PinMismatch` when the bytes are not
    the pinned ones — the offending file is removed so the next attempt starts clean."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / bundle.filename
    part = dest.with_name(dest.name + ".part")
    info = head(bundle.url, opener)
    check_remote(bundle, info)

    st = _state_for(part)
    if st.have > bundle.bytes:
        progress(f"{bundle.filename}: partial file is larger than the pin ({st.have} > {bundle.bytes}); restarting")
        part.unlink()
        st = _Stream()
    elif st.have:
        progress(f"{bundle.filename}: resuming at {st.have:,} of {bundle.bytes:,} bytes")

    attempt = 0
    have_at_failure = st.have
    while st.have < bundle.bytes:
        headers = {"Range": f"bytes={st.have}-"} if st.have else {}
        try:
            with opener(_request(bundle.url, headers=headers), timeout=120) as r:
                if st.have and r.status == 200:
                    progress(f"{bundle.filename}: server ignored the Range header; restarting from 0")
                    part.unlink(missing_ok=True)
                    st = _Stream()
                elif st.have and r.status != 206:
                    raise RuntimeError(f"{bundle.url}: unexpected status {r.status} for a Range request")
                total = _total_from(r.headers, st.have)
                if total is not None and total != bundle.bytes:
                    raise PinMismatch(f"{bundle.url}: server now reports {total} bytes, pinned {bundle.bytes}")
                _stream(r, part, st, bundle, progress)
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 416:
                # Range not satisfiable: the part already holds every byte (or more)
                if part.exists() and part.stat().st_size >= bundle.bytes:
                    break
                part.unlink(missing_ok=True)
                st = _Stream()
            elif isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 502, 503, 504):
                raise
            if st.have > have_at_failure:
                attempt = 0  # the link dropped, but the file grew: not a stall
            have_at_failure = st.have
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"{bundle.url}: gave up after {retries} resume attempts without progress "
                                   f"at byte {st.have}: {e}") from e
            wait = min(5.0 * attempt, 60.0)
            progress(f"{bundle.filename}: {type(e).__name__}: {e}; resuming in {wait:.0f}s (attempt {attempt}/{retries})")
            sleep(wait)
            on_disk = part.stat().st_size if part.exists() else 0
            if on_disk != st.have:  # a write failed part-way: the hash state no longer matches the file
                st = _state_for(part)

    hashes = st.hashes()
    try:
        verify(bundle, hashes)
    except PinMismatch:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    write_verified(dest, bundle, hashes, info)
    return dest, hashes, info


def _total_from(headers: Any, offset: int) -> int | None:
    cr = headers.get("Content-Range")
    if cr and "/" in cr:
        total = cr.rsplit("/", 1)[1]
        return int(total) if total.isdigit() else None
    length = headers.get("Content-Length")
    return offset + int(length) if length else None


def _stream(r: Any, part: Path, st: _Stream, bundle: Bundle, progress: Progress) -> None:
    report_every = 512 << 20
    next_report = (st.have // report_every + 1) * report_every
    t0 = time.monotonic()
    started = st.have
    with open(part, "ab") as out:
        while chunk := r.read(CHUNK):
            out.write(chunk)
            st.update(chunk)
            if st.have >= next_report:
                rate = (st.have - started) / max(time.monotonic() - t0, 1e-6) / (1 << 20)
                progress(f"{bundle.filename}: {st.have / (1 << 30):.2f} / {bundle.bytes / (1 << 30):.2f} GiB ({rate:.1f} MiB/s)")
                next_report += report_every


def verify(bundle: Bundle, hashes: Hashes) -> None:
    if hashes.bytes != bundle.bytes:
        raise PinMismatch(f"{bundle.filename}: {hashes.bytes} bytes, pinned {bundle.bytes}")
    if hashes.md5 != bundle.md5:
        raise PinMismatch(f"{bundle.filename}: md5 {hashes.md5} differs from pinned {bundle.md5}")
    if bundle.sha256 and hashes.sha256 != bundle.sha256:
        raise PinMismatch(f"{bundle.filename}: sha256 {hashes.sha256} differs from pinned {bundle.sha256}")


def hash_file(path: Path) -> Hashes:
    return _state_for(Path(path)).hashes()


def write_verified(dest: Path, bundle: Bundle, hashes: Hashes, info: RemoteInfo | None) -> Path:
    sidecar = dest.with_name(dest.name + VERIFIED_SUFFIX)
    meta = {
        "filename": bundle.filename, "url": bundle.url, "bytes": hashes.bytes, "md5": hashes.md5, "sha256": hashes.sha256,
        "pinned": {"bytes": bundle.bytes, "md5": bundle.md5, "sha256": bundle.sha256, "crc32c": bundle.crc32c},
        "server": {"etag": info.etag, "last_modified": info.last_modified} if info else {},
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    sidecar.write_text(json.dumps(meta, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return sidecar


def read_verified(dest: Path) -> dict[str, Any] | None:
    sidecar = dest.with_name(dest.name + VERIFIED_SUFFIX)
    return json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else None


def update_verified(dest: Path, **fields: Any) -> Path:
    """Merge ``fields`` into the sidecar (creating a minimal one if the zip was never
    downloaded by the engine, e.g. an extraction verified after the fact)."""
    sidecar = dest.with_name(dest.name + VERIFIED_SUFFIX)
    meta = read_verified(dest) or {"filename": dest.name}
    meta.update(fields)
    sidecar.write_text(json.dumps(meta, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return sidecar


def is_verified(bundle: Bundle, dest_dir: Path) -> bool:
    """A zip beside a sidecar whose hashes match the pin, with the pinned size on disk.
    The sidecar is trusted for the hash (re-hashing 22 GB on every call is not free);
    the size is re-checked because it is."""
    dest = Path(dest_dir) / bundle.filename
    meta = read_verified(dest)
    if meta is None or not dest.exists():
        return False
    return (dest.stat().st_size == bundle.bytes and meta.get("md5") == bundle.md5
            and (bundle.sha256 is None or meta.get("sha256") == bundle.sha256))


# ---------------------------------------------------------------- extraction

def extract(
    bundle: Bundle,
    zip_path: Path,
    data_dir: Path,
    *,
    progress: Progress = lambda s: None,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> Path:
    """Unpack ``zip_path`` so that ``data_dir/<extracted_dir>/`` holds the bundle files.
    Members are CRC-checked by ``zipfile`` as they are read and sha256-hashed as they
    are written; the directory appears under its final name only after the last member
    is written and every hash agrees with the pin and the bundle's own checksum list.
    A failed attempt removes its temporary directory so the disk is not left full."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    final = bundle.extracted_path(data_dir)
    tmp = data_dir / f".extract-{bundle.extracted_dir}"
    if tmp.exists():
        shutil.rmtree(tmp)
    prefix = bundle.extracted_dir + "/"
    hashes: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        for m in members:
            if m.filename.startswith("/") or ".." in Path(m.filename).parts:
                raise ValueError(f"{zip_path}: refusing member path {m.filename!r}")
        need = sum(m.file_size for m in members) + EXTRACT_MARGIN
        free = disk_usage(data_dir).free
        if free < need:
            raise RuntimeError(
                f"{bundle.filename}: not enough free space under {data_dir} to extract: "
                f"{free / (1 << 30):.1f} GiB free, {need / (1 << 30):.1f} GiB needed "
                f"(members {sum(m.file_size for m in members) / (1 << 30):.1f} GiB + {EXTRACT_MARGIN >> 30} GiB margin)"
            )
        tmp.mkdir(parents=True)
        try:
            for m in members:
                rel = m.filename[len(prefix):] if m.filename.startswith(prefix) else m.filename
                target = tmp / bundle.extracted_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                progress(f"{bundle.filename}: extracting {rel} ({m.file_size / (1 << 30):.2f} GiB)")
                sha = hashlib.sha256()
                with zf.open(m) as src, open(target, "wb") as out:
                    while chunk := src.read(8 << 20):
                        out.write(chunk)
                        sha.update(chunk)
                hashes[rel] = {"bytes": m.file_size, "sha256": sha.hexdigest()}
            members_report = check_members(bundle, tmp / bundle.extracted_dir, hashes)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    if final.exists():
        shutil.rmtree(final)
    (tmp / bundle.extracted_dir).rename(final)
    shutil.rmtree(tmp, ignore_errors=True)
    missing = bundle.missing_files(data_dir)
    if missing:
        raise ValueError(f"{bundle.filename}: extracted, but required files are missing: {missing}")
    # the sidecar lives in the data directory (where `engine rank` looks), which is
    # where fetch_all keeps the zip too
    update_verified(data_dir / bundle.filename, members=members_report,
                    members_verified_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return final


def read_checksum_list(path: Path) -> dict[str, str]:
    """``sha256sum`` output (``<hex>  <path>``) → basename-relative path → hex. The list
    inside 2406_hg38 spells paths as ``2406_hg38/<file>``; the directory prefix is dropped."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            rel = parts[-1]
            rel = rel.split("/", 1)[1] if "/" in rel else rel
            out[rel] = parts[0].lower()
    return out


def check_members(bundle: Bundle, extracted: Path, hashes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compare observed member hashes with the pin (``member_sha256``) and with the
    checksum list the bundle ships (``checksum_list``). Raises :class:`PinMismatch` on
    any disagreement; a member no list mentions is reported as unverified, not failed."""
    shipped: dict[str, str] = {}
    if bundle.checksum_list and (extracted / bundle.checksum_list).is_file():
        shipped = read_checksum_list(extracted / bundle.checksum_list)
    report: dict[str, dict[str, Any]] = {}
    for rel, h in sorted(hashes.items()):
        entry: dict[str, Any] = dict(h)
        pinned = bundle.member_sha256.get(rel)
        listed = shipped.get(rel)
        entry["matches_pin"] = (h["sha256"] == pinned) if pinned else None
        entry["matches_checksum_list"] = (h["sha256"] == listed) if listed else None
        if entry["matches_pin"] is False:
            raise PinMismatch(f"{bundle.filename}: {rel} sha256 {h['sha256']} differs from pinned {pinned}")
        if entry["matches_checksum_list"] is False:
            raise PinMismatch(f"{bundle.filename}: {rel} sha256 {h['sha256']} differs from the bundle's "
                              f"{bundle.checksum_list} ({listed})")
        report[rel] = entry
    absent = [rel for rel in bundle.member_sha256 if rel not in hashes]
    if absent:
        raise PinMismatch(f"{bundle.filename}: pinned members not in the bundle: {absent}")
    return report


def verify_extracted(bundle: Bundle, data_dir: Path, *, progress: Progress = lambda s: None) -> dict[str, dict[str, Any]]:
    """Hash an already-extracted bundle (one extracted before member hashing existed, or
    copied in by hand) against the pin and the shipped checksum list, and record the
    result in the sidecar. Reads every file once — minutes for the 30 GB variant store."""
    data_dir = Path(data_dir)
    extracted = bundle.extracted_path(data_dir)
    missing = bundle.missing_files(data_dir)
    if missing:
        raise FileNotFoundError(f"{bundle.filename}: nothing to verify, required files missing: {missing}")
    hashes: dict[str, dict[str, Any]] = {}
    for f in sorted(p for p in extracted.rglob("*") if p.is_file()):
        rel = str(f.relative_to(extracted))
        progress(f"{bundle.filename}: hashing {rel} ({f.stat().st_size / (1 << 30):.2f} GiB)")
        sha = hashlib.sha256()
        with open(f, "rb") as src:
            while chunk := src.read(8 << 20):
                sha.update(chunk)
        hashes[rel] = {"bytes": f.stat().st_size, "sha256": sha.hexdigest()}
    report = check_members(bundle, extracted, hashes)
    update_verified(data_dir / bundle.filename, members=report,
                    members_verified_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return report


def members_status(bundle: Bundle, data_dir: Path) -> dict[str, Any]:
    """What the sidecar says about the extracted members: how many were hashed, whether
    every pinned member's recorded sha256 equals the pin *as it is now* (a pin added
    after the extraction still counts — the phenotype members were pinned that way),
    which pinned members disagree, and when — without re-reading the files."""
    meta = read_verified(Path(data_dir) / bundle.filename) or {}
    members = meta.get("members") or {}
    pinned = list(bundle.member_sha256)
    recorded = {rel: members.get(rel, {}).get("sha256") for rel in pinned}
    checked = [rel for rel in pinned if recorded[rel] == bundle.member_sha256[rel]]
    mismatched = [rel for rel in pinned if recorded[rel] and recorded[rel] != bundle.member_sha256[rel]]
    return {
        "hashed": len(members),
        "pinned": len(pinned),
        "pinned_verified": len(checked),
        "pinned_mismatched": mismatched,
        "all_pinned_verified": bool(pinned) and len(checked) == len(pinned),
        "verified_at": meta.get("members_verified_at"),
    }


def _members_need_hashing(bundle: Bundle, data_dir: Path) -> bool:
    """``--verify-extracted`` re-hashes when the sidecar has no member hashes, or when a
    pinned member is not vouched for by the hashes it has."""
    status = members_status(bundle, data_dir)
    return status["hashed"] == 0 or (status["pinned"] > 0 and not status["all_pinned_verified"])


# ---------------------------------------------------------------- the whole set

def fetch_all(
    cfg: ExomiserConfig,
    ref_dir: Path,
    *,
    names: tuple[str, ...] | None = None,
    do_extract: bool = True,
    delete_zip: bool = False,
    verify_members: bool = False,
    progress: Progress = lambda s: None,
    opener: Opener = urllib.request.urlopen,
) -> dict[str, dict[str, Any]]:
    """Download, verify and (optionally) extract every bundle into ``ref_dir``, which
    then serves as ``--exomiser-data``. Smallest bundle first, so a problem with the
    code path shows up before the 22 GB transfer. ``verify_members`` re-hashes an
    extraction the sidecar has no member hashes for. Returns per-bundle facts."""
    ref_dir = Path(ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    chosen = [cfg.bundles[n] for n in (names or tuple(cfg.bundles))]
    chosen.sort(key=lambda b: b.bytes)
    report: dict[str, dict[str, Any]] = {}
    for b in chosen:
        zip_path = ref_dir / b.filename
        facts: dict[str, Any] = {"filename": b.filename, "bytes": b.bytes, "md5": b.md5}
        extracted_ok = not b.missing_files(ref_dir)
        if extracted_ok and not zip_path.exists():
            if read_verified(zip_path) is None:
                # unzipped by hand, or the sidecar was lost: hashing 30 GB against the pin
                # beats fetching 22 GB again
                progress(f"{b.filename}: extracted, zip absent and no {VERIFIED_SUFFIX} sidecar; hashing the "
                         f"extracted members against the pin instead of re-downloading")
                verify_extracted(b, ref_dir, progress=progress)
                facts.update(read_verified(zip_path) or {}, status="extracted-unverified-zip")
            else:
                progress(f"{b.filename}: already extracted and verified (zip removed after extraction)")
                facts.update(read_verified(zip_path) or {}, status="extracted")
                if verify_members and _members_need_hashing(b, ref_dir):
                    verify_extracted(b, ref_dir, progress=progress)
                    facts["members"] = (read_verified(zip_path) or {}).get("members")
            facts["extracted_dir"] = str(b.extracted_path(ref_dir))
            report[b.name] = facts
            continue
        if is_verified(b, ref_dir):
            progress(f"{b.filename}: already present and verified")
            facts.update(read_verified(zip_path) or {}, status="verified")
        elif zip_path.exists() and not zip_path.with_name(zip_path.name + ".part").exists():
            # a complete file that arrived by other means: hash it rather than fetch it again
            progress(f"{b.filename}: present without a sidecar; hashing {zip_path.stat().st_size / (1 << 30):.2f} GiB")
            try:
                hashes = hash_file(zip_path)
                verify(b, hashes)
            except PinMismatch as e:
                progress(f"{b.filename}: {e}; re-downloading")
                zip_path.unlink()
                t0 = time.monotonic()
                _, hashes, info = download(b, ref_dir, progress=progress, opener=opener)
                facts.update(sha256=hashes.sha256, etag=info.etag, last_modified=info.last_modified,
                             download_s=round(time.monotonic() - t0, 1), status="downloaded")
            else:
                write_verified(zip_path, b, hashes, None)
                facts.update(sha256=hashes.sha256, status="hashed")
                progress(f"{b.filename}: verified md5 {hashes.md5} · sha256 {hashes.sha256}")
        else:
            progress(f"{b.filename}: downloading {b.url} ({b.bytes / (1 << 30):.2f} GiB)")
            t0 = time.monotonic()
            _, hashes, info = download(b, ref_dir, progress=progress, opener=opener)
            facts.update(sha256=hashes.sha256, etag=info.etag, last_modified=info.last_modified,
                         download_s=round(time.monotonic() - t0, 1), status="downloaded")
            progress(f"{b.filename}: verified md5 {hashes.md5} · sha256 {hashes.sha256}")
        if do_extract:
            if extracted_ok:
                progress(f"{b.filename}: already extracted to {b.extracted_path(ref_dir)}")
                if verify_members and _members_need_hashing(b, ref_dir):
                    verify_extracted(b, ref_dir, progress=progress)
            else:
                t0 = time.monotonic()
                extract(b, zip_path, ref_dir, progress=progress)
                facts["extract_s"] = round(time.monotonic() - t0, 1)
                progress(f"{b.filename}: extracted to {b.extracted_path(ref_dir)}")
            facts["extracted_dir"] = str(b.extracted_path(ref_dir))
            facts["members"] = (read_verified(zip_path) or {}).get("members")
            if delete_zip:
                zip_path.unlink()
                facts["zip_deleted"] = True
                progress(f"{b.filename}: zip removed (hashes kept in {b.filename}{VERIFIED_SUFFIX})")
        report[b.name] = facts
    return report


if __name__ == "__main__":  # `python -m engine.rank.download …` — same command as `engine exomiser-download`
    from engine.rank.cli import exomiser_download

    exomiser_download()
