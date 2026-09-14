import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from engine.retrieve.http import Http, HttpCache, HttpError, RateLimiter
from engine.retrieve.store import EvidenceRecord, EvidenceStore, key_str, parse_key, safe_name


# ---------------------------------------------------------------- store

def _rec(i: int = 1, **kw) -> EvidenceRecord:
    base = dict(record_id=f"gnomad:1-11796321-G-A", source="gnomad", source_version="gnomad_r4",
                query={"variantId": "1-11796321-G-A"}, url="https://gnomad.broadinstitute.org/variant/1-11796321-G-A",
                retrieved_at="2026-09-12T19:00:00+00:00", payload={"af": 0.3, "n": i})
    base.update(kw)
    return EvidenceRecord(**base)


def test_record_bytes_are_deterministic():
    a, b = _rec(), _rec()
    assert a.to_json() == b.to_json()
    assert a.sha256 == b.sha256
    assert EvidenceRecord.from_json(a.to_json()) == a
    # key order in payload must not matter
    c = _rec(payload={"n": 1, "af": 0.3})
    assert c.to_json() == a.to_json()


def test_store_round_trip_and_index(tmp_path: Path):
    s = EvidenceStore(tmp_path / "evidence")
    p = s.put(_rec())
    assert p.parent.name == "gnomad" and p.suffix == ".json"
    assert s.exists("gnomad:1-11796321-G-A")
    assert s.get("gnomad:1-11796321-G-A") == _rec()
    assert s.get("gnomad:nope") is None
    s.put(_rec(record_id="clinvar:VCV000003520", source="clinvar"))
    assert s.count() == 2 and s.count("gnomad") == 1
    idx = json.loads(s.write_index().read_text())
    assert set(idx) == {"gnomad:1-11796321-G-A", "clinvar:VCV000003520"}
    assert idx["gnomad:1-11796321-G-A"]["url"].startswith("https://gnomad")
    assert idx["gnomad:1-11796321-G-A"]["sha256"] == _rec().sha256


def test_store_put_is_idempotent(tmp_path: Path):
    s = EvidenceStore(tmp_path)
    p = s.put(_rec())
    m1 = p.stat().st_mtime_ns
    s.put(_rec())
    assert p.stat().st_mtime_ns == m1  # identical bytes: untouched


def test_safe_name_and_keys():
    assert safe_name("vep:7:117559590:ATCT:A").startswith("vep_7_117559590_ATCT_A.")
    assert safe_name("a/b") != safe_name("a_b")  # digest disambiguates
    assert key_str(("7", 117559590, "ATCT", "A")) == "7:117559590:ATCT:A"
    assert parse_key("7:117559590:ATCT:A") == ("7", 117559590, "ATCT", "A")


# ---------------------------------------------------------------- http

class _Handler(BaseHTTPRequestHandler):
    calls: list[tuple[str, str, bytes]] = []
    fail_first = 0

    def log_message(self, *a):  # quiet
        pass

    def _send(self, status: int, body: dict, extra: dict | None = None):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        _Handler.calls.append(("GET", self.path, b""))
        if self.path.startswith("/missing"):
            return self._send(404, {"error": "no such record"})
        if self.path.startswith("/flaky"):
            if _Handler.fail_first > 0:
                _Handler.fail_first -= 1
                return self._send(429, {"error": "slow down"}, {"Retry-After": "0"})
            return self._send(200, {"ok": True, "path": self.path})
        if self.path.startswith("/boom"):
            return self._send(500, {"error": "boom"})
        if self.path.startswith("/cut"):
            if _Handler.fail_first > 0:
                # a body cut short: Content-Length promises more than is sent, then the connection closes
                _Handler.fail_first -= 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "4000")
                self.end_headers()
                self.wfile.write(b'{"ok": tr')
                self.wfile.flush()
                self.connection.close()
                return None
            return self._send(200, {"ok": True, "path": self.path})
        return self._send(200, {"ok": True, "path": self.path})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        _Handler.calls.append(("POST", self.path, body))
        return self._send(200, {"echo": json.loads(body)})


@pytest.fixture
def server():
    _Handler.calls = []
    _Handler.fail_first = 0
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _http(tmp_path: Path, **kw) -> Http:
    return Http(HttpCache(tmp_path / "cache"), limiter=RateLimiter(default_per_second=0), retries=2, **kw)


def test_cache_hit_preserves_original_retrieval_time(tmp_path: Path, server: str):
    h = _http(tmp_path)
    r1 = h.get(f"{server}/x", params={"b": 2, "a": 1})
    r2 = h.get(f"{server}/x", params={"a": 1, "b": 2})  # same request, different param order
    assert r1.from_cache is False and r2.from_cache is True
    assert r1.retrieved_at == r2.retrieved_at and r1.text == r2.text
    assert h.live_requests == 1 and h.cache.stats() == {"hits": 1, "misses": 1}


def test_offline_replays_cache_and_refuses_network(tmp_path: Path, server: str):
    h = _http(tmp_path)
    h.post(f"{server}/vep", {"variants": ["1 11796321 . G A . . ."]})
    off = _http(tmp_path, offline=True)
    r = off.post(f"{server}/vep", {"variants": ["1 11796321 . G A . . ."]})
    assert r.from_cache and r.json()["echo"]["variants"][0].startswith("1 11796321")
    with pytest.raises(RuntimeError, match="offline"):
        off.get(f"{server}/never-fetched")


def test_404_is_a_result_only_when_asked(tmp_path: Path, server: str):
    h = _http(tmp_path)
    with pytest.raises(HttpError) as e:
        h.get(f"{server}/missing")
    assert e.value.status == 404
    r = h.get(f"{server}/missing", cache_404=True)
    assert r.status == 404 and not r.from_cache
    r2 = h.get(f"{server}/missing", cache_404=True)
    assert r2.status == 404 and r2.from_cache


def test_retries_on_429_then_succeeds_and_never_caches_errors(tmp_path: Path, server: str):
    _Handler.fail_first = 1
    h = _http(tmp_path)
    r = h.get(f"{server}/flaky")
    assert r.status == 200 and h.live_requests == 2
    with pytest.raises(HttpError) as e:
        h.get(f"{server}/boom")
    assert e.value.status == 500
    # the 500 was not cached: a later call hits the network again
    n = h.live_requests
    with pytest.raises(HttpError):
        h.get(f"{server}/boom")
    assert h.live_requests > n


def test_a_body_cut_short_is_retried_like_a_5xx(tmp_path: Path, server: str):
    """Ensembl under load closes a POST mid-body (IncompleteRead); that is a transient
    failure, not a result, and must not kill a ten-minute retrieve."""
    _Handler.fail_first = 1
    h = _http(tmp_path)
    r = h.get(f"{server}/cut")
    assert r.status == 200 and json.loads(r.text)["ok"] is True and h.live_requests == 2


def test_rate_limiter_spaces_requests():
    import time
    rl = RateLimiter(per_second={"h": 50.0})
    t0 = time.monotonic()
    for _ in range(6):
        rl.wait("h")
    assert time.monotonic() - t0 >= 5 * (1 / 50.0) * 0.9
