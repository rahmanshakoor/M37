"""A small HTTP client with a content-addressed cache, rate limiting and retries.

Why not just ``requests``: reproducibility. Every response is cached by the hash of
the exact request (method, URL, params, body), and a cache hit returns the *original*
retrieval time — so a rerun with a warm cache produces byte-identical evidence
records. Only definitive responses are cached (2xx, and 404 when the caller says a
404 is meaningful); throttling and server errors are retried, never cached.

The cache lives outside the repository (``ENGINE_CACHE`` or ``--cache``). It may
contain patient variants in request bodies, so it is subject to the same deletion
obligation as the run directories.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import http.client
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine import __version__

DEFAULT_USER_AGENT = f"engine/{__version__} (rare-disease variant engine; research use)"
RETRY_STATUSES = {429, 500, 502, 503, 504}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class Response:
    status: int
    text: str
    retrieved_at: str
    from_cache: bool
    request_key: str
    headers: dict[str, str]

    def json(self) -> Any:
        return json.loads(self.text)


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str):
        super().__init__(f"HTTP {status} from {url}: {body[:300]}")
        self.status = status
        self.url = url
        self.body = body


class HttpCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    @staticmethod
    def key(method: str, url: str, params: dict | None, body: Any) -> str:
        blob = _canonical({"method": method.upper(), "url": url, "params": params or {}, "body": body})
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Response | None:
        p = self._path(key)
        if not p.exists():
            with self._lock:
                self.misses += 1
            return None
        d = json.loads(p.read_text())
        with self._lock:
            self.hits += 1
        return Response(d["status"], d["text"], d["retrieved_at"], True, key, d.get("headers", {}))

    def put(self, key: str, resp: Response, request: dict[str, Any]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "request": request,
            "status": resp.status,
            "headers": resp.headers,
            "retrieved_at": resp.retrieved_at,
            "text": resp.text,
        }, sort_keys=True, ensure_ascii=False))
        tmp.replace(p)  # atomic: a concurrent reader never sees a half-written entry

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


class RateLimiter:
    """Minimum interval between requests, per host."""

    def __init__(self, per_second: dict[str, float] | None = None, default_per_second: float = 3.0):
        self.per_second = dict(per_second or {})
        self.default = default_per_second
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        rate = self.per_second.get(host, self.default)
        if rate <= 0:
            return
        gap = 1.0 / rate
        with self._lock:  # serialises the reservation, so N threads still respect the gap
            last = self._last.get(host)
            now = time.monotonic()
            delay = gap - (now - last) if last is not None and now - last < gap else 0.0
            self._last[host] = now + delay
        if delay > 0:
            time.sleep(delay)


def default_cache_root() -> Path:
    env = os.environ.get("ENGINE_CACHE")
    return Path(env) if env else Path.cwd().parent / "cache"


class Http:
    def __init__(
        self,
        cache: HttpCache,
        *,
        limiter: RateLimiter | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 120.0,
        retries: int = 4,
        offline: bool = False,
        backoff_floor: dict[str, float] | None = None,
    ):
        self.cache = cache
        self.limiter = limiter or RateLimiter()
        self.user_agent = user_agent
        self.timeout = timeout
        self.retries = retries
        self.offline = offline
        """When True, a cache miss raises instead of hitting the network — for tests
        and for proving a rerun needs nothing from outside."""
        self.backoff_floor = dict(backoff_floor or {})
        """Per-host minimum sleep after a 429/5xx. Some hosts (gnomAD) keep a 429 alive
        for as long as you keep probing, so the floor must be generous."""
        self._lock = threading.Lock()
        self.live_requests = 0

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        cache_404: bool = False,
        cache_ok: bool = True,
        timeout: float | None = None,
    ) -> Response:
        """Perform (or replay) a request. Raises :class:`HttpError` on a non-retryable
        error status, or on 404 unless ``cache_404`` (then the 404 is returned)."""
        key = HttpCache.key(method, url, params, json_body)
        if cache_ok:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        if self.offline:
            raise RuntimeError(f"offline: no cached response for {method} {url}")

        full = url + ("?" + urllib.parse.urlencode(params, doseq=True) if params else "")
        hdrs = {"User-Agent": self.user_agent, "Accept": "application/json"}
        data = None
        if json_body is not None:
            data = _canonical(json_body).encode()
            hdrs["Content-Type"] = "application/json"
        hdrs.update(headers or {})
        host = urllib.parse.urlparse(url).netloc
        floor = self.backoff_floor.get(host, 0.0)

        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            self.limiter.wait(host)
            req = urllib.request.Request(full, data=data, headers=hdrs, method=method.upper())
            try:
                with self._lock:
                    self.live_requests += 1
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                    text = r.read().decode("utf-8", errors="replace")
                    resp = Response(r.status, text, _now(), False, key, _keep_headers(r.headers))
                if cache_ok:
                    self.cache.put(key, resp, {"method": method.upper(), "url": url, "params": params, "body": json_body})
                return resp
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace") if e.fp else ""
                if e.code == 404 and cache_404:
                    resp = Response(404, body, _now(), False, key, _keep_headers(e.headers))
                    if cache_ok:
                        self.cache.put(key, resp, {"method": method.upper(), "url": url, "params": params, "body": json_body})
                    return resp
                if e.code in RETRY_STATUSES and attempt < self.retries:
                    last_err = e
                    time.sleep(max(_backoff(attempt, e.headers.get("Retry-After")), floor))
                    continue
                raise HttpError(e.code, full, body) from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
                # HTTPException covers a body cut short mid-transfer (IncompleteRead) and a
                # malformed status line — transient on a busy public endpoint, retried like a 5xx
                last_err = e
                if attempt < self.retries:
                    time.sleep(max(_backoff(attempt, None), floor / 4))
                    continue
                raise
        raise RuntimeError(f"request failed after retries: {full}") from last_err

    def get(self, url: str, **kw: Any) -> Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, json_body: Any, **kw: Any) -> Response:
        return self.request("POST", url, json_body=json_body, **kw)


def _backoff(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), 120.0)
        except ValueError:
            pass
    return min(2.0 * (2 ** attempt), 60.0)


_KEEP = ("content-type", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset", "retry-after")


def _keep_headers(h: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in _KEEP:
        v = h.get(k) if hasattr(h, "get") else None
        if v:
            out[k] = str(v)
    return out
