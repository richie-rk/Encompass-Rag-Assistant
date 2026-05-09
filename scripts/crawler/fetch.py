"""HTTP client with on-disk HTML cache, polite delays, and retries.

Cache layout: <cache_dir>/<relative-url-path>.html  (one file per page).
Reruns hit the cache by default; pass force=True or use --no-cache to refetch.
"""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

USER_AGENT = (
    "EncompassRest-RAG-Assistant-Crawler/0.1 "
    "(+https://github.com/richie-rk/encompass-rag-assistant)"
)


@dataclass
class FetchResult:
    url: str
    html: str
    http_status: int
    cache_hit: bool
    body_hash: str
    cache_path: Path


class Fetcher:
    def __init__(
        self,
        cache_dir: Path,
        *,
        delay: float = 0.5,
        timeout: float = 30.0,
        retries: int = 3,
        force_refresh: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self.force_refresh = force_refresh

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._last_live_fetch: float = 0.0

    def cache_path_for(self, url: str) -> Path:
        # Map URL path to a cache file; index pages get _index.html
        parsed = urlparse(url)
        rel = parsed.path.lstrip("/")
        if not rel or rel.endswith("/"):
            rel = (rel + "_index").rstrip("/") if rel else "_index"
        return self.cache_dir / f"{rel}.html"

    def fetch(self, url: str) -> Optional[FetchResult]:
        path = self.cache_path_for(url)
        if path.exists() and not self.force_refresh:
            html = path.read_text(encoding="utf-8")
            return FetchResult(
                url=url, html=html, http_status=200, cache_hit=True,
                body_hash=_sha256(html), cache_path=path,
            )

        # Polite spacing between live fetches only
        now = time.monotonic()
        gap = now - self._last_live_fetch
        if gap < self.delay:
            time.sleep(self.delay - gap + random.uniform(0, 0.1))

        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout, allow_redirects=True)
                self._last_live_fetch = time.monotonic()

                if resp.status_code in (429, 500, 502, 503, 504):
                    backoff = min(2 ** attempt, 30) + random.uniform(0, 1)
                    time.sleep(backoff)
                    continue

                if resp.status_code != 200:
                    return FetchResult(
                        url=url, html="", http_status=resp.status_code,
                        cache_hit=False, body_hash="", cache_path=path,
                    )

                html = resp.text
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(html, encoding="utf-8")
                return FetchResult(
                    url=url, html=html, http_status=200, cache_hit=False,
                    body_hash=_sha256(html), cache_path=path,
                )
            except requests.RequestException as e:
                last_err = e
                self._last_live_fetch = time.monotonic()
                time.sleep(min(2 ** attempt, 30))

        if last_err is not None:
            raise last_err
        return None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
