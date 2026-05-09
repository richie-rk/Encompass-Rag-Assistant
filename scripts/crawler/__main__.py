"""CLI entry point for the Developer Connect crawler.

Usage:
    python -m scripts.crawler                        # full crawl, defaults
    python -m scripts.crawler --limit 10             # debug: first 10 pages
    python -m scripts.crawler --no-cache             # force refetch all pages
    python -m scripts.crawler --dry-run              # enumerate frontier only
    python -m scripts.crawler --include-hidden       # do not skip hidden:true
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn,
)

from .extract import extract_internal_links, extract_record
from .fetch import Fetcher
from .sidebar import FrontierEntry, SSR_PROPS_RE, enumerate_section


def _classify_empty(html: str) -> str:
    """When extraction returns None, classify why so the manifest is useful."""
    if SSR_PROPS_RE.search(html) is None:
        # ICE login page is what gets returned for auth-gated docs (no ssr-props).
        if "<title>Login</title>" in html or 'class="loginForm"' in html:
            return "auth_required (login page returned)"
        return "no_ssr_props"
    return "no_doc_or_custompage"

console = Console()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "scripts" / "data"
DEFAULT_BASE_URL = "https://developer.icemortgagetechnology.com"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="scripts.crawler",
        description="Crawl ICE MT Developer Connect docs into JSONL.",
    )
    p.add_argument("--section", default="developer-connect",
                   help="ReadMe section/project slug (default: developer-connect)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help=f"Site base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("--out", type=Path,
                   default=DEFAULT_DATA_DIR / "developer_connect.jsonl",
                   help="Output JSONL path")
    p.add_argument("--manifest", type=Path,
                   default=DEFAULT_DATA_DIR / "developer_connect_manifest.jsonl",
                   help="Per-URL fetch manifest path")
    p.add_argument("--cache-dir", type=Path,
                   default=DEFAULT_DATA_DIR / "cache",
                   help="HTML cache directory")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Minimum seconds between live fetches (default: 0.5)")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="Per-request timeout in seconds (default: 30)")
    p.add_argument("--retries", type=int, default=3,
                   help="Retries on 429/5xx/network errors (default: 3)")
    p.add_argument("--no-cache", action="store_true",
                   help="Refetch every URL even if cached")
    p.add_argument("--include-hidden", action="store_true",
                   help="Do not skip pages with hidden=true")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N pages (debug; 0 = no limit). Disables BFS discovery.")
    p.add_argument("--max-pages", type=int, default=1000,
                   help="Hard ceiling on total frontier including BFS-discovered (default: 1000)")
    p.add_argument("--no-discover", action="store_true",
                   help="Disable BFS link discovery; sidebar-only enumeration (legacy behavior)")
    p.add_argument("--dry-run", action="store_true",
                   help="Enumerate the frontier and exit; do not fetch pages")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold]Developer Connect crawler[/bold]  section={args.section}")
    console.print(f"  base_url   : {args.base_url}")
    console.print(f"  cache_dir  : {args.cache_dir}")
    console.print(f"  out        : {args.out}")
    console.print(f"  manifest   : {args.manifest}")
    console.print(f"  delay      : {args.delay}s   refresh={'yes' if args.no_cache else 'no'}   "
                  f"include_hidden={'yes' if args.include_hidden else 'no'}")

    fetcher = Fetcher(
        cache_dir=args.cache_dir / args.section,
        delay=args.delay, timeout=args.timeout, retries=args.retries,
        force_refresh=args.no_cache,
    )

    console.print("\n[yellow]Enumerating frontier from sidebar...[/yellow]")
    t0 = time.monotonic()
    frontier = enumerate_section(
        fetcher,
        base_url=args.base_url, section=args.section,
        skip_hidden=not args.include_hidden,
    )
    console.print(f"  -> {len(frontier)} entries in {time.monotonic()-t0:.1f}s")

    counts = {"guide": 0, "reference": 0, "changelog": 0}
    for e in frontier:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    console.print(f"  guides={counts.get('guide',0)}  refs={counts.get('reference',0)}  changelog={counts.get('changelog',0)}")

    # `--limit` truncates the initial frontier and disables BFS so debug runs are
    # deterministic in size. `--no-discover` keeps the full sidebar enumeration but
    # skips BFS — useful for legacy reproducibility.
    bfs_enabled = not (args.limit or args.no_discover)
    if args.limit:
        frontier = frontier[: args.limit]
        console.print(f"  [dim]limited to first {len(frontier)} entries (BFS disabled)[/dim]")
    elif args.no_discover:
        console.print(f"  [dim]BFS discovery disabled (--no-discover)[/dim]")
    else:
        console.print(f"  [dim]BFS discovery enabled (cap: {args.max_pages})[/dim]")

    if args.dry_run:
        console.print("\n[bold]Dry-run; first 15 frontier entries:[/bold]")
        for e in frontier[:15]:
            console.print(f"  [{e.kind:9s}] {e.url}  ({e.title})")
        return 0

    n_fetched = n_cache_hit = n_extracted = n_failed = n_empty = 0
    n_discovered = 0
    seen_urls: set[str] = {e.url for e in frontier}

    out_f = args.out.open("w", encoding="utf-8")
    man_f = args.manifest.open("w", encoding="utf-8")
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(),
            TimeElapsedColumn(), TimeRemainingColumn(), console=console,
        ) as bar:
            task = bar.add_task("Crawling", total=len(frontier))
            i = 0
            while i < len(frontier):
                entry = frontier[i]
                i += 1
                bar.update(task, description=f"Crawling [{entry.kind}] {entry.slug}")
                manifest_entry = {
                    "url": entry.url, "slug": entry.slug, "kind": entry.kind,
                    "fetched_at": _now_iso(),
                    "discovered_via": entry.sidebar_meta.get("discovered_via", "sidebar"),
                }
                try:
                    res = fetcher.fetch(entry.url)
                except Exception as e:
                    n_failed += 1
                    manifest_entry.update({"http_status": None, "cache_hit": False,
                                            "body_hash": None, "error": str(e)[:300]})
                    man_f.write(json.dumps(manifest_entry) + "\n")
                    bar.advance(task)
                    continue

                if res is None or res.http_status != 200:
                    n_failed += 1
                    manifest_entry.update({
                        "http_status": getattr(res, "http_status", None),
                        "cache_hit": False, "body_hash": None,
                        "error": f"non-200: {getattr(res, 'http_status', 'no-response')}",
                    })
                    man_f.write(json.dumps(manifest_entry) + "\n")
                    bar.advance(task)
                    continue

                n_fetched += 1
                if res.cache_hit:
                    n_cache_hit += 1

                record = extract_record(
                    res.html, entry,
                    base_url=args.base_url, section=args.section,
                )
                manifest_entry.update({
                    "http_status": res.http_status,
                    "cache_hit": res.cache_hit,
                    "body_hash": res.body_hash,
                })
                if record is None:
                    n_empty += 1
                    manifest_entry["error"] = _classify_empty(res.html)
                else:
                    n_extracted += 1
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()  # safe to ctrl-C without losing data

                    if bfs_enabled and len(frontier) < args.max_pages:
                        for link in extract_internal_links(
                            res.html,
                            base_url=args.base_url, section=args.section,
                            source_url=entry.url,
                        ):
                            if link.url in seen_urls:
                                continue
                            seen_urls.add(link.url)
                            frontier.append(link)
                            n_discovered += 1
                            if len(frontier) >= args.max_pages:
                                break
                        bar.update(task, total=len(frontier))

                man_f.write(json.dumps(manifest_entry) + "\n")
                man_f.flush()
                bar.advance(task)
    finally:
        out_f.close()
        man_f.close()

    console.rule("[bold]Summary[/bold]")
    console.print(f"  total frontier  : {len(frontier)}")
    if bfs_enabled:
        console.print(f"  via sidebar     : {len(frontier) - n_discovered}")
        console.print(f"  via BFS         : {n_discovered}")
    console.print(f"  fetched OK      : {n_fetched}  ({n_cache_hit} cache hits)")
    console.print(f"  extracted       : {n_extracted}")
    console.print(f"  empty/no-doc    : {n_empty}")
    console.print(f"  failed          : {n_failed}")
    console.print(f"\n  [green]Wrote {args.out}[/green]")
    console.print(f"  [green]Wrote {args.manifest}[/green]")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
