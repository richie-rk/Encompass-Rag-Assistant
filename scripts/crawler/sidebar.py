"""Enumerate the page frontier from one or two seed fetches.

Strategy:
  1. Fetch the section's welcome page; ssr-props.sidebars contains the entire
     guide + reference TOC for the whole section (no BFS, no 404 noise).
  2. Fetch the section's /changelog index; ssr-props.changelogs is the list of
     release-notes entries.
  3. Walk every body for changelog:slug references to pick up release subpages
     not exposed in the index.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .fetch import Fetcher

SSR_PROPS_RE = re.compile(
    r'<script id="ssr-props" type="application/json">(.*?)</script>',
    re.DOTALL,
)
CHANGELOG_REF_RE = re.compile(r'\(changelog:([a-z0-9][a-z0-9._-]*)\)', re.IGNORECASE)


@dataclass
class FrontierEntry:
    slug: str
    url: str
    kind: str  # "guide" | "reference" | "changelog"
    title: str
    category_title: Optional[str] = None
    category_slug: Optional[str] = None
    parent_slug: Optional[str] = None
    breadcrumb: List[str] = field(default_factory=list)
    hidden: bool = False
    deprecated: bool = False
    sidebar_meta: Dict = field(default_factory=dict)  # keep raw node for reference


def parse_ssr_props(html: str) -> Optional[Dict]:
    m = SSR_PROPS_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _walk_pages(
    nodes: Iterable[Dict],
    *,
    kind: str,
    base_url: str,
    section: str,
    category_title: Optional[str],
    category_slug: Optional[str],
    breadcrumb: List[str],
    parent_slug: Optional[str],
    skip_hidden: bool,
    out: List[FrontierEntry],
) -> None:
    url_segment = "docs" if kind == "guide" else "reference"
    for node in nodes or []:
        slug = node.get("slug")
        if not slug:
            continue
        title = node.get("title") or slug
        hidden = bool(node.get("hidden", False))
        deprecated = bool(node.get("deprecated", False))

        if hidden and skip_hidden:
            # don't enumerate children of a hidden page either
            continue

        url = f"{base_url}/{section}/{url_segment}/{slug}"
        crumb = [*breadcrumb, title]
        out.append(FrontierEntry(
            slug=slug, url=url, kind=kind, title=title,
            category_title=category_title, category_slug=category_slug,
            parent_slug=parent_slug, breadcrumb=crumb,
            hidden=hidden, deprecated=deprecated,
            sidebar_meta={k: v for k, v in node.items() if k != "pages"},
        ))
        children = node.get("pages") or []
        if children:
            _walk_pages(
                children, kind=kind, base_url=base_url, section=section,
                category_title=category_title, category_slug=category_slug,
                breadcrumb=crumb, parent_slug=slug,
                skip_hidden=skip_hidden, out=out,
            )


def enumerate_section(
    fetcher: Fetcher,
    *,
    base_url: str,
    section: str = "developer-connect",
    skip_hidden: bool = True,
) -> List[FrontierEntry]:
    base_url = base_url.rstrip("/")
    frontier: List[FrontierEntry] = []

    welcome_url = f"{base_url}/{section}/docs/welcome"
    res = fetcher.fetch(welcome_url)
    if res is None or res.http_status != 200:
        raise RuntimeError(f"Could not fetch welcome page: {welcome_url}")
    ssr = parse_ssr_props(res.html)
    if ssr is None:
        raise RuntimeError(f"No ssr-props in welcome page: {welcome_url}")

    sidebars = ssr.get("sidebars") or {}
    for cat in sidebars.get("docs") or []:
        _walk_pages(
            cat.get("pages") or [], kind="guide",
            base_url=base_url, section=section,
            category_title=cat.get("title"), category_slug=cat.get("slug"),
            breadcrumb=[cat.get("title") or ""], parent_slug=None,
            skip_hidden=skip_hidden, out=frontier,
        )
    for cat in sidebars.get("refs") or []:
        _walk_pages(
            cat.get("pages") or [], kind="reference",
            base_url=base_url, section=section,
            category_title=cat.get("title"), category_slug=cat.get("slug"),
            breadcrumb=[cat.get("title") or ""], parent_slug=None,
            skip_hidden=skip_hidden, out=frontier,
        )

    # Changelog: index page surfaces a `changelogs` list, plus body cross-links
    changelog_index = f"{base_url}/{section}/changelog"
    cl_res = fetcher.fetch(changelog_index)
    if cl_res is not None and cl_res.http_status == 200:
        cl_ssr = parse_ssr_props(cl_res.html) or {}
        seen_slugs: set[str] = set()
        for entry in cl_ssr.get("changelogs") or []:
            slug = entry.get("slug")
            if not slug or slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            if entry.get("hidden") and skip_hidden:
                continue
            frontier.append(FrontierEntry(
                slug=slug,
                url=f"{base_url}/{section}/changelog/{slug}",
                kind="changelog",
                title=entry.get("title") or slug,
                category_title="Release Notes",
                category_slug="changelog",
                breadcrumb=["Release Notes", entry.get("title") or slug],
                hidden=bool(entry.get("hidden", False)),
                deprecated=False,
                sidebar_meta={"type": entry.get("type"), "updatedAt": entry.get("updatedAt")},
            ))
            for ref_slug in CHANGELOG_REF_RE.findall(entry.get("body") or ""):
                if ref_slug in seen_slugs:
                    continue
                seen_slugs.add(ref_slug)
                frontier.append(FrontierEntry(
                    slug=ref_slug,
                    url=f"{base_url}/{section}/changelog/{ref_slug}",
                    kind="changelog",
                    title=ref_slug,  # filled in from doc when fetched
                    category_title="Release Notes",
                    category_slug="changelog",
                    breadcrumb=["Release Notes", ref_slug],
                ))

    return frontier
