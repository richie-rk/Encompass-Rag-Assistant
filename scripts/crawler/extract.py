"""Build a normalized PageRecord dict from a fetched page's ssr-props.

Input shapes encountered on Developer Connect:
  guide       -> ssr.doc            (type=basic, isReference=False)
  reference   -> ssr.doc            + ssr.oasDefinition when ReadMe has it
  changelog   -> ssr.changelogs[*]  on the index page (list shape)
                 ssr.changelog       on per-entry pages   (singular shape)
  custompage  -> ssr.custompage      (HTML/JS landing pages, low prose value)

Pages that return the ICE login form have no ssr-props at all; the caller
classifies that case from raw HTML.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional
from urllib.parse import urljoin

from .rewrite import rewrite_body
from .sidebar import FrontierEntry, parse_ssr_props


def _category_summary(doc: Dict) -> Dict[str, Optional[str]]:
    cat = doc.get("category") or {}
    return {
        "title": cat.get("title"),
        "slug": cat.get("slug"),
        "type": cat.get("type"),
        "id": cat.get("_id") or cat.get("id"),
    }


def _summarize_oas(oas: Dict) -> Optional[Dict]:
    """Strip ReadMe-internal noise but keep the full OpenAPI document.
    Returns None if no real paths or schemas are present.
    """
    if not isinstance(oas, dict):
        return None
    paths = oas.get("paths") or {}
    schemas = (oas.get("components") or {}).get("schemas") or {}
    if not paths and not schemas:
        return None
    return {
        "openapi": oas.get("openapi"),
        "info": oas.get("info"),
        "servers": oas.get("servers"),
        "paths": paths,
        "components": oas.get("components") or {},
    }


def extract_record(
    html: str,
    entry: FrontierEntry,
    *,
    base_url: str,
    section: str,
) -> Optional[Dict]:
    ssr = parse_ssr_props(html)
    if ssr is None:
        return None

    if entry.kind == "changelog":
        return _extract_changelog(ssr, entry, base_url=base_url, section=section)

    doc = ssr.get("doc") or {}
    if not doc:
        cp = ssr.get("custompage") or {}
        if cp:
            return _extract_custompage(cp, entry, base_url=base_url, section=section)
        return None

    raw_body = doc.get("body") or ""
    body_md = rewrite_body(raw_body, base_url=base_url, section=section)

    record = {
        "source": "developer_connect",
        "section": section,
        "kind": entry.kind,
        "slug": doc.get("slug") or entry.slug,
        "title": doc.get("title") or entry.title,
        "url": entry.url,
        "category": _category_summary(doc),
        "parent_slug": entry.parent_slug,
        "breadcrumb": entry.breadcrumb,
        "hidden": bool(doc.get("hidden", entry.hidden)),
        "deprecated": bool(doc.get("deprecated", entry.deprecated)),
        "type": doc.get("type"),
        "is_api": bool(doc.get("isApi")),
        "is_reference": bool(doc.get("isReference")),
        "excerpt": doc.get("excerpt") or "",
        "body_md": body_md,
        "body_md_chars": len(body_md),
        "raw_body_chars": len(raw_body),
        "updated_at": doc.get("updatedAt"),
        "created_at": doc.get("createdAt"),
        "revision": doc.get("revision"),
    }

    if entry.kind == "reference":
        record["oas"] = _summarize_oas(ssr.get("oasDefinition") or {})
        record["oas_public_url"] = ssr.get("oasPublicUrl")

    # ReadMe surfaces some content via fields outside `doc.body` and `oasDefinition`.
    # `doc.api` is the high-value one — for high-traffic endpoints (e.g., `get-loan`)
    # it carries rendered cURL/Python/JS code samples that the OAS spec alone does not.
    if doc.get("api"):
        record["doc_api"] = doc["api"]
    if doc.get("reusableContent"):
        record["reusable_content"] = doc["reusableContent"]
    if doc.get("tutorials"):
        record["doc_tutorials"] = doc["tutorials"]

    return record


def _extract_changelog(
    ssr: Dict,
    entry: FrontierEntry,
    *,
    base_url: str,
    section: str,
) -> Optional[Dict]:
    # Per-entry pages put the record at ssr.changelog (singular dict).
    # The /changelog index page puts a list at ssr.changelogs and the most
    # recently-published entry at ssr.changelog as well, so we prefer the
    # singular form first.
    match = ssr.get("changelog")
    if not match:
        cl_list = ssr.get("changelogs") or []
        match = next((c for c in cl_list if c.get("slug") == entry.slug), None)
    if match is None and "doc" in ssr:
        match = ssr.get("doc")

    if not match:
        return None

    raw_body = match.get("body") or ""
    body_md = rewrite_body(raw_body, base_url=base_url, section=section)

    return {
        "source": "developer_connect",
        "section": section,
        "kind": "changelog",
        "slug": match.get("slug") or entry.slug,
        "title": match.get("title") or entry.title,
        "url": entry.url,
        "category": {"title": "Release Notes", "slug": "changelog", "type": "changelog", "id": None},
        "parent_slug": None,
        "breadcrumb": entry.breadcrumb,
        "hidden": bool(match.get("hidden", entry.hidden)),
        "deprecated": False,
        "type": match.get("type"),
        "is_api": False,
        "is_reference": False,
        "excerpt": match.get("bodyPreview") or "",
        "body_md": body_md,
        "body_md_chars": len(body_md),
        "raw_body_chars": len(raw_body),
        "updated_at": match.get("updatedAt"),
        "created_at": match.get("createdAt"),
        "revision": match.get("revision"),
    }


# --- Link discovery (BFS) -----------------------------------------------------

# Markdown link `[text](url)` — negative lookbehind keeps image links `![alt](url)` out.
_LINK_RE = re.compile(r'(?<!!)\[(?:[^\]]*?)\]\(([^)\s]+)\)')

# HTML href attribute. ReadMe renders some nav/sidebar links only in the DOM —
# they aren't in the structured `sidebars` JSON, so we have to scrape hrefs too.
_HREF_RE = re.compile(r'\bhref=["\']([^"\'\s<>]+)["\']')

# ReadMe internal link macros: doc:slug, ref:slug, changelog:slug
_README_MACRO_RE = re.compile(r'^(doc|ref|changelog):([A-Za-z0-9][A-Za-z0-9._-]*)$')

_SLUG_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')

_MACRO_TO_PATHSEG = {"doc": "docs", "ref": "reference", "changelog": "changelog"}
_MACRO_TO_KIND = {"doc": "guide", "ref": "reference", "changelog": "changelog"}
_PATHSEG_TO_KIND = {"docs": "guide", "reference": "reference", "changelog": "changelog"}


def _collect_link_source_body(ssr: Dict) -> str:
    """Concatenate every body markdown a page might carry, so we can scan once."""
    parts: List[str] = []
    doc = ssr.get("doc") or {}
    if doc.get("body"):
        parts.append(doc["body"])
    cl = ssr.get("changelog") or {}
    if cl.get("body"):
        parts.append(cl["body"])
    for c in ssr.get("changelogs") or []:
        if c.get("body"):
            parts.append(c["body"])
    cp = ssr.get("custompage") or {}
    if cp.get("body"):
        parts.append(cp["body"])
    return "\n\n".join(parts)


def extract_internal_links(
    html: str,
    *,
    base_url: str,
    section: str,
    source_url: str,
) -> List[FrontierEntry]:
    """Discover internal cross-references in a fetched page's markdown body.

    Recognizes:
      - ReadMe macros: [X](doc:slug), [X](ref:slug), [X](changelog:slug)
      - Absolute URLs: https://.../<section>/(docs|reference|changelog)/<slug>
      - Relative URLs: /<section>/(docs|reference|changelog)/<slug>
      - Section-relative links resolved against source_url

    Caller is responsible for deduping against the existing frontier.
    """
    section_prefix = f"{base_url.rstrip('/')}/{section}/"
    seen_in_page: set[str] = set()
    discovered: List[FrontierEntry] = []

    def _try_classify(raw: str) -> Optional[tuple]:
        """Return (kind, slug, absolute_url) if `raw` is an internal page link."""
        macro = _README_MACRO_RE.match(raw)
        if macro:
            prefix = macro.group(1).lower()
            slug = macro.group(2)
            path_seg = _MACRO_TO_PATHSEG[prefix]
            kind = _MACRO_TO_KIND[prefix]
            return (kind, slug, f"{section_prefix}{path_seg}/{slug}")

        absolute = urljoin(source_url, raw)
        absolute = absolute.split("#", 1)[0].split("?", 1)[0]
        if not absolute.startswith(section_prefix):
            return None
        tail = absolute[len(section_prefix):].rstrip("/")
        parts = tail.split("/")
        if len(parts) != 2:
            return None
        path_seg, slug_candidate = parts
        if path_seg not in _PATHSEG_TO_KIND:
            return None
        if not _SLUG_RE.match(slug_candidate):
            return None
        return (_PATHSEG_TO_KIND[path_seg], slug_candidate, absolute)

    def _emit(classified):
        kind, slug, url = classified
        if url in seen_in_page:
            return
        seen_in_page.add(url)
        discovered.append(FrontierEntry(
            slug=slug, url=url, kind=kind, title=slug,
            category_title=None, category_slug=None,
            parent_slug=None,
            breadcrumb=[],
            hidden=False, deprecated=False,
            sidebar_meta={"discovered_via": "link_bfs"},
        ))

    # 1) Markdown links in the page body (author-written cross-refs)
    ssr = parse_ssr_props(html)
    body = _collect_link_source_body(ssr) if ssr else ""
    for m in _LINK_RE.finditer(body):
        c = _try_classify(m.group(1).strip())
        if c is not None:
            _emit(c)

    # 2) HTML hrefs across the whole rendered page (nav/sidebar/breadcrumbs).
    #    ReadMe surfaces some pages only in the rendered DOM, not in `sidebars` JSON,
    #    so this pass is what catches `custom-tools`, `plugins`, `webhook-subscriptions`,
    #    and similar slugs that aren't in the structured sidebar tree.
    for m in _HREF_RE.finditer(html):
        c = _try_classify(m.group(1).strip())
        if c is not None:
            _emit(c)

    return discovered


# --- Custompage extraction ----------------------------------------------------


def _extract_custompage(
    cp: Dict,
    entry: FrontierEntry,
    *,
    base_url: str,
    section: str,
) -> Optional[Dict]:
    raw_body = cp.get("body") or ""
    body_md = rewrite_body(raw_body, base_url=base_url, section=section)
    return {
        "source": "developer_connect",
        "section": section,
        "kind": "custompage",
        "slug": cp.get("slug") or entry.slug,
        "title": cp.get("title") or entry.title,
        "url": entry.url,
        "category": {"title": entry.category_title, "slug": entry.category_slug,
                     "type": "custompage", "id": None},
        "parent_slug": entry.parent_slug,
        "breadcrumb": entry.breadcrumb,
        "hidden": bool(cp.get("hidden", entry.hidden)),
        "deprecated": False,
        "type": "custompage",
        "is_api": False,
        "is_reference": False,
        "excerpt": "",
        "body_md": body_md,
        "body_md_chars": len(body_md),
        "raw_body_chars": len(raw_body),
        "html_chars": len(cp.get("html") or ""),
        "fullscreen": bool(cp.get("fullscreen")),
        "updated_at": cp.get("updatedAt"),
        "created_at": cp.get("createdAt"),
        "revision": cp.get("revision"),
    }
