"""Collapse JSONL records that are duplicate views of the same page.

ReadMe sometimes serves identical content at multiple URLs (e.g., `/docs/auth`
and `/reference/auth` both render the same `doc.body`). Our BFS picks up both;
left untreated they would create twin vectors in FAISS, surface the same chunk
twice in retrieval, and waste prompt context.

Detection:  sha256(body_md). Records with the same hash are byte-identical.

Tie-break (which URL wins when a hash collision is found):
  1. `discovered_via=sidebar` beats `discovered_via=link_bfs`. Sidebar entries
     are the URLs ReadMe officially navigates to.
  2. Longer `body_md` wins. Hashes match so this is academic, but if any
     rendering difference creeps in, keep the richer record.
  3. Lexicographic URL — deterministic so re-runs are reproducible.

The decision logic mirrors the ranking we sort the input list by; the first
record per hash wins.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple


def _content_hash(record: Dict) -> str:
    body = (record.get("body_md") or "").encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _sort_key(record: Dict) -> Tuple[bool, int, str]:
    via = record.get("discovered_via") or ""
    is_not_sidebar = via != "sidebar"             # False (sidebar) sorts first
    body_len_neg = -len(record.get("body_md") or "")
    return (is_not_sidebar, body_len_neg, record.get("url", ""))


def dedupe_by_content(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Return (kept, dropped). `dropped` is for logging — same shape as `kept`.

    Records with empty bodies are NOT collapsed against each other on the
    body_md hash alone (every empty-body record would otherwise dedupe to one).
    Empty-body records are passed through as-is; their content uniqueness
    comes from the OAS spec, which the chunker handles separately.
    """
    sorted_recs = sorted(records, key=_sort_key)
    seen: dict[str, Dict] = {}
    kept: List[Dict] = []
    dropped: List[Dict] = []
    for r in sorted_recs:
        body = r.get("body_md") or ""
        if not body.strip():
            kept.append(r)
            continue
        h = _content_hash(r)
        if h in seen:
            dropped.append(r)
            continue
        seen[h] = r
        kept.append(r)
    return kept, dropped
