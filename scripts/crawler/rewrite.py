"""Normalize ReadMe-flavored markdown so it stands on its own outside the hub.

Transforms applied:
  - `[label](doc:slug[#anchor])`        -> absolute /docs/<slug> URL
  - `[label](ref:slug[#anchor])`        -> absolute /reference/<slug> URL
  - `[label](changelog:slug[#anchor])`  -> absolute /changelog/<slug> URL
  - `<ComponentName />`                  -> removed (reusable-content placeholders)
  - `[block:html]{...}[/block]`          -> removed (decorative HTML embeds)

Everything else is left intact: [block:code], [block:image], [block:callout],
GFM tables, fenced code, and standard markdown all pass through.
"""
from __future__ import annotations

import re

LINK_MACRO_RE = re.compile(
    r'\(\s*(doc|ref|changelog):([A-Za-z0-9][A-Za-z0-9._-]*)(#[^)\s]+)?\s*\)'
)
SELF_CLOSING_COMPONENT_RE = re.compile(r'<([A-Z][A-Za-z0-9]*)\s*/>')
BLOCK_HTML_RE = re.compile(r'\[block:html\].*?\[/block\]', re.DOTALL)

KIND_TO_SEGMENT = {"doc": "docs", "ref": "reference", "changelog": "changelog"}


def rewrite_body(body: str, *, base_url: str, section: str) -> str:
    if not body:
        return ""
    base = f"{base_url.rstrip('/')}/{section}"

    def _link_sub(m: re.Match) -> str:
        kind, slug, anchor = m.group(1), m.group(2), m.group(3) or ""
        segment = KIND_TO_SEGMENT[kind]
        return f"({base}/{segment}/{slug}{anchor})"

    out = LINK_MACRO_RE.sub(_link_sub, body)
    out = BLOCK_HTML_RE.sub("", out)
    out = SELF_CLOSING_COMPONENT_RE.sub("", out)
    # collapse runs of >2 blank lines that the strips can leave behind
    out = re.sub(r'\n{3,}', '\n\n', out)
    return out.strip()
