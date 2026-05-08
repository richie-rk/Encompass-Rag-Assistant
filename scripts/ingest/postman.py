"""Flatten a Postman v2.1 collection into BM25-friendly entries.

The collection is a tree of folders (`item` arrays containing nested `item`
arrays) terminating in request leaves. We walk the tree, build one entry per
leaf, and emit a text doc that maximizes keyword surface for BM25:

    {folder_path}
    {request name}
    {METHOD} {url}
    {description}
    Headers: {key1, key2, ...}
    Query: {key1, key2, ...}
    Body fields: {key1, key2, ...}

Each entry also carries metadata for the gated retrieval flow at query time
(name + method + path + brief description go straight into the LLM prompt
when the gate fires).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class PostmanEntry:
    name: str
    method: str
    url: str
    description: str
    folder_path: List[str]
    bm25_text: str
    # Tokens for BM25 — pre-tokenized to keep the rest of the pipeline simple.
    tokens: List[str] = field(default_factory=list)


# --- URL normalization --------------------------------------------------------

def _render_url(url_obj: Any) -> str:
    """Postman urls can be a string OR an object with `raw`/`host`/`path`/`query`."""
    if isinstance(url_obj, str):
        return url_obj
    if not isinstance(url_obj, dict):
        return ""
    raw = url_obj.get("raw")
    if isinstance(raw, str) and raw:
        return raw
    host = url_obj.get("host") or []
    path = url_obj.get("path") or []
    if isinstance(host, list):
        host_s = ".".join(str(h) for h in host)
    else:
        host_s = str(host)
    if isinstance(path, list):
        path_s = "/".join(str(p) for p in path)
    else:
        path_s = str(path)
    return f"{host_s}/{path_s}".strip("/")


# --- Body / header / query field key extraction -------------------------------

def _body_keys(body: Optional[Dict]) -> List[str]:
    if not isinstance(body, dict):
        return []
    mode = body.get("mode")
    if mode == "urlencoded":
        return [str(p.get("key") or "") for p in (body.get("urlencoded") or []) if isinstance(p, dict)]
    if mode == "formdata":
        return [str(p.get("key") or "") for p in (body.get("formdata") or []) if isinstance(p, dict)]
    if mode == "raw":
        raw = body.get("raw") or ""
        # Try to parse as JSON and surface its top-level keys; otherwise empty.
        if isinstance(raw, str) and raw.strip().startswith(("{", "[")):
            try:
                parsed = json.loads(raw)
            except Exception:
                return []
            if isinstance(parsed, dict):
                return list(parsed.keys())
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return list(parsed[0].keys())
        return []
    return []


def _header_keys(headers: Optional[List[Dict]]) -> List[str]:
    if not isinstance(headers, list):
        return []
    return [str(h.get("key") or "") for h in headers if isinstance(h, dict) and h.get("key")]


def _query_keys(url_obj: Any) -> List[str]:
    if not isinstance(url_obj, dict):
        return []
    return [str(q.get("key") or "") for q in (url_obj.get("query") or [])
            if isinstance(q, dict) and q.get("key")]


def _path_variables(url_obj: Any) -> List[str]:
    """Path variables in Postman urls show up as `:name` segments or `variable[]`."""
    out: List[str] = []
    if isinstance(url_obj, dict):
        for v in url_obj.get("variable") or []:
            if isinstance(v, dict) and v.get("key"):
                out.append(str(v["key"]))
        for seg in url_obj.get("path") or []:
            if isinstance(seg, str) and seg.startswith(":"):
                out.append(seg[1:])
    return out


# --- BM25 doc + tokenization --------------------------------------------------

import re

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\d+")


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens; matches what we use for the JSONL BM25
    side so query tokenization is consistent across both indices.
    """
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _build_bm25_text(
    *, folder_path: List[str], name: str, method: str, url: str,
    description: str, header_keys: List[str], body_keys: List[str],
    query_keys: List[str], path_vars: List[str],
) -> str:
    parts: List[str] = []
    if folder_path:
        parts.append(" > ".join(folder_path))
    parts.append(name)
    parts.append(f"{method} {url}")
    if description:
        parts.append(description)
    if header_keys:
        parts.append("Headers: " + ", ".join(header_keys))
    if query_keys:
        parts.append("Query: " + ", ".join(query_keys))
    if path_vars:
        parts.append("Path vars: " + ", ".join(path_vars))
    if body_keys:
        parts.append("Body fields: " + ", ".join(body_keys))
    return "\n".join(parts)


# --- Tree walk ----------------------------------------------------------------

def _walk(items: Iterable[Dict], folder_path: List[str]) -> Iterable[PostmanEntry]:
    for it in items or []:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "")
        if "item" in it:  # folder
            yield from _walk(it["item"], folder_path + [name])
            continue
        req = it.get("request")
        if not isinstance(req, dict):
            continue
        method = str(req.get("method") or "GET").upper()
        url = _render_url(req.get("url"))
        description = str(req.get("description") or "")
        header_keys = _header_keys(req.get("header"))
        body_keys = _body_keys(req.get("body"))
        query_keys = _query_keys(req.get("url"))
        path_vars = _path_variables(req.get("url"))

        bm25_text = _build_bm25_text(
            folder_path=folder_path, name=name, method=method, url=url,
            description=description, header_keys=header_keys,
            body_keys=body_keys, query_keys=query_keys, path_vars=path_vars,
        )

        yield PostmanEntry(
            name=name, method=method, url=url, description=description,
            folder_path=list(folder_path), bm25_text=bm25_text,
            tokens=_tokenize(bm25_text),
        )


def load_postman_entries(collection_path: Path) -> List[PostmanEntry]:
    with collection_path.open(encoding="utf-8") as f:
        coll = json.load(f)
    return list(_walk(coll.get("item") or [], folder_path=[]))


def tokenize_query(query: str) -> List[str]:
    """Same tokenization the BM25 docs were built with — exposed so the
    retriever side can use it without importing a private."""
    return _tokenize(query)
