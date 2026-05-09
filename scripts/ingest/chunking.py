"""Chunking strategy for the JSONL records.

Strategy (locked-in earlier in the design conversation):
  - If the record's text is <= 2000 chars, keep it as a single chunk.
  - Otherwise split via RecursiveCharacterTextSplitter at 1500/200, preferring
    `## ` and `### ` headings as separators so we don't shred a single operation
    or example.
  - Every chunk is prepended with `Title: {title}\nSection: {breadcrumb}\n\n`
    so retrieval has the topical anchor even on sub-page chunks.

Per-record text source:
  - `type=endpoint` -> synthesized OAS markdown only (body_md is empty for ~95%).
  - `type=basic` reference page with body_md_chars < 200 AND OAS paths present
        -> body_md + synthesized OAS markdown appended.
  - everything else -> body_md as-is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from scripts.ingest.oas_synth import synthesize_endpoint_markdown


PAGE_WISE_THRESHOLD = 2000
SPLIT_CHUNK_SIZE = 1500
SPLIT_OVERLAP = 200
SPARSE_BODY_THRESHOLD = 200  # for basic/reference pages — append synth if shorter
MIN_CHUNK_SIZE = 500         # below this we merge into a neighbour to avoid tiny fragments


@dataclass
class Chunk:
    """One chunk's payload for embedding + indexing."""
    page_content: str
    metadata: Dict


# Paragraph-first separators. Heading-preference (`\n## `, `\n### `) was tempting
# but produced 50-char fragments at every sub-heading boundary, blowing chunk
# count by 4x without semantic gain. Headings still naturally align with `\n\n`
# in our synthesized output, so we lose nothing meaningful.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _build_record_text(record: Dict) -> str:
    """Decide the source text for one JSONL record."""
    body = record.get("body_md") or ""
    rec_type = record.get("type")
    has_oas_paths = bool((record.get("oas") or {}).get("paths"))

    if rec_type == "endpoint":
        return synthesize_endpoint_markdown(record)

    if rec_type == "basic" and has_oas_paths and len(body) < SPARSE_BODY_THRESHOLD:
        synth = synthesize_endpoint_markdown(record)
        if synth:
            return f"{body}\n\n{synth}" if body.strip() else synth

    return body


def _format_header(record: Dict) -> str:
    title = (record.get("title") or "").strip() or record.get("slug", "")
    breadcrumb = record.get("breadcrumb") or []
    section = " > ".join(b for b in breadcrumb if b) or "Documentation"
    return f"Title: {title}\nSection: {section}\n\n"


def _make_metadata(record: Dict, chunk_index: int, total_chunks: int) -> Dict:
    """Metadata travels with each chunk through FAISS load/save (pickled).

    Keep this small — every chunk pays for it. We pull URL, title, slug, kind,
    breadcrumb so the LLM can cite, and chunk_index for debugging retrieval hits.
    """
    return {
        "source": "developer_connect",
        "url": record["url"],
        "title": record.get("title", ""),
        "slug": record.get("slug", ""),
        "kind": record.get("kind", ""),
        "type": record.get("type") or "",
        "breadcrumb": record.get("breadcrumb") or [],
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }


def chunk_record(record: Dict) -> List[Chunk]:
    """Turn one JSONL record into 0..N chunks.

    Returns an empty list when the record has no chunkable text — caller can
    skip these without special-casing per-type filters.
    """
    text = _build_record_text(record)
    if not text or not text.strip():
        return []

    header = _format_header(record)

    if len(text) <= PAGE_WISE_THRESHOLD:
        page_content = header + text
        return [Chunk(page_content=page_content,
                      metadata=_make_metadata(record, 0, 1))]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SPLIT_CHUNK_SIZE,
        chunk_overlap=SPLIT_OVERLAP,
        separators=_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )
    pieces = splitter.split_text(text)
    pieces = _merge_small(pieces)
    total = len(pieces)
    return [
        Chunk(page_content=header + piece,
              metadata=_make_metadata(record, i, total))
        for i, piece in enumerate(pieces)
    ]


def _merge_small(pieces: List[str]) -> List[str]:
    """Walk through chunks left-to-right; merge any chunk under MIN_CHUNK_SIZE
    into its neighbour as long as the combined size stays under 2*chunk_size."""
    if not pieces:
        return pieces
    out: List[str] = [pieces[0]]
    cap = 2 * SPLIT_CHUNK_SIZE
    for p in pieces[1:]:
        if len(p) < MIN_CHUNK_SIZE and len(out[-1]) + len(p) < cap:
            out[-1] = out[-1] + "\n\n" + p
        elif len(out[-1]) < MIN_CHUNK_SIZE and len(out[-1]) + len(p) < cap:
            # tiny chunk at the front — fold it forward
            out[-1] = out[-1] + "\n\n" + p
        else:
            out.append(p)
    return out


def chunk_records(records: List[Dict]) -> List[Chunk]:
    """Batch helper. Skips records that produced no chunks."""
    out: List[Chunk] = []
    for r in records:
        out.extend(chunk_record(r))
    return out
