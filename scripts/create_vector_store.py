"""Build the three-index vector store for the RAG pipeline.

Inputs:
  - scripts/data/developer_connect.jsonl   (BFS-crawled documentation)
  - scripts/data/Encompass_Developer_Connect_postman_collection.json

Outputs (under VECTOR_STORE_PATH, default `vector_store/`):
  - jsonl_faiss/index.faiss + index.pkl   (LangChain FAISS-IP, Jina v3 vectors)
  - jsonl_bm25.pkl + jsonl_chunks.pkl     (BM25 over the same chunks)
  - postman_bm25.pkl + postman_entries.pkl (separate BM25 for endpoint surfacing)
  - stats.json                            (chunk/index counts for sanity)

Pipeline per record:
  filter -> dedupe -> chunk (page-wise/<=2K, otherwise 1500/200 with headers)
  -> embed via Jina v3 (task=retrieval.passage)
  -> FAISS-IP + BM25.

Run from the repo root:
    python -m scripts.create_vector_store
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

# Repo-root-relative imports work because we run from the repo root.
from scripts.ingest.chunking import Chunk, chunk_records
from scripts.ingest.dedupe import dedupe_by_content
from scripts.ingest.jina_embedder import JinaEmbedder
from scripts.ingest.postman import load_postman_entries, tokenize_query

console = Console()
load_dotenv()


# --- Paths --------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
JSONL_PATH = REPO_ROOT / "scripts" / "data" / "developer_connect.jsonl"
POSTMAN_PATH = REPO_ROOT / "scripts" / "data" / "Encompass_Developer_Connect_postman_collection.json"
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "vector_store"))


# --- Steps --------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _filter_records(records: List[Dict]) -> List[Dict]:
    """Drop record types we don't ingest."""
    out = []
    for r in records:
        if r.get("type") == "custompage":
            continue  # 4 records, all empty after extraction (deferred fix)
        out.append(r)
    return out


def _chunks_to_documents(chunks: List[Chunk]) -> List[Document]:
    return [Document(page_content=c.page_content, metadata=c.metadata) for c in chunks]


def _texts_fingerprint(texts: List[str]) -> str:
    """Stable hash of the chunk text list. If chunking parameters change, this
    changes — we use it to invalidate stale checkpoints."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _embed_with_checkpoint(
    texts: List[str], embedder: JinaEmbedder, checkpoint: Path,
    *, save_every: int = 512,
) -> np.ndarray:
    """Embed all `texts` with on-disk resumability.

    The Jina API can rate-limit mid-build; this lets a re-run pick up where it
    left off instead of re-paying for already-embedded chunks. The checkpoint
    file holds however many vectors we've successfully computed so far (in the
    same order as `texts`); a sidecar JSON tracks the chunk-list fingerprint
    so a checkpoint from a different chunking config doesn't silently apply.
    """
    meta_path = checkpoint.with_suffix(".meta.json")
    fingerprint = _texts_fingerprint(texts)

    if checkpoint.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            existing = np.load(checkpoint)
            if existing.shape[0] >= len(texts):
                console.print(f"  [dim]checkpoint complete ({existing.shape[0]} vectors); skipping embed[/dim]")
                return existing[: len(texts)]
            console.print(f"  [dim]resuming from checkpoint ({existing.shape[0]}/{len(texts)} embedded)[/dim]")
            vectors = existing
        else:
            console.print("  [yellow]checkpoint stale (chunks changed) — embedding from scratch[/yellow]")
            vectors = np.zeros((0, 1024), dtype=np.float32)
    else:
        vectors = np.zeros((0, 1024), dtype=np.float32)

    def _persist(v: np.ndarray) -> None:
        np.save(checkpoint, v)
        meta_path.write_text(
            json.dumps({"fingerprint": fingerprint, "count": int(v.shape[0])}),
            encoding="utf-8",
        )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(),
        TimeElapsedColumn(), TimeRemainingColumn(), console=console,
    ) as bar:
        task = bar.add_task("Embedding chunks", total=len(texts), completed=int(vectors.shape[0]))
        while vectors.shape[0] < len(texts):
            start = int(vectors.shape[0])
            end = min(start + save_every, len(texts))
            new_vecs = embedder.embed_passages(texts[start:end])
            vectors = np.vstack([vectors, new_vecs]) if vectors.size else new_vecs
            _persist(vectors)
            bar.update(task, completed=int(vectors.shape[0]))
    return vectors


def _build_jsonl_indices(
    documents: List[Document], vectors: np.ndarray, embedder: JinaEmbedder,
) -> "FAISS":
    """Build FAISS-IP from precomputed vectors so we don't re-embed on retry."""
    text_emb_pairs = list(zip(
        (d.page_content for d in documents),
        vectors.tolist(),
    ))
    metadatas = [d.metadata for d in documents]
    vector_store = FAISS.from_embeddings(
        text_emb_pairs,
        embedding=embedder.as_langchain(),
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=False,  # Jina v3 returns unit vectors already.
    )
    return vector_store


def _build_jsonl_bm25(documents: List[Document]) -> tuple[BM25Okapi, List[List[str]]]:
    tokenized = [tokenize_query(d.page_content) for d in documents]
    return BM25Okapi(tokenized), tokenized


def _build_postman_bm25(entries) -> BM25Okapi:
    return BM25Okapi([e.tokens for e in entries])


# --- Orchestrator -------------------------------------------------------------

def main() -> int:
    console.rule("[bold]Encompass RAG vector store build[/bold]")
    console.print(f"  jsonl   : {JSONL_PATH}")
    console.print(f"  postman : {POSTMAN_PATH}")
    console.print(f"  output  : {VECTOR_STORE_PATH.resolve()}")

    if not JSONL_PATH.exists():
        console.print(f"[red]Missing input: {JSONL_PATH}[/red]")
        return 1
    if not POSTMAN_PATH.exists():
        console.print(f"[red]Missing input: {POSTMAN_PATH}[/red]")
        return 1

    # 1) Load + filter + dedupe
    records = _load_jsonl(JSONL_PATH)
    console.print(f"\n[cyan]Loaded {len(records)} records[/cyan]")

    records = _filter_records(records)
    console.print(f"  after filter (drop custompage): {len(records)}")

    kept, dropped = dedupe_by_content(records)
    console.print(f"  after dedupe by body_md hash:   {len(kept)} (dropped {len(dropped)} dup{'s' if len(dropped)!=1 else ''})")
    if dropped:
        for r in dropped[:5]:
            console.print(f"    [dim]dropped:[/dim] {r['url']}")
        if len(dropped) > 5:
            console.print(f"    [dim]... and {len(dropped) - 5} more[/dim]")

    # 2) Chunk
    chunks = chunk_records(kept)
    console.print(f"\n[cyan]Chunked into {len(chunks)} chunks[/cyan]")
    if not chunks:
        console.print("[red]No chunks produced; aborting.[/red]")
        return 1
    documents = _chunks_to_documents(chunks)

    # Quick chunk-size sanity
    sizes = sorted(len(c.page_content) for c in chunks)
    console.print(f"  chunk char sizes: min={sizes[0]}  median={sizes[len(sizes)//2]}  "
                  f"p95={sizes[int(len(sizes)*0.95)]}  max={sizes[-1]}")

    # 3) Embed (with on-disk checkpoint) + FAISS index
    embedder = JinaEmbedder.from_env()
    console.print(f"\n[cyan]Jina backend: {embedder.backend}  model: {embedder.model_name}  device: {embedder.device}[/cyan]")
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    checkpoint = VECTOR_STORE_PATH / "_jsonl_embeddings_checkpoint.npy"
    texts = [d.page_content for d in documents]
    vectors = _embed_with_checkpoint(texts, embedder, checkpoint)
    vector_store = _build_jsonl_indices(documents, vectors, embedder)

    # 4) JSONL BM25
    console.print("\n[yellow]Building JSONL BM25...[/yellow]")
    jsonl_bm25, _ = _build_jsonl_bm25(documents)

    # 5) Postman
    console.print("\n[yellow]Loading + indexing Postman collection...[/yellow]")
    postman_entries = load_postman_entries(POSTMAN_PATH)
    postman_bm25 = _build_postman_bm25(postman_entries)
    console.print(f"  postman entries: {len(postman_entries)}")

    # 6) Save
    console.print(f"\n[yellow]Writing artifacts to {VECTOR_STORE_PATH}/...[/yellow]")
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    faiss_dir = VECTOR_STORE_PATH / "jsonl_faiss"
    vector_store.save_local(str(faiss_dir))

    with (VECTOR_STORE_PATH / "jsonl_bm25.pkl").open("wb") as f:
        pickle.dump(jsonl_bm25, f)
    with (VECTOR_STORE_PATH / "jsonl_chunks.pkl").open("wb") as f:
        # Persist chunk metadata so the retriever can map BM25 hits back to docs
        # without reloading the FAISS docstore.
        pickle.dump(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in documents],
            f,
        )
    with (VECTOR_STORE_PATH / "postman_bm25.pkl").open("wb") as f:
        pickle.dump(postman_bm25, f)
    with (VECTOR_STORE_PATH / "postman_entries.pkl").open("wb") as f:
        pickle.dump(postman_entries, f)

    stats = {
        "records_loaded": len(records),
        "records_after_dedupe": len(kept),
        "records_dropped_dupe": len(dropped),
        "chunks": len(chunks),
        "chunk_size_min": sizes[0],
        "chunk_size_median": sizes[len(sizes) // 2],
        "chunk_size_max": sizes[-1],
        "postman_entries": len(postman_entries),
        "embedder_backend": embedder.backend,
        "embedder_model": embedder.model_name,
        "embedder_device": embedder.device,
    }
    with (VECTOR_STORE_PATH / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    console.rule("[bold green]Build complete[/bold green]")
    for k, v in stats.items():
        console.print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
