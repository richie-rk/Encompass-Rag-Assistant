"""Three-index hybrid retriever.

Inputs (built by `scripts/create_vector_store.py`):
    vector_store/jsonl_faiss/index.faiss + index.pkl   FAISS-IP, Jina v3 vectors
    vector_store/jsonl_bm25.pkl                         BM25 over JSONL chunks
    vector_store/jsonl_chunks.pkl                       chunk texts + metadata
    vector_store/postman_bm25.pkl                       BM25 over Postman entries
    vector_store/postman_entries.pkl                    PostmanEntry objects

Retrieval flow per query:
    1. Embed with Jina v3 task=retrieval.query.
    2. FAISS-IP top-N + JSONL BM25 top-N → RRF fuse → top-K chunks.
    3. Postman BM25 top-5 → token-overlap filter (>=0.3) → score-ratio split:
       - top1/top2 >= 2.0 → emit only top-1
       - else → emit up to top-3
       Gate output is a separate slot, not merged into the chunk list.

Returns a `RetrievalResult` carrying both buckets so the prompt builder can
keep the `## Documentation` and `## Relevant API endpoint(s)` sections clean.
"""
from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from langchain_community.vectorstores import FAISS

# `scripts.ingest` is in the repo root's `scripts/` package. Make sure the repo
# root is on sys.path so we can import the embedder + tokenizer from there
# regardless of where the API server is launched from.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ingest.jina_embedder import JinaEmbedder  # noqa: E402
from scripts.ingest.postman import PostmanEntry, tokenize_query  # noqa: E402


# --- Tunables (env-overridable) -----------------------------------------------

DEFAULT_K = int(os.getenv("RETRIEVAL_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
RRF_INPUT_DEPTH = int(os.getenv("RRF_INPUT_DEPTH", "20"))  # top-N per ranker before fuse

POSTMAN_TOP_N = int(os.getenv("POSTMAN_TOP_N", "5"))
POSTMAN_OVERLAP_MIN = float(os.getenv("POSTMAN_OVERLAP_MIN", "0.3"))
POSTMAN_RATIO_MIN = float(os.getenv("POSTMAN_RATIO_MIN", "2.0"))
POSTMAN_MAX_CLUSTERED = int(os.getenv("POSTMAN_MAX_CLUSTERED", "3"))


# --- Result types -------------------------------------------------------------

@dataclass
class ChunkHit:
    """One JSONL chunk surfaced by the hybrid retriever."""
    page_content: str
    metadata: Dict[str, Any]
    score: float                          # RRF score
    sources: List[str] = field(default_factory=list)  # which rankers hit it


@dataclass
class EndpointHit:
    """One Postman entry surfaced past the gate."""
    entry: PostmanEntry
    score: float                          # raw BM25
    overlap: float                        # query/endpoint token overlap


@dataclass
class RetrievalResult:
    chunks: List[ChunkHit]
    endpoints: List[EndpointHit]


# --- Retriever ---------------------------------------------------------------

class HybridRetriever:
    """Loads three indices and runs RRF on the JSONL pair, gated BM25 on Postman."""

    def __init__(self, vector_store_path: str = "vector_store"):
        self.vector_store_path = Path(vector_store_path)
        self._embedder: JinaEmbedder | None = None
        self._faiss: FAISS | None = None
        self._jsonl_bm25 = None
        self._jsonl_chunks: List[Dict] = []
        self._postman_bm25 = None
        self._postman_entries: List[PostmanEntry] = []
        self.load_indices()

    # ----- Loading ------------------------------------------------------------

    def load_indices(self) -> None:
        """Load all four artifacts and the embedder."""
        self._embedder = JinaEmbedder.from_env()

        self._faiss = FAISS.load_local(
            str(self.vector_store_path / "jsonl_faiss"),
            self._embedder.as_langchain(),
            allow_dangerous_deserialization=True,
        )

        with (self.vector_store_path / "jsonl_bm25.pkl").open("rb") as f:
            self._jsonl_bm25 = pickle.load(f)
        with (self.vector_store_path / "jsonl_chunks.pkl").open("rb") as f:
            self._jsonl_chunks = pickle.load(f)
        with (self.vector_store_path / "postman_bm25.pkl").open("rb") as f:
            self._postman_bm25 = pickle.load(f)
        with (self.vector_store_path / "postman_entries.pkl").open("rb") as f:
            self._postman_entries = pickle.load(f)

        # Build a URL-based identity for chunks so FAISS hits and BM25 hits dedupe
        # cleanly. We use (url, chunk_index) — duplicates are caught at ingest by
        # the body_hash dedupe, so within the chunk list each (url, chunk_index)
        # is unique.
        self._chunk_id_by_idx: Dict[int, Tuple[str, int]] = {
            i: (c["metadata"].get("url", ""), c["metadata"].get("chunk_index", i))
            for i, c in enumerate(self._jsonl_chunks)
        }

    # ----- Public API ---------------------------------------------------------

    def retrieve(self, query: str, k: int = DEFAULT_K) -> RetrievalResult:
        chunks = self._retrieve_chunks(query, k)
        endpoints = self._retrieve_endpoints(query)
        return RetrievalResult(chunks=chunks, endpoints=endpoints)

    # ----- JSONL hybrid (RRF) -------------------------------------------------

    def _retrieve_chunks(self, query: str, k: int) -> List[ChunkHit]:
        # 1) Semantic — Jina v3 retrieval.query, FAISS-IP top-N
        qvec = self._embedder.embed_query(query)
        # similarity_search_with_score_by_vector returns (doc, score) sorted by
        # decreasing similarity for IP/cosine spaces.
        sem_hits: List[Tuple[Any, float]] = self._faiss.similarity_search_with_score_by_vector(
            qvec.tolist(), k=RRF_INPUT_DEPTH,
        )

        # Build url+chunk_index → rank for each ranker.
        sem_ranked: List[Tuple[Tuple[str, int], Any, float]] = []
        for rank, (doc, score) in enumerate(sem_hits, start=1):
            cid = (doc.metadata.get("url", ""), doc.metadata.get("chunk_index", 0))
            sem_ranked.append((cid, doc, float(score)))

        # 2) Keyword — BM25 over JSONL chunks
        toks = tokenize_query(query)
        bm25_scores = self._jsonl_bm25.get_scores(toks) if toks else np.zeros(len(self._jsonl_chunks))
        # Top-N indices by BM25 score
        top_n = min(RRF_INPUT_DEPTH, len(bm25_scores))
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_n]
        bm25_ranked: List[Tuple[Tuple[str, int], Dict, float]] = []
        for rank, idx in enumerate(bm25_top_idx, start=1):
            if bm25_scores[idx] <= 0:
                continue  # nothing to add — query had no overlap with this chunk
            cid = self._chunk_id_by_idx[idx]
            bm25_ranked.append((cid, self._jsonl_chunks[idx], float(bm25_scores[idx])))

        # 3) RRF fuse
        return self._rrf_fuse(sem_ranked, bm25_ranked, k=k)

    @staticmethod
    def _rrf_fuse(
        sem_ranked: List[Tuple[Tuple[str, int], Any, float]],
        bm25_ranked: List[Tuple[Tuple[str, int], Dict, float]],
        *, k: int,
    ) -> List[ChunkHit]:
        """Reciprocal Rank Fusion — score(d) = Σ 1 / (RRF_K + rank_r(d))."""
        scores: Dict[Tuple[str, int], float] = {}
        items: Dict[Tuple[str, int], Any] = {}
        sources: Dict[Tuple[str, int], List[str]] = {}

        for rank, (cid, doc, _) in enumerate(sem_ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            items.setdefault(cid, doc)
            sources.setdefault(cid, []).append("semantic")

        for rank, (cid, payload, _) in enumerate(bm25_ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            # Prefer FAISS Document objects over the chunks-pickle dict if both
            # rankers found this chunk (Document carries cleaner metadata typing).
            items.setdefault(cid, payload)
            sources.setdefault(cid, []).append("bm25")

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out: List[ChunkHit] = []
        for cid, rrf_score in ordered[:k]:
            payload = items[cid]
            # Both shapes (LangChain Document or chunks-pickle dict) expose
            # `page_content` and `metadata`. Normalize to dict access.
            page_content = getattr(payload, "page_content", None) or payload["page_content"]
            metadata = getattr(payload, "metadata", None) or payload["metadata"]
            out.append(ChunkHit(
                page_content=page_content,
                metadata=dict(metadata),
                score=rrf_score,
                sources=sources[cid],
            ))
        return out

    # ----- Postman gated retrieval --------------------------------------------

    def _retrieve_endpoints(self, query: str) -> List[EndpointHit]:
        toks = tokenize_query(query)
        if not toks:
            return []

        scores = self._postman_bm25.get_scores(toks)
        top_idx = np.argsort(scores)[::-1][:POSTMAN_TOP_N]

        # Filter by token overlap. Overlap = |query_tokens ∩ entry_tokens| / |query_tokens|.
        q_set = set(toks)
        candidates: List[Tuple[int, float, float]] = []  # (idx, bm25, overlap)
        for idx in top_idx:
            entry = self._postman_entries[idx]
            entry_set = set(entry.tokens)
            if not entry_set:
                continue
            overlap = len(q_set & entry_set) / len(q_set)
            if overlap < POSTMAN_OVERLAP_MIN:
                continue
            if scores[idx] <= 0:
                continue
            candidates.append((int(idx), float(scores[idx]), overlap))

        if not candidates:
            return []

        # Sort by BM25 score descending so ratio is computed on the right pair.
        candidates.sort(key=lambda c: c[1], reverse=True)

        if len(candidates) == 1:
            picks = candidates[:1]
        else:
            top1 = candidates[0][1]
            top2 = candidates[1][1] or 1e-9  # avoid /0 on degenerate cases
            if top1 / top2 >= POSTMAN_RATIO_MIN:
                picks = candidates[:1]            # one dominates
            else:
                picks = candidates[:POSTMAN_MAX_CLUSTERED]  # clustered → up to N

        return [EndpointHit(entry=self._postman_entries[i], score=s, overlap=o)
                for (i, s, o) in picks]
