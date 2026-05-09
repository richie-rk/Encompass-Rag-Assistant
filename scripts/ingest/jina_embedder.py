"""Jina v3 embedder with two backends.

Local (`sentence-transformers`) — works offline, downloads the model on first run.
API (`https://api.jina.ai/v1/embeddings`) — no model download, needs JINA_API_KEY.

Both produce 1024-dim, L2-normalized vectors. Both expose asymmetric retrieval
via Jina v3's task adapters: `retrieval.passage` for documents at ingest time,
`retrieval.query` for queries at retrieval time. The two adapters project Q and
D into matched-but-different regions of the same vector space.

Usage:
    embedder = JinaEmbedder.from_env()
    passage_vectors = embedder.embed_passages(["doc 1", "doc 2"])
    query_vector = embedder.embed_query("how do I refresh a token")

Also exposes a LangChain-compatible `Embeddings` adapter via `as_langchain()`
so the `FAISS.from_documents(...)` call site doesn't need to know which backend
is in use.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np


# --- Defaults from env --------------------------------------------------------

DEFAULT_MODEL = "jinaai/jina-embeddings-v3"
DEFAULT_DEVICE = "cpu"
DEFAULT_BACKEND = "local"
JINA_API_URL = "https://api.jina.ai/v1/embeddings"

PASSAGE_TASK = "retrieval.passage"
QUERY_TASK = "retrieval.query"


class JinaEmbedder:
    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        api_key: Optional[str] = None,
    ):
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.api_key = api_key
        self._local_model = None  # lazy init — avoid pulling torch on api-only paths

        if backend == "api" and not api_key:
            raise RuntimeError(
                "JINA_BACKEND=api requires JINA_API_KEY. Set it in .env or "
                "switch to JINA_BACKEND=local."
            )
        if backend not in ("local", "api"):
            raise ValueError(f"Unknown JINA_BACKEND={backend!r}; expected 'local' or 'api'.")

    @classmethod
    def from_env(cls) -> "JinaEmbedder":
        return cls(
            backend=os.getenv("JINA_BACKEND", DEFAULT_BACKEND).lower(),
            model_name=os.getenv("JINA_MODEL", DEFAULT_MODEL),
            device=os.getenv("JINA_DEVICE", DEFAULT_DEVICE),
            api_key=os.getenv("JINA_API_KEY") or None,
        )

    # --- Public API -----------------------------------------------------------

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts, task=PASSAGE_TASK)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._encode([text], task=QUERY_TASK)
        return vec[0]

    def as_langchain(self):
        """Return a LangChain `Embeddings`-conforming adapter.

        Built lazily so the import doesn't fail in environments without
        langchain-core installed.
        """
        from langchain_core.embeddings import Embeddings

        embedder = self

        class _LangChainEmbeddings(Embeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return embedder.embed_passages(texts).tolist()

            def embed_query(self, text: str) -> List[float]:
                return embedder.embed_query(text).tolist()

        return _LangChainEmbeddings()

    # --- Backends -------------------------------------------------------------

    def _encode(self, texts: List[str], *, task: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        if self.backend == "local":
            return self._encode_local(texts, task=task)
        return self._encode_api(texts, task=task)

    def _encode_local(self, texts: List[str], *, task: str) -> np.ndarray:
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
        # Small batch_size — Jina v3 on Windows CPU has been segfaulting at the
        # default batch_size=32. 8 is conservative; raise if you have headroom.
        # `task=` swaps in the right LoRA adapter; `normalize_embeddings=True`
        # gives unit vectors so FAISS-IP returns cosine similarity directly.
        embs = self._local_model.encode(
            texts,
            task=task,
            batch_size=8,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 32,
        )
        return embs.astype(np.float32)

    def _encode_api(self, texts: List[str], *, task: str) -> np.ndarray:
        # Jina free tier: 500 RPM but only ~1.6M TPM. With ~400 tokens/chunk a
        # batch of 32 is ~12K tokens/request — keeps us under TPM at ~120 req/min.
        # Larger batches trigger 429s mid-build.
        import requests
        import time
        import random

        BATCH = 32
        MAX_RETRIES = 6
        all_vecs: List[List[float]] = []
        body_template = {
            "model": self.model_name.split("/")[-1],
            "task": task,
            "embedding_type": "float",
            "normalized": True,
        }
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            body = {**body_template, "input": chunk}
            backoff = 1.0
            for attempt in range(MAX_RETRIES):
                resp = requests.post(
                    JINA_API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=120,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt == MAX_RETRIES - 1:
                        resp.raise_for_status()
                    # Honour Retry-After when present; otherwise exponential.
                    ra = resp.headers.get("Retry-After")
                    sleep = float(ra) if ra and ra.isdigit() else backoff + random.random()
                    time.sleep(sleep)
                    backoff = min(backoff * 2, 30.0)
                    continue
                resp.raise_for_status()
                data = resp.json().get("data") or []
                for entry in data:
                    all_vecs.append(entry["embedding"])
                break
        return np.asarray(all_vecs, dtype=np.float32)
