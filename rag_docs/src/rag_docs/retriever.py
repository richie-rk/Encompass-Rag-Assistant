import pickle
import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """Hybrid retriever using both FAISS and BM25"""

    def __init__(self, vector_store_path: str = "vector_store"):
        self.vector_store_path = Path(vector_store_path)
        self.load_indices()

    def load_indices(self):
        """Load FAISS, BM25, and metadata"""

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Load FAISS
        self.faiss_store = FAISS.load_local(
            str(self.vector_store_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Load BM25
        with open(self.vector_store_path / "bm25_index.pkl", 'rb') as f:
            self.bm25 = pickle.load(f)

        # Load metadata
        with open(self.vector_store_path / "documents_metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.6) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword search

        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for semantic search (0-1). Higher = more semantic
        """
        # Get semantic search results with scores
        semantic_docs = self.faiss_store.similarity_search_with_score(query, k=k * 2)

        # Get BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores
        semantic_scores = self._normalize_semantic_scores([score for _, score in semantic_docs])
        bm25_scores_norm = self._normalize_bm25_scores(bm25_scores)

        # Combine scores
        combined_scores = {}

        # Add semantic scores
        for i, (doc, score) in enumerate(semantic_docs):
            doc_id = id(doc)
            combined_scores[doc_id] = {
                'doc': doc,
                'semantic_score': semantic_scores[i] * alpha,
                'bm25_score': 0,
                'total_score': semantic_scores[i] * alpha
            }

        # Add BM25 scores
        for i, score in enumerate(bm25_scores_norm):
            if score > 0:  # Only consider non-zero scores
                # Match with FAISS documents
                doc = self.faiss_store.docstore._dict.get(str(i))
                if doc:
                    doc_id = id(doc)
                    if doc_id in combined_scores:
                        combined_scores[doc_id]['bm25_score'] = score * (1 - alpha)
                        combined_scores[doc_id]['total_score'] += score * (1 - alpha)
                    else:
                        combined_scores[doc_id] = {
                            'doc': doc,
                            'semantic_score': 0,
                            'bm25_score': score * (1 - alpha),
                            'total_score': score * (1 - alpha)
                        }

        # Sort by combined score
        ranked_results = sorted(
            combined_scores.values(),
            key=lambda x: x['total_score'],
            reverse=True
        )

        return [result['doc'] for result in ranked_results[:k]]

    def _normalize_semantic_scores(self, scores: List[float]) -> List[float]:
        """Normalize semantic similarity scores to 0-1"""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(max_score - s) / (max_score - min_score) for s in scores]

    def _normalize_bm25_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize BM25 scores to 0-1"""
        if scores.max() == 0:
            return scores
        return scores / scores.max()