"""APIDocumentationRAG — orchestrates retrieval + LLM for one query.

Pipeline:
    HybridRetriever.retrieve(query) -> chunks + endpoints
    _format_prompt(query, chunks, endpoints) -> single string
    self.llm.invoke(prompt) -> answer string

The previous implementation wrapped a HybridRetriever in a `CustomRetriever`
and ran it through LangChain's `RetrievalQA`. RetrievalQA stuffs all retrieved
documents into one `{context}` blob — that's a poor fit now that we have two
distinct prompt sections (`## Documentation` and `## Relevant API endpoint(s)`)
and the endpoints slot is conditional. We build the prompt directly here.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from rag_docs.retriever import ChunkHit, EndpointHit, HybridRetriever, RetrievalResult
from rag_docs.utils.logger import logger


DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"

SYSTEM_PROMPT = """You are an assistant for the Encompass Developer Connect API. \
Answer the user's question using only the documentation excerpts and (when present) \
the API endpoint metadata below. If the answer isn't in the provided context, say so \
plainly — do not invent fields, parameters, or endpoints.

When you reference specific information, cite it as [Source N] using the numbered \
sources below. When the user is asking how to call an API, point them at the \
endpoints listed under "Relevant API endpoint(s)" if any are present.
"""


class APIDocumentationRAG:
    def __init__(
        self,
        vector_store_path: str = "vector_store",
        model_name: Optional[str] = None,
        use_gemini: bool = False,
        gemini_api_key: Optional[str] = None,
    ):
        self.vector_store_path = vector_store_path
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.use_gemini = use_gemini
        self.gemini_api_key = gemini_api_key

        self.retriever: HybridRetriever | None = None
        self.llm = None

        self._load()

    # ----- Setup --------------------------------------------------------------

    def _load(self) -> None:
        try:
            self.retriever = HybridRetriever(self.vector_store_path)
            logger.info(f"Vector store loaded from {self.vector_store_path}")
            self._setup_llm()
        except Exception as e:
            logger.error(f"Error loading RAG system: {e}")
            raise

    def _setup_llm(self) -> None:
        # Lazy imports — users who only use one provider don't pay the cost
        # of installing the other's langchain integration package.
        temperature = float(os.getenv("TEMPERATURE", "0.1"))
        if self.use_gemini and self.gemini_api_key:
            from langchain_google_genai import GoogleGenerativeAI
            self.llm = GoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.gemini_api_key,
                temperature=temperature,
            )
            logger.info("LLM provider: Gemini")
        else:
            from langchain_ollama import OllamaLLM
            self.llm = OllamaLLM(model=self.model_name, temperature=temperature)
            logger.info(f"LLM provider: Ollama ({self.model_name})")

    # ----- Prompt formatting --------------------------------------------------

    @staticmethod
    def _format_chunk(idx: int, c: ChunkHit) -> str:
        title = c.metadata.get("title", "") or "(untitled)"
        breadcrumb = " > ".join(b for b in (c.metadata.get("breadcrumb") or []) if b) or "Documentation"
        url = c.metadata.get("url", "")
        return (
            f"[Source {idx}] Title: {title}  •  Section: {breadcrumb}\n"
            f"URL: {url}\n\n"
            f"{c.page_content.strip()}"
        )

    @staticmethod
    def _format_endpoint(e: EndpointHit) -> str:
        ent = e.entry
        folder = " > ".join(ent.folder_path) if ent.folder_path else ""
        head = f"- {ent.method} {ent.url} — {ent.name}"
        body_lines: List[str] = []
        if folder:
            body_lines.append(f"  Folder: {folder}")
        if ent.description:
            # Keep description tight in the prompt; full text stays in the response model.
            body_lines.append(f"  {ent.description.strip().splitlines()[0][:200]}")
        return "\n".join([head, *body_lines])

    @classmethod
    def _format_prompt(cls, query: str, result: RetrievalResult) -> str:
        sections: List[str] = [SYSTEM_PROMPT, ""]

        sections.append("## Documentation")
        if not result.chunks:
            sections.append("(no documentation matched this query)")
        else:
            for i, c in enumerate(result.chunks, start=1):
                sections.append(cls._format_chunk(i, c))
                sections.append("---")
            # Drop the trailing separator
            if sections[-1] == "---":
                sections.pop()

        if result.endpoints:
            sections.append("")
            sections.append("## Relevant API endpoint(s)")
            for e in result.endpoints:
                sections.append(cls._format_endpoint(e))

        sections.append("")
        sections.append("## Question")
        sections.append(query.strip())
        sections.append("")
        sections.append("## Answer")
        return "\n".join(sections)

    # ----- Query --------------------------------------------------------------

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        if not self.retriever or not self.llm:
            raise RuntimeError("RAG system not initialized")

        logger.info(f"Query: {question}")
        result = self.retriever.retrieve(question, k=k)
        prompt = self._format_prompt(question, result)
        answer = self.llm.invoke(prompt)
        # OllamaLLM returns str; GoogleGenerativeAI returns str too — keep it strict.
        if not isinstance(answer, str):
            answer = str(answer)

        return {
            "answer": answer,
            "sources": [self._chunk_to_source(c) for c in result.chunks],
            "relevant_endpoints": [self._endpoint_to_dict(e) for e in result.endpoints],
        }

    # ----- Serialization for the API layer -----------------------------------

    @staticmethod
    def _chunk_to_source(c: ChunkHit) -> Dict[str, Any]:
        m = c.metadata
        full = c.page_content
        return {
            "title": m.get("title", ""),
            "url": m.get("url", ""),
            "kind": m.get("kind", ""),
            "breadcrumb": list(m.get("breadcrumb") or []),
            "preview": full[:500],
            "full_content": full,
            "score": float(c.score),
        }

    @staticmethod
    def _endpoint_to_dict(e: EndpointHit) -> Dict[str, Any]:
        ent = e.entry
        return {
            "name": ent.name,
            "method": ent.method,
            "path": ent.url,
            "description": (ent.description or "").strip(),
            "folder_path": list(ent.folder_path or []),
        }
