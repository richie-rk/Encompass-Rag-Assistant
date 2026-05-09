from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from rag_docs.rag_class import APIDocumentationRAG
from rag_docs.utils.logger import logger

load_dotenv()


# --- Request / response models -----------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1,
                       description="The question to ask about the API documentation")


class Source(BaseModel):
    """A JSONL chunk surfaced by the hybrid retriever."""
    title: str = Field(default="", description="Page title")
    url: str = Field(default="", description="Canonical URL for the source page")
    kind: str = Field(default="", description="guide | reference | changelog")
    breadcrumb: List[str] = Field(default_factory=list,
                                  description="Sidebar breadcrumb segments")
    preview: str = Field(default="", description="First 500 chars of the chunk")
    full_content: str = Field(default="", description="Complete chunk text")
    score: float = Field(default=0.0,
                         description="Reciprocal Rank Fusion score (higher is better)")


class Endpoint(BaseModel):
    """A Postman entry surfaced by the gated retrieval flow."""
    name: str = Field(default="", description="Postman request name")
    method: str = Field(default="", description="HTTP method")
    path: str = Field(default="", description="URL or path template")
    description: str = Field(default="", description="One-line description")
    folder_path: List[str] = Field(default_factory=list,
                                   description="Postman folder path for context")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="LLM answer text")
    sources: List[Source] = Field(default_factory=list,
                                  description="JSONL chunks used to answer the query")
    relevant_endpoints: List[Endpoint] = Field(default_factory=list,
                                               description="Postman endpoints surfaced by the gate (0-3)")


# --- Lifespan / app ----------------------------------------------------------

def _resolve_vector_store_path() -> str:
    """Resolve the directory holding the FAISS + BM25 + chunk artifacts.

    Order:
        1. ``VECTOR_STORE_PATH`` env var — must point at an existing directory.
        2. ``./vector_store/`` if it exists (local build wins over remote).
        3. ``snapshot_download`` from HF — lands in the HF cache, returns that path.
    """
    explicit = os.getenv("VECTOR_STORE_PATH")
    if explicit:
        if not os.path.isdir(explicit):
            raise RuntimeError(
                f"VECTOR_STORE_PATH={explicit!r} is set but is not an existing directory."
            )
        return explicit

    if os.path.isdir("vector_store"):
        return "vector_store"

    repo_id = os.getenv("VECTOR_STORE_HF_REPO", "Richie-rk/encompass-developer-connect-index")
    revision = os.getenv("VECTOR_STORE_HF_REVISION", "main")
    logger.info(f"No local vector store found; fetching {repo_id}@{revision} from HF")
    try:
        return snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision)
    except Exception as e:
        raise RuntimeError(
            f"Vector store not available: no local copy and HF fetch failed ({e}). "
            "Either: (1) build locally with `python -m scripts.create_vector_store`, "
            "(2) ensure network access and HF auth (set HF_TOKEN if the dataset is private), "
            "or (3) set VECTOR_STORE_PATH to an existing directory."
        ) from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG system on startup."""
    vector_store_path = _resolve_vector_store_path()

    use_gemini = os.getenv("USE_GEMINI", "false").lower() == "true"
    gemini_api_key = os.getenv("GEMINI_API_KEY", None)

    logger.info(f"Loading vector store from: {vector_store_path}")
    try:
        app.state.rag_system = APIDocumentationRAG(
            vector_store_path=vector_store_path,
            use_gemini=use_gemini,
            gemini_api_key=gemini_api_key,
        )
        provider = "Gemini" if use_gemini else f"Ollama ({app.state.rag_system.model_name})"
        logger.info(f"RAG system initialized. LLM: {provider}")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

    yield

    logger.info("Shutting down.")


app = FastAPI(
    title="Encompass API Documentation Assistant",
    description="Hybrid RAG over Encompass Developer Connect docs (FAISS-IP + dual BM25 + Postman gate)",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_rag_system() -> APIDocumentationRAG:
    if not hasattr(app.state, "rag_system"):
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    return app.state.rag_system


# --- Endpoints ---------------------------------------------------------------

@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def get_answer(
    request: QueryRequest,
    rag_system: APIDocumentationRAG = Depends(get_rag_system),
):
    """Answer a question over the Encompass API documentation.

    Returns the LLM answer plus the documentation chunks used and any API
    endpoints the Postman gate decided are relevant (0-3).
    """
    try:
        result = rag_system.query(request.query)
        return QueryResponse(
            answer=result["answer"],
            sources=[Source(**s) for s in result.get("sources", [])],
            relevant_endpoints=[Endpoint(**e) for e in result.get("relevant_endpoints", [])],
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")


@app.get("/api/health", tags=["Health"])
async def health_check():
    has_rag = hasattr(app.state, "rag_system") and app.state.rag_system is not None
    return {
        "status": "healthy" if has_rag else "degraded",
        "message": "API Documentation Assistant is running",
        "rag_system_loaded": has_rag,
    }


@app.get("/api/config", tags=["Configuration"])
async def get_config():
    rag = app.state.rag_system  # lifespan raises on failure, so this is set
    return {
        "vector_store_path": rag.vector_store_path,
        "model_type": "Gemini" if rag.use_gemini else "Ollama",
        "ollama_model": rag.model_name,
        "temperature": float(os.getenv("TEMPERATURE", "0.1")),
        "embedder_backend": os.getenv("JINA_BACKEND", "local"),
        "embedder_model": os.getenv("JINA_MODEL", "jinaai/jina-embeddings-v3"),
    }


if __name__ == "__main__":
    # Use the full module path so this works when launched as
    # `python -m rag_docs.rag_app` from any directory (uvicorn imports the
    # ASGI app by string, and bare "rag_app" only resolves if the cwd is the
    # `rag_docs/src/rag_docs/` dir).
    uvicorn.run(
        "rag_docs.rag_app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
