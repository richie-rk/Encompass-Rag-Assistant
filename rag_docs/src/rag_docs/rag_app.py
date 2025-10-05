from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import os
from dotenv import load_dotenv

from rag_docs.rag_class import APIDocumentationRAG
from rag_docs.utils.logger import logger

# Load env variables
load_dotenv()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question to ask about the API documentation")


class Context7Source(BaseModel):
    content: str = Field(..., description="Preview of the source content")
    url: str = Field(default="", description="URL of the documentation")
    type: str = Field(default="", description="Type of documentation")
    full_content: str = Field(..., description="Full content of the source")


class PostmanSource(BaseModel):
    content: str = Field(..., description="Preview of the endpoint content")
    endpoint_name: str = Field(default="", description="Name of the endpoint")
    method: str = Field(default="", description="HTTP method")
    path: str = Field(default="", description="API path")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw endpoint data")
    full_content: str = Field(..., description="Full content of the source")


class CSVSource(BaseModel):
    content: str = Field(..., description="Preview of the source content")
    url: str = Field(default="", description="URL from the documentation")
    title: str = Field(default="", description="Title of the document")
    type: str = Field(default="", description="Type of documentation")
    full_content: str = Field(..., description="Full content of the source")


class EnhancedQueryResponse(BaseModel):
    answer: str = Field(..., description="The answer to the query")
    context7_sources: List[Context7Source] = Field(default_factory=list, description="Context7 documentation sources")
    postman_sources: List[PostmanSource] = Field(default_factory=list, description="Postman endpoint sources")
    csv_sources: List[CSVSource] = Field(default_factory=list, description="CSV documentation sources")
    total_sources_used: int = Field(default=0, description="Total number of sources used")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG system on startup"""

    # Check if vector store exists
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "vector_store")

    if not os.path.exists(vector_store_path):
        logger.error(f"Vector store not found at {vector_store_path}")
        logger.error("Please run 'python scripts/create_vector_store.py' first to create the vector store")
        raise RuntimeError(f"Vector store not found at {vector_store_path}")

    # Get configuration from environment
    model_name = os.getenv("OLLAMA_MODEL", "deepseek-coder-v2:16b")
    use_gemini = os.getenv("USE_GEMINI", "false").lower() == "true"
    gemini_api_key = os.getenv("GEMINI_API_KEY", None)

    # Log configuration
    logger.info(f"Loading vector store from: {vector_store_path}")
    logger.info(f"Model configuration: {'Gemini' if use_gemini else f'Ollama ({model_name})'}")

    try:
        # Initialize the RAG system with new parameters
        app.state.rag_system = APIDocumentationRAG(
            vector_store_path=vector_store_path,
            model_name=model_name,
            use_gemini=use_gemini,
            gemini_api_key=gemini_api_key
        )

        logger.info("RAG system initialized successfully!")

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        raise

    yield

    logger.info("Shutting down the application...")


app = FastAPI(
    title="Encompass API Documentation Assistant",
    description="A Retrieval-Augmented Generation (RAG) system for querying Encompass API documentation",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get the RAG system
def get_rag_system():
    if not hasattr(app.state, 'rag_system'):
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    return app.state.rag_system


@app.post("/api/query", response_model=EnhancedQueryResponse, tags=["Query"])
async def get_answer(request: QueryRequest, rag_system: APIDocumentationRAG = Depends(get_rag_system)):
    """
    Get an answer to a question about the Encompass API documentation.

    Returns the answer along with categorized source documents from Context7, Postman collections, and CSV docs.
    """
    try:
        # Query the RAG system
        result = rag_system.query(request.query)

        # Convert sources to response models
        context7_sources = [
            Context7Source(**source) for source in result.get("context7_sources", [])
        ]

        postman_sources = [
            PostmanSource(**source) for source in result.get("postman_sources", [])
        ]

        csv_sources = [
            CSVSource(**source) for source in result.get("csv_sources", [])
        ]

        return EnhancedQueryResponse(
            answer=result["result"],
            context7_sources=context7_sources,
            postman_sources=postman_sources,
            csv_sources=csv_sources,
            total_sources_used=len(context7_sources) + len(postman_sources) + len(csv_sources)
        )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/health", tags=["Health"])
async def health_check():
    """Check if the API is running and RAG system is loaded"""

    has_rag = hasattr(app.state, 'rag_system') and app.state.rag_system is not None

    return {
        "status": "healthy" if has_rag else "degraded",
        "message": "API Documentation Assistant is running",
        "rag_system_loaded": has_rag
    }


@app.get("/api/config", tags=["Configuration"])
async def get_config():
    """Get current configuration (non-sensitive)"""

    return {
        "vector_store_path": os.getenv("VECTOR_STORE_PATH", "vector_store"),
        "model_type": "Gemini" if os.getenv("USE_GEMINI", "false").lower() == "true" else "Ollama",
        "ollama_model": os.getenv("OLLAMA_MODEL", "deepseek-coder-v2:16b"),
        "temperature": float(os.getenv("TEMPERATURE", "0.1"))
    }


if __name__ == "__main__":
    uvicorn.run("rag_app:app", host="0.0.0.0", port=8000, reload=True)