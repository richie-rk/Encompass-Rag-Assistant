from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import uvicorn
import os

from rag_docs.rag_class import APIDocumentationRAG
from importlib.resources import files
from rag_docs.utils.logger import logger


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question to ask about the API documentation")


class SourceDocument(BaseModel):
    content: str = Field(..., description="Content of the source document")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The answer to the query")
    sources: List[SourceDocument] = Field(..., description="Source documents used to generate the answer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    csv_path = str(files("rag_docs.data").joinpath("documentation_data.csv"))
    postman_path = str(files("rag_docs.data").joinpath("Encompass_Developer_Connect_postman_collection.json"))
    vector_store_path = "vector_store"

    # Create global RAG system
    app.state.rag_system = APIDocumentationRAG(csv_path, postman_path)

    # Initialize/load the RAG system
    if os.path.exists(vector_store_path):
        logger.info(f"Loading existing vector store from {vector_store_path}...")
        success = app.state.rag_system.load_vector_store(vector_store_path)
        if not success:
            logger.info("Failed to load vector store. Initializing from scratch...")
            app.state.rag_system.initialize()
            app.state.rag_system.save_vector_store(vector_store_path)
    else:
        logger.info("No existing vector store found. Initializing from scratch...")
        app.state.rag_system.initialize()
        app.state.rag_system.save_vector_store(vector_store_path)

    logger.info("RAG system initialized and ready to use!")

    yield

    logger.info("Shutting down the application...")


app = FastAPI(
    title="API Documentation Assistant",
    description="A Retrieval-Augmented Generation (RAG) system for querying API documentation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Dependency to get the RAG system
def get_rag_system():
    return app.state.rag_system


@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def get_answer(request: QueryRequest, rag_system: APIDocumentationRAG = Depends(get_rag_system)):
    """
    Get an answer to a question about the API documentation.

    This endpoint uses a Retrieval-Augmented Generation (RAG) system to answer
    questions about the API documentation. It retrieves relevant information from
    the documentation and generates a natural language answer.
    """
    try:
        result = rag_system.query(request.query)

        answer = result["result"]
        sources = [
            SourceDocument(content=doc.page_content)
            for doc in result["source_documents"]
        ]

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:

        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/health", tags=["Health"])
async def health_check():
    """
    Check if the API is running.

    Returns a simple message indicating that the API is running.
    """
    return {"status": "healthy", "message": "API Documentation Assistant is running"}


if __name__ == "__main__":
    uvicorn.run("rag_app:app", host="0.0.0.0", port=8000, reload=True)
