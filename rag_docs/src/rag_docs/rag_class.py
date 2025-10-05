import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_google_genai import GoogleGenerativeAI
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from rag_docs.retriever import HybridRetriever
from rag_docs.utils.logger import logger


class CustomRetriever(BaseRetriever):
    """Custom retriever that wraps our HybridRetriever to be LangChain compatible"""

    hybrid_retriever: HybridRetriever

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Synchronous retrieval method"""
        return self.hybrid_retriever.hybrid_search(query, k=5)

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Asynchronous retrieval method"""
        return self._get_relevant_documents(query, run_manager=run_manager)


class APIDocumentationRAG:
    def __init__(self, vector_store_path: str = "vector_store",
                 model_name: str = "deepseek-coder-v2:16b",
                 use_gemini: bool = False,
                 gemini_api_key: Optional[str] = None):

        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.use_gemini = use_gemini
        self.gemini_api_key = gemini_api_key

        # Initialize components
        self.retriever = None
        self.llm = None
        self.qa_chain = None

        # Load vector store
        self.load_vector_store()

    def load_vector_store(self):
        """Load the vector store and setup retriever"""
        try:
            # Initialize hybrid retriever
            self.retriever = HybridRetriever(self.vector_store_path)
            logger.info(f"Vector store loaded from {self.vector_store_path}")

            # Setup LLM
            self.setup_llm()

            # Create QA chain
            self.create_rag_pipeline()

            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    def setup_llm(self):
        """Setup LLM based on configuration"""
        if self.use_gemini and self.gemini_api_key:
            self.llm = GoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.gemini_api_key,
                temperature=0.1
            )
            logger.info("Using Gemini API for LLM")
        else:
            self.llm = OllamaLLM(
                model=self.model_name,
                temperature=0.1
            )
            logger.info(f"Using Ollama model: {self.model_name}")

    def create_rag_pipeline(self):
        """Create the RAG pipeline with enhanced prompt"""
        prompt_template = """You are an expert on the Encompass API, helping developers with implementation and troubleshooting.

CONTEXT FROM DOCUMENTATION:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Provide a direct, accurate answer based on the documentation
2. If discussing an API endpoint, include:
   - HTTP method and path
   - Required parameters
   - Authentication requirements
   - Example request/response if available
3. For error troubleshooting, explain:
   - Common causes
   - How to fix the issue
   - What to check in the request
4. Include code examples when helpful
5. Clearly indicate if information comes from official docs, CSV documentation, or Postman examples

ANSWER:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create a proper LangChain retriever that wraps our hybrid retriever
        custom_retriever = CustomRetriever(hybrid_retriever=self.retriever)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=custom_retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        logger.info("RAG pipeline created successfully")

    def query(self, question: str) -> Dict[str, Any]:
        """Query with enhanced source attribution"""
        if not self.qa_chain:
            raise RuntimeError("RAG system not initialized")

        logger.info(f"Processing query: {question}")

        # Get answer and sources
        result = self.qa_chain.invoke({"query": question})

        # Process sources for better presentation
        processed_sources = self._process_sources(result.get("source_documents", []))

        return {
            "result": result["result"],
            "source_documents": result.get("source_documents", []),
            "context7_sources": processed_sources["context7"],
            "postman_sources": processed_sources["postman"],
            "csv_sources": processed_sources["csv_docs"]
        }

    def _process_sources(self, documents: List) -> Dict[str, List]:
        """Process and categorize source documents"""
        context7_sources = []
        postman_sources = []
        csv_sources = []

        for doc in documents:
            metadata = doc.metadata

            if metadata.get('source') == 'context7':
                context7_sources.append({
                    'content': doc.page_content[:500],  # Preview
                    'url': metadata.get('url', ''),
                    'type': metadata.get('type', ''),
                    'full_content': doc.page_content
                })
            elif metadata.get('source') == 'csv_docs':
                csv_sources.append({
                    'content': doc.page_content[:500],  # Preview
                    'url': metadata.get('url', ''),
                    'title': metadata.get('title', ''),
                    'type': metadata.get('type', ''),
                    'full_content': doc.page_content
                })
            elif metadata.get('source') == 'postman':
                # Parse the raw endpoint data if available
                raw_data = {}
                if metadata.get('raw_data'):
                    try:
                        raw_data = json.loads(metadata['raw_data'])
                    except:
                        pass

                postman_sources.append({
                    'content': doc.page_content[:500],
                    'endpoint_name': metadata.get('endpoint_name', ''),
                    'method': metadata.get('method', ''),
                    'path': metadata.get('path', ''),
                    'raw_data': raw_data,
                    'full_content': doc.page_content
                })

        return {
            'context7': context7_sources,
            'postman': postman_sources,
            'csv_docs': csv_sources
        }