import pandas as pd
import json
import os

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from rag_docs.utils.logger import logger


class APIDocumentationRAG:
    def __init__(self, csv_path, postman_path, model_name="qwen2.5-coder:7b"):
        self.csv_path = csv_path
        self.postman_path = postman_path
        self.model_name = model_name
        self.vector_store = None
        self.llm = None
        self.qa_chain = None

    def setup_ollama_model(self):
        """Set up the Ollama LLM with the specified model"""

        llm = OllamaLLM(model=self.model_name,
                        temperature=0)
        print(f"Ollama model '{self.model_name}' initialized")
        return llm

    def create_rag_pipeline(self):
        """Create the RAG pipeline"""
        prompt_template = """
You are an API documentation expert specializing in Encompass API. Your task is to analyze the provided context and question to deliver accurate, precise information.

Context Analysis Guidelines:
1. Carefully review all provided documentation sections
2. Identify relevant API endpoints, parameters, and return values
3. Note any prerequisites or dependencies
4. Pay attention to version-specific information

Available Context:
{context}

Query Analysis:
1. Question Type: [API endpoint | implementation | troubleshooting | conceptual]
2. Required Information: [list specific details needed]
3. Context Relevance: [rate how relevant the context is]

Question:
{question}

Response Format:
1. Direct answer first
2. Supporting evidence from context
3. Code examples if relevant
4. Important caveats or limitations

Answer:
"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        print("RAG pipeline created successfully")
        return qa_chain

    def load_vector_store(self, path="vector_store"):
        if not os.path.exists(path):
            print(f"Vector store path {path} does not exist.")
            return False

        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vector_store = FAISS.load_local(
                path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Vector store loaded from {path}")

            if self.llm is None:
                self.llm = self.setup_ollama_model()

            if self.qa_chain is None:
                self.qa_chain = self.create_rag_pipeline()

            return True
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False

    def query(self, question):
        """Query the RAG system"""
        if self.qa_chain is None:
            print("RAG system not initialized. Please run initialize() first.")
            return None

        logger.info(f"Question: {question}")
        logger.info("-" * 50)

        result = self.qa_chain({"query": question})

        return result
