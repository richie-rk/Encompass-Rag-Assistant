import os
import json
import pickle
import csv
import re
from pathlib import Path
from typing import List, Dict
import numpy as np
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.progress import track

console = Console()


class EncompassDocumentProcessor:
    """Process Context7, CSV documentation, and Postman data for vector store creation"""

    def __init__(self):
        self.api_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "],
            keep_separator=True
        )

        self.postman_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ", "],
            keep_separator=True
        )

    def process_csv_documentation(self, file_path: str) -> List[Document]:
        """Process CSV documentation file"""
        documents = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in track(reader, description="Processing CSV documentation..."):
                    url = row.get('url', '')
                    title = row.get('title', '')
                    content = row.get('content', '')

                    if not content or len(content.strip()) < 50:
                        continue

                    # Check if it's API endpoint content
                    if self._is_api_endpoint(content):
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': 'csv_docs',
                                'url': url,
                                'title': title,
                                'type': 'api_endpoint',
                                'chunk_size': len(content)
                            }
                        )
                        documents.append(doc)
                    else:
                        # Split longer content into chunks
                        chunks = self.api_splitter.split_text(content)
                        for i, chunk in enumerate(chunks):
                            doc = Document(
                                page_content=chunk,
                                metadata={
                                    'source': 'csv_docs',
                                    'url': url,
                                    'title': title,
                                    'type': 'documentation',
                                    'chunk_index': i,
                                    'total_chunks': len(chunks)
                                }
                            )
                            documents.append(doc)
        except Exception as e:
            console.print(f"[yellow]Warning: Error processing CSV file: {e}[/yellow]")

        return documents

    def process_context7_file(self, file_path: str) -> List[Document]:
        """Process Context7 documentation"""
        documents = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse Context7 format
            sections = content.split('\n\n---\n\n')

            for section in track(sections, description="Processing Context7 sections..."):
                url_match = re.search(r'URL:\s*(https?://[^\n]+)', section)
                url = url_match.group(1) if url_match else ""

                if self._is_api_endpoint(section):
                    doc = Document(
                        page_content=section,
                        metadata={
                            'source': 'context7',
                            'url': url,
                            'type': 'api_endpoint',
                            'chunk_size': len(section)
                        }
                    )
                    documents.append(doc)
                else:
                    chunks = self.api_splitter.split_text(section)
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                'source': 'context7',
                                'url': url,
                                'type': 'documentation',
                                'chunk_index': i,
                                'total_chunks': len(chunks)
                            }
                        )
                        documents.append(doc)
        except Exception as e:
            console.print(f"[yellow]Warning: Error processing Context7 file: {e}[/yellow]")

        return documents

    def process_postman_collection(self, file_path: str) -> List[Document]:
        """Process Postman collection"""
        documents = []

        try:
            with open(file_path, 'r') as f:
                collection = json.load(f)

            items = self._flatten_postman_items(collection)

            for item in track(items, description="Processing Postman endpoints..."):
                endpoint_text = self._format_endpoint_text(item)

                doc = Document(
                    page_content=endpoint_text,
                    metadata={
                        'source': 'postman',
                        'endpoint_name': item.get('name', ''),
                        'method': item.get('request', {}).get('method', ''),
                        'path': self._extract_path(item),
                        'type': 'endpoint',
                        'raw_data': json.dumps(item)
                    }
                )
                documents.append(doc)
        except Exception as e:
            console.print(f"[yellow]Warning: Error processing Postman file: {e}[/yellow]")

        return documents

    def _format_endpoint_text(self, item: Dict) -> str:
        """Format Postman endpoint as searchable text"""
        request = item.get('request', {})
        method = request.get('method', 'UNKNOWN')
        path = self._extract_path(item)
        description = item.get('description', '')

        headers = []
        for header in request.get('header', []):
            headers.append(f"{header.get('key')}: {header.get('value')}")

        body = ""
        if 'body' in request:
            body_data = request['body']
            if body_data.get('mode') == 'raw':
                body = body_data.get('raw', '')

        text = f"""
Endpoint: {item.get('name', 'Unknown')}
Method: {method}
Path: {path}
Description: {description}

Headers:
{chr(10).join(headers) if headers else 'None'}

Request Body:
{body if body else 'None'}

Authentication: {request.get('auth', {}).get('type', 'None')}
"""
        return text

    def _extract_path(self, item: Dict) -> str:
        """Extract path from Postman item"""
        request = item.get('request', {})
        url = request.get('url', {})

        if isinstance(url, dict):
            path_parts = url.get('path', [])
            return '/' + '/'.join(path_parts) if path_parts else url.get('raw', '')
        return str(url)

    def _flatten_postman_items(self, collection: Dict, items: List = None) -> List:
        """Recursively flatten Postman collection items"""
        if items is None:
            items = []

        if 'item' in collection:
            for item in collection['item']:
                if 'item' in item:
                    self._flatten_postman_items(item, items)
                else:
                    items.append(item)

        return items

    def _is_api_endpoint(self, content: str) -> bool:
        """Check if content describes an API endpoint"""
        patterns = [
            r'(GET|POST|PUT|DELETE|PATCH)\s+/',
            r'endpoint:',
            r'parameters:',
            r'response:'
        ]
        return any(re.search(p, content, re.I) for p in patterns)


def create_optimized_vector_store():
    """Main function to create the vector store"""
    console.print("[bold blue]Encompass RAG Vector Store Creator[/bold blue]")

    # Paths
    csv_docs_path = "data/documentation_data.csv"
    context7_path = "data/context7_llms.txt"
    postman_path = "data/Encompass_Developer_Connect_postman_collection.json"
    vector_store_path = "vector_store"

    # Process documents
    processor = EncompassDocumentProcessor()
    all_documents = []

    # Process CSV documentation if it exists
    if os.path.exists(csv_docs_path):
        console.print("\n[yellow]Processing CSV documentation...[/yellow]")
        csv_docs = processor.process_csv_documentation(csv_docs_path)
        console.print(f"✓ Processed {len(csv_docs)} CSV documents")
        all_documents.extend(csv_docs)
    else:
        console.print(f"[yellow]CSV documentation not found at {csv_docs_path}, skipping...[/yellow]")

    # Process Context7 documentation if it exists
    if os.path.exists(context7_path):
        console.print("\n[yellow]Processing Context7 documentation...[/yellow]")
        context7_docs = processor.process_context7_file(context7_path)
        console.print(f"✓ Processed {len(context7_docs)} Context7 documents")
        all_documents.extend(context7_docs)
    else:
        console.print(f"[yellow]Context7 file not found at {context7_path}, skipping...[/yellow]")

    # Process Postman collection if it exists
    if os.path.exists(postman_path):
        console.print("\n[yellow]Processing Postman collection...[/yellow]")
        postman_docs = processor.process_postman_collection(postman_path)
        console.print(f"✓ Processed {len(postman_docs)} Postman endpoints")
        all_documents.extend(postman_docs)
    else:
        console.print(f"[yellow]Postman collection not found at {postman_path}, skipping...[/yellow]")

    if not all_documents:
        console.print("[red]Error: No documents to process![/red]")
        return False

    console.print(f"\n[green]Total documents: {len(all_documents)}[/green]")

    # Create embeddings
    console.print("\n[yellow]Creating embeddings (this may take a while)...[/yellow]")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},  # Change to 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create FAISS vector store
    console.print("[yellow]Building FAISS index...[/yellow]")
    vector_store = FAISS.from_documents(all_documents, embeddings)

    # Create BM25 index for hybrid search
    console.print("[yellow]Creating BM25 index for hybrid search...[/yellow]")
    texts = [doc.page_content for doc in all_documents]
    tokenized_texts = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)

    # Save everything
    console.print("\n[yellow]Saving vector store and indices...[/yellow]")
    os.makedirs(vector_store_path, exist_ok=True)

    vector_store.save_local(vector_store_path)

    with open(f"{vector_store_path}/bm25_index.pkl", 'wb') as f:
        pickle.dump(bm25, f)

    with open(f"{vector_store_path}/documents_metadata.pkl", 'wb') as f:
        pickle.dump([doc.metadata for doc in all_documents], f)

    # Save statistics
    stats = {
        'total_documents': len(all_documents),
        'csv_documents': len([d for d in all_documents if d.metadata.get('source') == 'csv_docs']),
        'context7_documents': len([d for d in all_documents if d.metadata.get('source') == 'context7']),
        'postman_endpoints': len([d for d in all_documents if d.metadata.get('source') == 'postman']),
        'embedding_model': 'BAAI/bge-base-en-v1.5',
        'chunk_sizes': {
            'api_documentation': 1500,
            'postman_endpoints': 800
        }
    }

    with open(f"{vector_store_path}/stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    console.print(f"\n[bold green]✓ Vector store created successfully at '{vector_store_path}'[/bold green]")
    console.print("\nStatistics:")
    console.print(f"  • CSV documents: {stats['csv_documents']}")
    console.print(f"  • Context7 documents: {stats['context7_documents']}")
    console.print(f"  • Postman endpoints: {stats['postman_endpoints']}")
    console.print(f"  • Total chunks: {len(all_documents)}")

    return True


if __name__ == "__main__":
    create_optimized_vector_store()