import os
import pandas as pd
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from importlib.resources import files
from rich import print


# project_root = Path(__file__).resolve().parents[2]  # As script folder is placed in the same level as the repo root
# csv_path = project_root / "scripts" / "docs" / "documentation_data.csv"
# postman_path = project_root / "scripts" / "Encompass_Developer_Connect_postman_collection.json"


def setup_vector_store():
    """Set up the vector store for the RAG system"""
    print("Setting up vector store...")
    vector_store_path = str(files("rag_docs.src.rag_docs").joinpath("vector_store"))
    csv_path = str(files("rag_docs.src.rag_docs.data").joinpath("documentation_data.csv"))
    postman_path = str(
        files("rag_docs.src.rag_docs.data").joinpath("Encompass_Developer_Connect_postman_collection.json"))

    if not os.path.exists(csv_path):
        print(f"ERROR: Documentation CSV file not found at {csv_path}")
        return False

    if not os.path.exists(postman_path):
        print(f"ERROR: Postman collection file not found at {postman_path}")
        return False

    if os.path.exists(vector_store_path):
        print(f"Vector store already exists at {vector_store_path}")
        choice = input("Do you want to recreate it? (y/n): ").lower()
        if choice != 'y':
            print("Setup aborted.")
            return False

    print("Loading documentation data...")
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} documentation pages")
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return False

    print("Loading Postman collection...")
    try:
        with open(postman_path, 'r') as f:
            collection = json.load(f)
        print(f"Loaded Postman collection: {collection.get('info', {}).get('name', 'Unknown')}")
    except Exception as e:
        print(f"Error loading Postman collection: {str(e)}")
        return False

    print("Processing documentation...")
    doc_texts = []
    for text in df['content'].tolist():
        # Convert to string if not already a string
        if not isinstance(text, str):
            text = str(text)
        doc_texts.append(text)

    print("Processing Postman collection...")
    postman_texts = process_postman_collection(collection)

    print("Combine all texts")
    all_texts = doc_texts + postman_texts

    print("Splitting texts into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = []
    for text in all_texts:

        if not isinstance(text, str):
            text = str(text)
        chunks.extend(text_splitter.split_text(text))

    print(f"Split into {len(chunks)} chunks")

    print("Creating vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    print(f"Saving vector store to {vector_store_path}...")
    os.makedirs(vector_store_path, exist_ok=True)
    vector_store.save_local(vector_store_path)

    print("Vector store setup completed successfully!")
    return True


def process_postman_collection(collection):
    """Process Postman collection into text format"""
    texts = []

    if 'info' in collection:
        info = collection['info']
        texts.append(f"Collection Name: {info.get('name', '')}\nDescription: {info.get('description', '')}")

    if 'item' in collection:
        for item in collection['item']:
            process_item(item, texts)

    return texts


def process_item(item, texts, parent_path=""):
    """Process a Postman collection item (endpoint or folder)"""

    if 'item' in item:
        folder_name = item.get('name', 'Unnamed Folder')
        folder_path = f"{parent_path}/{folder_name}" if parent_path else folder_name

        if 'description' in item:
            texts.append(f"Folder: {folder_path}\nDescription: {item['description']}")

        for nested_item in item['item']:
            process_item(nested_item, texts, folder_path)

    elif 'request' in item:
        endpoint_name = item.get('name', 'Unnamed Endpoint')
        endpoint_path = f"{parent_path}/{endpoint_name}" if parent_path else endpoint_name

        request = item['request']
        method = request.get('method', 'UNKNOWN')

        url = ""
        if isinstance(request.get('url'), dict):
            url_parts = request['url'].get('path', [])
            url = '/'.join(url_parts) if url_parts else request['url'].get('raw', '')
        else:
            url = request.get('url', '')

        headers = []
        if 'header' in request:
            for header in request['header']:
                headers.append(f"{header.get('key', '')}: {header.get('value', '')}")

        body = ""
        if 'body' in request:
            if request['body'].get('mode') == 'raw':
                body = request['body'].get('raw', '')
            elif request['body'].get('mode') == 'formdata':
                form_items = []
                for form_item in request['body'].get('formdata', []):
                    form_items.append(f"{form_item.get('key', '')}: {form_item.get('value', '')}")
                body = '\n'.join(form_items)

        description = item.get('description', '')

        endpoint_text = f"""
Endpoint: {endpoint_path}
Method: {method}
URL: {url}
Description: {description}
Headers: {', '.join(headers) if headers else 'None'}
Body: {body}
"""
        texts.append(endpoint_text)

    return texts


if __name__ == "__main__":
    setup_vector_store()
