# Encompass RAG Assistant
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Retrieval-Augmented Generation (RAG) system for querying Encompass API documentation with natural language.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)


## Overview

The Encompass RAG Assistant is a powerful tool that allows users to query Encompass API documentation using natural language. It leverages a vector database of 3000+ Encompass URLs and Postman collections to provide accurate, contextual answers to API-related questions.

This application combines modern RAG techniques with a user-friendly interface to make Encompass API documentation more accessible and easier to navigate.

## Features

- **Natural Language Queries**: Ask questions about Encompass API in plain English
- **Context-Aware Responses**: Answers include relevant context from documentation
- **Source References**: All responses include links to original documentation
- **Code Examples**: Provides relevant code snippets when appropriate
- **User-Friendly Interface**: Clean, intuitive Streamlit UI
- **Fast API Backend**: Robust FastAPI implementation for reliable performance

## Architecture

The system consists of three main components:

1. **Vector Store**: 
   - Built using FAISS with MiniLM-L6-v2 embeddings
   - Contains 3000+ Encompass URLs and Postman collection data
   - Enables semantic search across documentation

2. **RAG System**:
   - Retrieves relevant documentation chunks based on query similarity
   - Uses Ollama with Phi-2 (2.7B parameter model) for generation
   - Custom prompt template for accurate API documentation responses

3. **User Interface**:
   - Streamlit-based chat interface
   - Two-column layout with input and conversation history
   - Dark theme for reduced eye strain

![Architecture](https://github.com/user-attachments/assets/f6c01fcf-391f-4781-99c9-df33b168d947)


## Installation

### Prerequisites
- Python 3.9+
- Ollama (for local LLM hosting)
- 8GB+ RAM recommended

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/encompass-rag-assistant.git
   cd encompass-rag-assistant
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Ollama and download the Phi-2 model:
   ```bash
   # Follow Ollama installation instructions at https://ollama.ai/
   ollama pull phi2
   ```

5. Create the vector store (if not already created):
   ```bash
   python scripts/create_vector_store.py
   ```

## Usage

1. Start the FastAPI backend:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

2. In a separate terminal, start the Streamlit UI:
   ```bash
   streamlit run app/ui.py
   ```

3. Open your browser and navigate to http://localhost:8501

4. Enter your question about Encompass API in the text area and click "Submit Question"

# Configuration Roadmap

The application will be configurable through environment variables or a `.env` file. Below is the planned configuration system:

## Environment Variables

### Model Configuration
- [ ] MODEL_NAME=qwen2.5-coder:7b
- [ ] TEMPERATURE=0.0

### Vector Store Configuration
- [ ] VECTOR_STORE_PATH=./vector_store
- [ ] EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

### API Configuration
- [ ] HOST=0.0.0.0
- [ ] PORT=8000

### Logging Configuration
- [x] LOG_LEVEL=INFO
- [x] LOG_FORMAT=%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s
- [x] LOG_FILE=logs/app.log
- [x] LOG_MAX_SIZE=10485760
- [x] LOG_BACKUP_COUNT=5

## Implementation Status

- [x] Logging configuration system
- [ ] Model configuration system
- [ ] Vector store configuration system
- [ ] API configuration system

> **Note:** This configuration system is currently under development. Some features are already implemented (marked with [x]), while others are planned for future releases (marked with [ ]). For now, please refer to the installation instructions for current setup requirements.

Built with ❤️ for Encompass API users
