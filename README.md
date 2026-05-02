# Retrieval-Augmented Generation Chatbot

![RAG System Plan](https://github.com/kinola-IQ/Retrieval-Augmented-Generation-chat-system/blob/main/Design%20plans/RAG%20system%20design.png)

## Overview

This repository contains a prototype RAG chatbot backend. It is designed as a FastAPI service that can:

- load Hugging Face text-generation models with `transformers`
- ingest and split PDF documents for embeddings
- generate embeddings via Hugging Face embedding models
- expose API routes under `/v1`
- support vector DB configuration patterns (Pinecone config is present but may require implementation)

## Architecture

- `api/server.py`: FastAPI application entry point and app factory.
- `api/routes/chatbot.py`: router stub for chatbot-related endpoints.
- `core/generation/llm.py`: wrapper around Hugging Face model loading and inference pipeline.
- `core/retrieval/embeddings.py`: PDF loading, document splitting, query splitting, and embedding creation.
- `core/retrieval/vector_store.py`: vector store implementation.
- `core/utils/config.py`: centralized environment variable configuration for Hugging Face and Pinecone.
- `scripts/ingest_data.py`: script for preprocessing and indexing documents.

## Dependencies

The repository specifies minimum versions for dependencies:

- `fastapi>=0.104.1`
- `uvicorn>=0.24.0`
- `transformers>=5.0.0rc3`
- `tenacity>=8.2.3`
- `python-dotenv>=1.2.2`
- `langchain>=0.2.5`
- `langchain-community>=0.3.27`
- `langchain-experimental>=0.0.61`
- `tiktoken>=0.5.2`
- `pydantic>=2.13.3`
- `pytest>=9.0.3`
- `ragas>=0.4.3`
- `langsmith>=0.8.0`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file or export these variables before running the app:

```env
HUGGINGFACE_API_KEY=<your-huggingface-api-key>
HUGGINGFACE_MODEL_NAME=<model-name>
HUGGINGFACE_TASK=<task-name>
HUGGINGFACE_EMBEDDING_MODEL=<embedding-model-name>
HUGGINGFACE_EVAL_MODEL=<eval-model-name>

PINECONE_API_KEY=<your-pinecone-api-key>
PINECONE_ENVIRONMENT=<pinecone-environment>
PINECONE_INDEX=<pinecone-index>
PINECONE_NAMESPACE=<pinecone-namespace>
```

## Running the API

Start the API server with Uvicorn:

```bash
uvicorn api.server:server --reload --host 127.0.0.1 --port 8000
```

The API is mounted under `/v1`.

## Data Ingestion

To preprocess and index documents:

```bash
python scripts/ingest_data.py
```

## Running Tests

Run the test suite with pytest:

```bash
pytest
```

## Project Status

- `api/routes/chatbot.py` currently defines an empty router and does not expose any endpoint implementations yet.
- `core/generation/llm.py` sets up a Hugging Face pipeline with retry support.
- `core/retrieval/embeddings.py` supports PDF loading and document/query splitting via LangChain.
- `core/utils/config.py` requires the specified environment variables and will raise if they are missing.

## Notes

- The repository is currently a scaffolding for a RAG pipeline and may require additional implementation in route definitions and vector store integration.
- `api/server.py` uses FastAPI lifecycle management and a retry decorator around app creation.

## Test Coverage

Basic test files exist in `tests/test_api.py`, `tests/test_generation.py`, and `tests/test_retrieval.py`, but the current repository state contains placeholder implementations only.
