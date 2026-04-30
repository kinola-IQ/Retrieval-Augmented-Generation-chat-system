# Retrieval-Augmented Generation Chatbot

![RAG System Plan](https://github.com/kinola-IQ/Retrieval-Augmented-Generation-chat-system/blob/852601f22e7b7f2872801ad5dd7d5e5bb0f03f34/System%20Design)
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
- `core/utils/config.py`: centralized environment variable configuration for Hugging Face and Pinecone.
- `scripts/injest_data.py`: placeholder script for preprocessing and indexing documents.

## Dependencies

The repository pins exact versions for reproducible builds:

- `fastapi==0.104.1`
- `uvicorn==0.24.0`
- `transformers==4.36.2`
- `tenacity==8.2.3`
- `python-dotenv==1.0.0`
- `langchain==0.1.0`
- `langchain-community==0.0.10`
- `langchain-experimental==0.0.50`
- `tiktoken==0.5.2`
- `pydantic==2.5.0`
- `pytest==7.4.3`

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

PINECONE_API_KEY=<your-pinecone-api-key>
PINECONE_ENVIRONMENT=<pinecone-environment>
PINECONE_INDEX_NAME=<pinecone-index-name>
```

## Running the API

Start the API server with Uvicorn:

```bash
uvicorn api.server:server --reload --host 127.0.0.1 --port 8000
```

The API is mounted under `/v1`.

## Project Status

- `api/routes/chatbot.py` currently defines an empty router and does not expose any endpoint implementations yet.
- `core/generation/llm.py` sets up a Hugging Face pipeline with retry support.
- `core/retrieval/embeddings.py` supports PDF loading and document/query splitting via LangChain.
- `core/utils/config.py` requires the specified environment variables and will raise if they are missing.

## Notes

- The repository is currently a scaffolding for a RAG pipeline and may require additional implementation in route definitions and vector store integration.
- `api/server.py` uses FastAPI lifecycle management and a retry decorator around app creation.

## Test Coverage

Basic test files exist in `tests/test_api.py` and `tests/test_retrieval.py`, but the current repository state contains placeholder docstrings only.
