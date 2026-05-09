# Retrieval-Augmented Generation (RAG) Chatbot

FastAPI backend for question answering with retrieval from Pinecone and generation through Hugging Face models. The system ingests PDFs, chunks text with LangChain utilities, embeds chunks, stores vectors in Pinecone, and uses retrieved context to ground model responses.

![RAG system design](Design%20plans/System%20Design.png)

## What this repository contains

1. **Ingestion pipeline**: PDF loading, chunking, embedding creation, and Pinecone upsert.
2. **RAG chat pipeline**: query embedding, similarity search, prompt construction, answer generation, and source packaging.
3. **FastAPI service**: app startup/lifespan management, chat route, and health route.
4. **CI + containerization**: Pylint workflow and image build/publish workflow.

## Project layout

| Path | Purpose |
|------|---------|
| `api/server.py` | FastAPI app creation and startup lifespan (`make_connections`). |
| `api/routes/chatbot.py` | Chat and health endpoint handlers. |
| `api/middleware/schema.py` | `UserRequest` and `ChatResponse` models. |
| `core/generation/rag_pipeline.py` | Main retrieve -> prompt -> generate orchestration. |
| `core/generation/llm.py` | Hugging Face model/pipeline wrapper. |
| `core/retrieval/retriever.py` | Query embedding + Pinecone similarity search. |
| `core/retrieval/embeddings.py` | PDF loading/chunking + embedding generation. |
| `core/retrieval/vectore_store.py` | Vector DB wrapper around Pinecone operations. |
| `core/utils/config.py` | Environment-backed configuration access. |
| `core/utils/startup.py` | Shared connection bootstrap and readiness state. |
| `scripts/ingest_data.py` | Sequential and parallel ingestion functions. |
| `scripts/benchmark.py` | Correctness evaluation script with row-level error handling and safe score normalization. |
| `tests/` | Test modules for generation/retrieval plus API test placeholder. |

Sample document artifacts are tracked with DVC under `data/raw/`.

## Install

From the repository root:

```bash
pip install -r requirements.txt
```

If Pinecone is not installed transitively in your environment, install the client explicitly:

```bash
pip install pinecone
```

## Required environment variables

Configuration is loaded in `core/utils/config.py` using `os.environ[...]` (missing keys will raise errors immediately when accessed).

```env
# Hugging Face
HUGGINGFACE_API_KEY=<token>
HUGGINGFACE_MODEL_NAME=<generative-model-id>
HUGGINGFACE_TASK=<pipeline-task>
HUGGINGFACE_EMBEDDING_MODEL=<embedding-model-id>
HUGGINGFACE_EVAL_MODEL=<eval-model-id>

# Pinecone
PINECONE_API_KEY=<key>
PINECONE_ENVIRONMENT=<environment>
PINECONE_INDEX=<index-name>
PINECONE_NAMESPACE=<namespace>

# Optional integration
GOOGLE_GENAI_API_KEY=<google-genai-api-key>

# Optional benchmark/export settings (current code reads these exact names)
PATH=<csv-output-directory>
FILENAME=<csv-filename>
```

`PATH` is read directly by code for CSV export constants; this can conflict with your system `PATH`. Treat this as current behavior, not best practice.

## Run locally

Run from the repository root:

```bash
uvicorn api.server:server --reload --host 127.0.0.1 --port 8000
```

The ASGI app variable is `server` in `api/server.py`.

## API surface

`api/routes/chatbot.py` defines:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat` | Body `{ "prompt": "<question>" }`; returns generated answer + sources. |
| `GET` | `/v1/services/health` | Returns service health and initialization status. |

Routes are mounted from `api/routes/chatbot.py` onto the app in `api/server.py` under the `/v1` prefix.

## Ingest documents

`scripts/ingest_data.py` accesses `get_resources()` at import time, so Pinecone resources must already be initialized before those functions run.

Module entry point:

```bash
python -m scripts.ingest_data
```

If ingestion fails with resource-initialization errors, initialize resources first using the same startup path used by the API lifespan (`make_connections` in `core/utils/startup.py`), then execute ingestion calls.

## Tests and CI

Run tests:

```bash
pytest
```

CI linting is configured in `.github/workflows/pylint.yml` and currently runs Pylint on Python `3.10` through `3.14`.

## Container and publishing

- `Dockerfile` builds a multi-stage Python 3.11 image and runs Uvicorn on port `8501`.
- `.github/workflows/package-image.yml` builds on push to `main` and publishes to GHCR and Docker Hub.
- Docker entrypoint runs `uvicorn api.server:server ...` on port `8501`.

## Current status

- Core retrieval and generation flow is implemented.
- Ingestion utilities are implemented (sequential and async upsert patterns).
- `scripts/benchmark.py` now includes defensive error handling (initialization guards, row-level exception handling, and file-level failure handling).
- Some components are placeholders or thin (`api/middleware/auth.py`, `tests/test_api.py`).
- Operational correctness depends on Pinecone index compatibility with your embedding model and namespace configuration consistency.
