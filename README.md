# Retrieval-Augmented Generation (RAG) Chatbot

A FastAPI backend that answers questions using **retrieval from a Pinecone vector index** and **local text generation** via Hugging Face [Transformers](https://huggingface.co/docs/transformers) pipelines. PDFs are chunked with LangChain (semantic or recursive splitting), embedded with Hugging Face embedding models, and upserted into Pinecone for similarity search at query time.

![RAG system design](Design%20plans/System%20Design.png)

## What it does

1. **Ingestion** — Load PDFs, split into chunks, embed with `HuggingFaceEmbeddings`, upsert vectors and metadata (`source`, `text`) into a Pinecone namespace.
2. **Chat** — Embed the user prompt, query Pinecone for top‑k chunks, build a grounded prompt, run the configured generation model, return **answer text** plus **source citations** (text, source file, score).
3. **Health** — Report whether startup initialization (Pinecone client) completed successfully.

## API (`/v1`)

| Method | Path | Description |
|--------|------|---------------|
| `POST` | `/v1/chat` | JSON body: `{ "prompt": "<user question>" }`. Response: `{ "response": "<answer>", "sources": [ { "text", "source", "score" }, ... ] }`. |
| `GET` | `/v1/services/health` | Returns `{ "status", "service", "services_initialized" }` when dependencies are ready; `500` / `408` on failure or timeout. |

Request and response shapes are defined in `api/middleware/schema.py` (`UserRequest`, `ChatResponse`).

## Project layout

| Path | Role |
|------|------|
| `api/server.py` | FastAPI app factory, lifespan hook, mounts routes under `/v1`. |
| `api/routes/chatbot.py` | `/chat` and `/services/health` handlers. |
| `api/middleware/schema.py` | Pydantic models for API I/O. |
| `core/generation/rag_pipeline.py` | End-to-end RAG: retrieve → prompt → generate → format sources. |
| `core/generation/llm.py` | Hugging Face `pipeline` wrapper with retries (`HUGGINGFACE` provider). |
| `core/retrieval/retriever.py` | Query embedding and vector similarity search against the live index. |
| `core/retrieval/embeddings.py` | PDF loading (`PyPDFLoader`), chunking, batch embedding for ingestion. |
| `core/retrieval/vectore_store.py` | `VectorStore` wrapper around Pinecone (query, upsert, delete, stats). |
| `core/utils/config.py` | Environment-based config for Hugging Face and Pinecone. |
| `core/utils/startup.py` | Startup connection handling and shared `VECTOR_DB` client. |
| `scripts/ingest_data.py` | Batch and parallel ingestion into Pinecone. |
| `scripts/benchmark.py` | Placeholder for evaluation / benchmarking. |
| `tests/` | Test modules (`test_api`, `test_generation`, `test_retrieval`); many are stubs or mocks. |

Sample data is tracked with **DVC** under `data/raw/` (see `.dvc` and `*.dvc` files).

## Dependencies

Install from the repository root:

```bash
pip install -r requirements.txt
```

The code also imports the **Pinecone** Python client (`from pinecone import Pinecone`). If it is not pulled in transitively, install it explicitly (see [Pinecone Python SDK](https://docs.pinecone.io/guides/get-started/install)).

Pinned / declared in `requirements.txt` include FastAPI, Uvicorn, Transformers, LangChain community + experimental, tiktoken, Pydantic, pytest, Ragas, LangSmith, tenacity, and python-dotenv. **PyTorch** (or another backend supported by your chosen models) is required at runtime for Transformers and embedding models.

## Environment variables

Create a `.env` file in the project root (or export variables). All of the following are read in `core/utils/config.py`; missing keys raise at access time.

```env
# Hugging Face — generation, embeddings, and optional eval model name
HUGGINGFACE_API_KEY=<token>
HUGGINGFACE_MODEL_NAME=<generative-model-id>
HUGGINGFACE_TASK=text-generation
HUGGINGFACE_EMBEDDING_MODEL=<embedding-model-id>
HUGGINGFACE_EVAL_MODEL=<eval-model-id>

# Pinecone
PINECONE_API_KEY=<key>
PINECONE_ENVIRONMENT=<environment>
PINECONE_INDEX=<index-name>
PINECONE_NAMESPACE=<namespace>
```

Use the same embedding model for ingestion and retrieval so query vectors match the index.

## Run the API

From the **repository root** (so `api` and `core` resolve as packages):

```bash
uvicorn api.server:server --reload --host 127.0.0.1 --port 8000
```

The ASGI app instance is named `server` in `api/server.py`.

## Ingest documents

`scripts/ingest_data.py` pulls the live Pinecone client via `get_resources()` at import time, which matches the object initialized in the API lifespan (`core.utils.startup.make_connections`). In practice you either need that initialization to have run first, or you should call `make_connections()` (async) before importing the ingest module, then invoke `ingest_in_batches` / `ingest_in_parallel` with your PDF path and display name.

From the repo root, imports use parent-package paths (`from ..core...`); run ingest logic as a module so `..` resolves to the project root:

```bash
python -m scripts.ingest_data
```

If the module exits immediately on import because `VECTOR_DB` is unset, initialize Pinecone the same way `api/server.py` does in `lifespan`, then run your ingestion calls (or add a small `__main__` block that awaits `make_connections()` before importing heavy logic).

## Tests

```bash
pytest
```

CI runs **Pylint** on Python 3.8–3.10 (`.github/workflows/pylint.yml`).

## Status and notes

- **Implemented:** RAG pipeline, chat and health routes, PDF chunking and embedding utilities, Pinecone-oriented ingestion and vector wrapper, structured logging and helpers (timeouts, timers).
- **Thin / placeholder:** `scripts/benchmark.py`, `api/middleware/auth.py`, and several test files are minimal stubs.
- **Operational detail:** Ensure your Pinecone index exists, dimensions match your embedding model, and namespaces used at ingest match those used in `RAGPipeline` / config before expecting useful chat results.
