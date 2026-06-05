# Multimodal RAG

A backend service for **retrieval-augmented generation over your own documents**.
Upload files, the system asynchronously chunks and embeds them into a vector store,
and you can ask questions that are answered **with citations pointing back to the
exact page** of the source.

Built to run entirely on **open-source, local models** (no API keys required).

> **Status:** PDF text ingestion + retrieval with page-level citations,
> streaming answers, and production hardening (auth, rate limiting, logging).

## Why this design

The interesting engineering here is **not** "call an embedding API". It's the
production-shaped backend around it:

- **Asynchronous ingestion pipeline** — uploads kick off a staged Celery job
  (`extract → chunk → embed → index`) with live per-stage status, retries, and
  idempotent re-indexing. Long work never blocks a request.
- **Direct-to-storage uploads** — the API hands out presigned URLs so file bytes
  go straight to object storage; the API never buffers large files.
- **Provenance-rich citations** — every chunk records where it came from
  (file, page, character span), so answers cite the precise source location.
- **Swappable model providers** — the embedder and LLM sit behind small
  interfaces, so "local model vs. hosted API" is a config change, not a rewrite.
- **Streaming answers (SSE)** — `/query/stream` streams the answer token-by-token
  and emits citations once the full answer is known.
- **Production hardening** — API-key auth, Redis-backed rate limiting, structured
  logging with per-request IDs, and consistent JSON error responses.

## Architecture

```
                 presigned PUT
   client ───────────────────────────────►  MinIO (object storage)
     │                                            ▲
     │ 1. POST /documents (register + get URL)    │ download bytes
     │ 2. POST /documents/{id}/ingest             │
     ▼                                            │
  FastAPI  ──enqueue──►  Redis  ──►  Celery worker┘
     │                                  │  extract → chunk → embed → index
     │ 5. POST /query                   ▼
     └──────────────►  Postgres + pgvector  (metadata + vectors + provenance)
                                  │
                                  ▼
                      Ollama (local LLM)  →  answer + citations
```

## Tech stack

| Concern            | Choice                                   |
| ------------------ | ---------------------------------------- |
| API                | FastAPI                                   |
| Async jobs         | Celery + Redis                            |
| Metadata + vectors | Postgres + `pgvector`                     |
| Object storage     | MinIO (S3-compatible)                     |
| Text embeddings    | `BAAI/bge-small-en-v1.5` (local, CPU)     |
| LLM generation     | Ollama (`qwen2.5:3b-instruct`, local)     |
| PDF parsing        | PyMuPDF                                   |

## Prerequisites

1. **Docker + Docker Compose**
2. **Ollama** running on the host with the model pulled:

```bash
ollama pull qwen2.5:3b-instruct
```

The app reaches Ollama at `http://host.docker.internal:11434` from inside Docker.

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

Services:

- API + docs: http://localhost:8000/docs
- MinIO console: http://localhost:9001 (user/pass from `.env`)

Run the end-to-end demo (uploads a PDF, waits for ingestion, asks a question):

```bash
pip install httpx
python scripts/demo.py path/to/file.pdf "What is this document about?"
```

## API

| Method | Path                             | Purpose                                   |
| ------ | -------------------------------- | ----------------------------------------- |
| POST   | `/api/documents`                 | Register a doc, get a presigned upload URL |
| POST   | `/api/documents/{id}/ingest`     | Start async ingestion                     |
| GET    | `/api/documents`                 | List documents                            |
| GET    | `/api/documents/{id}`            | Document status                           |
| GET    | `/api/documents/{id}/job`        | Live ingestion job status                 |
| POST   | `/api/query`                     | Ask a question, get an answer + citations |
| POST   | `/api/query/stream`              | Same, streamed token-by-token over SSE    |

When `API_KEY` is set in `.env`, send it as the `X-API-Key` header on every
`/api/*` request. Every response carries an `X-Request-ID` for log correlation.

## Tests

```bash
docker compose exec api sh -c "pip install -r requirements-dev.txt && python -m pytest"
```
