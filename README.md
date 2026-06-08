# Multimodal RAG

A backend service for **retrieval-augmented generation over your own documents**.
Upload PDFs or images, the system asynchronously chunks and embeds them into a
vector store, and you can ask questions that are answered **with citations
pointing back to the source** (page, file, or image).

Built to run entirely on **open-source, local models** (no API keys required).

> **Status:** PDF and image ingestion, CLIP text→image search, vision-LLM
> answers, page-level citations, streaming, and production hardening (auth,
> rate limiting, logging).

## Features

- **Asynchronous ingestion pipeline** — uploads kick off a staged Celery job
  (`extract → chunk → embed → index`) with live per-stage status, retries, and
  idempotent re-indexing. Long work never blocks a request.
- **Direct-to-storage uploads** — the API hands out presigned URLs so file bytes
  go straight to object storage; the API never buffers large files.
- **Multimodal (text + images)** — upload images, or let the pipeline pull
  images out of PDF pages. CLIP encodes images and text into one shared space,
  so a plain-text query can retrieve semantically matching images. Retrieved
  images are then handed to a **vision LLM** (Gemma4:12b), so you can ask "what
  is in this image?" and get a grounded answer — not just a similarity match.
- **Provenance-rich citations** — every chunk stores where it came from (file,
  page, character span, on-page bounding box). Query responses return page,
  snippet, score, and a presigned link to the source file or image.
- **Pluggable model backends** — embedders and the LLM sit behind small
  interfaces; alternate implementations (e.g. a hosted API) can be swapped in
  without touching ingestion or retrieval code.
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
     │ 2. PUT file to presigned URL               │
     │ 3. POST /documents/{id}/ingest             │
     ▼                                            │
  FastAPI  ──enqueue──►  Redis  ──►  Celery worker ─┘
     │                                  │
     │                                  │ extract → chunk → embed → index
     │ 4. POST /query                   │ (BGE text + CLIP images)
     └──────────────►  Postgres + pgvector
                                  │
                                  ▼
              Ollama (vision LLM + retrieved images)  →  answer + citations
```

## Tech stack

| Concern            | Choice                                   |
| ------------------ | ---------------------------------------- |
| API                | FastAPI                                   |
| Async jobs         | Celery + Redis                            |
| Metadata + vectors | Postgres + `pgvector`                     |
| Object storage     | MinIO (S3-compatible)                     |
| Text embeddings    | `BAAI/bge-small-en-v1.5` (local, CPU)     |
| Image embeddings   | CLIP `clip-ViT-B-32` (local, shared text+image space) |
| LLM generation     | Ollama (`gemma4:12b`, local **vision** model) |
| PDF / image parsing| PyMuPDF + Pillow                          |

## Prerequisites

1. **Docker + Docker Compose**
2. **Ollama** running on the host with a **vision-capable** model pulled:

```bash
ollama pull gemma4:12b
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

Supported uploads: PDF, PNG, JPEG, WebP (`ALLOWED_MIME_TYPES` in `.env`).

Run the end-to-end demo (uploads a PDF, waits for ingestion, asks a question):

```bash
pip install httpx
python scripts/demo.py samples/sample.pdf "What is this document about?"
```

Image search demo (upload images, query by text description via CLIP):

```bash
python scripts/demo_image_search.py "a red square" samples/red_square.png samples/blue_circle.png
```

Put sample files under `samples/` locally.

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
| POST   | `/api/search/images`             | Find images by a text description (CLIP)   |

`POST /api/query` and `/api/query/stream` accept optional `document_id` (scope
to one doc), `top_k`, and `include_images` (default `true`). `/search/images` hits include bounding boxes when
available.

When `API_KEY` is set in `.env`, send it as the `X-API-Key` header on every
`/api/*` request. Every response carries an `X-Request-ID` for log correlation.

## Tests

```bash
docker compose exec api sh -c "pip install -r requirements-dev.txt && python -m pytest"
```
