# Multimodal RAG

A local-first backend for retrieval-augmented generation over PDFs and images.
It ingests documents asynchronously, retrieves relevant text and visual content,
and generates grounded answers with source citations.

## Highlights

- PDF text extraction and image ingestion with page-level provenance
- BGE text retrieval and CLIP text-to-image search
- Vision-language answers through Gemma 4 running on Ollama
- Citations with source page, snippet, similarity score, and presigned URL
- Background ingestion with progress tracking, retries, and idempotent re-indexing
- Server-Sent Events (SSE) for streaming answers
- Optional API-key authentication, Redis rate limiting, and request-ID logging

## Architecture

```text
Client
  ├─ register document ───────────────► FastAPI
  ├─ upload with presigned URL ───────► MinIO
  └─ start ingestion ─────────────────► Celery + Redis
                                            │
                              extract → chunk → embed → index
                                            │
                                            ▼
                                  PostgreSQL + pgvector
                                            │
Client ◄──── answer + citations ◄──── FastAPI ────► Ollama
```

Text chunks and image chunks use separate vector spaces:

- `BAAI/bge-small-en-v1.5` for text retrieval
- `clip-ViT-B-32` for cross-modal image retrieval
- `gemma4:e2b` for grounded text and vision responses

## Stack

| Component | Technology |
| --- | --- |
| API | FastAPI |
| Background jobs | Celery + Redis |
| Database and vectors | PostgreSQL + pgvector |
| Object storage | MinIO |
| Document processing | PyMuPDF + Pillow |
| Local models | Sentence Transformers + Ollama |

## Quick start

Requirements:

- Docker and Docker Compose
- Ollama running on the host

Pull the vision model:

```bash
ollama pull gemma4:e2b
```

Configure and start the application:

```bash
cp .env.example .env
docker compose up --build
```

The API documentation is available at <http://localhost:8000/docs> and the
MinIO console at <http://localhost:9001>.

Supported uploads are PDF, PNG, JPEG, and WebP.

## Demo

Place test files in `samples/`, then run the PDF question-answering flow:

```bash
pip install httpx
python scripts/demo.py samples/sample.pdf "What is this document about?"
```

Run text-to-image search with CLIP:

```bash
python scripts/demo_image_search.py "a red square" samples/red_square.png samples/blue_circle.png
```

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/documents` | Register a document and receive a presigned upload URL |
| `POST` | `/api/documents/{id}/ingest` | Start asynchronous ingestion |
| `GET` | `/api/documents` | List documents |
| `GET` | `/api/documents/{id}` | Get document status |
| `GET` | `/api/documents/{id}/job` | Get the latest ingestion job |
| `POST` | `/api/query` | Generate an answer with citations |
| `POST` | `/api/query/stream` | Stream an answer and citations over SSE |
| `POST` | `/api/search/images` | Search indexed images using text |

Query requests support optional `document_id`, `top_k`, and `include_images`
fields. When `API_KEY` is configured, include it in the `X-API-Key` header.

## Tests

```bash
docker compose exec api sh -c "pip install -r requirements-dev.txt && python -m pytest"
```
