import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import get_session
from app.core.security import rate_limit, require_api_key
from app.core.storage import presigned_put_url
from app.ingest.tasks import ingest_document
from app.models import Document, File, IngestJob
from app.models.schemas import (
    DocumentView,
    ImageSearchRequest,
    ImageSearchResponse,
    JobView,
    QueryRequest,
    QueryResponse,
    UploadRequest,
    UploadResponse,
)
from app.models.tables import DocStatus
from app.retrieval.answer import answer_question, search_images, stream_answer_events

router = APIRouter(dependencies=[Depends(require_api_key), Depends(rate_limit)])


@router.post("/documents", response_model=UploadResponse)
def create_document(req: UploadRequest, session: Session = Depends(get_session)):
    """Register a document and hand back a presigned URL so the client uploads
    the bytes straight to object storage (the API never buffers the file)."""
    if req.mime_type not in settings.allowed_mimes:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported mime type '{req.mime_type}'. Allowed: {sorted(settings.allowed_mimes)}",
        )

    document = Document(title=req.title or req.filename, status=DocStatus.PENDING)
    session.add(document)
    session.flush()

    storage_key = f"{document.id}/{req.filename}"
    file = File(
        document_id=document.id,
        storage_key=storage_key,
        filename=req.filename,
        mime_type=req.mime_type,
    )
    session.add(file)
    session.commit()

    return UploadResponse(
        document_id=document.id,
        file_id=file.id,
        upload_url=presigned_put_url(storage_key),
        storage_key=storage_key,
    )


@router.post("/documents/{document_id}/ingest", response_model=JobView)
def start_ingest(document_id: uuid.UUID, session: Session = Depends(get_session)):
    """Call after the client has PUT the file. Enqueues the async pipeline."""
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(404, "Document not found")

    job = IngestJob(document_id=document_id)
    session.add(job)
    session.commit()

    ingest_document.delay(str(document_id))
    session.refresh(job)
    return JobView.model_validate(job)


@router.get("/documents", response_model=list[DocumentView])
def list_documents(session: Session = Depends(get_session)):
    docs = session.query(Document).order_by(Document.created_at.desc()).all()
    return [DocumentView.model_validate(d) for d in docs]


@router.get("/documents/{document_id}", response_model=DocumentView)
def get_document(document_id: uuid.UUID, session: Session = Depends(get_session)):
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(404, "Document not found")
    return DocumentView.model_validate(document)


@router.get("/documents/{document_id}/job", response_model=JobView)
def get_job(document_id: uuid.UUID, session: Session = Depends(get_session)):
    job = (
        session.query(IngestJob)
        .filter(IngestJob.document_id == document_id)
        .order_by(IngestJob.created_at.desc())
        .first()
    )
    if not job:
        raise HTTPException(404, "No ingest job for this document")
    return JobView.model_validate(job)


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, session: Session = Depends(get_session)):
    if not req.question.strip():
        raise HTTPException(400, "Question must not be empty")
    return answer_question(
        session, req.question, req.document_id, req.top_k, req.include_images
    )


@router.post("/query/stream")
def query_stream(req: QueryRequest):
    """Same as /query but streams the answer token-by-token over SSE, then emits
    a final `citations` event once the full answer is known."""
    if not req.question.strip():
        raise HTTPException(400, "Question must not be empty")
    return StreamingResponse(
        stream_answer_events(req.question, req.document_id, req.top_k, req.include_images),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/search/images", response_model=ImageSearchResponse)
def search_images_endpoint(req: ImageSearchRequest, session: Session = Depends(get_session)):
    """Cross-modal search: find images by a text description, ranked by CLIP
    visual-semantic similarity."""
    if not req.query.strip():
        raise HTTPException(400, "Query must not be empty")
    return search_images(session, req.query, req.document_id, req.top_k)
