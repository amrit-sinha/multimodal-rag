import uuid

from app.core.celery_app import celery
from app.core.db import SessionLocal
from app.core.storage import download_bytes, upload_bytes
from app.ingest.chunk import chunk_pages
from app.ingest.extract import ExtractedImage, extract_pdf, extract_pdf_images, load_image
from app.models import Chunk, Document, File, IngestJob
from app.models.tables import DocStatus, JobStage, JobStatus, Modality
from app.providers import get_embedder, get_image_embedder


def _set_stage(session, job: IngestJob, stage: JobStage) -> None:
    job.stage = stage
    job.status = JobStatus.RUNNING
    session.commit()


def _build_text_chunks(doc_uuid, file: File, data: bytes) -> list[Chunk]:
    pages, page_count = extract_pdf(data)
    file.page_count = page_count
    text_chunks = chunk_pages(pages)
    if not text_chunks:
        return []

    vectors = get_embedder().embed_documents([c.content for c in text_chunks])
    return [
        Chunk(
            document_id=doc_uuid,
            file_id=file.id,
            modality=Modality.TEXT,
            content=c.content,
            page_no=c.page_no,
            char_start=c.char_start,
            char_end=c.char_end,
            embedding=vec,
        )
        for c, vec in zip(text_chunks, vectors)
    ]


def _build_image_chunks(doc_uuid, file: File, images: list[ExtractedImage]) -> list[Chunk]:
    if not images:
        return []

    vectors = get_image_embedder().embed_images([im.image for im in images])
    chunks: list[Chunk] = []
    for idx, (im, vec) in enumerate(zip(images, vectors)):
        # Directly-uploaded images already live in storage; crops pulled out of a
        # PDF are uploaded here so the citation can link to a viewable image.
        if im.page_no is None:
            image_key = file.storage_key
        else:
            image_key = f"{doc_uuid}/extracted/p{im.page_no}_{idx}.png"
            upload_bytes(image_key, im.data, "image/png")

        page_label = f" (page {im.page_no})" if im.page_no else ""
        chunks.append(
            Chunk(
                document_id=doc_uuid,
                file_id=file.id,
                modality=Modality.IMAGE,
                content=f"Image from {file.filename}{page_label}",
                page_no=im.page_no,
                bbox=im.bbox,
                image_key=image_key,
                image_embedding=vec,
            )
        )
    return chunks


@celery.task(bind=True, max_retries=2, default_retry_delay=10)
def ingest_document(self, document_id: str) -> dict:
    """Durable, staged ingestion: extract -> chunk -> embed -> index.
    Handles PDFs (text + embedded images) and standalone image uploads."""
    session = SessionLocal()
    doc_uuid = uuid.UUID(document_id)
    job = (
        session.query(IngestJob)
        .filter(IngestJob.document_id == doc_uuid)
        .order_by(IngestJob.created_at.desc())
        .first()
    )
    document = session.get(Document, doc_uuid)
    file = session.query(File).filter(File.document_id == doc_uuid).first()

    if not (job and document and file):
        session.close()
        raise ValueError(f"Missing job/document/file for {document_id}")

    job.attempts += 1
    document.status = DocStatus.PROCESSING
    session.commit()

    try:
        _set_stage(session, job, JobStage.EXTRACTING)
        data = download_bytes(file.storage_key)
        is_pdf = file.mime_type == "application/pdf"

        _set_stage(session, job, JobStage.CHUNKING)
        if is_pdf:
            images = extract_pdf_images(data)
        else:
            images = [load_image(data)]

        _set_stage(session, job, JobStage.EMBEDDING)
        text_chunks = _build_text_chunks(doc_uuid, file, data) if is_pdf else []
        image_chunks = _build_image_chunks(doc_uuid, file, images)

        all_chunks = text_chunks + image_chunks
        if not all_chunks:
            raise ValueError("No indexable text or images found in document")

        # Idempotent: clear any prior chunks for this doc before re-indexing.
        _set_stage(session, job, JobStage.INDEXING)
        session.query(Chunk).filter(Chunk.document_id == doc_uuid).delete()
        session.add_all(all_chunks)

        job.stage = JobStage.DONE
        job.status = JobStatus.SUCCEEDED
        job.error = None
        document.status = DocStatus.READY
        session.commit()
        return {
            "document_id": document_id,
            "text_chunks": len(text_chunks),
            "image_chunks": len(image_chunks),
        }

    except Exception as exc:  # noqa: BLE001
        session.rollback()
        job.status = JobStatus.FAILED
        job.error = str(exc)
        document.status = DocStatus.FAILED
        session.commit()
        raise self.retry(exc=exc)
    finally:
        session.close()
