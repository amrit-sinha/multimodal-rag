import uuid

from app.core.celery_app import celery
from app.core.db import SessionLocal
from app.core.storage import download_bytes
from app.ingest.chunk import chunk_pages
from app.ingest.extract import extract_pdf
from app.models import Chunk, Document, File, IngestJob
from app.models.tables import DocStatus, JobStage, JobStatus, Modality
from app.providers import get_embedder


def _set_stage(session, job: IngestJob, stage: JobStage) -> None:
    job.stage = stage
    job.status = JobStatus.RUNNING
    session.commit()


@celery.task(bind=True, max_retries=2, default_retry_delay=10)
def ingest_document(self, document_id: str) -> dict:
    """Durable, staged ingestion: extract -> chunk -> embed -> index.
    Each stage updates the job row so the API can report live progress."""
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
        # Extract
        _set_stage(session, job, JobStage.EXTRACTING)
        data = download_bytes(file.storage_key)
        pages, page_count = extract_pdf(data)
        file.page_count = page_count
        session.commit()
        if not pages:
            raise ValueError("No extractable text found in document")

        # Chunk
        _set_stage(session, job, JobStage.CHUNKING)
        text_chunks = chunk_pages(pages)

        # Embed
        _set_stage(session, job, JobStage.EMBEDDING)
        embedder = get_embedder()
        vectors = embedder.embed_documents([c.content for c in text_chunks])

        # Index (idempotent: clear any prior chunks for this doc first)
        _set_stage(session, job, JobStage.INDEXING)
        session.query(Chunk).filter(Chunk.document_id == doc_uuid).delete()
        session.bulk_save_objects(
            [
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
        )

        job.stage = JobStage.DONE
        job.status = JobStatus.SUCCEEDED
        job.error = None
        document.status = DocStatus.READY
        session.commit()
        return {"document_id": document_id, "chunks": len(text_chunks)}

    except Exception as exc:  # noqa: BLE001
        session.rollback()
        job.status = JobStatus.FAILED
        job.error = str(exc)
        document.status = DocStatus.FAILED
        session.commit()
        raise self.retry(exc=exc)
    finally:
        session.close()
