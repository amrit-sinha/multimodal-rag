import uuid
from datetime import datetime, timezone
from enum import StrEnum

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.core.db import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class DocStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class JobStage(StrEnum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    DONE = "done"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    owner: Mapped[str] = mapped_column(String(255), default="demo")
    title: Mapped[str] = mapped_column(String(512))
    status: Mapped[DocStatus] = mapped_column(String(32), default=DocStatus.PENDING)
    created_at: Mapped[datetime] = mapped_column(default=_now, server_default=func.now())

    files: Mapped[list["File"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    jobs: Mapped[list["IngestJob"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class File(Base):
    __tablename__ = "files"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    storage_key: Mapped[str] = mapped_column(String(1024))
    filename: Mapped[str] = mapped_column(String(512))
    mime_type: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="files")


class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    stage: Mapped[JobStage] = mapped_column(String(32), default=JobStage.QUEUED)
    status: Mapped[JobStatus] = mapped_column(String(32), default=JobStatus.PENDING)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_now, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=_now, onupdate=_now, server_default=func.now())

    document: Mapped["Document"] = relationship(back_populates="jobs")


class Chunk(Base):
    """A single retrievable unit that carries full provenance back to its
    source location for citations."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    file_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"), index=True)
    modality: Mapped[Modality] = mapped_column(String(16), default=Modality.TEXT)
    content: Mapped[str] = mapped_column(Text)

    # Provenance: where this chunk came from inside the source file.
    page_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Storage key of a viewable image (the upload itself, or a crop extracted
    # from a PDF page). Only set for image chunks.
    image_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Text chunks carry a bge embedding; image chunks carry a CLIP embedding.
    # Each lives in its own vector space, so they're stored in separate columns.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dim), nullable=True
    )
    image_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.image_embedding_dim), nullable=True
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")
