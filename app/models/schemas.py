import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UploadRequest(BaseModel):
    filename: str
    mime_type: str = "application/pdf"
    title: str | None = None


class UploadResponse(BaseModel):
    document_id: uuid.UUID
    file_id: uuid.UUID
    upload_url: str
    storage_key: str


class JobView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    stage: str
    status: str
    attempts: int
    error: str | None
    updated_at: datetime


class DocumentView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    status: str
    created_at: datetime


class QueryRequest(BaseModel):
    question: str
    document_id: uuid.UUID | None = None
    top_k: int | None = None


class Citation(BaseModel):
    marker: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    file_id: uuid.UUID
    page_no: int | None
    snippet: str
    score: float
    source_url: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
