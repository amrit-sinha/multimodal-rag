import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models import Chunk


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


def vector_search(
    session: Session,
    query_vector: list[float],
    top_k: int,
    document_id: uuid.UUID | None = None,
) -> list[Retrieved]:
    """Dense retrieval over text chunks via pgvector cosine distance (bge space)."""
    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
    query = session.query(Chunk, distance).filter(Chunk.embedding.isnot(None))
    if document_id is not None:
        query = query.filter(Chunk.document_id == document_id)
    rows = query.order_by(distance).limit(top_k).all()
    return [Retrieved(chunk=chunk, score=1.0 - float(dist)) for chunk, dist in rows]


def image_search(
    session: Session,
    query_vector: list[float],
    top_k: int,
    document_id: uuid.UUID | None = None,
) -> list[Retrieved]:
    """Cross-modal retrieval: a CLIP-encoded text query searches image chunks in
    the shared CLIP space, ranking images by visual-semantic similarity."""
    distance = Chunk.image_embedding.cosine_distance(query_vector).label("distance")
    query = session.query(Chunk, distance).filter(Chunk.image_embedding.isnot(None))
    if document_id is not None:
        query = query.filter(Chunk.document_id == document_id)
    rows = query.order_by(distance).limit(top_k).all()
    return [Retrieved(chunk=chunk, score=1.0 - float(dist)) for chunk, dist in rows]
