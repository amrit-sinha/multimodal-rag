import base64
import json
import re
import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import SessionLocal
from app.core.storage import download_bytes, presigned_get_url
from app.models import File
from app.models.schemas import Citation, ImageHit, ImageSearchResponse, QueryResponse
from app.models.tables import Modality
from app.providers import get_embedder, get_image_embedder, get_llm
from app.retrieval.search import Retrieved, image_search, vector_search

SYSTEM_PROMPT = (
    "You are a careful multimodal research assistant. You are given numbered "
    "sources that are either text passages or attached images. Read the text "
    "passages AND visually examine the attached images to answer the question — "
    "describe what you actually see in the images. The attached images are given "
    "in the same order as the image-numbered sources. Cite every claim with "
    "inline markers like [1] or [2] that match the source numbers (image sources "
    "are numbered too). If the question cannot be answered from the sources or "
    "images, say so plainly. Do not invent sources."
)

NO_RESULTS = "I couldn't find anything relevant in the indexed documents."


@dataclass
class Source:
    retrieved: Retrieved
    modality: str  # "text" | "image"


def _retrieve_text(session, question, document_id, top_k) -> list[Retrieved]:
    top_k = top_k or settings.top_k
    query_vector = get_embedder().embed_query(question)
    return vector_search(session, query_vector, top_k, document_id)


def _retrieve_images(session, question, document_id) -> list[Retrieved]:
    if settings.image_top_k <= 0:
        return []
    query_vector = get_image_embedder().embed_text(question)
    results = image_search(session, query_vector, settings.image_top_k, document_id)
    # For a global query, drop weak matches so unrelated images aren't attached.
    # When scoped to a document the caller asked about it explicitly, so keep them.
    if document_id is None:
        results = [r for r in results if r.score >= settings.image_score_min]
    return results


def _gather_sources(session, question, document_id, top_k, include_images) -> list[Source]:
    sources = [Source(r, Modality.TEXT) for r in _retrieve_text(session, question, document_id, top_k)]
    if include_images:
        sources += [Source(r, Modality.IMAGE) for r in _retrieve_images(session, question, document_id)]
    return sources


def _location(chunk) -> str:
    return f"page {chunk.page_no}" if chunk.page_no else "unknown location"


def _build_context(sources: list[Source]) -> str:
    blocks = []
    for i, s in enumerate(sources, start=1):
        chunk = s.retrieved.chunk
        if s.modality == Modality.IMAGE:
            blocks.append(f"[{i}] IMAGE ({_location(chunk)}) — examine the attached image to answer")
        else:
            blocks.append(f"[{i}] (source: {_location(chunk)})\n{chunk.content}")
    return "\n\n".join(blocks)


def _build_prompt(question: str, sources: list[Source]) -> str:
    context = _build_context(sources)
    return f"Sources:\n{context}\n\nQuestion: {question}\n\nAnswer with inline [n] citations:"


def _image_payload(sources: list[Source]) -> list[str]:
    """Base64-encode the retrieved images, in source order, for the vision LLM."""
    payload = []
    for s in sources:
        if s.modality == Modality.IMAGE and s.retrieved.chunk.image_key:
            data = download_bytes(s.retrieved.chunk.image_key)
            payload.append(base64.b64encode(data).decode())
    return payload


def extract_markers(answer: str) -> set[int]:
    """Pull inline citation markers like [1], [2] out of the model's answer."""
    return {int(m) for m in re.findall(r"\[(\d+)\]", answer)}


def _citations_for(session: Session, sources: list[Source], answer: str) -> list[Citation]:
    cited_markers = extract_markers(answer)
    citations: list[Citation] = []
    for i, s in enumerate(sources, start=1):
        if cited_markers and i not in cited_markers:
            continue
        chunk = s.retrieved.chunk
        if s.modality == Modality.IMAGE:
            source_url = presigned_get_url(chunk.image_key) if chunk.image_key else None
        else:
            file = session.get(File, chunk.file_id)
            source_url = presigned_get_url(file.storage_key) if file else None
        snippet = chunk.content[:300] + ("..." if len(chunk.content) > 300 else "")
        citations.append(
            Citation(
                marker=i,
                modality=str(s.modality),
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                file_id=chunk.file_id,
                page_no=chunk.page_no,
                snippet=snippet,
                score=round(s.retrieved.score, 4),
                source_url=source_url,
            )
        )
    return citations


def answer_question(
    session: Session,
    question: str,
    document_id: uuid.UUID | None = None,
    top_k: int | None = None,
    include_images: bool = True,
) -> QueryResponse:
    sources = _gather_sources(session, question, document_id, top_k, include_images)
    if not sources:
        return QueryResponse(answer=NO_RESULTS, citations=[])

    prompt = _build_prompt(question, sources)
    answer = get_llm().generate(SYSTEM_PROMPT, prompt, images=_image_payload(sources))
    citations = _citations_for(session, sources, answer)
    return QueryResponse(answer=answer, citations=citations)


def search_images(
    session: Session,
    query: str,
    document_id: uuid.UUID | None = None,
    top_k: int | None = None,
) -> ImageSearchResponse:
    """Text-to-image cross-modal search using the CLIP shared embedding space."""
    top_k = top_k or settings.top_k
    query_vector = get_image_embedder().embed_text(query)
    results = image_search(session, query_vector, top_k, document_id)

    hits: list[ImageHit] = []
    for r in results:
        image_url = presigned_get_url(r.chunk.image_key) if r.chunk.image_key else None
        hits.append(
            ImageHit(
                chunk_id=r.chunk.id,
                document_id=r.chunk.document_id,
                page_no=r.chunk.page_no,
                score=round(r.score, 4),
                image_url=image_url,
                bbox=r.chunk.bbox,
            )
        )
    return ImageSearchResponse(query=query, hits=hits)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_answer_events(
    question: str,
    document_id: uuid.UUID | None = None,
    top_k: int | None = None,
    include_images: bool = True,
):
    """SSE generator: streams answer tokens, then emits resolved citations.
    Manages its own DB session because the request scope has ended by the time
    the body streams."""
    session = SessionLocal()
    try:
        sources = _gather_sources(session, question, document_id, top_k, include_images)
        if not sources:
            yield _sse("token", {"text": NO_RESULTS})
            yield _sse("citations", {"citations": []})
            yield _sse("done", {})
            return

        prompt = _build_prompt(question, sources)
        images = _image_payload(sources)
        collected: list[str] = []
        for token in get_llm().stream(SYSTEM_PROMPT, prompt, images=images):
            collected.append(token)
            yield _sse("token", {"text": token})

        answer = "".join(collected)
        citations = _citations_for(session, sources, answer)
        yield _sse("citations", {"citations": [c.model_dump(mode="json") for c in citations]})
        yield _sse("done", {})
    finally:
        session.close()
