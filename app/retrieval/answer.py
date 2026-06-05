import json
import re
import uuid

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import SessionLocal
from app.core.storage import presigned_get_url
from app.models import File
from app.models.schemas import Citation, QueryResponse
from app.providers import get_embedder, get_llm
from app.retrieval.search import Retrieved, vector_search

SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the user's question using ONLY "
    "the numbered sources provided. Cite every claim with inline markers like [1] "
    "or [2] that refer to the source numbers. If the sources do not contain the "
    "answer, say so plainly. Do not invent facts or sources."
)

NO_RESULTS = "I couldn't find anything relevant in the indexed documents."


def _build_context(retrieved: list[Retrieved]) -> str:
    blocks = []
    for i, r in enumerate(retrieved, start=1):
        loc = f"page {r.chunk.page_no}" if r.chunk.page_no else "unknown location"
        blocks.append(f"[{i}] (source: {loc})\n{r.chunk.content}")
    return "\n\n".join(blocks)


def _retrieve(
    session: Session,
    question: str,
    document_id: uuid.UUID | None,
    top_k: int | None,
) -> list[Retrieved]:
    top_k = top_k or settings.top_k
    query_vector = get_embedder().embed_query(question)
    return vector_search(session, query_vector, top_k, document_id)


def _build_prompt(question: str, retrieved: list[Retrieved]) -> str:
    context = _build_context(retrieved)
    return f"Sources:\n{context}\n\nQuestion: {question}\n\nAnswer with inline [n] citations:"


def extract_markers(answer: str) -> set[int]:
    """Pull inline citation markers like [1], [2] out of the model's answer."""
    return {int(m) for m in re.findall(r"\[(\d+)\]", answer)}


def _citations_for(session: Session, retrieved: list[Retrieved], answer: str) -> list[Citation]:
    cited_markers = extract_markers(answer)
    citations: list[Citation] = []
    for i, r in enumerate(retrieved, start=1):
        if cited_markers and i not in cited_markers:
            continue
        file = session.get(File, r.chunk.file_id)
        source_url = presigned_get_url(file.storage_key) if file else None
        snippet = r.chunk.content[:300] + ("..." if len(r.chunk.content) > 300 else "")
        citations.append(
            Citation(
                marker=i,
                chunk_id=r.chunk.id,
                document_id=r.chunk.document_id,
                file_id=r.chunk.file_id,
                page_no=r.chunk.page_no,
                snippet=snippet,
                score=round(r.score, 4),
                source_url=source_url,
            )
        )
    return citations


def answer_question(
    session: Session,
    question: str,
    document_id: uuid.UUID | None = None,
    top_k: int | None = None,
) -> QueryResponse:
    retrieved = _retrieve(session, question, document_id, top_k)
    if not retrieved:
        return QueryResponse(answer=NO_RESULTS, citations=[])

    prompt = _build_prompt(question, retrieved)
    answer = get_llm().generate(SYSTEM_PROMPT, prompt)
    citations = _citations_for(session, retrieved, answer)
    return QueryResponse(answer=answer, citations=citations)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_answer_events(
    question: str,
    document_id: uuid.UUID | None = None,
    top_k: int | None = None,
):
    """Server-Sent Events generator: streams answer tokens as they're produced,
    then emits the resolved citations. Manages its own DB session because the
    request scope has ended by the time the body streams."""
    session = SessionLocal()
    try:
        retrieved = _retrieve(session, question, document_id, top_k)
        if not retrieved:
            yield _sse("token", {"text": NO_RESULTS})
            yield _sse("citations", {"citations": []})
            yield _sse("done", {})
            return

        prompt = _build_prompt(question, retrieved)
        collected: list[str] = []
        for token in get_llm().stream(SYSTEM_PROMPT, prompt):
            collected.append(token)
            yield _sse("token", {"text": token})

        answer = "".join(collected)
        citations = _citations_for(session, retrieved, answer)
        yield _sse("citations", {"citations": [c.model_dump(mode="json") for c in citations]})
        yield _sse("done", {})
    finally:
        session.close()
