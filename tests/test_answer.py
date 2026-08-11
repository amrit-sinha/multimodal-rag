import uuid
from types import SimpleNamespace

import app.retrieval.answer as answer_mod
from app.retrieval.answer import (
    Source,
    _build_context,
    _build_prompt,
    _citations_for,
    extract_markers,
)
from app.retrieval.search import Retrieved


def _text_source(page_no: int, content: str, score: float) -> Source:
    chunk = SimpleNamespace(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        file_id=uuid.uuid4(),
        page_no=page_no,
        content=content,
    )
    return Source(Retrieved(chunk=chunk, score=score), "text")


def test_extract_markers():
    assert extract_markers("foo [1] bar [3] baz [1]") == {1, 3}
    assert extract_markers("sources [1-4]") == {1, 2, 3, 4}
    assert extract_markers("no markers here") == set()


def test_build_context_numbers_sources():
    sources = [_text_source(1, "first", 0.9), _text_source(2, "second", 0.8)]
    ctx = _build_context(sources)
    assert "[1]" in ctx and "[2]" in ctx
    assert "page 1" in ctx and "page 2" in ctx


def test_build_context_marks_image_sources():
    img_chunk = SimpleNamespace(
        id=uuid.uuid4(), document_id=uuid.uuid4(), file_id=uuid.uuid4(), page_no=3, content="img"
    )
    sources = [Source(Retrieved(chunk=img_chunk, score=0.5), "image")]
    ctx = _build_context(sources)
    assert "attached image" in ctx
    assert "page 3" in ctx


def test_build_prompt_includes_question():
    sources = [_text_source(1, "content", 0.9)]
    prompt = _build_prompt("What is X?", sources)
    assert "What is X?" in prompt


def test_search_images_builds_hits(monkeypatch):
    monkeypatch.setattr(
        answer_mod, "get_image_embedder", lambda: SimpleNamespace(embed_text=lambda q: [0.1] * 512)
    )
    fake_chunk = SimpleNamespace(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        page_no=2,
        image_key="docs/img.png",
        bbox={"x0": 1, "y0": 2, "x1": 3, "y1": 4},
    )
    monkeypatch.setattr(
        answer_mod, "image_search", lambda *a, **k: [Retrieved(chunk=fake_chunk, score=0.812)]
    )
    monkeypatch.setattr(answer_mod, "presigned_get_url", lambda key: f"http://img/{key}")

    resp = answer_mod.search_images(session=None, query="a red square")
    assert resp.query == "a red square"
    assert len(resp.hits) == 1
    hit = resp.hits[0]
    assert hit.page_no == 2
    assert hit.score == 0.812
    assert hit.image_url == "http://img/docs/img.png"
    assert hit.bbox == {"x0": 1, "y0": 2, "x1": 3, "y1": 4}


def test_citations_only_include_referenced_markers(monkeypatch):
    monkeypatch.setattr(answer_mod, "presigned_get_url", lambda key: "http://example/url")

    sources = [
        _text_source(1, "relevant passage", 0.91),
        _text_source(2, "unreferenced passage", 0.70),
    ]
    session = SimpleNamespace(get=lambda model, _id: SimpleNamespace(storage_key="k"))

    citations = _citations_for(session, sources, "The answer is grounded [1].")
    assert len(citations) == 1
    assert citations[0].marker == 1
    assert citations[0].modality == "text"
    assert citations[0].page_no == 1
    assert citations[0].source_url == "http://example/url"


def test_image_citation_uses_image_key(monkeypatch):
    monkeypatch.setattr(answer_mod, "presigned_get_url", lambda key: f"http://img/{key}")
    img_chunk = SimpleNamespace(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        file_id=uuid.uuid4(),
        page_no=None,
        content="Image from cat.png",
        image_key="docs/cat.png",
    )
    sources = [Source(Retrieved(chunk=img_chunk, score=0.77), "image")]

    citations = _citations_for(session=None, sources=sources, answer="It shows a cat [1].")
    assert len(citations) == 1
    assert citations[0].modality == "image"
    assert citations[0].source_url == "http://img/docs/cat.png"
