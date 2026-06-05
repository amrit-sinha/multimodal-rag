import uuid
from types import SimpleNamespace

import app.retrieval.answer as answer_mod
from app.retrieval.answer import _build_context, _build_prompt, _citations_for, extract_markers
from app.retrieval.search import Retrieved


def _fake_retrieved(page_no: int, content: str, score: float) -> Retrieved:
    chunk = SimpleNamespace(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        file_id=uuid.uuid4(),
        page_no=page_no,
        content=content,
    )
    return Retrieved(chunk=chunk, score=score)


def test_extract_markers():
    assert extract_markers("foo [1] bar [3] baz [1]") == {1, 3}
    assert extract_markers("no markers here") == set()


def test_build_context_numbers_sources():
    retrieved = [_fake_retrieved(1, "first", 0.9), _fake_retrieved(2, "second", 0.8)]
    ctx = _build_context(retrieved)
    assert "[1]" in ctx and "[2]" in ctx
    assert "page 1" in ctx and "page 2" in ctx


def test_build_prompt_includes_question():
    retrieved = [_fake_retrieved(1, "content", 0.9)]
    prompt = _build_prompt("What is X?", retrieved)
    assert "What is X?" in prompt


def test_citations_only_include_referenced_markers(monkeypatch):
    monkeypatch.setattr(answer_mod, "presigned_get_url", lambda key: "http://example/url")

    retrieved = [
        _fake_retrieved(1, "relevant passage", 0.91),
        _fake_retrieved(2, "unreferenced passage", 0.70),
    ]

    # Fake session whose .get() returns a file-like object with a storage_key.
    session = SimpleNamespace(get=lambda model, _id: SimpleNamespace(storage_key="k"))

    citations = _citations_for(session, retrieved, "The answer is grounded [1].")
    assert len(citations) == 1
    assert citations[0].marker == 1
    assert citations[0].page_no == 1
    assert citations[0].source_url == "http://example/url"
