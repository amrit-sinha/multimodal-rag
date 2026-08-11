from app.ingest.chunk import _split_page, chunk_pages
from app.ingest.extract import Page


def test_chunks_preserve_page_numbers():
    pages = [Page(page_no=1, text="alpha " * 400), Page(page_no=2, text="beta " * 400)]
    chunks = chunk_pages(pages)
    pages_seen = {c.page_no for c in chunks}
    assert pages_seen == {1, 2}


def test_chunks_are_non_empty_and_bounded():
    pages = [Page(page_no=1, text="word " * 1000)]
    chunks = chunk_pages(pages)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c.content.strip()
        assert c.char_start < c.char_end


def test_short_page_produces_single_chunk():
    pages = [Page(page_no=1, text="just a short sentence.")]
    chunks = chunk_pages(pages)
    assert len(chunks) == 1
    assert chunks[0].content == "just a short sentence."
    assert chunks[0].page_no == 1


def test_offsets_are_within_page_text():
    text = "lorem ipsum dolor sit amet " * 200
    pages = [Page(page_no=3, text=text)]
    chunks = chunk_pages(pages)
    for c in chunks:
        assert 0 <= c.char_start < len(text)
        assert c.char_end <= len(text)


def test_overlap_starts_on_word_boundary_and_offsets_match():
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    chunks = _split_page(Page(page_no=1, text=text), size=24, overlap=8)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.char_start == 0 or text[chunk.char_start - 1].isspace()
        assert text[chunk.char_start:chunk.char_end] == chunk.content
