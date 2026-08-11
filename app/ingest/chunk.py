from dataclasses import dataclass

from app.core.config import settings
from app.ingest.extract import Page


@dataclass
class TextChunk:
    content: str
    page_no: int
    char_start: int  # offset within the page text
    char_end: int


def _split_page(page: Page, size: int, overlap: int) -> list[TextChunk]:
    """Sliding character window over a single page, breaking on whitespace
    where possible so we don't cut words in half. Offsets are page-relative,
    which is what we store as provenance for citations."""
    text = page.text
    chunks: list[TextChunk] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text.rfind(" ", start + overlap, end)
            if window != -1:
                end = window
        raw = text[start:end]
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw) - len(raw.rstrip())
        content = raw.strip()
        if content:
            chunks.append(
                TextChunk(
                    content=content,
                    page_no=page.page_no,
                    char_start=start + left_trim,
                    char_end=end - right_trim,
                )
            )
        if end >= n:
            break
        start = max(end - overlap, start + 1)
        # The overlap may land inside a word; advance to the next boundary so
        # citations and snippets never start with a partial word.
        if start > 0 and not text[start - 1].isspace():
            boundary = text.find(" ", start, end)
            if boundary != -1:
                start = boundary + 1

    return chunks


def chunk_pages(pages: list[Page]) -> list[TextChunk]:
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    chunks: list[TextChunk] = []
    for page in pages:
        chunks.extend(_split_page(page, size, overlap))
    return chunks
