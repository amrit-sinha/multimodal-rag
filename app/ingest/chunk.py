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
        content = text[start:end].strip()
        if content:
            chunks.append(
                TextChunk(content=content, page_no=page.page_no, char_start=start, char_end=end)
            )
        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_pages(pages: list[Page]) -> list[TextChunk]:
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    chunks: list[TextChunk] = []
    for page in pages:
        chunks.extend(_split_page(page, size, overlap))
    return chunks
