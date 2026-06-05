from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class Page:
    page_no: int  # 1-indexed
    text: str


def extract_pdf(data: bytes) -> tuple[list[Page], int]:
    """Extract text per page from a PDF. Returns (pages, page_count)."""
    pages: list[Page] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append(Page(page_no=i + 1, text=text))
        page_count = doc.page_count
    return pages, page_count
