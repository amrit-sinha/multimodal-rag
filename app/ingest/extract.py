import io
from dataclasses import dataclass

import fitz  # PyMuPDF
from PIL import Image


@dataclass
class Page:
    page_no: int  # 1-indexed
    text: str


@dataclass
class ExtractedImage:
    page_no: int | None
    image: Image.Image
    data: bytes  # PNG-encoded bytes, for storage + viewing
    bbox: dict | None


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


# Skip tiny images (icons, bullets, separators) that aren't worth indexing.
_MIN_IMAGE_PIXELS = 64


def extract_pdf_images(data: bytes) -> list[ExtractedImage]:
    """Pull embedded raster images out of each PDF page, keeping page number and
    on-page bounding box as provenance."""
    out: list[ExtractedImage] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            for img in page.get_images(full=True):
                xref = img[0]
                base = doc.extract_image(xref)
                try:
                    pil = Image.open(io.BytesIO(base["image"])).convert("RGB")
                except Exception:
                    continue
                if pil.width < _MIN_IMAGE_PIXELS or pil.height < _MIN_IMAGE_PIXELS:
                    continue

                rects = page.get_image_rects(xref)
                bbox = None
                if rects:
                    r = rects[0]
                    bbox = {"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1}

                out.append(
                    ExtractedImage(
                        page_no=i + 1,
                        image=pil,
                        data=_to_png(pil),
                        bbox=bbox,
                    )
                )
    return out


def load_image(data: bytes) -> ExtractedImage:
    """Wrap a directly-uploaded image file as a single extractable image."""
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    return ExtractedImage(page_no=None, image=pil, data=_to_png(pil), bbox=None)


def _to_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
