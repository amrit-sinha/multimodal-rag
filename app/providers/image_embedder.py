from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

from app.core.config import settings

if TYPE_CHECKING:
    from PIL.Image import Image


class ImageEmbedder(Protocol):
    """CLIP-style encoder that maps both images and text into one shared vector
    space, so a text query can retrieve semantically similar images."""

    dim: int

    def embed_images(self, images: list["Image"]) -> list[list[float]]: ...

    def embed_text(self, text: str) -> list[float]: ...


class CLIPEmbedder:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name or settings.clip_model)
        self.dim = settings.image_embedding_dim

    def embed_images(self, images: list["Image"]) -> list[list[float]]:
        vectors = self.model.encode(images, normalize_embeddings=True, batch_size=16)
        return [v.tolist() for v in vectors]

    def embed_text(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()


@lru_cache
def get_image_embedder() -> ImageEmbedder:
    return CLIPEmbedder()
