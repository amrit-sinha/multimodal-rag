from functools import lru_cache
from typing import Protocol

from app.core.config import settings


class Embedder(Protocol):
    """Anything that turns text into vectors. Swap the impl (local model vs.
    hosted API) without touching the ingestion or retrieval code."""

    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class BGEEmbedder:
    """Local sentence-transformers embedder (BAAI/bge-small-en-v1.5).

    BGE recommends prefixing queries with an instruction for retrieval; passages
    are embedded as-is. Embeddings are L2-normalized so cosine == dot product.
    """

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name or settings.embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            self.QUERY_INSTRUCTION + text, normalize_embeddings=True
        )
        return vector.tolist()


@lru_cache
def get_embedder() -> Embedder:
    return BGEEmbedder()
