from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres
    postgres_user: str = "rag"
    postgres_password: str = "rag"
    postgres_db: str = "rag"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # Redis / Celery
    redis_url: str = "redis://redis:6379/0"

    # MinIO
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "rag-uploads"
    minio_secure: bool = False
    minio_public_endpoint: str = "localhost:9000"

    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    # CLIP shares a text+image embedding space, separate from the bge text space.
    clip_model: str = "clip-ViT-B-32"
    image_embedding_dim: int = 512
    ollama_base_url: str = "http://host.docker.internal:11434"
    # Multimodal (vision) model so the LLM can actually "see" retrieved images.
    llm_model: str = "gemma4:e2b"
    # Keep the model resident in Ollama between calls so large models don't pay a
    # multi-minute cold-load on every request.
    ollama_keep_alive: str = "10m"
    # Ollama GPU layers to offload. Set to 0 on 4GB GPUs when large vision models
    # crash CUDA (stack-buffer overrun). Unset = Ollama default (use GPU).
    ollama_num_gpu: int | None = None

    # Retrieval / chunking
    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 150
    # How many images to feed the vision LLM per query, and the minimum CLIP
    # similarity required to attach one during an unscoped (global) query.
    image_top_k: int = 2
    image_score_min: float = 0.22

    # Security / ops
    api_key: str = ""  # empty disables auth (dev convenience)
    rate_limit_per_minute: int = 60  # <= 0 disables
    log_level: str = "INFO"
    max_upload_mb: int = 50
    allowed_mime_types: str = "application/pdf,image/png,image/jpeg,image/webp"

    @property
    def allowed_mimes(self) -> set[str]:
        return {m.strip() for m in self.allowed_mime_types.split(",") if m.strip()}

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
