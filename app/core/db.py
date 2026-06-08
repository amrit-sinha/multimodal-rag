from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a scoped DB session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# Idempotent, non-destructive migrations applied after create_all. Keeps
# existing data when the schema evolves (a lightweight stand-in for Alembic).
_MIGRATIONS = (
    "ALTER TABLE chunks ALTER COLUMN embedding DROP NOT NULL",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS image_embedding vector(%d)"
    % settings.image_embedding_dim,
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS image_key varchar(1024)",
)


def init_db() -> None:
    """Create the pgvector extension and all tables. Called on API startup."""
    # Import models so they register on Base.metadata before create_all.
    from app import models  # noqa: F401

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        for stmt in _MIGRATIONS:
            conn.execute(text(stmt))
