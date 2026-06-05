import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routes import router
from app.core.db import init_db
from app.core.logging import configure_logging
from app.core.storage import ensure_bucket

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    ensure_bucket()
    logger.info("Startup complete")
    yield


app = FastAPI(title="Multimodal RAG", version="0.1.0", lifespan=lifespan)
app.add_middleware(RequestContextMiddleware)
register_exception_handlers(app)
app.include_router(router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
