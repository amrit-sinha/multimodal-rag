from celery import Celery

from app.core.config import settings

celery = Celery(
    "multimodal_rag",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.ingest.tasks"],
)

celery.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
)
