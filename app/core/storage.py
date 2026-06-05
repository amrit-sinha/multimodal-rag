from datetime import timedelta

from minio import Minio

from app.core.config import settings


def _client(public: bool = False) -> Minio:
    """Internal client talks to the docker service; public client mints URLs
    against the host-reachable endpoint so the browser can use them.

    `region` is pinned so presigning never makes a live region-lookup call: the
    public endpoint (localhost) isn't reachable from inside the container, so
    that lookup would otherwise fail.
    """
    endpoint = settings.minio_public_endpoint if public else settings.minio_endpoint
    return Minio(
        endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
        region="us-east-1",
    )


client = _client()


def ensure_bucket() -> None:
    if not client.bucket_exists(settings.minio_bucket):
        client.make_bucket(settings.minio_bucket)


def presigned_put_url(object_key: str, expires_minutes: int = 30) -> str:
    """URL the client PUTs the raw file to, bypassing our API for the bytes."""
    return _client(public=True).presigned_put_object(
        settings.minio_bucket, object_key, expires=timedelta(minutes=expires_minutes)
    )


def presigned_get_url(object_key: str, expires_minutes: int = 60) -> str:
    return _client(public=True).presigned_get_object(
        settings.minio_bucket, object_key, expires=timedelta(minutes=expires_minutes)
    )


def download_bytes(object_key: str) -> bytes:
    response = client.get_object(settings.minio_bucket, object_key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()
