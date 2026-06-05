import logging
import time

import redis
from fastapi import Header, HTTPException, Request

from app.core.config import settings

logger = logging.getLogger("app.security")

# Single shared client; redis-py connection pool is thread-safe.
_redis = redis.Redis.from_url(settings.redis_url)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """API-key gate. Disabled when API_KEY is unset, so local dev stays easy."""
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def rate_limit(request: Request) -> None:
    """Fixed-window per-client limiter backed by Redis. Fails open if Redis is
    unavailable so a limiter outage never takes down the API."""
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return

    client_ip = request.client.host if request.client else "unknown"
    window = int(time.time() // 60)
    key = f"ratelimit:{client_ip}:{window}"

    try:
        count = _redis.incr(key)
        if count == 1:
            _redis.expire(key, 60)
    except redis.RedisError:
        logger.warning("Rate limiter unavailable; allowing request")
        return

    if count > limit:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again in a minute.",
            headers={"Retry-After": "60"},
        )
