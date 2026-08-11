import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import request_id_ctx

logger = logging.getLogger("app.errors")


def _payload(detail, request_id: str) -> dict:
    return {"detail": detail, "request_id": request_id}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exc_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(exc.detail, request_id_ctx.get()),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_payload(exc.errors(), request_id_ctx.get()),
        )

    @app.exception_handler(RuntimeError)
    async def runtime_handler(request: Request, exc: RuntimeError):
        rid = request_id_ctx.get()
        msg = str(exc)
        if msg.startswith("Ollama"):
            logger.error("Ollama error on %s %s: %s", request.method, request.url.path, msg)
            return JSONResponse(status_code=502, content=_payload(msg, rid))
        logger.exception("Unhandled RuntimeError on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content=_payload("Internal server error", rid))

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        rid = request_id_ctx.get()
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_payload("Internal server error", rid),
        )
