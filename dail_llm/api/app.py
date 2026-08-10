"""FastAPI application and single-origin frontend host."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from dail_llm import __version__
from dail_llm.api.runtime import BusyError, InferenceGate, RateLimiter
from dail_llm.api.schemas import (
    AttentionRequest,
    AttentionResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from dail_llm.api.service import ModelService, PromptValidationError
from dail_llm.config import (
    DEVICE,
    MAX_CONCURRENT_INFERENCE,
    MAX_QUEUED_INFERENCE,
    PLOTS_DIR,
    PROJECT_ROOT,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
        }
        for key in ("request_id", "method", "path", "status", "elapsed_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


logger = logging.getLogger("dail_llm.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def create_app(load_model_on_start: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_service = None
        app.state.model_error = None
        if load_model_on_start:
            try:
                app.state.model_service = await anyio.to_thread.run_sync(ModelService)
                logger.info("model_loaded")
            except Exception as exc:  # Keep health endpoint available on failure.
                app.state.model_error = str(exc)
                logger.exception("model_load_failed")
        yield

    app = FastAPI(
        title="Dáil LLM API",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.gate = InferenceGate(MAX_CONCURRENT_INFERENCE, MAX_QUEUED_INFERENCE)
    app.state.rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)

    cors_origins = [
        origin.strip()
        for origin in os.getenv("DAIL_CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))[:128]
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_failed", extra={"request_id": request_id})
            raise
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self' http://localhost:8000; "
            "font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        )
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "elapsed_ms": elapsed_ms,
            },
        )
        return response

    def service(request: Request) -> ModelService:
        model_service = getattr(request.app.state, "model_service", None)
        if model_service is None:
            raise HTTPException(
                status_code=503,
                detail=getattr(request.app.state, "model_error", None) or "Model is not ready.",
            )
        return model_service

    async def enforce_rate_limit(request: Request) -> None:
        allowed, retry_after = await request.app.state.rate_limiter.allow(_client_key(request))
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Request limit reached. Try again shortly.",
                headers={"Retry-After": str(retry_after)},
            )

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health(request: Request):
        model_service = getattr(request.app.state, "model_service", None)
        error = getattr(request.app.state, "model_error", None)
        return {
            "status": "ready" if model_service is not None else "degraded",
            "version": __version__,
            "model_loaded": model_service is not None,
            "device": str(model_service.wrapper.device) if model_service else DEVICE,
            "error": error,
        }

    @app.get("/api/v1/model")
    async def model_metadata(request: Request):
        return service(request).metadata()

    @app.get("/api/v1/evaluation")
    async def evaluation(request: Request):
        try:
            return service(request).evaluation()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/v1/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest, request: Request):
        await enforce_rate_limit(request)
        try:
            async with request.app.state.gate.slot():
                return await anyio.to_thread.run_sync(
                    service(request).generate,
                    payload.prompt,
                    payload.max_new_tokens,
                    payload.temperature,
                )
        except BusyError as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": "3"},
            ) from exc
        except PromptValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/attention", response_model=AttentionResponse)
    async def attention(payload: AttentionRequest, request: Request):
        await enforce_rate_limit(request)
        try:
            async with request.app.state.gate.slot():
                return await anyio.to_thread.run_sync(
                    service(request).attention,
                    payload.prompt,
                    payload.layer,
                    payload.head,
                )
        except BusyError as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": "3"},
            ) from exc
        except PromptValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    frontend_dist = Path(
        os.getenv("DAIL_FRONTEND_DIST", str(PROJECT_ROOT / "frontend" / "dist"))
    )
    if PLOTS_DIR.exists():
        app.mount("/research", StaticFiles(directory=PLOTS_DIR), name="research")
    if (frontend_dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend(path: str):
        candidate = frontend_dist / path
        if path and candidate.is_file() and frontend_dist in candidate.resolve().parents:
            return FileResponse(candidate)
        index = frontend_dist / "index.html"
        if index.exists():
            return FileResponse(index)
        return JSONResponse(
            {"detail": "Frontend has not been built. Use the Vite development server."},
            status_code=404,
        )

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(
        "dail_llm.api.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=1,
    )


if __name__ == "__main__":
    run()
