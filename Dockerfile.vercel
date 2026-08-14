FROM node:22-alpine AS frontend-builder
WORKDIR /build/frontend
RUN corepack enable && corepack prepare pnpm@10.15.0 --activate
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DAIL_CHECKPOINT_PATH=/app/outputs/checkpoints/model_best.pt \
    DAIL_FRONTEND_DIST=/app/frontend/dist \
    DAIL_DEVICE=cpu \
    DAIL_MAX_CONCURRENT=1 \
    PORT=8000

WORKDIR /app
RUN useradd --create-home --uid 10001 dail
COPY pyproject.toml README.md config.py ./
COPY dail_llm/ ./dail_llm/
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3" && \
    python -m pip install .

COPY outputs/ ./outputs/
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist/
RUN chown -R dail:dail /app
USER dail
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python -c "import json,urllib.request; r=urllib.request.urlopen('http://127.0.0.1:$PORT/api/v1/health', timeout=3); assert json.load(r)['model_loaded']"
CMD ["python", "-m", "dail_llm.api"]
