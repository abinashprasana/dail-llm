# Dáil LLM

Dáil LLM is a compact, character-level transformer trained on a prepared extract of Dáil Éireann debates. The application pairs the existing PyTorch checkpoint with a FastAPI service and a React interface for text generation, held-out evaluation, and causal-attention inspection.

The public interface is factual by design. Model figures come from the checkpoint, corpus figures come from `outputs/dataset_manifest.json`, and evaluation figures come from `outputs/evaluation_results.json`.

## Run the complete product

Docker starts the compiled interface and API on one origin:

```bash
docker compose up --build
```

Open `http://localhost:8000`. The container uses one Uvicorn worker so the checkpoint is loaded only once.

## Verified model record

| Property | Value |
|---|---:|
| Checkpoint | `model_best.pt` |
| Parameters | 3,271,168 |
| Decoder layers | 4 |
| Attention heads | 8 |
| Embedding width | 256 |
| Context window | 256 characters |
| Vocabulary | Character-level |

The latest deterministic evaluation of the existing checkpoint reports:

| Held-out metric | Value |
|---|---:|
| Cross-entropy | 1.4030 |
| Perplexity | 4.0675 |
| Bits per character | 2.0242 |
| Next-character accuracy | 58.67% |

BLEU is deliberately not reported. Comparing an open-ended continuation with a randomly chosen reference does not provide useful evidence for this model. Repeated word-trigram rate is recorded for each deterministic sample, and unavailable metrics are stored as `null` rather than zero.

## Corpus provenance

The active checkpoint uses a 6.02 MB normalized corpus containing 6,295,637 characters. The extraction accepted 9,080 speeches dated from 15 February to 25 April 1950, then divided the prepared text into 90% training, 5% validation, and 5% test material. These are the measured facts for this checkpoint; they are not claims about the full source archive.

Source: Proksch, S.O. and Slapin, J.B. (2010), *Database of Parliamentary Speeches in Ireland, 1919–2013*, Harvard Dataverse. [https://doi.org/10.7910/DVN/6MZN76](https://doi.org/10.7910/DVN/6MZN76)

The 3.44 GB source file is excluded from the image and repository artifacts. The deployable image contains only the selected checkpoint, manifest, evaluation JSON, research figures, API code, and compiled frontend.

## Product surfaces

- `/` explains the model, corpus record, architecture, and held-out evidence. Its desktop hero uses a lazy-loaded 3D parliamentary chamber. Mobile, reduced-motion, low-power, and WebGL-failure paths use a designed SVG scene.
- `/lab` provides generation, evaluation, and attention tools. Generation remains non-streaming at the API boundary; the returned passage is revealed progressively in the browser.
- `/api/docs` exposes the versioned HTTP contract.

The interface uses locally bundled Newsreader and Manrope fonts, native scrolling, visible focus states, and motion that follows the user's reduced-motion setting.

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Readiness, service version, model state, and device |
| `GET` | `/api/v1/model` | Checkpoint identity, architecture, parameters, and corpus manifest |
| `POST` | `/api/v1/generate` | Generate 50–500 new characters from a supported prompt |
| `GET` | `/api/v1/evaluation` | Structured held-out metrics and deterministic samples |
| `POST` | `/api/v1/attention` | Character labels and attention matrices for one or every head |

Generation and attention share a bounded inference gate. One request runs at a time by default, two may wait, and additional requests receive HTTP 429. Each client IP may make five generation or attention requests per minute unless the limits are changed.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DAIL_CHECKPOINT_PATH` | `outputs/checkpoints/model_best.pt` | Checkpoint loaded during application startup |
| `DAIL_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or another PyTorch device |
| `DAIL_MAX_CONCURRENT` | `1` | Simultaneous inference jobs |
| `DAIL_MAX_QUEUED` | `2` | Requests allowed to wait for inference |
| `DAIL_RATE_LIMIT_REQUESTS` | `5` | Per-IP requests in each rate window |
| `DAIL_RATE_LIMIT_WINDOW` | `60` | Rate window in seconds |
| `DAIL_CORS_ORIGINS` | `http://localhost:5173` | Development origins, comma-separated |
| `PORT` | `8000` | HTTP port |

Copy `.env.example` when running outside Docker.

## Local development

Python 3.12 and Node 22 are the supported runtimes.

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3"
python -m pip install -e ".[dev]"
```

Start the API:

```bash
uvicorn dail_llm.api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal:

```bash
cd frontend
corepack enable
pnpm install
pnpm dev
```

Open `http://localhost:5173`. Vite proxies API and research-figure requests to FastAPI.

Research and data utilities have separate dependencies:

```bash
python -m pip install -e ".[research]"
python -m dail_llm.data.extract_dail
python -m dail_llm.data.dataset_builder
python -m dail_llm.eval.evaluate
```

The canonical configuration is `dail_llm/config.py`. The root `config.py` remains only as a compatibility import for older commands. The previous Streamlit surface is retained in `legacy/streamlit_app.py` and is not installed in production.

## Checks

```bash
python -m ruff check dail_llm tests
python -m pytest
cd frontend
pnpm lint
pnpm typecheck
pnpm test
pnpm build
pnpm exec playwright install chromium
pnpm test:e2e
```

Playwright covers 1440×900, 1024×768, 768×1024, and 390×844 viewports, including compact navigation, overflow, reduced motion, and automated accessibility checks. GitHub Actions also constructs the Docker image.

## Repository map

```text
dail_llm/api/       FastAPI application, model service, limits, and schemas
dail_llm/data/      extraction, chunking, split creation, and tokenizer
dail_llm/eval/      held-out metrics and deterministic evaluation record
dail_llm/model/     transformer architecture, training, and CLI generation
frontend/           React, TypeScript, Vite, Tailwind, Motion, and R3F
outputs/            checkpoint, verified manifests, evaluation, and figures
legacy/             superseded Streamlit interface
tests/              backend unit, contract, reliability, and checkpoint tests
```

## Scope

Dáil LLM predicts characters; it does not retrieve sources, verify statements, or represent the views of Dáil Éireann. Its short context and compact architecture limit coherence. Generated passages should be read as model behavior, not parliamentary records or factual claims.
