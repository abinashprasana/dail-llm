"""Canonical configuration for Dáil LLM."""
from __future__ import annotations

import os
from pathlib import Path

import torch

PROJECT_NAME = "Dáil LLM"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data
RAW_DATA_PATH = PROJECT_ROOT / "dataverse_files" / "dail_debates_clean.txt"
SOURCE_DATA_PATH = PROJECT_ROOT / "dataverse_files" / "Dail_debates_1919-2013.tab"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_MSG_PATH = PROCESSED_DATA_DIR / "train.txt"
VAL_MSG_PATH = PROCESSED_DATA_DIR / "val.txt"
TEST_MSG_PATH = PROCESSED_DATA_DIR / "test.txt"
CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"
DB_PATH = DATA_DIR / "texts.db"

# Outputs
CKPT_DIR = OUTPUTS_DIR / "checkpoints"
CKPT_PATH = Path(os.getenv("DAIL_CHECKPOINT_PATH", str(CKPT_DIR / "model_best.pt")))
PLOTS_DIR = OUTPUTS_DIR / "plots"
EVAL_RESULTS_PATH = OUTPUTS_DIR / "evaluation_results.md"
EVAL_RESULTS_JSON_PATH = OUTPUTS_DIR / "evaluation_results.json"
DATASET_MANIFEST_PATH = OUTPUTS_DIR / "dataset_manifest.json"
TRAINING_HISTORY_PATH = OUTPUTS_DIR / "training_history.json"

# Data processing
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150
VAL_RATIO = 0.05
TEST_RATIO = 0.05
DATASET_NAME = "Dáil Éireann Parliamentary Debates 1919-2013 (Harvard Dataverse)"
DATASET_CITATION = (
    "Proksch, S.O. and Slapin, J.B. (2010). Database of Parliamentary "
    "Speeches in Ireland, 1919-2013. Harvard Dataverse."
)
DATASET_DOI = "https://doi.org/10.7910/DVN/6MZN76"

# Model
BLOCK_SIZE = 256
EMBED_DIM = 256
N_LAYERS = 4
N_HEADS = 8
DROPOUT = 0.1

# Training
BATCH_SIZE = 32
MAX_STEPS = 2000
EVAL_EVERY = 200
LEARNING_RATE = 3e-4

# Runtime
_requested_device = os.getenv("DAIL_DEVICE", "auto").lower()
if _requested_device == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
elif _requested_device == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"
else:
    DEVICE = _requested_device

MAX_CONCURRENT_INFERENCE = max(1, int(os.getenv("DAIL_MAX_CONCURRENT", "1")))
MAX_QUEUED_INFERENCE = max(0, int(os.getenv("DAIL_MAX_QUEUED", "2")))
RATE_LIMIT_REQUESTS = max(1, int(os.getenv("DAIL_RATE_LIMIT_REQUESTS", "5")))
RATE_LIMIT_WINDOW_SECONDS = max(1, int(os.getenv("DAIL_RATE_LIMIT_WINDOW", "60")))
