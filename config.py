"""
Central configuration for Dáil LLM — Irish Parliamentary Transformer.
"""
from pathlib import Path
import torch

PROJECT_NAME = "Dáil LLM — Irish Parliamentary Transformer"

# Project root is the directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
RAW_DATA_PATH = PROJECT_ROOT / "dataverse_files" / "dail_debates_clean.txt"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_MSG_PATH = PROCESSED_DATA_DIR / "train.txt"
VAL_MSG_PATH = PROCESSED_DATA_DIR / "val.txt"
TEST_MSG_PATH = PROCESSED_DATA_DIR / "test.txt"
CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"
DB_PATH = DATA_DIR / "texts.db"

# ---------------------------------------------------------------------------
# Model checkpoints / outputs
# ---------------------------------------------------------------------------
CKPT_DIR = OUTPUTS_DIR / "checkpoints"
CKPT_PATH = CKPT_DIR / "model.pt"
PLOTS_DIR = OUTPUTS_DIR / "plots"
EVAL_RESULTS_PATH = OUTPUTS_DIR / "evaluation_results.md"

# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150
VAL_RATIO = 0.05
TEST_RATIO = 0.05

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------
DATASET_NAME = "Dáil Éireann Parliamentary Debates 1919-2013 (Harvard Dataverse)"

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
BLOCK_SIZE = 256      # Context window (characters)
EMBED_DIM = 256       # Embedding dimension
N_LAYERS = 4          # Transformer blocks
N_HEADS = 8           # Attention heads
DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
MAX_STEPS = 2000
EVAL_EVERY = 200
LEARNING_RATE = 3e-4

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
