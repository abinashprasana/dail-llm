"""
Central configuration for the legacy src/ package (kept for reference).
The canonical package is dail_llm/.
"""
from pathlib import Path
import torch

# Base Paths
# Assuming src/config.py is in src/, so parents[1] is the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data Paths
RAW_DATA_PATH = DATA_DIR / "raw" / "dail_debates_clean.txt"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_MSG_PATH = PROCESSED_DATA_DIR / "train.txt"
VAL_MSG_PATH = PROCESSED_DATA_DIR / "val.txt"
CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"
DB_PATH = DATA_DIR / "texts.db"

# Model Checkpoints
CKPT_DIR = OUTPUTS_DIR / "checkpoints"
CKPT_PATH = CKPT_DIR / "model.pt"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# Data Processing Config
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150
VAL_RATIO = 0.1

# Model Hyperparameters
BLOCK_SIZE = 128      # Context window
EMBED_DIM = 128       # Embedding dimension
N_LAYERS = 2         # Transformer blocks
N_HEADS = 4          # Attention heads
DROPOUT = 0.1

# Training Hyperparameters
BATCH_SIZE = 32
MAX_STEPS = 2000
EVAL_EVERY = 200
LEARNING_RATE = 3e-4

# Inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
