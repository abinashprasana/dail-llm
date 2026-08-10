"""
Script to prepare the Dáil Debates dataset.
Reads raw text, cleans it, chunks it, and saves to disk/DB.
"""
import json
import re
import sqlite3
import logging
from pathlib import Path
from typing import List, Tuple

from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, DB_PATH,
    CHUNKS_PATH, TRAIN_MSG_PATH, VAL_MSG_PATH,
    CHUNK_CHARS, CHUNK_OVERLAP, VAL_RATIO
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

START_RE = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*", re.IGNORECASE)
END_RE = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*", re.IGNORECASE)


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    return path.read_text(encoding="utf-8", errors="replace")


def strip_gutenberg_header_footer(text: str) -> str:
    lines = text.splitlines()
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if START_RE.search(line):
            start_idx = i + 1
            break

    for i, line in enumerate(lines):
        if END_RE.search(line):
            end_idx = i
            break

    # Fallback: if markers missing, just return original
    if start_idx is None or end_idx is None or start_idx >= end_idx:
        logger.warning("Gutenberg markers not found, using full text.")
        return text

    return "\n".join(lines[start_idx:end_idx]).strip()


def normalize(text: str) -> str:
    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse trailing spaces
    text = "\n".join([ln.rstrip() for ln in text.splitlines()])
    # collapse too many blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # fix weird spaced punctuation (very light)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_chars: int = 1000, overlap: int = 150) -> List[str]:
    """
    Create overlapping chunks by character count.
    Keeps paragraphs somewhat by chunking on boundaries when possible.
    """
    if chunk_chars <= overlap:
        raise ValueError("chunk_chars must be > overlap")

    chunks = []
    i = 0
    n = len(text)

    while i < n:
        j = min(i + chunk_chars, n)

        # try to end on a newline boundary to keep paragraphs nicer
        boundary = text.rfind("\n", i, j)
        if boundary != -1 and boundary > i + int(chunk_chars * 0.6):
            j = boundary

        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)

        i = max(j - overlap, i + 1)

    return chunks


def train_val_split(text: str, val_ratio: float = 0.1) -> Tuple[str, str]:
    n = len(text)
    cut = int(n * (1 - val_ratio))
    train = text[:cut].strip()
    val = text[cut:].strip()
    return train, val


def build_sqlite(chunks: List[str], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path.as_posix())
    try:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL
            )
        """)
        cur.execute("DELETE FROM chunks")
        cur.executemany(
            "INSERT INTO chunks(chunk_index, text) VALUES (?, ?)",
            [(i, c) for i, c in enumerate(chunks)]
        )
        con.commit()
    finally:
        con.close()


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading raw data from {RAW_DATA_PATH}")
    raw = read_text(RAW_DATA_PATH)
    core = strip_gutenberg_header_footer(raw)
    clean = normalize(core)

    clean_path = PROCESSED_DATA_DIR / "dail_clean.txt"
    clean_path.write_text(clean, encoding="utf-8")
    logger.info(f"Saved clean text to {clean_path}")

    # Create chunks for RAG
    chunks = chunk_text(clean, chunk_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP)

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks):
            f.write(json.dumps({"chunk_index": i, "text": ch}, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")

    # Build SQLite DB for RAG
    build_sqlite(chunks, DB_PATH)
    logger.info(f"Built SQLite database at {DB_PATH}")

    # Create Train/Val split
    train, val = train_val_split(clean, val_ratio=VAL_RATIO)
    TRAIN_MSG_PATH.write_text(train, encoding="utf-8")
    VAL_MSG_PATH.write_text(val, encoding="utf-8")
    logger.info(f"Saved split: Train ({len(train)} chars) / Val ({len(val)} chars)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Failed to process dataset: {e}")
        raise
