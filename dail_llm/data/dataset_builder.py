"""
Prepare the Dáil Debates dataset for training.

Reads the cleaned plain-text file produced by extract_dail.py,
normalises whitespace, builds overlapping RAG chunks, creates a
90 / 5 / 5 train / val / test split, and persists everything to disk.
"""
import json
import re
import sqlite3
import logging
from pathlib import Path
from typing import List, Tuple

from config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, DB_PATH,
    CHUNKS_PATH, TRAIN_MSG_PATH, VAL_MSG_PATH, TEST_MSG_PATH,
    CHUNK_CHARS, CHUNK_OVERLAP, VAL_RATIO, TEST_RATIO,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    return path.read_text(encoding="utf-8", errors="replace")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(ln.rstrip() for ln in text.splitlines())
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_chars: int = 1000, overlap: int = 150) -> List[str]:
    if chunk_chars <= overlap:
        raise ValueError("chunk_chars must be > overlap")
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(i + chunk_chars, n)
        boundary = text.rfind("\n", i, j)
        if boundary != -1 and boundary > i + int(chunk_chars * 0.6):
            j = boundary
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i = max(j - overlap, i + 1)
    return chunks


def train_val_test_split(text: str, val_ratio: float, test_ratio: float) -> Tuple[str, str, str]:
    n = len(text)
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))
    return text[:train_end].strip(), text[train_end:val_end].strip(), text[val_end:].strip()


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
    clean = normalize(raw)

    clean_path = PROCESSED_DATA_DIR / "dail_clean.txt"
    clean_path.write_text(clean, encoding="utf-8")
    logger.info(f"Saved clean text ({len(clean):,} chars) to {clean_path}")

    chunks = chunk_text(clean, chunk_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks):
            f.write(json.dumps({"chunk_index": i, "text": ch}, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(chunks)} RAG chunks to {CHUNKS_PATH}")

    build_sqlite(chunks, DB_PATH)
    logger.info(f"Built SQLite database at {DB_PATH}")

    train, val, test = train_val_test_split(clean, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)
    TRAIN_MSG_PATH.write_text(train, encoding="utf-8")
    VAL_MSG_PATH.write_text(val, encoding="utf-8")
    TEST_MSG_PATH.write_text(test, encoding="utf-8")
    logger.info(
        f"Split sizes — Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,} chars"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise
