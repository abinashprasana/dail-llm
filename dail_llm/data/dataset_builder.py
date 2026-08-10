"""
Prepare the Dáil Debates dataset for training.

Reads the cleaned plain-text file produced by extract_dail.py,
normalises whitespace, builds overlapping retrieval chunks, creates a
90 / 5 / 5 train / val / test split, and persists everything to disk.
"""
import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path

from dail_llm.config import (
    CHUNK_CHARS,
    CHUNK_OVERLAP,
    CHUNKS_PATH,
    DATASET_CITATION,
    DATASET_DOI,
    DATASET_MANIFEST_PATH,
    DATASET_NAME,
    DB_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_PATH,
    TEST_MSG_PATH,
    TEST_RATIO,
    TRAIN_MSG_PATH,
    VAL_MSG_PATH,
    VAL_RATIO,
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


def chunk_text(text: str, chunk_chars: int = 1000, overlap: int = 150) -> list[str]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if chunk_chars <= overlap:
        raise ValueError("chunk_chars must be greater than overlap")
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(i + chunk_chars, n)
        boundary = text.rfind("\n", i, j)
        if boundary != -1 and boundary > i + int(chunk_chars * 0.6):
            j = boundary
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(j - overlap, i + 1)
    return chunks


def train_val_test_split(text: str, val_ratio: float, test_ratio: float) -> tuple[str, str, str]:
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("validation and test ratios must be in [0, 1)")
    if val_ratio + test_ratio >= 1:
        raise ValueError("validation and test ratios must sum to less than 1")
    n = len(text)
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))
    return text[:train_end].strip(), text[train_end:val_end].strip(), text[val_end:].strip()


def build_sqlite(chunks: list[str], db_path: Path) -> None:
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


def rebuild_chunk_assets(clean: str) -> list[str]:
    """Rebuild the derived retrieval files without touching training splits."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    chunks = chunk_text(clean, chunk_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP)
    with CHUNKS_PATH.open("w", encoding="utf-8") as file_handle:
        for index, chunk in enumerate(chunks):
            payload = {"chunk_index": index, "text": chunk}
            file_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    build_sqlite(chunks, DB_PATH)
    return chunks


def rebuild_chunks_and_manifest() -> None:
    """Refresh chunks, SQLite, and manifest facts while preserving split files."""
    clean = normalize(read_text(RAW_DATA_PATH))
    chunks = rebuild_chunk_assets(clean)
    train = read_text(TRAIN_MSG_PATH)
    val = read_text(VAL_MSG_PATH)
    test = read_text(TEST_MSG_PATH)
    _write_manifest(clean, chunks, train, val, test)
    logger.info("Rebuilt %s chunks and SQLite without changing training splits", len(chunks))


def _write_manifest(clean: str, chunks: list[str], train: str, val: str, test: str) -> None:
    manifest: dict = {}
    if DATASET_MANIFEST_PATH.exists():
        try:
            manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}

    manifest.update({
        "schema_version": 1,
        "dataset": {
            "name": DATASET_NAME,
            "citation": DATASET_CITATION,
            "doi": DATASET_DOI,
            "full_date_range": "1919-2013",
        },
        "corpus": {
            **manifest.get("corpus", {}),
            "clean_bytes": len(clean.encode("utf-8")),
            "clean_characters": len(clean),
            "sha256": hashlib.sha256(clean.encode("utf-8")).hexdigest(),
        },
        "processing": {
            "chunk_characters": CHUNK_CHARS,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunk_count": len(chunks),
            "validation_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "split_characters": {
                "train": len(train),
                "validation": len(val),
                "test": len(test),
            },
        },
    })
    DATASET_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading raw data from {RAW_DATA_PATH}")
    raw = read_text(RAW_DATA_PATH)
    clean = normalize(raw)

    clean_path = PROCESSED_DATA_DIR / "dail_clean.txt"
    clean_path.write_text(clean, encoding="utf-8")
    logger.info(f"Saved clean text ({len(clean):,} chars) to {clean_path}")

    chunks = rebuild_chunk_assets(clean)
    logger.info(f"Saved {len(chunks)} retrieval chunks to {CHUNKS_PATH}")
    logger.info(f"Built SQLite database at {DB_PATH}")

    train, val, test = train_val_test_split(clean, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)
    TRAIN_MSG_PATH.write_text(train, encoding="utf-8")
    VAL_MSG_PATH.write_text(val, encoding="utf-8")
    TEST_MSG_PATH.write_text(test, encoding="utf-8")
    _write_manifest(clean, chunks, train, val, test)
    logger.info(
        f"Split sizes — Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,} chars"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise
