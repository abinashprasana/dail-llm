"""
Extract and clean English speeches from the Dáil Debates dataset.

Usage:
    python -m dail_llm.data.extract_dail
    # or directly:
    python dail_llm/data/extract_dail.py

Reads:  dataverse_files/Dail_debates_1919-2013.tab  (3.44 GB, tab-delimited)
Writes: dataverse_files/dail_debates_clean.txt       (~6 MB of clean English text)

Dependencies: ftfy  (pip install ftfy)
"""

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from dail_llm.config import (
    DATASET_CITATION,
    DATASET_DOI,
    DATASET_MANIFEST_PATH,
    DATASET_NAME,
)

try:
    import ftfy
except ImportError:
    print("ERROR: ftfy is not installed. Run: pip install ftfy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths (resolved relative to project root, regardless of where script runs)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
# dail_llm/data/extract_dail.py  -> parent.parent.parent = project root
PROJECT_ROOT = _HERE.parent.parent.parent
TAB_FILE = PROJECT_ROOT / "dataverse_files" / "Dail_debates_1919-2013.tab"
OUT_FILE = PROJECT_ROOT / "dataverse_files" / "dail_debates_clean.txt"

# ---------------------------------------------------------------------------
# Extraction settings
# ---------------------------------------------------------------------------
TARGET_BYTES = 6 * 1024 * 1024   # Stop once we have 6 MB of clean text
MIN_SPEECH_LEN = 50              # Skip speeches shorter than this
MAX_NON_ASCII_RATIO = 0.40       # Skip speeches with >40% non-ASCII characters
CUTOFF_YEAR = 1950               # Only keep speeches from this year onwards
PROGRESS_EVERY = 100_000         # Print progress every N rows


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 1.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text)


def _clean_speech(text: str) -> str:
    text = text.strip()
    # Replace internal newlines with a space
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text


def extract(
    tab_file: Path = TAB_FILE,
    out_file: Path = OUT_FILE,
    manifest_path: Path = DATASET_MANIFEST_PATH,
) -> None:
    if not tab_file.exists():
        print(f"ERROR: Source file not found:\n  {tab_file}")
        sys.exit(1)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows_processed = 0
    speeches_kept = 0
    bytes_written = 0
    date_min: str | None = None
    date_max: str | None = None
    speaker_counts: Counter = Counter()
    last_speech_sample = ""

    print(f"Reading: {tab_file}")
    print(f"Writing: {out_file}")
    print(f"Target:  {TARGET_BYTES / 1024 / 1024:.1f} MB\n")

    with (
        tab_file.open(encoding="utf-8", errors="replace") as fh,
        out_file.open("w", encoding="utf-8") as out,
    ):
        reader = csv.reader(fh, delimiter="\t")

        # Read and validate header
        try:
            header = next(reader)
        except StopIteration:
            print("ERROR: File is empty.")
            return

        # Locate column indices robustly
        col = {name: i for i, name in enumerate(header)}
        required = {"speech", "date", "member_name"}
        missing = required - col.keys()
        if missing:
            print(f"ERROR: Missing expected columns: {missing}")
            print(f"Available columns: {list(col.keys())}")
            sys.exit(1)

        idx_speech = col["speech"]
        idx_date = col["date"]
        idx_name = col["member_name"]

        for row in reader:
            rows_processed += 1

            # Progress report
            if rows_processed % PROGRESS_EVERY == 0:
                mb = bytes_written / 1024 / 1024
                print(
                    f"  Rows: {rows_processed:,}  |  Kept: {speeches_kept:,}  "
                    f"|  Size: {mb:.2f} MB"
                )
                if last_speech_sample:
                    print(f"  Sample: {last_speech_sample[:120]!r}")
                print()

            if bytes_written >= TARGET_BYTES:
                break

            # Guard against short rows
            if len(row) <= max(idx_speech, idx_date, idx_name):
                continue

            date_str = row[idx_date].strip()
            raw_speech = row[idx_speech]
            speaker = row[idx_name].strip()

            # --- Filter: year cutoff ---
            try:
                year = int(date_str[:4])
            except (ValueError, IndexError):
                continue
            if year < CUTOFF_YEAR:
                continue

            # --- Fix encoding ---
            fixed = ftfy.fix_text(raw_speech)

            # --- Filter: minimum length ---
            if len(fixed) < MIN_SPEECH_LEN:
                continue

            # --- Filter: non-ASCII ratio (Irish language proxy) ---
            if _non_ascii_ratio(fixed) > MAX_NON_ASCII_RATIO:
                continue

            # --- Clean ---
            clean = _clean_speech(fixed)
            if len(clean) < MIN_SPEECH_LEN:
                continue

            # --- Write ---
            chunk = clean + "\n\n"
            out.write(chunk)
            chunk_bytes = len(chunk.encode("utf-8"))
            bytes_written += chunk_bytes
            speeches_kept += 1
            last_speech_sample = clean

            # Track metadata
            if date_min is None or date_str < date_min:
                date_min = date_str
            if date_max is None or date_str > date_max:
                date_max = date_str
            speaker_counts[speaker] += 1

    source_bytes = out_file.read_bytes()
    source_text = out_file.read_text(encoding="utf-8")
    clean_sha256 = hashlib.sha256(source_bytes).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset": {
            "name": DATASET_NAME,
            "citation": DATASET_CITATION,
            "doi": DATASET_DOI,
            "full_date_range": "1919-2013",
        },
        "corpus": {
            "minimum_year": CUTOFF_YEAR,
            "selected_date_min": date_min,
            "selected_date_max": date_max,
            "rows_processed": rows_processed,
            "accepted_speeches": speeches_kept,
            "source_file_bytes": len(source_bytes),
            "source_file_characters": len(source_text),
            "source_sha256": clean_sha256,
            "target_bytes": TARGET_BYTES,
            "minimum_speech_characters": MIN_SPEECH_LEN,
            "maximum_non_ascii_ratio": MAX_NON_ASCII_RATIO,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    mb_written = bytes_written / 1024 / 1024
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Total rows processed : {rows_processed:,}")
    print(f"  Speeches kept        : {speeches_kept:,}")
    print(f"  Total characters     : {bytes_written:,}  ({mb_written:.2f} MB)")
    print(f"  Date range           : {date_min} to {date_max}")
    print(f"  Output file          : {out_file}")
    print(f"  Dataset manifest     : {manifest_path}")
    print()
    print("  Top 5 speakers by speech count:")
    for name, count in speaker_counts.most_common(5):
        print(f"    {count:>6,}  {name}")
    print("=" * 60)


if __name__ == "__main__":
    extract()
