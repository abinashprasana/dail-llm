"""
Single entry point for the Dáil LLM pipeline.

    python train_pipeline.py

Runs in order:
  1. extract_dail.py   — if dail_debates_clean.txt does not exist
  2. dataset_builder.py — prepare train/val/test splits + RAG chunks
  3. train.py          — train the character-level transformer
  4. evaluate.py       — generate evaluation report

Then prints instructions to launch the Streamlit app.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLEAN_TXT = ROOT / "dataverse_files" / "dail_debates_clean.txt"
TRAIN_TXT = ROOT / "data" / "processed" / "train.txt"


def run(module: str, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\nERROR: '{module}' exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)


def main():
    print("\nDáil LLM — Irish Parliamentary Transformer — Full Training Pipeline")
    print("=" * 60)

    # Step 1 — Extract dataset (skip if already done)
    if CLEAN_TXT.exists():
        size_mb = CLEAN_TXT.stat().st_size / 1024 / 1024
        print(f"\nStep 1 — Skipped: {CLEAN_TXT.name} already exists ({size_mb:.2f} MB)")
    else:
        run("dail_llm.data.extract_dail", "Step 1 — Extracting Dáil dataset")

    # Step 2 — Prepare data splits
    run("dail_llm.data.dataset_builder", "Step 2 — Building dataset splits")

    # Step 3 — Train
    run("dail_llm.model.train", "Step 3 — Training the transformer")

    # Step 4 — Evaluate
    run("dail_llm.eval.evaluate", "Step 4 — Generating evaluation report")

    print("\n" + "=" * 60)
    print("  Training complete.")
    print("=" * 60)
    print("\nTo launch the Streamlit dashboard, run:")
    print("\n    streamlit run dail_llm/app/streamlit_app.py\n")


if __name__ == "__main__":
    main()
