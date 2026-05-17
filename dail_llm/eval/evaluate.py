"""
Standalone evaluation runner.

    python -m dail_llm.eval.evaluate

Loads the best checkpoint, runs perplexity + BLEU + repetition on
the held-out test set, generates 5 sample texts, and saves a
markdown results table to outputs/evaluation_results.md.
"""
from __future__ import annotations
import random
import time
from pathlib import Path

import torch

from config import (
    CKPT_DIR, TEST_MSG_PATH, EVAL_RESULTS_PATH, DEVICE,
)
from dail_llm.data.tokenizer import CharTokenizer
from dail_llm.model.transformer import DailTransformerLM
from dail_llm.eval.metrics import (
    calculate_perplexity,
    calculate_bleu,
    calculate_repetition_score,
)

SEED_PROMPTS = [
    "The Minister for",
    "In this House",
    "The question before us",
    "I wish to raise",
    "On the matter of",
]
GENERATE_TOKENS = 200
GENERATE_TEMPERATURE = 0.8


def load_model(ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path.as_posix(), map_location=device)
    stoi = ckpt["vocab"]
    cfg = ckpt["config"]

    tokenizer = CharTokenizer("")
    tokenizer.stoi = stoi
    tokenizer.itos = {i: ch for ch, i in stoi.items()}

    model = DailTransformerLM(
        vocab_size=len(stoi),
        block_size=cfg["block_size"],
        embed_dim=cfg["embed_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, cfg


def main():
    device = torch.device(DEVICE)
    ckpt_path = CKPT_DIR / "model_best.pt"
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / "model.pt"

    print(f"Loading checkpoint: {ckpt_path}")
    model, tokenizer, cfg = load_model(ckpt_path, device)
    block_size = cfg["block_size"]

    # ------------------------------------------------------------------
    # Load test set
    # ------------------------------------------------------------------
    if not TEST_MSG_PATH.exists():
        raise FileNotFoundError(
            f"Test split not found at {TEST_MSG_PATH}. Run dataset_builder.py first."
        )
    test_text = TEST_MSG_PATH.read_text(encoding="utf-8", errors="replace")
    test_ids = tokenizer.encode(test_text)
    print(f"Test set: {len(test_text):,} chars / {len(test_ids):,} tokens")

    # ------------------------------------------------------------------
    # Perplexity on test set
    # ------------------------------------------------------------------
    print("Calculating perplexity...")
    ppl = calculate_perplexity(model, test_ids, block_size, device)
    print(f"  Perplexity: {ppl:.2f}")

    # ------------------------------------------------------------------
    # Generate 5 samples
    # ------------------------------------------------------------------
    print("Generating samples...")
    generated: list[str] = []
    for prompt in SEED_PROMPTS:
        valid_prompt = "".join(c for c in prompt if c in tokenizer.stoi)
        if not valid_prompt:
            valid_prompt = " "
        idx = tokenizer.encode(valid_prompt).unsqueeze(0).to(device)
        out = model.generate(idx, max_new_tokens=GENERATE_TOKENS, temperature=GENERATE_TEMPERATURE)[0]
        generated.append(tokenizer.decode(out))

    # ------------------------------------------------------------------
    # BLEU: compare generated samples against random test-set windows
    # ------------------------------------------------------------------
    refs: list[str] = []
    window = GENERATE_TOKENS
    for _ in SEED_PROMPTS:
        max_start = max(0, len(test_text) - window)
        start = random.randint(0, max_start)
        refs.append(test_text[start: start + window])

    bleu = calculate_bleu(generated, refs)
    print(f"  BLEU: {bleu:.4f}")

    # ------------------------------------------------------------------
    # Repetition scores
    # ------------------------------------------------------------------
    rep_scores = [calculate_repetition_score(g) for g in generated]
    avg_rep = sum(rep_scores) / len(rep_scores)
    print(f"  Avg repetition score: {avg_rep:.4f}")

    # ------------------------------------------------------------------
    # Build markdown report
    # ------------------------------------------------------------------
    lines: list[str] = [
        "# Evaluation Results — Dáil LLM\n",
        f"**Dataset:** Dáil Éireann Parliamentary Debates 1919-2013\n",
        f"**Checkpoint:** {ckpt_path.name}\n",
        "",
        "## Scores\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Perplexity | {ppl:.2f} |",
        f"| Corpus BLEU | {bleu:.4f} |",
        f"| Avg Repetition Score | {avg_rep:.4f} |",
        "",
        "## Generated Samples\n",
    ]

    for i, (prompt, text, rep) in enumerate(zip(SEED_PROMPTS, generated, rep_scores), 1):
        lines += [
            f"### Sample {i} — prompt: *\"{prompt}\"*\n",
            f"*Repetition score: {rep:.4f}*\n",
            "```",
            text.strip(),
            "```",
            "",
        ]

    report = "\n".join(lines)

    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_PATH.write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\nSaved to: {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
