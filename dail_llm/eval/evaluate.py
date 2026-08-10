"""Evaluate the trained character model and write JSON plus Markdown reports."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from dail_llm.config import (
    CKPT_DIR,
    DATASET_MANIFEST_PATH,
    DEVICE,
    EVAL_RESULTS_JSON_PATH,
    EVAL_RESULTS_PATH,
    TEST_MSG_PATH,
)
from dail_llm.data.tokenizer import CharTokenizer
from dail_llm.eval.metrics import (
    calculate_language_model_metrics,
    calculate_repetition_score,
)
from dail_llm.model.transformer import DailTransformerLM

SEED = 42
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
    ckpt = torch.load(ckpt_path.as_posix(), map_location=device, weights_only=False)
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


def _markdown_report(report: dict) -> str:
    metrics = report["metrics"]
    def render(value: float | None, spec: str) -> str:
        return "N/A" if value is None else format(value, spec)

    lines = [
        "# Evaluation Results - Dáil LLM",
        "",
        f"**Checkpoint:** {report['checkpoint']['name']}",
        "",
        "## Held-out test metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Cross-entropy | {render(metrics['cross_entropy'], '.4f')} |",
        f"| Perplexity | {render(metrics['perplexity'], '.2f')} |",
        f"| Bits per character | {render(metrics['bits_per_character'], '.4f')} |",
        f"| Next-character accuracy | {render(metrics['next_character_accuracy'], '.2%')} |",
        "",
        "## Deterministic samples",
        "",
    ]
    for index, sample in enumerate(report["samples"], 1):
        repetition = sample["repeated_word_trigram_rate"]
        repetition_label = "N/A" if repetition is None else f"{repetition:.4f}"
        lines.extend([
            f"### Sample {index}: \"{sample['prompt']}\"",
            "",
            f"*Repeated word-trigram rate: {repetition_label}*",
            "",
            "```text",
            sample["text"].strip(),
            "```",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(DEVICE)
    ckpt_path = CKPT_DIR / "model_best.pt"
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / "model.pt"

    print(f"Loading checkpoint: {ckpt_path}")
    model, tokenizer, cfg = load_model(ckpt_path, device)

    if not TEST_MSG_PATH.exists():
        raise FileNotFoundError(
            f"Test split not found at {TEST_MSG_PATH}. Run dataset_builder.py first."
        )
    test_text = TEST_MSG_PATH.read_text(encoding="utf-8", errors="replace")
    test_ids = tokenizer.encode(test_text)
    print(f"Test set: {len(test_ids):,} supported characters")

    print("Calculating held-out metrics...")
    metrics = calculate_language_model_metrics(
        model,
        test_ids,
        cfg["block_size"],
        device,
    )

    print("Generating deterministic samples...")
    samples: list[dict] = []
    for prompt in SEED_PROMPTS:
        valid_prompt, _ = tokenizer.filter_supported(prompt)
        if not valid_prompt:
            valid_prompt = " "
        idx = tokenizer.encode(valid_prompt).unsqueeze(0).to(device)
        out = model.generate(
            idx,
            max_new_tokens=GENERATE_TOKENS,
            temperature=GENERATE_TEMPERATURE,
        )[0]
        text = tokenizer.decode(out)
        continuation = text[len(valid_prompt):]
        samples.append({
            "prompt": valid_prompt,
            "text": text,
            "continuation": continuation,
            "repeated_word_trigram_rate": calculate_repetition_score(continuation),
        })

    manifest = None
    if DATASET_MANIFEST_PATH.exists():
        manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "checkpoint": {
            "name": ckpt_path.name,
            "parameters": sum(p.numel() for p in model.parameters()),
            "config": cfg,
        },
        "test_characters": len(test_ids),
        "metrics": metrics,
        "generation": {
            "new_characters": GENERATE_TOKENS,
            "temperature": GENERATE_TEMPERATURE,
        },
        "samples": samples,
        "dataset": manifest,
    }

    EVAL_RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_JSON_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    EVAL_RESULTS_PATH.write_text(_markdown_report(report), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved JSON report to: {EVAL_RESULTS_JSON_PATH}")
    print(f"Saved Markdown report to: {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
