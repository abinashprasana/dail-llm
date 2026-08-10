"""
Evaluation metrics for the Dáil LLM.
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def calculate_language_model_metrics(
    model,
    data_tensor: torch.Tensor,
    block_size: int,
    device,
    batch_size: int = 16,
) -> dict[str, float | None]:
    """Calculate token-weighted loss, perplexity, bits/character and accuracy."""
    model.eval()
    starts = list(range(0, len(data_tensor) - block_size - 1, block_size))
    if not starts:
        return {
            "cross_entropy": None,
            "perplexity": None,
            "bits_per_character": None,
            "next_character_accuracy": None,
        }

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset: offset + batch_size]
            x = torch.stack([
                data_tensor[i: i + block_size] for i in batch_starts
            ]).to(device)
            y = torch.stack([
                data_tensor[i + 1: i + block_size + 1] for i in batch_starts
            ]).to(device)
            logits, _ = model(x)
            total_loss += F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            ).item()
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    return {
        "cross_entropy": avg_loss,
        "perplexity": math.exp(avg_loss),
        "bits_per_character": avg_loss / math.log(2),
        "next_character_accuracy": total_correct / total_tokens,
    }


def calculate_perplexity(model, data_tensor: torch.Tensor, block_size: int, device) -> float:
    """
    Evaluate perplexity of `model` on `data_tensor`.

    Slides a window of `block_size` tokens across the tensor, averages the
    cross-entropy loss over all positions, then returns exp(avg_loss).
    """
    value = calculate_language_model_metrics(
        model, data_tensor, block_size, device
    )["perplexity"]
    return float("inf") if value is None else value


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def calculate_bleu(generated_texts: list[str], reference_texts: list[str]) -> float:
    """
    Corpus-level BLEU score using nltk.

    Both lists are lists of strings.  Each string is word-tokenised
    (split on whitespace) before scoring.

    Returns a float in [0, 1].
    """
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    except ImportError as e:
        raise ImportError("nltk is required: pip install nltk") from e

    smoother = SmoothingFunction().method1

    references = [[ref.split()] for ref in reference_texts]
    hypotheses = [gen.split() for gen in generated_texts]

    # Pad shorter list so lengths match
    min_len = min(len(references), len(hypotheses))
    references = references[:min_len]
    hypotheses = hypotheses[:min_len]

    if not hypotheses:
        return 0.0

    score = corpus_bleu(references, hypotheses, smoothing_function=smoother)
    return float(score)


# ---------------------------------------------------------------------------
# Repetition
# ---------------------------------------------------------------------------

def calculate_repetition_score(text: str) -> float | None:
    """
    Fraction of 3-grams that appear more than once.
    A high score means the text is repetitive.
    """
    tokens = re.findall(r"\w+|\S", text.lower())
    if len(tokens) < 3:
        return None
    seen: set[tuple[str, str, str]] = set()
    repeats = 0
    total = len(tokens) - 2
    for i in range(total):
        gram = tuple(tokens[i: i + 3])
        if gram in seen:
            repeats += 1
        else:
            seen.add(gram)
    return repeats / total


# Keep old function name as alias
repetition_score = calculate_repetition_score
