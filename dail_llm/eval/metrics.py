"""
Evaluation metrics for the Dáil LLM.
"""
from __future__ import annotations
import math
import re
from typing import List

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def calculate_perplexity(model, data_tensor: torch.Tensor, block_size: int, device) -> float:
    """
    Evaluate perplexity of `model` on `data_tensor`.

    Slides a window of `block_size` tokens across the tensor, averages the
    cross-entropy loss over all positions, then returns exp(avg_loss).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, len(data_tensor) - block_size - 1, block_size):
            x = data_tensor[i: i + block_size].unsqueeze(0).to(device)
            y = data_tensor[i + 1: i + block_size + 1].unsqueeze(0).to(device)
            _, loss = model(x, y)
            if loss is not None:
                total_loss += loss.item()
                n_batches += 1

    if n_batches == 0:
        return float("inf")

    avg_loss = total_loss / n_batches
    return math.exp(avg_loss)


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def calculate_bleu(generated_texts: List[str], reference_texts: List[str]) -> float:
    """
    Corpus-level BLEU score using nltk.

    Both lists are lists of strings.  Each string is word-tokenised
    (split on whitespace) before scoring.

    Returns a float in [0, 1].
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
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

def calculate_repetition_score(text: str) -> float:
    """
    Fraction of 3-grams that appear more than once.
    A high score means the text is repetitive.
    """
    tokens = re.findall(r"\w+|\S", text.lower())
    if len(tokens) < 50:
        return 0.0
    seen: dict = {}
    repeats = 0
    total = 0
    for i in range(len(tokens) - 3):
        gram = tuple(tokens[i: i + 3])
        total += 1
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] == 2:
            repeats += 1
    return repeats / max(total, 1)


# Keep old function name as alias
repetition_score = calculate_repetition_score
