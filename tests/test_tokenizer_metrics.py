import math

import pytest
import torch

from dail_llm.data.tokenizer import CharTokenizer
from dail_llm.eval.metrics import calculate_language_model_metrics, calculate_repetition_score
from dail_llm.model.transformer import DailTransformerLM


def test_tokenizer_reports_unsupported_characters_once():
    tokenizer = CharTokenizer("abc ")
    filtered, unsupported = tokenizer.filter_supported("a😀b😀é")
    assert filtered == "ab"
    assert unsupported == ["😀", "é"]


def test_repetition_rate_handles_short_and_repeated_text():
    assert calculate_repetition_score("one two") is None
    assert calculate_repetition_score("one two three four") == 0
    assert calculate_repetition_score("one two three one two three") == pytest.approx(0.25)


def test_language_model_metrics_are_finite():
    torch.manual_seed(1)
    model = DailTransformerLM(
        vocab_size=5,
        block_size=4,
        embed_dim=8,
        n_layers=1,
        n_heads=2,
        dropout=0,
    )
    data = torch.tensor([0, 1, 2, 3, 4] * 4)
    metrics = calculate_language_model_metrics(model, data, 4, torch.device("cpu"), batch_size=2)
    assert math.isfinite(metrics["cross_entropy"])
    assert metrics["perplexity"] > 1
    assert metrics["bits_per_character"] > 0
    assert 0 <= metrics["next_character_accuracy"] <= 1


def test_language_model_metrics_are_null_when_unavailable():
    model = DailTransformerLM(
        vocab_size=5,
        block_size=8,
        embed_dim=8,
        n_layers=1,
        n_heads=2,
        dropout=0,
    )
    metrics = calculate_language_model_metrics(
        model,
        torch.tensor([0, 1, 2]),
        8,
        torch.device("cpu"),
    )
    assert all(value is None for value in metrics.values())
