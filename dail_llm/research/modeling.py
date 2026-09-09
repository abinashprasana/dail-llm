"""Versioned tokenizer, speech-local windows, and optional conditioning."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from dail_llm.model.transformer import DailTransformerLM


@contextmanager
def evaluating(model):
    previous = model.training
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        model.train(previous)


class ResearchTokenizer:
    version = 2

    def __init__(self, characters):
        self.characters = list(characters)
        self.stoi = {char: i + 2 for i, char in enumerate(self.characters)}
        self.unknown, self.padding = 0, 1

    @classmethod
    def fit(cls, rows):
        return cls(sorted(set("".join(r["text"] for r in rows if r["split"] == "train"))))

    @property
    def size(self):
        return len(self.characters) + 2

    def encode(self, text):
        return [self.stoi.get(char, self.unknown) for char in text]

    def label(self, index):
        return ["<unk>", "<pad>", *self.characters][index]

    def state(self):
        return {"version": self.version, "characters": self.characters}


def windows(rows, tokenizer, block_size):
    """Nonoverlapping target ranges, one preceding character, complete tails.

    Target offset zero has no prefix and is deliberately not scored. Every
    other target has exactly one owner; all methods use this same convention.
    """
    result = []
    for row in rows:
        ids = tokenizer.encode(row["text"])
        masked = set(row.get("masked_positions", []))
        excluded_targets = set(row.get("excluded_targets", masked))
        for start in range(0, len(ids) - 1, block_size):
            x, y = ids[start : start + block_size], ids[start + 1 : start + block_size + 1]
            x = x[: len(y)]
            x = [tokenizer.unknown if start + i in masked else value for i, value in enumerate(x)]
            y = [-100 if start + i + 1 in excluded_targets else value for i, value in enumerate(y)]
            result.append(
                {
                    "row": row,
                    "start": start,
                    "length": len(y),
                    "x": x + [tokenizer.padding] * (block_size - len(x)),
                    "y": y + [-100] * (block_size - len(y)),
                }
            )
    return result


class ResearchModel(DailTransformerLM):
    def __init__(self, vocab_size, architecture, mode="none", categories=None):
        super().__init__(vocab_size=vocab_size, **architecture)
        self.mode = mode
        self.categories = categories or []
        self.category_ids = {name: i + 1 for i, name in enumerate(self.categories)}
        if mode != "none":
            self.metadata_embedding = nn.Embedding(
                len(self.categories) + 1, architecture["embed_dim"]
            )
            nn.init.normal_(self.metadata_embedding.weight, std=0.02)

    def category(self, row):
        return self.category_ids.get(row.get(self.mode, ""), 0)

    def features(self, idx, metadata=None):
        x = self.token_emb(idx) + self.pos_emb(torch.arange(idx.shape[1], device=idx.device))[None]
        if self.mode != "none":
            if metadata is None:
                raise ValueError("Conditioned model requires metadata")
            x = x + self.metadata_embedding(metadata)[:, None, :]
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def predict(self, idx, metadata=None):
        hidden = self.features(idx, metadata)
        return self.head(hidden), hidden


def batch(items, model):
    return (
        torch.tensor([w["x"] for w in items]),
        torch.tensor([w["y"] for w in items]),
        torch.tensor([model.category(w["row"]) for w in items]),
    )
