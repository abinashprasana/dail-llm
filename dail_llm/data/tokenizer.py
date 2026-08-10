"""
Character-level tokenizer.
"""
from __future__ import annotations

import torch


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def filter_supported(self, text: str) -> tuple[str, list[str]]:
        """Return supported text and the unique unsupported characters."""
        supported: list[str] = []
        unsupported: list[str] = []
        seen: set[str] = set()
        for char in text:
            if char in self.stoi:
                supported.append(char)
            elif char not in seen:
                unsupported.append(char)
                seen.add(char)
        return "".join(supported), unsupported

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s if c in self.stoi], dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)
