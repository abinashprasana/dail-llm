"""Model-facing operations used by the HTTP layer."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from dail_llm.config import (
    DATASET_MANIFEST_PATH,
    EVAL_RESULTS_JSON_PATH,
)
from dail_llm.inference import ModelWrapper


class PromptValidationError(ValueError):
    pass


def _visible_label(char: str) -> str:
    return {" ": "space", "\n": "↵", "\t": "⇥"}.get(char, char)


class ModelService:
    def __init__(self):
        self.wrapper = ModelWrapper()
        self.checkpoint_sha256 = hashlib.sha256(
            self.wrapper.checkpoint_path.read_bytes()
        ).hexdigest()

    @property
    def model(self):
        if self.wrapper.model is None:
            raise RuntimeError("Model is not loaded.")
        return self.wrapper.model

    @property
    def tokenizer(self):
        if self.wrapper.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")
        return self.wrapper.tokenizer

    def _prepare_prompt(self, prompt: str) -> tuple[str, list[str]]:
        filtered, unsupported = self.tokenizer.filter_supported(prompt)
        if not filtered:
            raise PromptValidationError(
                "The prompt contains no characters supported by this model."
            )
        return filtered, unsupported

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> dict:
        filtered, unsupported = self._prepare_prompt(prompt)
        started = time.perf_counter()
        idx = self.tokenizer.encode(filtered).unsqueeze(0).to(self.wrapper.device)
        output = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )[0]
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return {
            "text": self.tokenizer.decode(output),
            "prompt": filtered,
            "generated_characters": max_new_tokens,
            "elapsed_ms": elapsed_ms,
            "filtered_characters": unsupported,
        }

    def attention(self, prompt: str, layer: int, head: int | None) -> dict:
        filtered, unsupported = self._prepare_prompt(prompt)
        n_layers = len(self.model.blocks)
        n_heads = self.model.blocks[0].attn.n_heads
        if layer >= n_layers:
            raise PromptValidationError(f"Layer must be between 0 and {n_layers - 1}.")
        if head is not None and head >= n_heads:
            raise PromptValidationError(f"Head must be between 0 and {n_heads - 1}.")

        ids = self.tokenizer.encode(filtered)[: self.model.block_size]
        ids = ids.unsqueeze(0).to(self.wrapper.device)
        with torch.no_grad():
            _, _, all_attention = self.model(ids, return_attention_weights=True)

        layer_attention = all_attention[layer][0].detach().cpu()
        selected = layer_attention if head is None else layer_attention[head: head + 1]
        labels = [_visible_label(self.tokenizer.itos[int(i)]) for i in ids[0].cpu()]
        return {
            "prompt": filtered,
            "labels": labels,
            "layer": layer,
            "head": head,
            "matrices": selected.tolist(),
            "filtered_characters": unsupported,
        }

    def metadata(self) -> dict:
        config = self.wrapper.config
        manifest = _read_json(DATASET_MANIFEST_PATH)
        return {
            "name": "Dáil LLM",
            "checkpoint": {
                "name": self.wrapper.checkpoint_path.name,
                "sha256": self.checkpoint_sha256,
            },
            "architecture": {
                **config,
                "vocab_size": self.tokenizer.vocab_size,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "type": "character-level decoder-only transformer",
            },
            "dataset": manifest,
        }

    def evaluation(self) -> dict:
        report = _read_json(EVAL_RESULTS_JSON_PATH)
        if report is None:
            raise FileNotFoundError("Structured evaluation results are not available.")
        return report


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
