"""
High-level interface for loading a checkpoint and generating text.
"""
from pathlib import Path

import torch

from dail_llm.config import CKPT_PATH, DEVICE
from dail_llm.data.tokenizer import CharTokenizer
from dail_llm.model.transformer import DailTransformerLM


class ModelWrapper:
    def __init__(self, checkpoint_path: Path = CKPT_PATH, device: str = DEVICE):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model: DailTransformerLM | None = None
        self.tokenizer: CharTokenizer | None = None
        self.config: dict = {}
        self._load_checkpoint()

    def _load_checkpoint(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")
        print(f"Loading checkpoint from {self.checkpoint_path}...")
        ckpt = torch.load(
            self.checkpoint_path.as_posix(),
            map_location=self.device,
            weights_only=False,
        )
        self.config = ckpt["config"]
        stoi = ckpt["vocab"]
        self.tokenizer = CharTokenizer("")
        self.tokenizer.stoi = stoi
        self.tokenizer.itos = {i: ch for ch, i in stoi.items()}
        self.model = DailTransformerLM(
            vocab_size=len(stoi),
            block_size=self.config["block_size"],
            embed_dim=self.config["embed_dim"],
            n_layers=self.config["n_layers"],
            n_heads=self.config["n_heads"],
            dropout=self.config["dropout"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 250, temperature: float = 1.0) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded.")
        filtered_prompt, _ = self.tokenizer.filter_supported(prompt)
        if not filtered_prompt:
            raise ValueError("Prompt does not contain any characters supported by this model.")
        idx = self.tokenizer.encode(filtered_prompt).unsqueeze(0).to(self.device)
        out_idx = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )[0]
        return self.tokenizer.decode(out_idx)
