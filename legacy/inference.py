"""
Inference logic for the Dáil LLM (legacy src/ package).
Handles loading checkpoints, initializing the model, and generating text.
"""
import torch
from pathlib import Path
from typing import Tuple, Optional

from src.config import CKPT_PATH, DEVICE
from src.model.transformer import CharTokenizer, TinyTransformerLM


class ModelWrapper:
    """
    A wrapper class to handle model loading and text generation.
    """
    def __init__(self, checkpoint_path: Path = CKPT_PATH, device: str = DEVICE):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model: Optional[TinyTransformerLM] = None
        self.tokenizer: Optional[CharTokenizer] = None
        self.config: dict = {}

        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load model weights and config from the checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")

        print(f"Loading checkpoint from {self.checkpoint_path}...")
        ckpt = torch.load(self.checkpoint_path.as_posix(), map_location=self.device)

        self.config = ckpt["config"]
        vocab_stoi = ckpt["vocab"]

        # Reconstruct Tokenizer
        # We need a dummy text to init, but we'll override the vocab immediately
        self.tokenizer = CharTokenizer("")
        self.tokenizer.stoi = vocab_stoi
        self.tokenizer.itos = {i: ch for ch, i in vocab_stoi.items()}

        # Reconstruct Model
        self.model = TinyTransformerLM(
            vocab_size=len(vocab_stoi),
            block_size=self.config["block_size"],
            embed_dim=self.config["embed_dim"],
            n_layers=self.config["n_layers"],
            n_heads=self.config["n_heads"],
            dropout=self.config["dropout"],
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 250, temperature: float = 1.0) -> str:
        """
        Generate text starting from a prompt.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded.")

        # Encode
        idx = self.tokenizer.encode(prompt).unsqueeze(0).to(self.device)

        # Generate
        out_idx = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )[0]

        # Decode
        return self.tokenizer.decode(out_idx)
