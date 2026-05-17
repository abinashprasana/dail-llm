"""
CLI: generate text from a trained checkpoint.

    python -m dail_llm.model.generate --prompt "The Minister for"
"""
import argparse
from pathlib import Path
import torch

from config import CKPT_PATH, DEVICE
from dail_llm.data.tokenizer import CharTokenizer
from dail_llm.model.transformer import DailTransformerLM


def load_checkpoint(path: Path) -> dict:
    return torch.load(path.as_posix(), map_location="cpu")


def main():
    ap = argparse.ArgumentParser(description="Generate text from a trained Dáil LLM checkpoint")
    ap.add_argument("--prompt", type=str, default="The Minister for")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--checkpoint", type=str, default=str(CKPT_PATH))
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path)
    stoi = ckpt["vocab"]
    config = ckpt["config"]

    tokenizer = CharTokenizer("")
    tokenizer.stoi = stoi
    tokenizer.itos = {i: ch for ch, i in stoi.items()}

    device = torch.device(DEVICE)
    model = DailTransformerLM(
        vocab_size=len(stoi),
        block_size=config["block_size"],
        embed_dim=config["embed_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = tokenizer.encode(args.prompt).unsqueeze(0).to(device)
    out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature)[0]

    print("\n=== OUTPUT ===\n")
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
