import argparse
from pathlib import Path
import torch

from .transformer import CharTokenizer, TinyTransformerLM

CKPT_PATH = Path("outputs/checkpoints/model.pt")


def load_checkpoint(path: Path):
    ckpt = torch.load(path.as_posix(), map_location="cpu")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="The Minister for")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    if not CKPT_PATH.exists():
        raise FileNotFoundError("Checkpoint not found. Train first: python -m src.model.train")

    ckpt = load_checkpoint(CKPT_PATH)
    stoi = ckpt["vocab"]
    config = ckpt["config"]

    # rebuild tokenizer object
    tokenizer = CharTokenizer(" ")  # dummy
    tokenizer.stoi = stoi
    tokenizer.itos = {i: ch for ch, i in stoi.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyTransformerLM(
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
    text = tokenizer.decode(out)

    print("\n=== OUTPUT ===\n")
    print(text)


if __name__ == "__main__":
    main()
