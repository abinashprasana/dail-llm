"""
Training script for the Dáil LLM (legacy src/ package).
"""
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from src.config import (
    TRAIN_MSG_PATH, VAL_MSG_PATH, CKPT_DIR, PLOTS_DIR,
    BLOCK_SIZE, EMBED_DIM, N_LAYERS, N_HEADS, DROPOUT,
    BATCH_SIZE, MAX_STEPS, EVAL_EVERY, LEARNING_RATE
)
from src.model.transformer import CharTokenizer, TinyTransformerLM


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def make_batch(data_ids: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    # random starting positions
    ix = torch.randint(0, len(data_ids) - block_size - 1, (batch_size,))
    x = torch.stack([data_ids[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data_ids[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_ids, val_ids, batch_size, block_size, device):
    model.eval()
    out = {}
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        losses = []
        for _ in range(20):
            x, y = make_batch(ids, batch_size, block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


def save_checkpoint(model, tokenizer, config: dict, path: Path):
    torch.save({
        "model_state": model.state_dict(),
        "vocab": tokenizer.stoi,
        "config": config
    }, path)


def train(args):
    # Determine device with safety checks
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  WARNING: You requested '--device cuda', but your PyTorch installation does not support CUDA.")
        print("   Falling back to CPU so the training can proceed.")
        device = torch.device("cpu")
    elif args.device:
        device = torch.device(args.device)
    else:
        # Auto-detect
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    train_text = load_text(TRAIN_MSG_PATH)
    val_text = load_text(VAL_MSG_PATH)

    tokenizer = CharTokenizer(train_text)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    # Config dict for saving
    config = {
        "block_size": args.block_size,
        "embed_dim": args.embed_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
    }

    model = TinyTransformerLM(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train_losses = []
    val_losses = []
    steps = []
    best_val_loss = float("inf")
    t0 = time.time()

    pbar = tqdm(range(1, args.max_steps + 1))
    for step in pbar:
        x, y = make_batch(train_ids, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            pbar.set_description(f"loss {loss.item():.4f}")

        if step % args.eval_every == 0:
            losses = estimate_loss(model, train_ids, val_ids, args.batch_size, args.block_size, device)
            steps.append(step)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])

            pbar.write(f"Step {step}: train={losses['train']:.4f}, val={losses['val']:.4f}")

            # Save latest
            save_checkpoint(model, tokenizer, config, CKPT_DIR / "model_latest.pt")

            # Save best
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(model, tokenizer, config, CKPT_DIR / "model_best.pt")
                # Also save as the main model.pt for the app
                save_checkpoint(model, tokenizer, config, CKPT_DIR / "model.pt")

    # plot
    plt.figure()
    plt.plot(steps, train_losses, label="train")
    plt.plot(steps, val_losses, label="val")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plot_path = PLOTS_DIR / "loss.png"
    plt.savefig(plot_path.as_posix(), dpi=150)

    print(f"\n✅ Training complete.")
    print(f"Saved best model to: {CKPT_DIR / 'model.pt'}")
    print(f"Saved plot to: {plot_path}")
    print(f"Time: {(time.time()-t0)/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description="Train Dáil LLM (legacy)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--block_size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    parser.add_argument("--n_layers", type=int, default=N_LAYERS)
    parser.add_argument("--n_heads", type=int, default=N_HEADS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. 'cpu', 'cuda')")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
