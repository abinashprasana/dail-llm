"""
Training script for the Dáil Éireann Character-Level Transformer.

    python -m dail_llm.model.train
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import (
    TRAIN_MSG_PATH, VAL_MSG_PATH, TEST_MSG_PATH,
    CKPT_DIR, PLOTS_DIR,
    BLOCK_SIZE, EMBED_DIM, N_LAYERS, N_HEADS, DROPOUT,
    BATCH_SIZE, MAX_STEPS, EVAL_EVERY, LEARNING_RATE,
)
from dail_llm.data.tokenizer import CharTokenizer
from dail_llm.model.transformer import DailTransformerLM
from dail_llm.eval.metrics import calculate_perplexity
from dail_llm.visualisation.training_plots import plot_loss_curve, plot_val_perplexity


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def make_batch(data_ids: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    ix = torch.randint(0, len(data_ids) - block_size - 1, (batch_size,))
    x = torch.stack([data_ids[i: i + block_size] for i in ix]).to(device)
    y = torch.stack([data_ids[i + 1: i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_ids, val_ids, batch_size, block_size, device):
    model.eval()
    out = {}
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        losses = [
            model(*make_batch(ids, batch_size, block_size, device))[1].item()
            for _ in range(20)
        ]
        out[split] = float(np.mean(losses))
    model.train()
    return out


def save_checkpoint(model, tokenizer, config: dict, path: Path):
    torch.save({"model_state": model.state_dict(), "vocab": tokenizer.stoi, "config": config}, path)


def train(args):
    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_text = load_text(TRAIN_MSG_PATH)
    val_text = load_text(VAL_MSG_PATH)

    tokenizer = CharTokenizer(train_text)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    # Also encode test set so we can save its tensor for evaluate.py
    if TEST_MSG_PATH.exists():
        test_text = load_text(TEST_MSG_PATH)
        test_ids = tokenizer.encode(test_text)
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(test_ids, CKPT_DIR / "test_ids.pt")
        print(f"Saved test tensor ({len(test_ids):,} tokens) to {CKPT_DIR / 'test_ids.pt'}")

    config = {
        "block_size": args.block_size,
        "embed_dim": args.embed_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
    }

    model = DailTransformerLM(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_perplexities: list[float] = []
    steps: list[int] = []
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
            val_ppl = calculate_perplexity(model, val_ids, args.block_size, device)

            steps.append(step)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            val_perplexities.append(val_ppl)

            pbar.write(
                f"Step {step}: train={losses['train']:.4f}  val={losses['val']:.4f}"
                f"  val_ppl={val_ppl:.1f}"
            )

            save_checkpoint(model, tokenizer, config, CKPT_DIR / "model_latest.pt")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(model, tokenizer, config, CKPT_DIR / "model_best.pt")
                save_checkpoint(model, tokenizer, config, CKPT_DIR / "model.pt")

    # Plots
    plot_loss_curve(steps, train_losses, val_losses, PLOTS_DIR / "loss.png")
    plot_val_perplexity(steps, val_perplexities, PLOTS_DIR / "val_perplexity.png")

    elapsed = (time.time() - t0) / 60
    print(f"\nTraining complete in {elapsed:.1f} min")
    print(f"Best model saved to: {CKPT_DIR / 'model.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train the Dáil Éireann Character LM")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--block_size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    parser.add_argument("--n_layers", type=int, default=N_LAYERS)
    parser.add_argument("--n_heads", type=int, default=N_HEADS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
