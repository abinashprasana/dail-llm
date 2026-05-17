"""
Training curve plotting utilities.
"""
from __future__ import annotations
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def plot_loss_curve(
    steps: List[int],
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_losses, label="train loss", marker="o", markersize=3)
    ax.plot(steps, val_losses, label="val loss", marker="s", markersize=3)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"Saved loss plot to {save_path}")


def plot_val_perplexity(
    steps: List[int],
    perplexities: List[float],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, perplexities, label="val perplexity", color="darkorange", marker="^", markersize=4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Perplexity")
    ax.set_title("Validation Perplexity over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"Saved perplexity plot to {save_path}")
