"""
Attention weight visualisation using seaborn heatmaps.

    from dail_llm.visualisation.attention_viz import visualise_attention
    visualise_attention(model, tokenizer, "The Minister for Finance", layer=0, head=0)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError as e:
    raise ImportError("seaborn is required: pip install seaborn") from e


def visualise_attention(
    model,
    tokenizer,
    prompt: str,
    layer: int = 0,
    head: int = 0,
    save_path: Optional[Path | str] = None,
) -> None:
    """
    Plot the attention weight matrix for one layer and one head.

    Parameters
    ----------
    model     : DailTransformerLM (in eval mode)
    tokenizer : CharTokenizer
    prompt    : Input string (kept short — <= block_size chars)
    layer     : Which transformer layer to visualise (0-indexed)
    head      : Which attention head to visualise (0-indexed)
    save_path : If given, save figure there; otherwise call plt.show()
    """
    model.eval()
    device = next(model.parameters()).device

    # Encode and truncate to block_size
    ids = tokenizer.encode(prompt)
    ids = ids[: model.block_size].unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, all_att = model(ids, return_attention_weights=True)

    # all_att[layer] shape: (1, n_heads, T, T)
    att = all_att[layer][0, head].cpu().numpy()   # (T, T)
    chars = [tokenizer.itos[i.item()] for i in ids[0]]

    # Replace whitespace chars with visible symbols for axis labels
    labels = [repr(c)[1:-1] if c in (" ", "\n", "\t") else c for c in chars]

    fig_size = max(6, len(chars) // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        att,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cmap="Blues",
        vmin=0.0,
        vmax=att.max(),
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_title(f"Layer {layer}, Head {head} — Attention Weights", fontsize=12)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending from)")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
    else:
        plt.show()


def visualise_all_heads(
    model,
    tokenizer,
    prompt: str,
    layer: int = 0,
    save_path: Optional[Path | str] = None,
) -> None:
    """
    Plot all attention heads for one layer in a single grid figure.
    """
    model.eval()
    device = next(model.parameters()).device

    ids = tokenizer.encode(prompt)
    ids = ids[: model.block_size].unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, all_att = model(ids, return_attention_weights=True)

    att_layer = all_att[layer][0].cpu().numpy()   # (n_heads, T, T)
    n_heads = att_layer.shape[0]
    chars = [tokenizer.itos[i.item()] for i in ids[0]]
    labels = [repr(c)[1:-1] if c in (" ", "\n", "\t") else c for c in chars]

    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols
    cell = max(3, len(chars) // 3)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell, rows * cell))

    # Flatten for easy indexing
    if n_heads == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in row]

    for h, ax in enumerate(axes_flat):
        if h < n_heads:
            sns.heatmap(
                att_layer[h],
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                cmap="Blues",
                cbar=False,
                linewidths=0.2,
                linecolor="white",
            )
            ax.set_title(f"Head {h}", fontsize=9)
            ax.tick_params(axis="both", labelsize=6)
        else:
            ax.set_visible(False)

    fig.suptitle(f"Layer {layer} — All Heads", fontsize=13)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
    else:
        plt.show()
