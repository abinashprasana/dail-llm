"""
Dáil LLM — Character-Level Transformer.

The CharTokenizer lives in dail_llm/data/tokenizer.py.
This module re-exports it for convenience.
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dail_llm.data.tokenizer import CharTokenizer  # noqa: F401  (re-export)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float, block_size: int):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0),
        )

    def forward(
        self, x: torch.Tensor, return_attention_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, H, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v                                                  # (B, H, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)          # (B, T, C)
        out = self.dropout(self.proj(out))

        if return_attention_weights:
            return out, att                                            # att: (B, H, T, T)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout, block_size)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout)

    def forward(
        self, x: torch.Tensor, return_attention_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_attention_weights:
            attn_out, att_weights = self.attn(self.ln1(x), return_attention_weights=True)
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            return x, att_weights          # att_weights: (B, H, T, T)
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DailTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 256,
        embed_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)

        # ModuleList (not Sequential) so we can pass kwargs per block
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout, block_size)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ):
        B, T = idx.shape
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            if targets is not None:
                targets = targets[:, -self.block_size:]

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(idx.size(1), device=idx.device))
        x = tok + pos.unsqueeze(0)

        all_att_weights: list[torch.Tensor] = []
        for block in self.blocks:
            if return_attention_weights:
                x, att_w = block(x, return_attention_weights=True)
                all_att_weights.append(att_w)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        if return_attention_weights:
            return logits, loss, all_att_weights   # list length = n_layers, each (B, H, T, T)
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
