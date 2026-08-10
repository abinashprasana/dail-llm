from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharTokenizer:
    """
    A simple character-level tokenizer.
    """
    def __init__(self, text: str):
        """
        Initialize the tokenizer by building the vocabulary from the given text.
        """
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> torch.Tensor:
        """Encode a string into a tensor of token IDs."""
        return torch.tensor([self.stoi[c] for c in s if c in self.stoi], dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Decode a tensor or list of token IDs back into a string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    def __init__(self, embed_dim: int, n_heads: int, dropout: float, block_size: int):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,heads,T,hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,heads,T,T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v  # (B,heads,T,hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B,T,C)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Simple position-wise feed-forward network.
    """
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
    """
    Standard Transformer block: LayerNorm -> SelfAttn -> LayerNorm -> FFN.
    Residual connections are applied after each sub-layer.
    """
    def __init__(self, embed_dim: int, n_heads: int, dropout: float, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout, block_size)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    """
    A tiny Transformer Language Model.
    """
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        embed_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, dropout, block_size) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            if targets is not None:
                targets = targets[:, -self.block_size:]

        tok = self.token_emb(idx)  # (B,T,C)
        pos = self.pos_emb(torch.arange(idx.size(1), device=idx.device))  # (T,C)
        x = tok + pos.unsqueeze(0)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate new tokens given a context `idx`.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # Focus on the last time step
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
