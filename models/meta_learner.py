import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model, n_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(self, x):
        """x: (n_tokens, d_model)"""
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MetaLearnerTransformer(nn.Module):
    """
    Small transformer — the cortex of the universal module.

    Input tokens (7 for Phase 2):
        [gru_memory]        1 token: integrated world state
        [reward_signal]     1 token: projected reward (placeholder for critic signals)
        [zero_placeholder]  1 token: placeholder for accumulated cerebellar correction
        [tf_mem]            1 token: transformer memory (persists across firings)
        [think_0]           1 token: action output
        [think_1]           1 token: memory update output
        [think_2]           1 token: upstream output (unused in Phase 2)
    """

    def __init__(self, d_meta=128, n_heads=2, n_layers=2, mlp_dim=256, n_tokens=7):
        super().__init__()
        self.d_meta = d_meta

        # Learned initial vectors for think tokens and transformer memory
        self.think_0_init = nn.Parameter(torch.randn(d_meta) * 0.01)
        self.think_1_init = nn.Parameter(torch.randn(d_meta) * 0.01)
        self.think_2_init = nn.Parameter(torch.randn(d_meta) * 0.01)
        self.tf_mem_init = nn.Parameter(torch.randn(d_meta) * 0.01)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(n_tokens, d_meta) * 0.01)

        # Transformer layers (pre-norm)
        self.layers = nn.ModuleList([
            TransformerBlock(d_meta, n_heads, mlp_dim)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_meta)

    def forward(self, tokens):
        """
        tokens: (n_tokens, d_meta)
        Returns: (n_tokens, d_meta) hidden states at all positions
        """
        x = tokens + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
