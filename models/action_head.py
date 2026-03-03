import torch
import torch.nn as nn

from config import GATE_SCALE


class ActionHead(nn.Module):
    """
    Takes think_0 hidden state, produces patch write coefficients.

    Output: n_patches * 2 * (n_basis + 1) values
        4 patches × 2 weight matrices = 8 writes
        8 basis coefficients + 1 gate per write = 9 values per write
        Total: 8 × 9 = 72
    """

    def __init__(self, d_meta=128, n_patches=4, n_basis=8):
        super().__init__()
        self.n_patches = n_patches
        self.n_basis = n_basis
        n_writes = n_patches * 2
        n_outputs = n_writes * (n_basis + 1)

        self.head = nn.Sequential(
            nn.Linear(d_meta, 64),
            nn.GELU(),
            nn.Linear(64, n_outputs),
        )

    def forward(self, think_0_hidden):
        """
        think_0_hidden: (d_meta,)
        Returns: list of (patch_idx, weight_name, coefficients, gate)
        """
        raw = self.head(think_0_hidden)

        writes = []
        idx = 0
        for patch_i in range(self.n_patches):
            for weight_name in ["down", "up"]:
                coefficients = raw[idx:idx + self.n_basis]
                gate = torch.sigmoid(raw[idx + self.n_basis]) * GATE_SCALE
                idx += self.n_basis + 1
                writes.append((patch_i, weight_name, coefficients, gate))

        return writes
