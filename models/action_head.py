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

    def __init__(self, d_meta=128, n_patches=4, n_basis=8,
                 obs_conditioned=False):
        super().__init__()
        self.n_patches = n_patches
        self.n_basis = n_basis
        self.obs_conditioned = obs_conditioned
        n_writes = n_patches * 2
        n_outputs = n_writes * (n_basis + 1)

        input_dim = d_meta * 2 if obs_conditioned else d_meta
        self.fc1 = nn.Linear(input_dim, 64)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(64, n_outputs)

        if obs_conditioned:
            # Zero-init the observation half so action head starts
            # identical to Phase 3 (ignoring gru_memory)
            with torch.no_grad():
                self.fc1.weight[:, d_meta:].zero_()

    def forward(self, think_0_hidden, gru_memory=None):
        """
        think_0_hidden: (d_meta,)
        gru_memory: optional (d_meta,) — direct observation for conditioning
        Returns: list of (patch_idx, weight_name, coefficients, gate)
        """
        if self.obs_conditioned and gru_memory is not None:
            x = torch.cat([think_0_hidden, gru_memory])
        else:
            x = think_0_hidden
        raw = self.fc2(self.act(self.fc1(x)))

        writes = []
        idx = 0
        for patch_i in range(self.n_patches):
            for weight_name in ["down", "up"]:
                coefficients = raw[idx:idx + self.n_basis]
                gate = torch.sigmoid(raw[idx + self.n_basis]) * GATE_SCALE
                idx += self.n_basis + 1
                writes.append((patch_i, weight_name, coefficients, gate))

        return writes
