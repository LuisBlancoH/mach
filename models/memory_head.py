import torch
import torch.nn as nn


class MemoryHead(nn.Module):
    """Updates transformer memory from think_1 hidden state via gated update."""

    def __init__(self, d_meta=128):
        super().__init__()
        self.gate = nn.Linear(d_meta, d_meta)
        self.candidate = nn.Linear(d_meta, d_meta)

    def forward(self, think_1_hidden, tf_mem):
        """
        Gated update of transformer memory.
        think_1_hidden: (d_meta,)
        tf_mem: (d_meta,)
        Returns: updated tf_mem (d_meta,)
        """
        g = torch.sigmoid(self.gate(think_1_hidden))
        c = torch.tanh(self.candidate(think_1_hidden))
        return (1 - g) * tf_mem + g * c
