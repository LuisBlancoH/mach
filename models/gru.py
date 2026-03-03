import torch
import torch.nn as nn


class SimpleGRU(nn.Module):
    """
    Integrates projected Qwen hidden states over tokens.
    Phase 2 simplified: no surprise gating.
    Full version (Phase 4+) adds: surprise = ||cerebellar_error|| modulating update gate.
    """

    def __init__(self, d_meta=128):
        super().__init__()
        self.gru_cell = nn.GRUCell(d_meta, d_meta)
        self.memory = None

    def reset(self):
        self.memory = None

    def integrate(self, projected_input):
        """
        projected_input: (d_meta,) — one projected Qwen hidden state
        Updates internal memory. Call once per observation.
        Returns: current memory state (d_meta,)
        """
        if self.memory is None:
            self.memory = torch.zeros_like(projected_input)
        self.memory = self.gru_cell(
            projected_input.unsqueeze(0), self.memory.unsqueeze(0)
        ).squeeze(0)
        return self.memory

    def get_memory(self):
        return self.memory
