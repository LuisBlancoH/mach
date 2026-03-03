import torch.nn as nn


class ObservationProjection(nn.Module):
    """Projects Qwen hidden states to meta-learner dimension."""

    def __init__(self, d_model, d_meta=128):
        super().__init__()
        self.proj = nn.Linear(d_model, d_meta, bias=False)

    def forward(self, hidden_states):
        """
        hidden_states: (batch, seq_len, d_model) or (batch, d_model)
        Returns: (batch, d_meta)
        """
        if hidden_states.dim() == 3:
            # Take last token's hidden state
            hidden_states = hidden_states[:, -1, :]
        return self.proj(hidden_states)
