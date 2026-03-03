import torch.nn as nn


class RewardProjection(nn.Module):
    """
    Projects raw reward signals into a token for the meta-learner.
    Phase 2 signals: [last_reward, cumulative_reward, firing_index]
    In later phases, this is replaced by critic signals (value, TD error).
    """

    def __init__(self, n_signals=3, d_meta=128):
        super().__init__()
        self.proj = nn.Linear(n_signals, d_meta)

    def forward(self, signals):
        """
        signals: (3,) or (batch, 3)
        Returns: (d_meta,) or (batch, d_meta)
        """
        return self.proj(signals)
