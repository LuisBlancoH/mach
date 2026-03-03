"""
Critic (basal ganglia analogue) for MACH Phase 3.

Evaluates mean-pooled meta-learner transformer hidden states to produce
a scalar value estimate. Action-conditional: because hidden states include
think tokens (which encode the proposed action), the critic evaluates
"this state WITH this proposed action, how good?" rather than just state value.

CriticSignalProjection replaces Phase 2's RewardProjection, projecting
[value, td_error] into a token for the meta-learner.
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    """MLP value estimator over mean-pooled transformer hidden states."""

    def __init__(self, d_meta=128, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_meta, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_states):
        """
        hidden_states: (n_tokens, d_meta) — transformer output
        Returns: scalar value estimate
        """
        pooled = hidden_states.mean(dim=0)  # (d_meta,)
        return self.net(pooled).squeeze(-1)  # scalar


class CriticSignalProjection(nn.Module):
    """
    Projects critic signals into a token for the meta-learner.
    Replaces RewardProjection from Phase 2.

    Inputs are detached — the critic gets gradient from critic loss directly,
    not from the transformer forward pass that consumes this token.
    """

    def __init__(self, d_meta=128):
        super().__init__()
        self.proj = nn.Linear(2, d_meta)

    def forward(self, value, td_error):
        """
        value: scalar tensor — critic's value estimate from previous firing
        td_error: scalar tensor — TD error from previous step
        Returns: (d_meta,) token
        """
        signals = torch.stack([value.detach(), td_error.detach()])
        return self.proj(signals)
