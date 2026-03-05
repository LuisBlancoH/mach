import torch
import torch.nn as nn


class ErrorProjection(nn.Module):
    """
    Project demo error signals to meta-learner input space.

    Error features are scalar summaries of how wrong the current patches are
    on demo problems: avg CE loss, fraction correct, avg confidence, step index.
    These are computed with torch.no_grad() — gradient flows through the
    projection weights, not through the demo evaluation.
    """

    def __init__(self, n_error_features=4, d_meta=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_error_features, d_meta),
            nn.GELU(),
            nn.Linear(d_meta, d_meta),
        )

    def forward(self, error_features):
        """error_features: (n_error_features,) → (d_meta,)"""
        return self.proj(error_features)
