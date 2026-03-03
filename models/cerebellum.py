"""
Cerebellum analogue for MACH Phase 4.

Predicts the next projected observation from current GRU memory.
Prediction error provides two signals:
  1. surprise (scalar) = ||prediction_error|| -> gates GRU input
  2. correction (d_meta vector) = projected prediction error -> token for meta-learner

The predictor is trained online (supervised, separate optimizer).
The correction_proj is meta-trained (CE loss gradient flows through transformer position 2).
"""

import torch
import torch.nn as nn


class Cerebellum(nn.Module):
    """
    Predicts next projected observation and computes prediction error.

    predictor: MLP(d_meta -> hidden -> d_meta) -- online supervised
    correction_proj: Linear(d_meta -> d_meta) -- meta-trained
    """

    def __init__(self, d_meta=128, hidden_dim=128):
        super().__init__()
        self.d_meta = d_meta

        self.predictor = nn.Sequential(
            nn.Linear(d_meta, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_meta),
        )

        self.correction_proj = nn.Linear(d_meta, d_meta)

    def predict(self, gru_memory):
        """
        Predict next projected observation from current GRU memory.
        gru_memory: (d_meta,)
        Returns: predicted_obs (d_meta,) — in predictor's graph (for online training)
        """
        return self.predictor(gru_memory.detach())

    def compute_error(self, predicted_obs, actual_obs):
        """
        Compute prediction error, surprise, and correction.

        predicted_obs: (d_meta,) — from predict() at previous step
        actual_obs: (d_meta,) — current projected observation

        Returns:
            surprise: scalar — L2 norm of error (detached)
            correction: (d_meta,) — projected error (in meta-training graph)
            pred_loss: scalar — MSE for online predictor training
        """
        error = actual_obs.detach() - predicted_obs.detach()
        surprise = error.norm().detach()
        correction = self.correction_proj(error.detach())
        pred_loss = ((predicted_obs - actual_obs.detach()) ** 2).mean()

        return surprise, correction, pred_loss
