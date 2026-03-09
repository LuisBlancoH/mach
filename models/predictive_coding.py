"""
Predictive Coding Patches for frozen LLM augmentation.

The brain never really processes raw sensory input — it only processes its
own errors. Each cortical column maintains a prediction of what it expects,
and only the mismatch (prediction error) drives computation and correction.

Architecture:
  - PredictiveCodingPatch: one cortical column (predict → compare → correct)
  - PredictiveCodingNetwork: hierarchy of patches (top-down predictions, bottom-up errors)
  - PredictiveCodingPatchedModel: hooks into frozen Qwen

Top-down predictions flow from abstract (layer 34) to concrete (layer 9).
Bottom-up prediction errors flow from concrete to abstract.
Predictions use the PREVIOUS forward pass's state (like the brain).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _clamp_grad(grad):
    """Clamp gradients at the Qwen boundary to prevent bfloat16 explosion."""
    return grad.clamp(-1.0, 1.0)


class PredictiveCodingPatch(nn.Module):
    """One cortical column: predict → compare → precision-weight → correct.

    Each patch:
    1. Compresses Qwen's hidden state to representation space (bottom-up)
    2. Compares to top-down prediction from the patch above
    3. Precision-weights the error (learned attention to what matters)
    4. Projects weighted error back as correction to Qwen's residual stream
    5. Updates its representation and generates prediction for the patch below
    """

    def __init__(self, d_model, d_repr=128):
        super().__init__()
        self.d_model = d_model
        self.d_repr = d_repr

        # Bottom-up compression: Qwen hidden state → representation space
        self.compress = nn.Linear(d_model, d_repr)

        # Top-down prediction: my representation → what level below should see
        self.predict_down = nn.Linear(d_repr, d_repr)

        # Precision weighting: which errors to trust (brain's version of attention)
        # Input: [compressed_actual, prediction_error] → precision weights
        self.precision = nn.Linear(d_repr * 2, d_repr)

        # Error → correction: project precision-weighted error back to residual stream
        self.error_to_correction = nn.Linear(d_repr, d_model)

        # Initialize correction projection near zero (patches start as near-identity)
        nn.init.zeros_(self.error_to_correction.weight)
        nn.init.zeros_(self.error_to_correction.bias)

        # Persistent state (survives across forward passes, like cortical activity)
        self.representation = None  # (batch, d_repr) current belief
        self.prediction = None      # (batch, d_repr) prediction for level below

        # Diagnostics
        self._last_error_norm = 0.0
        self._last_precision_mean = 0.0
        self._last_weighted_error_norm = 0.0

    def reset(self):
        """Clear persistent state."""
        self.representation = None
        self.prediction = None

    def forward(self, actual_hidden, top_down_prediction):
        """Process one layer of Qwen's hidden states.

        Args:
            actual_hidden: Qwen's hidden state at this layer (batch, seq, d_model)
            top_down_prediction: prediction from the patch above (batch, d_repr) or None

        Returns:
            correction: additive correction to Qwen's residual stream (batch, seq, d_model)
        """
        # 1. Bottom-up: compress Qwen's actual hidden state
        compressed = self.compress(actual_hidden.float())  # (batch, seq, d_repr)

        # 2. Prediction error
        if top_down_prediction is not None:
            # Expand prediction to match sequence length
            pred = top_down_prediction.unsqueeze(1).expand_as(compressed)
            error = compressed - pred
        else:
            # Top-most patch: no prediction from above, full surprise
            error = compressed

        # 3. Precision weighting (learned inverse-variance)
        precision_input = torch.cat([compressed, error], dim=-1)  # (batch, seq, 2*d_repr)
        precision_weights = torch.sigmoid(self.precision(precision_input))  # [0, 1]
        weighted_error = error * precision_weights

        # 4. Update representation: detached running belief
        # Pool over sequence for the persistent representation
        compressed_mean = compressed.detach().mean(dim=1)  # (batch, d_repr)
        if self.representation is not None:
            self.representation = 0.8 * self.representation + 0.2 * compressed_mean
        else:
            self.representation = compressed_mean

        # 5. Generate prediction for level below (used NEXT forward pass)
        self.prediction = self.predict_down(self.representation)  # (batch, d_repr)

        # 6. Project weighted error to correction in Qwen's space
        correction = self.error_to_correction(weighted_error)  # (batch, seq, d_model)

        # Diagnostics
        self._last_error_norm = error.detach().norm().item()
        self._last_precision_mean = precision_weights.detach().mean().item()
        self._last_weighted_error_norm = weighted_error.detach().norm().item()

        return correction.to(actual_hidden.dtype)


class PredictiveCodingNetwork(nn.Module):
    """Hierarchy of predictive coding patches.

    Patches are ordered from concrete (early Qwen layer) to abstract (late layer).
    Top-down predictions flow: patch3 → patch2 → patch1 → patch0
    Bottom-up errors flow: patch0 → patch1 → patch2 → patch3

    Predictions use the PREVIOUS forward pass state — like the brain,
    which is always predicting slightly ahead of incoming sensory data.
    """

    def __init__(self, d_model, d_repr=128, patch_layers=None):
        super().__init__()
        if patch_layers is None:
            patch_layers = [9, 18, 27, 34]
        self.patch_layers = patch_layers
        self.d_repr = d_repr

        self.patches = nn.ModuleList([
            PredictiveCodingPatch(d_model, d_repr) for _ in patch_layers
        ])

        # Stored predictions from previous forward pass
        # predictions[i] = what patch i+1 predicted for patch i
        self._predictions = [None] * len(patch_layers)

    def reset(self):
        """Clear all state. Call between eval runs or training restarts."""
        for patch in self.patches:
            patch.reset()
        self._predictions = [None] * len(self.patch_layers)

    def get_correction(self, patch_idx, actual_hidden):
        """Compute correction for one patch layer.

        Called by the hook during Qwen's forward pass.
        Uses the prediction from the PREVIOUS forward pass.
        """
        # Top-down prediction from the patch above (from previous forward pass)
        if patch_idx == len(self.patches) - 1:
            top_down = None  # top-most patch: no prediction from above
        else:
            top_down = self._predictions[patch_idx]

        # Compute correction via predictive coding
        correction = self.patches[patch_idx](actual_hidden, top_down)

        # Store this patch's prediction for the level below (used next forward pass)
        self._predictions[patch_idx] = self.patches[patch_idx].prediction

        return correction

    def get_prediction_loss(self):
        """Auxiliary loss: how well did higher patches predict lower patches?

        Drives the hierarchy to build accurate models of Qwen's computation.
        """
        loss = 0.0
        count = 0
        for i in range(len(self.patches) - 1):
            # Patch i+1 predicted what patch i should see
            prediction = self._predictions[i]
            actual = self.patches[i].representation
            if prediction is not None and actual is not None:
                loss = loss + F.mse_loss(prediction, actual.detach())
                count += 1
        return loss / max(count, 1)

    def get_diagnostics(self):
        """Return diagnostic dict for logging."""
        diag = {}
        for i, patch in enumerate(self.patches):
            diag[f"pc/patch{i}_error_norm"] = patch._last_error_norm
            diag[f"pc/patch{i}_precision_mean"] = patch._last_precision_mean
            diag[f"pc/patch{i}_weighted_error_norm"] = patch._last_weighted_error_norm
            if patch.representation is not None:
                diag[f"pc/patch{i}_repr_norm"] = patch.representation.norm().item()
        # Prediction accuracy
        for i in range(len(self.patches) - 1):
            pred = self._predictions[i]
            actual = self.patches[i].representation
            if pred is not None and actual is not None:
                cos = F.cosine_similarity(
                    pred.flatten(), actual.detach().flatten(), dim=0
                ).item()
                diag[f"pc/pred_cos_{i+1}_to_{i}"] = cos
        return diag


class PredictiveCodingPatchedModel(nn.Module):
    """Hooks a PredictiveCodingNetwork into a frozen Qwen model."""

    def __init__(self, base_model, pc_network):
        super().__init__()
        self.base_model = base_model
        self.pc_network = pc_network
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for i, layer_idx in enumerate(self.pc_network.patch_layers):
            layer = self.base_model.model.layers[layer_idx]
            handle = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(handle)

    def _make_hook(self, patch_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            correction = self.pc_network.get_correction(patch_idx, h)
            h_new = h + correction.to(h.dtype)

            # Gradient clamping at Qwen boundary
            if h_new.requires_grad:
                h_new.register_hook(_clamp_grad)

            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new
        return hook

    @property
    def device(self):
        return self.base_model.device

    def forward(self, input_ids, labels=None, attention_mask=None):
        return self.base_model(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask
        )

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
