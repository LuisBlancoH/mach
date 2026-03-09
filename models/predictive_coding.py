"""
Predictive Coding Patches for frozen LLM augmentation.

The brain never really processes raw sensory input — it only processes its
own errors. Each cortical column predicts what it expects, and only the
mismatch (prediction error) drives computation and correction.

Two-pass architecture (faithful to cortical processing):
  1. FEEDFORWARD SWEEP: Qwen runs normally, hidden states captured (observation)
  2. RECURRENT SETTLING: PC hierarchy iterates on captured states —
     top-down predictions meet bottom-up evidence, errors converge
  3. CORRECTED PASS: Qwen runs again with settled corrections applied

Top-down predictions flow from abstract (layer 34) to concrete (layer 9).
Bottom-up prediction errors flow from concrete to abstract.
Settling iterates on the SAME input (like the brain's ~200ms processing).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _clamp_grad(grad):
    """Clamp gradients at the Qwen boundary to prevent bfloat16 explosion."""
    return grad.clamp(-1.0, 1.0)


class PredictiveCodingPatch(nn.Module):
    """One cortical column.

    Compresses Qwen's hidden state, computes prediction error against
    top-down prediction, precision-weights the error, and projects
    the correction back to Qwen's residual stream.
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
        # Input: [compressed_actual, error] → precision weights
        self.precision = nn.Linear(d_repr * 2, d_repr)

        # Error → correction: project precision-weighted error back to residual stream
        self.error_to_correction = nn.Linear(d_repr, d_model)

        # Initialize correction projection near zero (patches start as near-identity)
        nn.init.zeros_(self.error_to_correction.weight)
        nn.init.zeros_(self.error_to_correction.bias)

        # Diagnostics (set during settle)
        self._last_error_norm = 0.0
        self._last_precision_mean = 0.0
        self._last_weighted_error_norm = 0.0

    def compute_error(self, compressed, prediction):
        """Compute precision-weighted prediction error.

        Args:
            compressed: compressed bottom-up features (batch, seq, d_repr)
            prediction: top-down prediction (batch, d_repr) or None

        Returns:
            weighted_error: precision-weighted error (batch, seq, d_repr)
            representation: updated belief (batch, d_repr)
        """
        if prediction is not None:
            pred = prediction.unsqueeze(1).expand_as(compressed)
            error = compressed - pred
        else:
            # Top-most patch: no prediction, everything is error
            error = compressed

        # Precision weighting (learned inverse-variance)
        precision_input = torch.cat([compressed, error], dim=-1)
        precision_weights = torch.sigmoid(self.precision(precision_input))
        weighted_error = error * precision_weights

        # Representation: prediction + weighted evidence (Bayesian update)
        # Pool over sequence for the persistent representation
        if prediction is not None:
            representation = prediction + weighted_error.mean(dim=1)
        else:
            representation = compressed.mean(dim=1)

        # Diagnostics
        self._last_error_norm = error.detach().norm().item()
        self._last_precision_mean = precision_weights.detach().mean().item()
        self._last_weighted_error_norm = weighted_error.detach().norm().item()

        return weighted_error, representation


class PredictiveCodingNetwork(nn.Module):
    """Hierarchy of predictive coding patches with iterative settling.

    Two-pass architecture:
    1. Capture: hooks store Qwen hidden states without modification
    2. Settle: iterate top-down predictions and bottom-up errors on captured states
    3. Correct: hooks apply the settled corrections to Qwen's second forward pass

    The settling loop (step 2) is the "recurrent processing" that the brain
    does for ~200ms when perceiving a stimulus. It runs entirely in the small
    PC representation space (d_repr=128), so it's cheap.
    """

    def __init__(self, d_model, d_repr=128, patch_layers=None, n_settle=3):
        super().__init__()
        if patch_layers is None:
            patch_layers = [9, 18, 27, 34]
        self.patch_layers = patch_layers
        self.d_model = d_model
        self.d_repr = d_repr
        self.n_settle = n_settle

        self.patches = nn.ModuleList([
            PredictiveCodingPatch(d_model, d_repr) for _ in patch_layers
        ])

        # State set during two-pass forward
        self._captured = {}          # {patch_idx: hidden_state} from first pass
        self._corrections = {}       # {patch_idx: correction} from settling
        self._mode = "capture"       # "capture" | "correct"

    def reset(self):
        """Clear all state."""
        self._captured.clear()
        self._corrections.clear()
        self._mode = "capture"

    def settle(self):
        """Run predictive coding hierarchy on captured states until convergence.

        This is the brain's ~200ms recurrent processing of a single stimulus.
        Top-down predictions and bottom-up errors iterate until the hierarchy
        agrees on an interpretation.

        Returns:
            prediction_loss: auxiliary loss for training the predictive model
        """
        n = len(self.patches)

        # Compress all captured hidden states (gradient flows through compress)
        compressed = {}
        for i in range(n):
            h = self._captured.get(i)
            if h is None:
                continue
            compressed[i] = self.patches[i].compress(h.float())

        # Initialize representations from compressed activations
        representations = {}
        for i in range(n):
            if i in compressed:
                representations[i] = compressed[i].mean(dim=1)  # (batch, d_repr)

        # Iterative settling: top-down predictions meet bottom-up errors
        for t in range(self.n_settle):
            # Top-down pass: generate predictions (abstract → concrete)
            predictions = {}
            for i in range(n - 1, 0, -1):
                if i in representations:
                    predictions[i - 1] = self.patches[i].predict_down(
                        representations[i]
                    )

            # Bottom-up pass: compute errors, update representations
            for i in range(n):
                if i not in compressed:
                    continue
                pred = predictions.get(i)  # None for top-most patch
                weighted_error, new_repr = self.patches[i].compute_error(
                    compressed[i], pred
                )
                representations[i] = new_repr

        # After settling: compute final corrections from final errors
        # Re-run one last error computation with settled predictions
        final_predictions = {}
        for i in range(n - 1, 0, -1):
            if i in representations:
                final_predictions[i - 1] = self.patches[i].predict_down(
                    representations[i]
                )

        prediction_loss = 0.0
        pred_count = 0
        for i in range(n):
            if i not in compressed:
                continue
            pred = final_predictions.get(i)
            weighted_error, _ = self.patches[i].compute_error(compressed[i], pred)

            # Correction: project weighted error back to Qwen's space
            correction = self.patches[i].error_to_correction(weighted_error)
            self._corrections[i] = correction

            # Prediction loss: how well did higher patches predict this one?
            if pred is not None and i in representations:
                actual_repr = compressed[i].detach().mean(dim=1)
                prediction_loss = prediction_loss + F.mse_loss(pred, actual_repr)
                pred_count += 1

        prediction_loss = prediction_loss / max(pred_count, 1)
        return prediction_loss

    def get_diagnostics(self):
        """Return diagnostic dict for logging."""
        diag = {}
        for i, patch in enumerate(self.patches):
            diag[f"pc/patch{i}_error_norm"] = patch._last_error_norm
            diag[f"pc/patch{i}_precision_mean"] = patch._last_precision_mean
            diag[f"pc/patch{i}_weighted_error_norm"] = patch._last_weighted_error_norm
        # Correction norms
        for i, corr in self._corrections.items():
            if corr is not None:
                diag[f"pc/patch{i}_correction_norm"] = corr.detach().norm().item()
        return diag


class PredictiveCodingPatchedModel(nn.Module):
    """Two-pass model: capture → settle → correct.

    Pass 1 (capture): Qwen runs normally, hooks store hidden states
    Pass 2 (correct): Qwen runs again, hooks apply settled corrections

    Between the two passes, the PC hierarchy settles — iterating
    top-down predictions and bottom-up errors on the captured states.
    """

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

            if self.pc_network._mode == "capture":
                # Phase 1: just observe, don't modify
                self.pc_network._captured[patch_idx] = h.detach()
                return output

            elif self.pc_network._mode == "correct":
                # Phase 3: apply pre-computed correction
                correction = self.pc_network._corrections.get(patch_idx)
                if correction is not None:
                    h_new = h + correction.to(h.dtype)
                    if h_new.requires_grad:
                        h_new.register_hook(_clamp_grad)
                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new

            return output
        return hook

    @property
    def device(self):
        return self.base_model.device

    def forward(self, input_ids, labels=None, attention_mask=None):
        """Two-pass forward: capture → settle → correct.

        Pass 1: Run Qwen, capture hidden states at patch layers (no modification)
        Settle: PC hierarchy iterates on captured states
        Pass 2: Run Qwen again with corrections applied → compute loss
        """
        # Phase 1: Feedforward sweep (observation only)
        self.pc_network._mode = "capture"
        self.pc_network._captured.clear()
        self.pc_network._corrections.clear()
        with torch.no_grad():
            self.base_model(input_ids=input_ids)

        # Phase 2: Recurrent settling (cheap — only PC network, not Qwen)
        prediction_loss = self.pc_network.settle()

        # Phase 3: Corrected forward pass (gradients flow through corrections)
        self.pc_network._mode = "correct"
        outputs = self.base_model(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask
        )

        # Store prediction loss for the training loop to use
        self._last_prediction_loss = prediction_loss

        return outputs

    def generate(self, *args, **kwargs):
        """Generate with two-pass: capture → settle → generate."""
        input_ids = kwargs.get('input_ids', args[0] if args else None)
        if input_ids is None:
            return self.base_model.generate(*args, **kwargs)

        # Capture pass
        self.pc_network._mode = "capture"
        self.pc_network._captured.clear()
        self.pc_network._corrections.clear()
        with torch.no_grad():
            self.base_model(input_ids=input_ids)

        # Settle
        with torch.no_grad():
            self.pc_network.settle()

        # Generate with corrections
        self.pc_network._mode = "correct"
        return self.base_model.generate(*args, **kwargs)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
