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

Two learning modes:
  - GRADIENT: Standard backprop through the two-pass architecture
  - HEBBIAN+RPE: Brain-faithful local learning rules:
      * compress, predict_down: self-supervised (prediction errors drive updates)
      * error_to_correction, precision: TD-error gated (reward shapes output)
    Prediction errors ARE the natural eligibility traces — no manufactured traces.
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

    In Hebbian mode, weights are stored as buffers (not parameters) and
    updated via local learning rules gated by TD error.
    """

    def __init__(self, d_model, d_repr=128, hebbian=False):
        super().__init__()
        self.d_model = d_model
        self.d_repr = d_repr
        self.hebbian = hebbian

        # Normalize compressed representations (bounded firing rates)
        self.compress_norm = nn.LayerNorm(d_repr)

        if hebbian:
            # Weights as buffers — updated by Hebbian rules, not optimizer
            # compress: d_model → d_repr
            self.register_buffer('compress_W', torch.zeros(d_repr, d_model))
            self.register_buffer('compress_b', torch.zeros(d_repr))
            nn.init.kaiming_uniform_(self.compress_W)

            # predict_down: d_repr → d_repr
            self.register_buffer('predict_down_W', torch.zeros(d_repr, d_repr))
            self.register_buffer('predict_down_b', torch.zeros(d_repr))
            nn.init.kaiming_uniform_(self.predict_down_W)

            # precision: d_repr*2 → d_repr
            self.register_buffer('precision_W', torch.zeros(d_repr, d_repr * 2))
            self.register_buffer('precision_b', torch.zeros(d_repr))
            nn.init.kaiming_uniform_(self.precision_W)

            # error_to_correction: d_repr → d_model (starts at zero)
            self.register_buffer('correction_W', torch.zeros(d_model, d_repr))
            self.register_buffer('correction_b', torch.zeros(d_model))

            # Learned learning rates (as parameters — meta-learned)
            self.eta_compress = nn.Parameter(torch.tensor(-3.0))      # sigmoid → ~0.05
            self.eta_predict = nn.Parameter(torch.tensor(-3.0))
            self.eta_precision = nn.Parameter(torch.tensor(-4.0))     # smaller — TD-gated
            self.eta_correction = nn.Parameter(torch.tensor(-4.0))
        else:
            # Standard nn.Linear (gradient-based training)
            self.compress = nn.Linear(d_model, d_repr)
            self.predict_down = nn.Linear(d_repr, d_repr)
            self.precision = nn.Linear(d_repr * 2, d_repr)
            self.error_to_correction = nn.Linear(d_repr, d_model)
            nn.init.normal_(self.error_to_correction.weight, std=0.01)
            nn.init.zeros_(self.error_to_correction.bias)

        # Diagnostics (set during settle)
        self._last_error_norm = 0.0
        self._last_precision_mean = 0.0
        self._last_weighted_error_norm = 0.0

        # Traces for Hebbian update (set during settle, consumed by hebbian_step)
        self._trace_compressed = None      # (batch, seq, d_repr)
        self._trace_prediction = None      # (batch, d_repr) or None
        self._trace_error = None           # (batch, seq, d_repr)
        self._trace_weighted_error = None  # (batch, seq, d_repr)
        self._trace_hidden_in = None       # (batch, seq, d_model) — Qwen hidden state
        self._trace_precision_input = None # (batch, seq, d_repr*2)

    def _linear(self, x, W, b):
        """Apply linear transform using buffer weights."""
        return F.linear(x, W, b)

    def do_compress(self, h):
        """Bottom-up compression: Qwen hidden state → representation space."""
        if self.hebbian:
            return self.compress_norm(self._linear(h, self.compress_W, self.compress_b))
        return self.compress_norm(self.compress(h))

    def do_predict_down(self, representation):
        """Top-down prediction: my representation → what level below should see."""
        if self.hebbian:
            return self._linear(representation, self.predict_down_W, self.predict_down_b)
        return self.predict_down(representation)

    def do_precision(self, precision_input):
        """Precision weighting."""
        if self.hebbian:
            return self._linear(precision_input, self.precision_W, self.precision_b)
        return self.precision(precision_input)

    def do_error_to_correction(self, weighted_error):
        """Project precision-weighted error back to Qwen's residual stream."""
        if self.hebbian:
            return self._linear(weighted_error, self.correction_W, self.correction_b)
        return self.error_to_correction(weighted_error)

    def compute_error(self, compressed, prediction, hidden_in=None):
        """Compute precision-weighted prediction error.

        Args:
            compressed: compressed bottom-up features (batch, seq, d_repr)
            prediction: top-down prediction (batch, d_repr) or None
            hidden_in: original Qwen hidden state (for Hebbian traces)

        Returns:
            weighted_error: precision-weighted error (batch, seq, d_repr)
            representation: updated belief (batch, d_repr)
        """
        if prediction is not None:
            pred = prediction.unsqueeze(1).expand_as(compressed)
            error = compressed - pred
        else:
            error = compressed

        precision_input = torch.cat([compressed, error], dim=-1)
        precision_weights = torch.sigmoid(self.do_precision(precision_input))
        weighted_error = error * precision_weights

        if prediction is not None:
            representation = prediction + weighted_error.mean(dim=1)
        else:
            representation = compressed.mean(dim=1)

        # Diagnostics
        self._last_error_norm = error.detach().norm().item()
        self._last_precision_mean = precision_weights.detach().mean().item()
        self._last_weighted_error_norm = weighted_error.detach().norm().item()

        # Store traces for Hebbian update (only final settle iteration matters)
        if self.hebbian:
            self._trace_compressed = compressed.detach()
            self._trace_prediction = prediction.detach() if prediction is not None else None
            self._trace_error = error.detach()
            self._trace_weighted_error = weighted_error.detach()
            self._trace_hidden_in = hidden_in
            self._trace_precision_input = precision_input.detach()

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

    Supports both gradient-based and Hebbian+RPE learning modes.
    """

    def __init__(self, d_model, d_repr=128, patch_layers=None, n_settle=3,
                 hebbian=False, correction_scale=0.1):
        super().__init__()
        if patch_layers is None:
            patch_layers = [9, 18, 27, 34]
        self.patch_layers = patch_layers
        self.d_model = d_model
        self.d_repr = d_repr
        self.n_settle = n_settle
        self.hebbian = hebbian
        self.correction_scale = correction_scale

        self.patches = nn.ModuleList([
            PredictiveCodingPatch(d_model, d_repr, hebbian=hebbian)
            for _ in patch_layers
        ])

        if hebbian:
            # Simple critic for TD error (value function)
            self.critic = nn.Sequential(
                nn.Linear(d_repr * len(patch_layers), 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            self._last_value = None  # V(s_t-1) for TD error

        # State set during two-pass forward
        self._captured = {}          # {patch_idx: hidden_state} from first pass
        self._corrections = {}       # {patch_idx: correction} from settling
        self._mode = "capture"       # "capture" | "correct"

    def reset(self):
        """Clear all state."""
        self._captured.clear()
        self._corrections.clear()
        self._mode = "capture"

    def reset_episode(self):
        """Reset episode-level state (for Hebbian mode)."""
        self.reset()
        if self.hebbian:
            self._last_value = None
            for patch in self.patches:
                patch._trace_compressed = None
                patch._trace_prediction = None
                patch._trace_error = None
                patch._trace_weighted_error = None
                patch._trace_hidden_in = None
                patch._trace_precision_input = None

    def settle(self):
        """Run predictive coding hierarchy on captured states until convergence.

        Returns:
            prediction_loss: auxiliary loss for training the predictive model
        """
        n = len(self.patches)

        # Compress all captured hidden states
        compressed = {}
        for i in range(n):
            h = self._captured.get(i)
            if h is None:
                continue
            compressed[i] = self.patches[i].do_compress(h.float())

        # Initialize representations from compressed activations
        representations = {}
        for i in range(n):
            if i in compressed:
                representations[i] = compressed[i].mean(dim=1)

        # Iterative settling: top-down predictions meet bottom-up errors
        for t in range(self.n_settle):
            predictions = {}
            for i in range(n - 1, 0, -1):
                if i in representations:
                    predictions[i - 1] = self.patches[i].do_predict_down(
                        representations[i]
                    )

            for i in range(n):
                if i not in compressed:
                    continue
                pred = predictions.get(i)
                hidden_in = self._captured.get(i)
                weighted_error, new_repr = self.patches[i].compute_error(
                    compressed[i], pred,
                    hidden_in=hidden_in.detach() if hidden_in is not None else None
                )
                representations[i] = new_repr

        # Final corrections from settled errors
        final_predictions = {}
        for i in range(n - 1, 0, -1):
            if i in representations:
                final_predictions[i - 1] = self.patches[i].do_predict_down(
                    representations[i]
                )

        prediction_loss = 0.0
        pred_count = 0
        for i in range(n):
            if i not in compressed:
                continue
            pred = final_predictions.get(i)
            hidden_in = self._captured.get(i)
            weighted_error, _ = self.patches[i].compute_error(
                compressed[i], pred,
                hidden_in=hidden_in.detach() if hidden_in is not None else None
            )

            correction = self.patches[i].do_error_to_correction(weighted_error)
            self._corrections[i] = correction * self.correction_scale

            if pred is not None and i in representations:
                actual_repr = compressed[i].detach().mean(dim=1)
                prediction_loss = prediction_loss + F.mse_loss(pred, actual_repr)
                pred_count += 1

        prediction_loss = prediction_loss / max(pred_count, 1)
        # Clamp to prevent unbounded MSE from destabilizing training
        prediction_loss = torch.clamp(prediction_loss, max=10.0)

        # Compute critic value for Hebbian TD error
        if self.hebbian:
            repr_cat = torch.cat(
                [representations[i] for i in range(n) if i in representations],
                dim=-1
            ).detach()
            self._current_value = self.critic(repr_cat).squeeze(-1)

        return prediction_loss

    def hebbian_step(self, reward, device):
        """Apply Hebbian+RPE updates to all PC patch buffers.

        Four update rules:
        1. compress: self-supervised — prediction error drives compression learning
           delta_W = eta * error^T @ hidden_in  (learn to compress what reduces error)
        2. predict_down: self-supervised — minimize prediction error
           delta_W = eta * error^T @ representation  (predict what you'll see)
        3. precision: TD-gated — learn which errors to trust based on reward
           delta_W = td_error * precision_grad @ precision_input
        4. error_to_correction: TD-gated — learn what corrections help
           delta_W = td_error * correction_grad @ weighted_error

        Args:
            reward: scalar reward for this problem
            device: torch device
        """
        if not self.hebbian:
            return 0.0

        reward_t = torch.tensor(reward, dtype=torch.float32, device=device)

        # Compute TD error: δ = r + γ·V(s_t) - V(s_{t-1})
        gamma = 0.5
        current_v = self._current_value.detach() if self._current_value is not None else torch.tensor(0.0, device=device)
        last_v = self._last_value.detach() if self._last_value is not None else torch.tensor(0.0, device=device)
        td_error = reward_t + gamma * current_v.mean() - last_v.mean()
        self._last_value = self._current_value.detach() if self._current_value is not None else None

        # Critic loss for training the value function
        if self._current_value is not None:
            critic_loss = F.mse_loss(self._current_value.mean(), reward_t)
        else:
            critic_loss = torch.tensor(0.0, device=device)

        n = len(self.patches)
        for i in range(n):
            patch = self.patches[i]
            if patch._trace_compressed is None:
                continue

            # Learning rates (meta-learned via sigmoid)
            eta_c = torch.sigmoid(patch.eta_compress) * 0.01    # max 0.01
            eta_p = torch.sigmoid(patch.eta_predict) * 0.01
            eta_pr = torch.sigmoid(patch.eta_precision) * 0.005  # smaller — TD-gated
            eta_cr = torch.sigmoid(patch.eta_correction) * 0.005

            # --- 1. compress (self-supervised): learn to compress what reduces error ---
            # delta_W = eta * mean(error)^T @ mean(hidden_in)
            # Error = compressed - prediction, so compress should move toward prediction
            if patch._trace_error is not None and patch._trace_hidden_in is not None:
                err_mean = patch._trace_error.mean(dim=1)    # (batch, d_repr)
                h_mean = patch._trace_hidden_in.float().mean(dim=1)  # (batch, d_model)
                # Outer product: (d_repr, d_model)
                delta = torch.einsum('bi,bj->ij', err_mean, h_mean) / err_mean.shape[0]
                patch.compress_W.add_(-eta_c * delta)  # negative: reduce error

            # --- 2. predict_down (self-supervised): minimize prediction error ---
            # The prediction at level i comes from level i+1's predict_down
            # Update: delta_W = eta * error_below^T @ representation_above
            # This is handled at the network level (see below)

            # --- 3. precision (TD-gated): learn which errors to trust ---
            # Positive TD error → increase precision on errors that were active
            # Negative TD error → decrease precision
            if patch._trace_precision_input is not None and patch._trace_error is not None:
                prec_in = patch._trace_precision_input.mean(dim=1)  # (batch, d_repr*2)
                err_sign = patch._trace_error.sign().mean(dim=1)    # (batch, d_repr)
                delta = torch.einsum('bi,bj->ij', err_sign, prec_in) / prec_in.shape[0]
                patch.precision_W.add_(eta_pr * td_error * delta)

            # --- 4. error_to_correction (TD-gated): learn what corrections help ---
            # Positive TD error → reinforce the correction direction
            if patch._trace_weighted_error is not None and patch._trace_hidden_in is not None:
                we_mean = patch._trace_weighted_error.mean(dim=1)  # (batch, d_repr)
                # correction_W: (d_model, d_repr)
                # We want to reinforce the correction that was produced
                # correction = W @ weighted_error, so delta_W ∝ correction_direction @ weighted_error^T
                # But simpler: just do td_error * weighted_error outer product with itself
                delta = torch.einsum('bi,bj->ji', we_mean, we_mean) / we_mean.shape[0]
                # Scale by 1/norm to prevent explosion
                delta_norm = delta.norm()
                if delta_norm > 0:
                    delta = delta / delta_norm
                patch.correction_W.add_(eta_cr * td_error * delta)

        # --- predict_down updates (cross-level) ---
        # Level i+1's predict_down should predict level i's compressed activation
        for i in range(n - 1):
            patch_above = self.patches[i + 1]
            patch_below = self.patches[i]
            if patch_below._trace_compressed is None:
                continue

            eta_p = torch.sigmoid(patch_above.eta_predict) * 0.01
            # Target: what level below actually saw (compressed)
            target = patch_below._trace_compressed.detach().mean(dim=1)  # (batch, d_repr)
            # Prediction: what level above predicted
            pred = patch_below._trace_prediction  # (batch, d_repr) or None
            if pred is not None:
                pred_error = target - pred  # (batch, d_repr)
                # predict_down input was the representation of level above
                # We need to reconstruct it — use compressed mean of above
                if patch_above._trace_compressed is not None:
                    repr_above = patch_above._trace_compressed.detach().mean(dim=1)
                    # predict_down_W: (d_repr, d_repr)
                    delta = torch.einsum('bi,bj->ij', pred_error, repr_above) / pred_error.shape[0]
                    patch_above.predict_down_W.add_(eta_p * delta)

        return critic_loss

    def get_diagnostics(self):
        """Return diagnostic dict for logging."""
        diag = {}
        for i, patch in enumerate(self.patches):
            diag[f"pc/patch{i}_error_norm"] = patch._last_error_norm
            diag[f"pc/patch{i}_precision_mean"] = patch._last_precision_mean
            diag[f"pc/patch{i}_weighted_error_norm"] = patch._last_weighted_error_norm
        for i, corr in self._corrections.items():
            if corr is not None:
                diag[f"pc/patch{i}_correction_norm"] = corr.detach().norm().item()
        diag["pc/correction_scale"] = self.correction_scale
        # Hebbian-specific diagnostics
        if self.hebbian:
            for i, patch in enumerate(self.patches):
                diag[f"pc/patch{i}_eta_compress"] = torch.sigmoid(patch.eta_compress).item()
                diag[f"pc/patch{i}_eta_predict"] = torch.sigmoid(patch.eta_predict).item()
                diag[f"pc/patch{i}_eta_precision"] = torch.sigmoid(patch.eta_precision).item()
                diag[f"pc/patch{i}_eta_correction"] = torch.sigmoid(patch.eta_correction).item()
                diag[f"pc/patch{i}_compress_W_norm"] = patch.compress_W.norm().item()
                diag[f"pc/patch{i}_correction_W_norm"] = patch.correction_W.norm().item()
            if self._last_value is not None:
                diag["pc/critic_value"] = self._last_value.mean().item()
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
