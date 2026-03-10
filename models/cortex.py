"""
Cortex: a predictive coding network that reads from Qwen and produces output.

Architecture:
  Qwen runs fully intact (frozen, untouched). Hidden states at multiple layers
  are captured as "sensory input" — like V1, V2, V4, IT projecting to
  association cortex simultaneously.

  The cortex (this module) receives these multi-scale observations, processes
  them through a predictive coding hierarchy, and produces a CORRECTION to
  Qwen's final hidden state. Like prefrontal cortex modulating sensory processing.

  Qwen layer 9  ──→ ╮
  Qwen layer 18 ──→ │  Cortex        ──→  correction
  Qwen layer 27 ──→ │  (PC hierarchy)     ↓
  Qwen layer 34 ──→ ╯               qwen_final + correction → LM head → tokens

No patching. No corruption. Qwen computes normally.
The cortex learns what to ADD to Qwen's thinking.

Learning:
  - Cortex weights: Hebbian + reward modulation (brain-faithful)
  - Input projections + output projection: gradient descent (the "genome")
  - Qwen: completely frozen, never modified
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorticalLayer(nn.Module):
    """One layer of the cortex's predictive coding hierarchy.

    Same as brain.py's CorticalLayer but adapted for the Qwen-integrated
    architecture. Each layer:
      - Compresses input from below
      - Predicts what layer below should see (top-down)
      - Computes precision-weighted prediction errors
      - Updates activations through settling
      - Learns via Hebbian rules gated by reward
    """

    def __init__(self, d_below, d_this, has_top_down=True):
        super().__init__()
        self.d_below = d_below
        self.d_this = d_this
        self.has_top_down = has_top_down

        # Bottom-up recognition
        self.register_buffer('up_W', torch.zeros(d_this, d_below))
        self.register_buffer('up_b', torch.zeros(d_this))
        nn.init.kaiming_uniform_(self.up_W)

        # Top-down prediction
        if has_top_down:
            self.register_buffer('down_W', torch.zeros(d_below, d_this))
            self.register_buffer('down_b', torch.zeros(d_below))
            nn.init.kaiming_uniform_(self.down_W)

        # Lateral recurrence
        self.register_buffer('lateral_W', torch.zeros(d_this, d_this))
        self.register_buffer('lateral_b', torch.zeros(d_this))
        nn.init.orthogonal_(self.lateral_W)
        self.lateral_W.mul_(0.1)

        # Normalization (bounded firing rates)
        self.norm = nn.LayerNorm(d_this)

        # Precision: learned per-dimension inverse-variance
        self.precision_logit = nn.Parameter(torch.zeros(d_this))

        # State
        self._activations = None
        self._prediction_error = None
        self._input_from_below = None
        self._prediction_from_above = None

        # Diagnostics
        self._last_error_norm = 0.0
        self._last_activation_norm = 0.0

    def reset(self):
        self._activations = None
        self._prediction_error = None
        self._input_from_below = None
        self._prediction_from_above = None

    def compress(self, x_below):
        """Bottom-up compression."""
        self._input_from_below = x_below
        compressed = F.linear(x_below, self.up_W, self.up_b)
        return self.norm(compressed)

    def predict_below(self):
        """Top-down prediction."""
        if not self.has_top_down or self._activations is None:
            return None
        return F.linear(self._activations, self.down_W, self.down_b)

    def compute_error(self, compressed, prediction_from_above):
        """Compute precision-weighted prediction error, update activations."""
        self._prediction_from_above = prediction_from_above

        if prediction_from_above is not None:
            error = compressed - prediction_from_above
        else:
            if self._activations is not None:
                error = compressed - self._activations
            else:
                error = compressed

        precision = torch.sigmoid(self.precision_logit)
        weighted_error = error * precision
        self._prediction_error = weighted_error

        lateral = F.linear(
            self._activations if self._activations is not None else compressed,
            self.lateral_W, self.lateral_b
        )

        base = self._activations if self._activations is not None else torch.zeros_like(compressed)
        self._activations = self.norm(base + 0.5 * weighted_error + 0.1 * lateral)

        self._last_error_norm = error.detach().norm().item()
        self._last_activation_norm = self._activations.detach().norm().item()

        return weighted_error

    def hebbian_update(self, reward, eta=0.001):
        """Hebbian learning gated by reward."""
        if self._input_from_below is None:
            return

        batch = self._input_from_below.shape[0]

        # Recognition: reduce prediction error
        if self._prediction_error is not None:
            err = self._prediction_error.detach()
            inp = self._input_from_below.detach()
            delta_up = torch.einsum('bi,bj->ij', err, inp) / batch
            self.up_W.add_(eta * reward * delta_up)

        # Prediction: self-supervised (always update)
        if self.has_top_down and self._activations is not None:
            act = self._activations.detach()
            target = self._input_from_below.detach()
            prediction = F.linear(act, self.down_W, self.down_b)
            pred_error = target - prediction
            delta_down = torch.einsum('bi,bj->ij', pred_error, act) / batch
            self.down_W.add_(eta * delta_down)

        # Lateral: reward-gated
        if self._activations is not None:
            act = self._activations.detach()
            delta_lat = torch.einsum('bi,bj->ij', act, act) / batch
            delta_lat.fill_diagonal_(0.0)
            self.lateral_W.mul_(0.999)
            self.lateral_W.add_(eta * 0.1 * reward * delta_lat)

        # Homeostatic plasticity
        max_norm = 30.0
        for W in [self.up_W, self.lateral_W]:
            norm = W.norm()
            if norm > max_norm:
                W.mul_(max_norm / norm)
        if self.has_top_down:
            norm = self.down_W.norm()
            if norm > max_norm:
                self.down_W.mul_(max_norm / norm)


class Cortex(nn.Module):
    """Predictive coding cortex that reads from Qwen.

    Observes Qwen's hidden states at multiple layers (multi-scale sensory input),
    processes through its own PC hierarchy, and produces a final representation
    for Qwen's LM head to generate from.

    The cortex never modifies Qwen's computation. It only reads and thinks.
    """

    def __init__(self, d_model, d_cortical=512, n_layers=4, n_settle=5,
                 observe_layers=None):
        super().__init__()
        if observe_layers is None:
            observe_layers = [9, 18, 27, 34]
        self.observe_layers = observe_layers
        self.d_model = d_model
        self.d_cortical = d_cortical
        self.n_layers = n_layers
        self.n_settle = n_settle

        # === Genome: gradient-trained I/O projections ===

        # Multi-scale input projections (one per observed Qwen layer)
        # Each projects d_model → d_cortical
        self.input_projs = nn.ModuleList([
            nn.Linear(d_model, d_cortical) for _ in observe_layers
        ])

        # Fuse multi-scale inputs into single cortical input
        self.fuse = nn.Linear(d_cortical * len(observe_layers), d_cortical)

        # Output projection: cortical activation → correction in Qwen's space
        # The cortex doesn't replace Qwen's output — it corrects it.
        # Like prefrontal cortex modulating sensory processing.
        # Last layer zero-init so corrections start at zero (baseline = Qwen).
        self.output_proj = nn.Sequential(
            nn.Linear(d_cortical, d_cortical),
            nn.GELU(),
            nn.Linear(d_cortical, d_model),
        )
        # Zero-init last layer: cortex starts as identity on Qwen's output
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        # === Plastic cortex: Hebbian-trained hierarchy ===

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            has_top_down = (i < n_layers - 1)
            self.layers.append(
                CorticalLayer(d_cortical, d_cortical, has_top_down)
            )

        # Critic for TD learning
        self.critic = nn.Sequential(
            nn.Linear(d_cortical, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._last_value = None

        # State
        self._observations = {}  # {layer_idx: hidden_state}
        self._last_correction_norm = 0.0

    def reset(self):
        """Reset all cortical state."""
        self._observations.clear()
        for layer in self.layers:
            layer.reset()

    def reset_episode(self):
        """Reset episode-level state."""
        self.reset()
        self._last_value = None

    def observe(self, layer_idx, hidden_state):
        """Store an observation from a Qwen layer.

        Called by hooks during Qwen's forward pass. Read-only — does not
        modify Qwen's computation.

        Args:
            layer_idx: which observed layer (index into observe_layers)
            hidden_state: (batch, seq, d_model) detached hidden state
        """
        self._observations[layer_idx] = hidden_state.detach()

    def think(self, qwen_final_hidden):
        """Process observations through the PC hierarchy per-position.

        Takes multi-scale Qwen observations, fuses them, settles the
        cortical hierarchy independently at each sequence position, and
        produces a correction to Qwen's final hidden state.

        output = qwen_final + cortex_correction

        At init, correction ≈ 0 (zero-init last layer), so we start at
        Qwen's baseline accuracy. The cortex learns what to ADD.

        Args:
            qwen_final_hidden: (batch, seq, d_model) Qwen's final hidden state

        Returns:
            output_hidden: (batch, seq, d_model) corrected hidden state for LM head
        """
        batch, seq, _ = qwen_final_hidden.shape

        # Project each observed layer into cortical space
        projected = []
        for i in range(len(self.observe_layers)):
            obs = self._observations.get(i)
            if obs is None:
                obs = torch.zeros(batch, seq, self.d_model,
                                  device=qwen_final_hidden.device)
            projected.append(self.input_projs[i](obs.float()))

        # Fuse: (batch, seq, d_cortical * n_obs) → (batch, seq, d_cortical)
        fused = self.fuse(torch.cat(projected, dim=-1))

        # Reshape to process all positions as a batch
        # (batch, seq, d_cortical) → (batch * seq, d_cortical)
        cortical_input = fused.view(batch * seq, self.d_cortical)

        # Settle the PC hierarchy (operates on all positions simultaneously)
        x = cortical_input
        for layer in self.layers:
            x = layer.compress(x)
            layer._activations = x

        for t in range(self.n_settle):
            predictions = {}
            for i in range(len(self.layers) - 1, 0, -1):
                pred = self.layers[i].predict_below()
                if pred is not None:
                    predictions[i - 1] = pred

            x = cortical_input
            for i, layer in enumerate(self.layers):
                compressed = layer.compress(x)
                pred_from_above = predictions.get(i)
                layer.compute_error(compressed, pred_from_above)
                x = layer._activations

        # (batch * seq, d_cortical)
        top_activation = self.layers[-1]._activations

        # Cortex produces a CORRECTION to Qwen's output
        # Residual: output = qwen + correction (correction starts at 0)
        correction = self.output_proj(top_activation)
        correction = correction.view(batch, seq, self.d_model)
        self._last_correction_norm = correction.detach().norm().item()
        output_hidden = qwen_final_hidden.float() + correction
        return output_hidden.to(qwen_final_hidden.dtype)

    def compute_value(self):
        """Critic value from top-layer activation."""
        top = self.layers[-1]._activations
        if top is None:
            return torch.tensor(0.0)
        return self.critic(top.detach()).squeeze(-1)

    def hebbian_step(self, reward, device=None):
        """Apply Hebbian updates to cortical layers."""
        current_value = self.compute_value()
        if self._last_value is not None:
            td_error = reward + 0.5 * current_value.mean().item() - self._last_value.mean().item()
        else:
            td_error = reward
        self._last_value = current_value.detach()

        for layer in self.layers:
            layer.hebbian_update(td_error, eta=0.001)

        # Critic loss
        if current_value.requires_grad:
            target = torch.tensor(reward, device=current_value.device, dtype=torch.float32)
            critic_loss = F.mse_loss(current_value.mean(), target)
        else:
            critic_loss = torch.tensor(0.0)

        return td_error, critic_loss

    def get_diagnostics(self):
        """Return diagnostic dict."""
        diag = {}
        for i, layer in enumerate(self.layers):
            diag[f"cortex/layer{i}_error_norm"] = layer._last_error_norm
            diag[f"cortex/layer{i}_activation_norm"] = layer._last_activation_norm
            diag[f"cortex/layer{i}_up_W_norm"] = layer.up_W.norm().item()
            if layer.has_top_down:
                diag[f"cortex/layer{i}_down_W_norm"] = layer.down_W.norm().item()
            diag[f"cortex/layer{i}_lateral_W_norm"] = layer.lateral_W.norm().item()
            diag[f"cortex/layer{i}_precision"] = torch.sigmoid(
                layer.precision_logit
            ).mean().item()
        diag["cortex/output_proj_norm"] = sum(
            p.norm().item() for p in self.output_proj.parameters()
        )
        diag["cortex/correction_norm"] = self._last_correction_norm
        if self._last_value is not None:
            diag["cortex/critic_value"] = self._last_value.mean().item()
        return diag


class CortexModel(nn.Module):
    """Qwen + Cortex: frozen LLM with a reading brain on top.

    Qwen runs fully intact. Hooks capture hidden states at observed layers
    (read-only). After Qwen finishes, the cortex processes the observations
    and produces a modified final hidden state for generation.

    No patching. No corruption. Qwen computes normally.
    """

    def __init__(self, base_model, cortex):
        super().__init__()
        self.base_model = base_model
        self.cortex = cortex
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register read-only hooks on Qwen layers."""
        for i, layer_idx in enumerate(self.cortex.observe_layers):
            layer = self.base_model.model.layers[layer_idx]
            handle = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(handle)

    def _make_hook(self, obs_idx):
        """Create a read-only hook that captures hidden states."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            # Read-only: capture but never modify
            self.cortex.observe(obs_idx, h)
            return output  # pass through unchanged
        return hook

    @property
    def device(self):
        return self.base_model.device

    def forward(self, input_ids, labels=None, attention_mask=None):
        """Forward pass: Qwen runs intact, cortex modifies final hidden state.

        1. Qwen forward (hooks capture observations, no modification)
        2. Cortex processes observations → modified hidden state
        3. LM head generates from modified hidden state → loss
        """
        self.cortex.reset()

        # Run Qwen fully intact — hooks capture observations
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Get Qwen's final hidden state (what it would normally generate from)
        qwen_final = outputs.hidden_states[-1]  # (batch, seq, d_model)

        # Cortex: observe → think → propose
        cortex_output = self.cortex.think(qwen_final)

        # Generate logits from cortex's output using Qwen's LM head
        logits = self.base_model.lm_head(cortex_output)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Store for training loop
        self._last_loss = loss
        self._last_logits = logits

        return type(outputs)(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def generate(self, input_ids, max_new_tokens=20, **kwargs):
        """Generate with cortex: Qwen runs intact, cortex modifies output.

        For generation, we run Qwen normally to get observations,
        then use cortex to modify the final hidden state, then
        use Qwen's LM head for next-token prediction.
        """
        device = input_ids.device
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            self.cortex.reset()

            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=generated,
                    output_hidden_states=True,
                )

            qwen_final = outputs.hidden_states[-1]
            cortex_output = self.cortex.think(qwen_final)

            # Only need logits for last position
            with torch.no_grad():
                logits = self.base_model.lm_head(cortex_output[:, -1:, :])

            next_token = logits.argmax(dim=-1)  # (batch, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS
            if next_token.item() == self.base_model.config.eos_token_id:
                break

        return generated

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
