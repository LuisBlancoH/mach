"""
Minimal viable brain: a standalone predictive coding network.

No Qwen. No transformers. No backprop in the core.

Each cortical layer:
  - Compresses input from below (bottom-up recognition)
  - Predicts what the layer below should see (top-down generation)
  - Computes prediction error (surprise)
  - Updates activations through iterative settling
  - Learns via Hebbian rules gated by reward (neuromodulation)

The settling loop IS the computation. No single forward pass —
the network iterates until its predictions match its inputs.

Architecture:
  Input → Embed → [Layer 0] ↔ [Layer 1] ↔ ... ↔ [Layer N] → Readout → Output
                   ↕ settle    ↕ settle          ↕ settle

Learning:
  - Internal weights: Hebbian + reward modulation (local, no backprop)
  - Embeddings + readout: gradient descent (the "genome" — pre-structured I/O)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorticalLayer(nn.Module):
    """One layer of the predictive coding hierarchy.

    Each layer maintains:
      - activations: current belief about the world at this level
      - up_weights: compress input from below into this layer's space
      - down_weights: predict what the layer below should look like
      - lateral_weights: recurrent connections within this layer

    Learning is Hebbian: connections that fire together wire together,
    gated by a global reward signal (neuromodulation).
    """

    def __init__(self, d_below, d_this, has_top_down=True):
        super().__init__()
        self.d_below = d_below
        self.d_this = d_this
        self.has_top_down = has_top_down

        # Bottom-up recognition: compress input from layer below
        self.register_buffer('up_W', torch.zeros(d_this, d_below))
        self.register_buffer('up_b', torch.zeros(d_this))
        nn.init.kaiming_uniform_(self.up_W)

        # Top-down prediction: predict what layer below should see
        if has_top_down:
            self.register_buffer('down_W', torch.zeros(d_below, d_this))
            self.register_buffer('down_b', torch.zeros(d_below))
            nn.init.kaiming_uniform_(self.down_W)

        # Lateral recurrence: within-layer dynamics
        self.register_buffer('lateral_W', torch.zeros(d_this, d_this))
        self.register_buffer('lateral_b', torch.zeros(d_this))
        nn.init.orthogonal_(self.lateral_W)
        self.lateral_W.mul_(0.1)  # start weak

        # Layer norm (bounded firing rates)
        self.norm = nn.LayerNorm(d_this)

        # Precision: learned inverse-variance per dimension
        self.precision_logit = nn.Parameter(torch.zeros(d_this))

        # State (set during processing)
        self.register_buffer('_activations', torch.zeros(1, d_this))
        self._prediction_error = None
        self._input_from_below = None
        self._prediction_from_above = None

        # Diagnostics
        self._last_error_norm = 0.0
        self._last_activation_norm = 0.0

    def reset(self):
        """Clear activations between inputs."""
        self._activations.zero_()
        self._prediction_error = None
        self._input_from_below = None
        self._prediction_from_above = None

    def compress(self, x_below):
        """Bottom-up: compress input from layer below.

        Args:
            x_below: (batch, d_below) input from the layer below
        """
        self._input_from_below = x_below
        compressed = F.linear(x_below, self.up_W, self.up_b)
        return self.norm(compressed)

    def predict_below(self):
        """Top-down: predict what the layer below should look like.

        Returns:
            prediction: (batch, d_below) or None if no top-down
        """
        if not self.has_top_down:
            return None
        return F.linear(self._activations, self.down_W, self.down_b)

    def compute_error(self, compressed, prediction_from_above):
        """Compute prediction error and update activations.

        Args:
            compressed: bottom-up input (batch, d_this)
            prediction_from_above: what the layer above predicts we should be,
                                    or None for the top layer
        """
        self._prediction_from_above = prediction_from_above

        if prediction_from_above is not None:
            error = compressed - prediction_from_above
        else:
            # Top layer: everything is error (no prediction to compare to)
            error = compressed - self._activations

        # Precision weighting
        precision = torch.sigmoid(self.precision_logit)
        weighted_error = error * precision

        self._prediction_error = weighted_error

        # Update activations: blend current state with new evidence
        # Lateral recurrence adds temporal dynamics
        lateral = F.linear(self._activations, self.lateral_W, self.lateral_b)
        self._activations = self.norm(
            self._activations + 0.5 * weighted_error + 0.1 * lateral
        )

        # Diagnostics
        self._last_error_norm = error.detach().norm().item()
        self._last_activation_norm = self._activations.detach().norm().item()

        return weighted_error

    def hebbian_update(self, reward, eta=0.001):
        """Update weights using Hebbian learning gated by reward.

        Three update rules:
        1. up_W (recognition): learn to compress inputs that reduce error
        2. down_W (prediction): learn to predict what layer below looks like
        3. lateral_W (recurrence): learn temporal dynamics

        All gated by reward: positive reward strengthens active connections,
        negative reward weakens them.

        Args:
            reward: scalar reward signal (neuromodulator)
            eta: learning rate
        """
        if self._input_from_below is None:
            return

        batch = self._input_from_below.shape[0]

        # 1. Recognition weights: reduce prediction error
        # delta_W = eta * reward * error^T @ input_below
        if self._prediction_error is not None:
            err = self._prediction_error.detach()
            inp = self._input_from_below.detach()
            delta_up = torch.einsum('bi,bj->ij', err, inp) / batch
            self.up_W.add_(eta * reward * delta_up)

        # 2. Prediction weights: predict layer below from activations
        if self.has_top_down and self._input_from_below is not None:
            act = self._activations.detach()
            target = self._input_from_below.detach()
            prediction = F.linear(act, self.down_W, self.down_b)
            pred_error = target - prediction
            delta_down = torch.einsum('bi,bj->ij', pred_error, act) / batch
            # Prediction is self-supervised: always update (not gated by reward)
            self.down_W.add_(eta * delta_down)

        # 3. Lateral weights: stabilize useful activation patterns
        if self._activations is not None:
            act = self._activations.detach()
            # Simple Hebbian: neurons that fire together wire together
            delta_lat = torch.einsum('bi,bj->ij', act, act) / batch
            # Subtract diagonal to prevent self-excitation
            delta_lat.fill_diagonal_(0.0)
            # Decay + reward-gated update
            self.lateral_W.mul_(0.999)  # slow decay
            self.lateral_W.add_(eta * 0.1 * reward * delta_lat)


class Brain(nn.Module):
    """A minimal predictive coding brain.

    No transformers, no frozen base model. Just a hierarchy of cortical
    layers that learn through prediction errors and reward modulation.

    Architecture:
      Token → Embedding → CorticalLayer stack → Readout → Logits

    The embedding and readout are trained with gradient descent (the "genome").
    The cortical layers learn with Hebbian rules (the plastic brain).
    """

    def __init__(self, vocab_size, d_embed=64, d_cortical=256,
                 n_layers=4, n_settle=5, max_seq_len=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_cortical = d_cortical
        self.n_layers = n_layers
        self.n_settle = n_settle
        self.max_seq_len = max_seq_len

        # === The "genome": pre-structured I/O (gradient-trained) ===

        # Token embeddings
        self.embed = nn.Embedding(vocab_size, d_embed)

        # Position embeddings (simple learned)
        self.pos_embed = nn.Embedding(max_seq_len, d_embed)

        # Flatten sequence into single vector for cortical processing
        # Brain doesn't process sequences — it processes a gestalt
        self.seq_compress = nn.Linear(d_embed * max_seq_len, d_cortical)

        # Output readout: cortical activation → token logits
        self.readout = nn.Linear(d_cortical, vocab_size)

        # === The "brain": plastic cortical hierarchy (Hebbian-trained) ===

        # Build cortical layers bottom-up
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            d_below = d_cortical  # all layers same size for simplicity
            d_this = d_cortical
            has_top_down = (i < n_layers - 1)  # top layer has no top-down
            self.layers.append(CorticalLayer(d_below, d_this, has_top_down))

        # Value estimate for TD learning (part of the "genome")
        self.critic = nn.Sequential(
            nn.Linear(d_cortical, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._last_value = None

    def reset(self):
        """Reset all cortical state between inputs."""
        for layer in self.layers:
            layer.reset()
        self._last_value = None

    def _embed_sequence(self, token_ids):
        """Embed token sequence into a single cortical input vector.

        Args:
            token_ids: (batch, seq_len) token indices

        Returns:
            x: (batch, d_cortical) flattened embedded sequence
        """
        batch, seq_len = token_ids.shape

        # Pad or truncate to max_seq_len
        if seq_len < self.max_seq_len:
            padding = torch.zeros(
                batch, self.max_seq_len - seq_len,
                dtype=token_ids.dtype, device=token_ids.device
            )
            token_ids = torch.cat([token_ids, padding], dim=1)
        elif seq_len > self.max_seq_len:
            token_ids = token_ids[:, :self.max_seq_len]

        # Token + position embeddings
        positions = torch.arange(self.max_seq_len, device=token_ids.device)
        x = self.embed(token_ids) + self.pos_embed(positions)

        # Flatten sequence into single vector
        x = x.view(batch, -1)  # (batch, d_embed * max_seq_len)
        x = self.seq_compress(x)  # (batch, d_cortical)

        return x

    def settle(self, cortical_input):
        """Run the predictive coding settling loop.

        This IS the computation. The hierarchy iterates, passing predictions
        down and errors up, until it converges on an interpretation.

        Args:
            cortical_input: (batch, d_cortical) embedded input

        Returns:
            top_activation: (batch, d_cortical) settled top-layer activation
        """
        # Bottom-up initial pass: compress input through all layers
        x = cortical_input
        for layer in self.layers:
            x = layer.compress(x)
            layer._activations = x

        # Iterative settling: top-down predictions meet bottom-up errors
        for t in range(self.n_settle):
            # Top-down pass: generate predictions (abstract → concrete)
            predictions = {}
            for i in range(len(self.layers) - 1, 0, -1):
                pred = self.layers[i].predict_below()
                if pred is not None:
                    predictions[i - 1] = pred

            # Bottom-up pass: compute errors, update activations
            x = cortical_input  # always start from raw input at bottom
            for i, layer in enumerate(self.layers):
                compressed = layer.compress(x)
                pred_from_above = predictions.get(i)
                layer.compute_error(compressed, pred_from_above)
                # Next layer's input is this layer's activations
                x = layer._activations

        return self.layers[-1]._activations

    def forward(self, token_ids):
        """Process input through the brain.

        Args:
            token_ids: (batch, seq_len) input token indices

        Returns:
            logits: (batch, vocab_size) output logits
        """
        # Embed sequence
        cortical_input = self._embed_sequence(token_ids)

        # Settle the hierarchy
        top_activation = self.settle(cortical_input)

        # Read out answer
        logits = self.readout(top_activation)

        return logits

    def compute_value(self):
        """Compute critic value from top-layer activation."""
        top = self.layers[-1]._activations.detach()
        return self.critic(top).squeeze(-1)

    def hebbian_step(self, reward):
        """Apply Hebbian updates to all cortical layers.

        Args:
            reward: scalar reward (+1 correct, -0.5 incorrect)
        """
        # TD error for smarter credit assignment
        current_value = self.compute_value()
        if self._last_value is not None:
            td_error = reward + 0.5 * current_value.mean().item() - self._last_value.mean().item()
        else:
            td_error = reward
        self._last_value = current_value.detach()

        # Update each layer with Hebbian rule
        for layer in self.layers:
            layer.hebbian_update(td_error, eta=0.001)

        return td_error

    def get_diagnostics(self):
        """Return diagnostic dict."""
        diag = {}
        for i, layer in enumerate(self.layers):
            diag[f"brain/layer{i}_error_norm"] = layer._last_error_norm
            diag[f"brain/layer{i}_activation_norm"] = layer._last_activation_norm
            diag[f"brain/layer{i}_up_W_norm"] = layer.up_W.norm().item()
            if layer.has_top_down:
                diag[f"brain/layer{i}_down_W_norm"] = layer.down_W.norm().item()
            diag[f"brain/layer{i}_lateral_W_norm"] = layer.lateral_W.norm().item()
            diag[f"brain/layer{i}_precision_mean"] = torch.sigmoid(
                layer.precision_logit
            ).mean().item()
        if self._last_value is not None:
            diag["brain/critic_value"] = self._last_value.mean().item()
        return diag
