"""
Columnar cortex: biologically-inspired cortical columns with meta-learned plasticity.

Each cortical column is a small predictive coding unit (~64 dims) that:
  - Receives bottom-up evidence
  - Compares with top-down predictions (prediction error)
  - Integrates lateral context from neighbor columns (attention)
  - Settles iteratively on a belief
  - Learns via Hebbian rules with meta-learned learning rates

Columns are organized into cortical areas (hierarchical levels).
Within an area: lateral attention between columns.
Between areas: bottom-up evidence flows up, top-down predictions flow down.

The "genome" (gradient-trained): meta-parameters (etas, decay, precision, gate_bias),
    lateral attention weights, input/output projections.
The "brain" (Hebbian-trained): W_error and W_predict per column.

Meta-learning: gradient flows through Hebbian updates to learn the learning rules.
    loss → forward → W_eff → W_delta → eta, decay
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorticalArea(nn.Module):
    """A cortical area: n_columns columns at the same hierarchical level.

    Each column maintains its own belief and plastic weights.
    Columns communicate laterally via multi-head attention.

    Plastic weights per column (Hebbian buffers + in-graph deltas):
      - W_error: (d_col, d_col) — prediction error → belief update
      - W_predict: (d_col, d_col) — belief → top-down prediction

    Meta-learned parameters per column (genome):
      - eta_error, eta_predict: learning rates (log-scale)
      - decay: forgetting rate for Hebbian deltas
      - gate_bias: initial conservatism (how readily to update beliefs)
      - precision: per-dimension confidence in prediction errors
    """

    def __init__(self, n_columns, d_col, n_heads=4):
        super().__init__()
        self.n_columns = n_columns
        self.d_col = d_col

        # === Plastic weights (per column, Hebbian-learned) ===
        # W_error: error → belief update
        self.register_buffer(
            'W_error_base', torch.zeros(n_columns, d_col, d_col)
        )
        # W_predict: belief → top-down prediction
        self.register_buffer(
            'W_predict_base', torch.zeros(n_columns, d_col, d_col)
        )
        # Initialize
        for i in range(n_columns):
            nn.init.kaiming_uniform_(self.W_error_base[i])
            nn.init.kaiming_uniform_(self.W_predict_base[i])

        # === Meta-learned parameters (per column, genome) ===
        self.eta_error = nn.Parameter(torch.full((n_columns,), -3.0))
        self.eta_predict = nn.Parameter(torch.full((n_columns,), -3.0))
        self.decay = nn.Parameter(torch.zeros(n_columns))
        self.gate_bias = nn.Parameter(torch.full((n_columns,), -1.0))
        self.precision = nn.Parameter(torch.zeros(n_columns, d_col))

        # === Lateral attention (genome) ===
        # Columns attend to each other — Q/K/V projections are gradient-learned
        self.lateral_attn = nn.MultiheadAttention(
            d_col, n_heads, batch_first=True
        )

        # === Layer norm (bounded firing rates) ===
        self.norm = nn.LayerNorm(d_col)

        # === Inhibitory competition (genome) ===
        # Temperature controls sharpness: low = winner-take-all, high = all equal
        # Starts high (explore) and genome learns the right level
        self.competition_temp = nn.Parameter(torch.tensor(2.0))

        # === State ===
        self.beliefs = None        # (batch, n_cols, d_col)
        self.W_error_delta = None  # (n_cols, d_col, d_col) — stays in graph
        self.W_predict_delta = None
        self._last_error = None
        self._last_bottom_up = None

        # Diagnostics
        self._last_error_norm = 0.0
        self._last_belief_norm = 0.0
        self._last_sparsity = 0.0

    def reset(self):
        """Reset beliefs between inputs."""
        self.beliefs = None
        self._last_error = None
        self._last_bottom_up = None

    def reset_episode(self):
        """Reset episode state (Hebbian deltas)."""
        self.reset()
        self.W_error_delta = None
        self.W_predict_delta = None

    @property
    def W_error_eff(self):
        """Effective error weights = base + accumulated Hebbian delta."""
        if self.W_error_delta is not None:
            return self.W_error_base + self.W_error_delta
        return self.W_error_base

    @property
    def W_predict_eff(self):
        """Effective prediction weights = base + accumulated Hebbian delta."""
        if self.W_predict_delta is not None:
            return self.W_predict_base + self.W_predict_delta
        return self.W_predict_base

    def settle_step(self, bottom_up, top_down=None, think_round=0):
        """One settling iteration for all columns simultaneously.

        Args:
            bottom_up: (batch, n_cols, d_col) evidence from below
            top_down: (batch, n_cols, d_col) prediction from above, or None
            think_round: current thinking round (higher = sharper competition)
        """
        # Initialize beliefs, or reinitialize if batch size changed
        # (happens during generation as sequence grows)
        if self.beliefs is None or self.beliefs.shape[0] != bottom_up.shape[0]:
            self.beliefs = bottom_up.clone()

        # 1. Prediction error: what surprised me?
        prediction = torch.einsum(
            'nij,bnj->bni', self.W_predict_eff, self.beliefs
        )
        error = bottom_up - prediction

        # 2. Precision-weight the error
        precision = torch.sigmoid(self.precision)  # (n_cols, d_col)
        weighted_error = error * precision.unsqueeze(0)

        # 3. Error → update signal
        error_signal = torch.einsum(
            'nij,bnj->bni', self.W_error_eff, weighted_error
        )

        # 4. Lateral attention: what do my neighbors think?
        lateral, _ = self.lateral_attn(
            self.beliefs, self.beliefs, self.beliefs
        )

        # 5. Gate: large error → open gate → update belief
        error_mag = weighted_error.norm(dim=-1, keepdim=True)
        gate = torch.sigmoid(
            error_mag + self.gate_bias.view(1, -1, 1)
        )

        # 6. Candidate belief update
        candidate = error_signal + lateral
        if top_down is not None:
            candidate = candidate + top_down

        # 7. Gated update with normalization
        new_beliefs = (1 - gate) * self.beliefs + gate * candidate
        b, n, d = new_beliefs.shape
        self.beliefs = self.norm(
            new_beliefs.reshape(b * n, d)
        ).reshape(b, n, d)

        # 8. Inhibitory competition: columns compete for activation
        # Temperature decreases with thinking rounds (explore → sharpen)
        # temp = base_temp / (1 + round), so later rounds are sharper
        temp = F.softplus(self.competition_temp) / (1.0 + think_round)
        strengths = self.beliefs.norm(dim=-1)  # (batch, n_cols)
        competition = F.softmax(strengths / (temp + 1e-6), dim=-1)  # (batch, n_cols)
        self.beliefs = self.beliefs * competition.unsqueeze(-1)

        # Store for Hebbian learning
        self._last_error = weighted_error
        self._last_bottom_up = bottom_up

        # Diagnostics
        self._last_error_norm = weighted_error.detach().norm().item()
        self._last_belief_norm = self.beliefs.detach().norm().item()
        self._last_sparsity = (competition.max(dim=-1).values.mean().item()
                               - 1.0 / self.n_columns)

    def get_predictions(self):
        """Generate top-down predictions from current beliefs."""
        if self.beliefs is None:
            return None
        return torch.einsum(
            'nij,bnj->bni', self.W_predict_eff, self.beliefs
        )

    def hebbian_update(self, reward):
        """Meta-learned Hebbian update. Graph stays alive for meta-learning.

        The outer products (content of update) use detached activations.
        The scaling (eta, decay) stays in the graph so gradients reach them.

        Gradient path: loss → forward → W_eff → W_delta → eta, decay
        """
        if self._last_error is None or self.beliefs is None:
            return

        batch = self.beliefs.shape[0]
        eta_err = F.softplus(self.eta_error)      # (n_cols,)
        eta_pred = F.softplus(self.eta_predict)    # (n_cols,)
        d = torch.sigmoid(self.decay)              # (n_cols,)

        # --- W_error: reward-gated ---
        # Strengthen error→belief pathways that led to good outcomes
        belief_det = self.beliefs.detach()
        error_det = self._last_error.detach()
        delta_err = torch.einsum(
            'bni,bnj->nij', belief_det, error_det
        ) / batch
        # Scale by meta-learned eta and reward (in graph)
        delta_err = eta_err.view(-1, 1, 1) * reward * delta_err

        if self.W_error_delta is not None:
            self.W_error_delta = (
                d.view(-1, 1, 1) * self.W_error_delta + delta_err
            )
        else:
            self.W_error_delta = delta_err

        # --- W_predict: self-supervised (always update) ---
        # Minimize prediction error regardless of reward
        bottom_up_det = self._last_bottom_up.detach()
        with torch.no_grad():
            pred = torch.einsum(
                'nij,bnj->bni', self.W_predict_eff, belief_det
            )
        pred_error = bottom_up_det - pred
        delta_pred = torch.einsum(
            'bni,bnj->nij', pred_error, belief_det
        ) / batch
        delta_pred = eta_pred.view(-1, 1, 1) * delta_pred

        if self.W_predict_delta is not None:
            self.W_predict_delta = (
                d.view(-1, 1, 1) * self.W_predict_delta + delta_pred
            )
        else:
            self.W_predict_delta = delta_pred

        # --- Homeostatic plasticity ---
        max_delta_norm = 10.0
        for attr in ['W_error_delta', 'W_predict_delta']:
            delta = getattr(self, attr)
            if delta is not None:
                norms = delta.norm(dim=(-2, -1), keepdim=True)
                scale = torch.clamp(
                    max_delta_norm / (norms + 1e-8), max=1.0
                )
                setattr(self, attr, (delta * scale).detach())


class ColumnarCortex(nn.Module):
    """Full cortex: hierarchy of cortical areas with columnar organization.

    Architecture:
      Qwen observations → input projection → Area 0 columns
                                                ↕ settle + lateral attn
      Area 0 beliefs → Area 1 columns (bottom-up: 1:1)
                        ↕ settle + lateral attn
                        Area 1 predictions → Area 0 (top-down: 1:1)
      ...
      Top area beliefs → output projection → correction to Qwen

    The settling loop runs the full hierarchy for n_settle iterations,
    with bottom-up evidence, top-down predictions, and lateral attention
    all operating simultaneously.
    """

    def __init__(self, d_model, d_col=64, n_columns=40, n_areas=3,
                 n_settle=5, n_think=3, n_heads=4, observe_layers=None,
                 think_epsilon=1e-3):
        super().__init__()
        if observe_layers is None:
            observe_layers = [9, 18, 27, 34]

        self.observe_layers = observe_layers
        self.d_model = d_model
        self.d_col = d_col
        self.n_columns = n_columns
        self.n_areas = n_areas
        self.n_settle = n_settle
        self.n_think = n_think            # max reasoning rounds
        self.think_epsilon = think_epsilon  # early stop when beliefs stabilize

        n_obs = len(observe_layers)

        # === Input: Qwen observations → column inputs ===
        # Shared projection per observation layer: d_model → d_col
        self.obs_projs = nn.ModuleList([
            nn.Linear(d_model, d_col) for _ in range(n_obs)
        ])
        # Per-column fusion: (d_col * n_obs) → d_col
        # Batched as a single parameter tensor for efficiency
        self.fuse_W = nn.Parameter(
            torch.randn(n_columns, d_col, d_col * n_obs) * 0.02
        )
        self.fuse_b = nn.Parameter(torch.zeros(n_columns, d_col))

        # === Re-entry: feed cortex output back as new input ===
        # This is cortical re-entry — the cortex processes its own conclusions.
        # Projects top-area beliefs back into column input space so each
        # thinking round sees its own previous output as new evidence.
        self.reentry_proj = nn.Linear(n_columns * d_col, n_columns * d_col)
        # Initialize near-zero so first round is driven by observations
        nn.init.normal_(self.reentry_proj.weight, std=0.01)
        nn.init.zeros_(self.reentry_proj.bias)

        # === Cortical areas (hierarchy) ===
        self.areas = nn.ModuleList([
            CorticalArea(n_columns, d_col, n_heads) for _ in range(n_areas)
        ])

        # === Output: top area → correction in Qwen's space ===
        self.output_proj = nn.Linear(n_columns * d_col, d_model)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # === Critic (TD learning) ===
        self.critic = nn.Sequential(
            nn.Linear(n_columns * d_col, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._last_value = None

        # === State ===
        self._observations = {}
        self._last_correction_norm = 0.0
        self._last_think_rounds = 0

    def reset(self):
        """Reset between inputs."""
        self._observations.clear()
        for area in self.areas:
            area.reset()

    def reset_episode(self):
        """Reset between episodes (clears Hebbian deltas)."""
        self._observations.clear()
        self._last_value = None
        for area in self.areas:
            area.reset_episode()

    def observe(self, obs_idx, hidden_state):
        """Capture a Qwen hidden state (read-only)."""
        self._observations[obs_idx] = hidden_state.detach()

    def _settle_once(self, column_input, think_round=0):
        """Run one settling pass through the full hierarchy.

        Args:
            column_input: (batch*seq, n_columns, d_col) sensory input
            think_round: current thinking round (for competition sharpening)
        """
        for area_idx in range(self.n_areas):
            area = self.areas[area_idx]

            # Bottom-up input
            if area_idx == 0:
                bottom_up = column_input
            else:
                bottom_up = self.areas[area_idx - 1].beliefs
                # Fallback if no beliefs or stale shape from previous seq
                if (bottom_up is None or
                        bottom_up.shape[0] != column_input.shape[0]):
                    bottom_up = column_input

            # Top-down prediction from area above
            top_down = None
            if area_idx < self.n_areas - 1:
                top_down = self.areas[area_idx + 1].get_predictions()
                # Shape guard: skip stale predictions from previous seq length
                if top_down is not None and top_down.shape[0] != bottom_up.shape[0]:
                    top_down = None

            area.settle_step(bottom_up, top_down, think_round=think_round)

    def think(self, qwen_final_hidden):
        """Process observations through the columnar hierarchy.

        Three nested loops:
          1. Thinking (outer): multiple reasoning rounds on the same input.
             Beliefs persist across rounds — each round sees the same
             evidence but with deeper understanding from prior rounds.
             Stops early when beliefs stabilize (gate closes naturally).
          2. Settling (inner): within each thinking round, iterate the
             hierarchy to convergence (bottom-up/top-down/lateral).
          3. Generation (caller): across tokens, beliefs persist as
             working memory.

        Args:
            qwen_final_hidden: (batch, seq, d_model)
        Returns:
            output: (batch, seq, d_model) = qwen_final + correction
        """
        batch, seq, _ = qwen_final_hidden.shape
        device = qwen_final_hidden.device

        # --- 1. Project observations to column space ---
        projected = []
        for i in range(len(self.observe_layers)):
            obs = self._observations.get(i)
            if obs is None:
                obs = torch.zeros(
                    batch, seq, self.d_model, device=device
                )
            # Slice observations to match seq (e.g., last position only during generate)
            if obs.shape[1] != seq:
                obs = obs[:, -seq:, :]
            projected.append(self.obs_projs[i](obs.float()))

        # (batch, seq, d_col * n_obs)
        obs_concat = torch.cat(projected, dim=-1)
        # (batch * seq, d_col * n_obs)
        obs_flat = obs_concat.view(batch * seq, -1)

        # Per-column fusion: (batch*seq, n_columns, d_col)
        column_input = (
            torch.einsum('nij,bj->bni', self.fuse_W, obs_flat)
            + self.fuse_b.unsqueeze(0)
        )

        # Clear observations (already projected)
        self._observations.clear()

        # --- 2. Thinking loop: reason via cortical re-entry ---
        # Round 1: settle on observations (perception)
        # Round 2+: settle on observations + own previous output (reasoning)
        # Each round, the cortex processes something DIFFERENT because its
        # own conclusions feed back as new evidence. This is reasoning.
        self._last_think_rounds = 0
        effective_input = column_input

        for think_round in range(self.n_think):
            # Snapshot beliefs before this round (for convergence check)
            if self.areas[-1].beliefs is not None:
                prev_beliefs = self.areas[-1].beliefs.detach().clone()
            else:
                prev_beliefs = None

            # Settle: n_settle iterations of bottom-up/top-down/lateral
            # Competition sharpens with each thinking round
            for t in range(self.n_settle):
                self._settle_once(effective_input, think_round=think_round)

            self._last_think_rounds = think_round + 1

            # Early stopping: beliefs stabilized?
            if (prev_beliefs is not None
                    and self.areas[-1].beliefs is not None
                    and prev_beliefs.shape == self.areas[-1].beliefs.shape):
                belief_change = (
                    self.areas[-1].beliefs.detach() - prev_beliefs
                ).norm().item()
                if belief_change < self.think_epsilon:
                    break  # cortex has settled — done thinking

            # Re-entry: feed own conclusions back as new evidence
            # Next round sees observations + what the cortex just concluded
            top_beliefs = self.areas[-1].beliefs
            reentry = self.reentry_proj(
                top_beliefs.reshape(top_beliefs.shape[0], -1)
            ).reshape(top_beliefs.shape[0], self.n_columns, self.d_col)
            effective_input = column_input + reentry

        # --- 3. Output ---
        top_beliefs = self.areas[-1].beliefs  # (batch*seq, n_cols, d_col)
        top_flat = top_beliefs.reshape(batch * seq, -1)

        correction = self.output_proj(top_flat).view(batch, seq, self.d_model)
        self._last_correction_norm = correction.detach().norm().item()

        output = qwen_final_hidden.float() + correction

        # Detach beliefs: preserve values as working memory but cut the graph.
        # Each token gets a fresh graph for backward, while beliefs carry
        # forward as persistent state (like biological working memory —
        # the content persists but doesn't drag computation history).
        for area in self.areas:
            if area.beliefs is not None:
                area.beliefs = area.beliefs.detach()

        return output.to(qwen_final_hidden.dtype)

    def compute_value(self):
        """Critic value estimate from top area beliefs."""
        if self.areas[-1].beliefs is None:
            return torch.tensor(0.0)
        top = self.areas[-1].beliefs.detach()
        top_flat = top.reshape(top.shape[0], -1)
        return self.critic(top_flat).squeeze(-1)

    def hebbian_step(self, reward, device=None):
        """Apply Hebbian updates to all areas, compute TD error."""
        current_value = self.compute_value()
        if self._last_value is not None:
            td_error = (
                reward
                + 0.5 * current_value.mean().item()
                - self._last_value.mean().item()
            )
        else:
            td_error = reward
        self._last_value = current_value.detach()

        for area in self.areas:
            area.hebbian_update(td_error)

        return td_error

    def get_diagnostics(self):
        """Return diagnostic dict for logging."""
        diag = {}
        for a, area in enumerate(self.areas):
            diag[f"cortex/area{a}_error_norm"] = area._last_error_norm
            diag[f"cortex/area{a}_belief_norm"] = area._last_belief_norm
            if area.W_error_delta is not None:
                diag[f"cortex/area{a}_W_err_delta"] = (
                    area.W_error_delta.detach().norm().item()
                )
            if area.W_predict_delta is not None:
                diag[f"cortex/area{a}_W_pred_delta"] = (
                    area.W_predict_delta.detach().norm().item()
                )
            diag[f"cortex/area{a}_eta_error"] = (
                F.softplus(area.eta_error).mean().item()
            )
            diag[f"cortex/area{a}_eta_predict"] = (
                F.softplus(area.eta_predict).mean().item()
            )
            diag[f"cortex/area{a}_decay"] = (
                torch.sigmoid(area.decay).mean().item()
            )
            diag[f"cortex/area{a}_precision"] = (
                torch.sigmoid(area.precision).mean().item()
            )
            diag[f"cortex/area{a}_competition_temp"] = (
                F.softplus(area.competition_temp).item()
            )
            diag[f"cortex/area{a}_sparsity"] = area._last_sparsity
        diag["cortex/correction_norm"] = self._last_correction_norm
        diag["cortex/think_rounds"] = self._last_think_rounds
        return diag


class ColumnarCortexModel(nn.Module):
    """Qwen + ColumnarCortex: frozen LLM with a columnar brain.

    Qwen runs fully intact. Hooks capture hidden states (read-only).
    The columnar cortex processes observations and produces a correction
    to Qwen's final hidden state for generation.
    """

    def __init__(self, base_model, cortex):
        super().__init__()
        self.base_model = base_model
        self.cortex = cortex
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for i, layer_idx in enumerate(self.cortex.observe_layers):
            layer = self.base_model.model.layers[layer_idx]
            handle = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(handle)

    def _make_hook(self, obs_idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            self.cortex.observe(obs_idx, h)
            return output
        return hook

    @property
    def device(self):
        return self.base_model.device

    def forward(self, input_ids, labels=None, attention_mask=None):
        # Don't reset cortex here — beliefs persist within a problem.
        # The training loop calls cortex.reset() between problems.

        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Process only last position through cortex (matches generate behavior)
        # This ensures train and test use the same cortex processing mode.
        qwen_final_last = outputs.hidden_states[-1][:, -1:, :]
        cortex_output = self.cortex.think(qwen_final_last)
        # Get logits only for last position
        logits = self.base_model.lm_head(cortex_output)

        loss = None
        if labels is not None:
            # logits is (batch, 1, vocab) — predict next token after last position
            # For token-by-token training, labels should have the target at position 0
            shift_logits = logits[:, 0, :]  # (batch, vocab)
            # We don't use labels here — the training loop handles CE directly
            pass

        self._last_loss = loss
        self._last_logits = logits

        return type(outputs)(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def generate(self, input_ids, max_new_tokens=20, **kwargs):
        device = input_ids.device
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Don't reset — beliefs carry across tokens (working memory)

            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=generated,
                    output_hidden_states=True,
                )

            # Only pass last position to cortex — beliefs accumulate
            # across tokens as working memory (batch dim stays consistent)
            qwen_final_last = outputs.hidden_states[-1][:, -1:, :]
            cortex_output = self.cortex.think(qwen_final_last)

            with torch.no_grad():
                logits = self.base_model.lm_head(cortex_output)

            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.base_model.config.eos_token_id:
                break

        return generated

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
