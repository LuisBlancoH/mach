import torch
import torch.nn as nn

from models.observation import ObservationProjection
from models.gru import SimpleGRU, SurpriseGatedGRU
from models.basis_vectors import BasisVectors
from models.meta_learner import MetaLearnerTransformer
from models.action_head import ActionHead
from models.memory_head import MemoryHead
from models.reward_projection import RewardProjection
from models.critic import Critic, CriticSignalProjection
from models.cerebellum import Cerebellum


class DifferentiablePatch(nn.Module):
    """
    Cortical patch with differentiable accumulated writes.
    Base weights are fixed at zero (buffers, not parameters).
    Meta-learner writes accumulate as delta tensors that remain in the
    computational graph, enabling gradient flow back to the meta-learner.
    """

    def __init__(self, d_model, hidden_dim=256):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.act = nn.GELU()

        # Base weights fixed at zero — not parameters
        self.register_buffer('down_base', torch.zeros(hidden_dim, d_model))
        self.register_buffer('up_base', torch.zeros(d_model, hidden_dim))

        # Accumulated deltas — set externally, part of computational graph
        self.delta_down = None
        self.delta_up = None

    def reset_deltas(self):
        """Call at start of each episode."""
        self.delta_down = torch.zeros(
            self.hidden_dim, self.d_model, device=self.down_base.device
        )
        self.delta_up = torch.zeros(
            self.d_model, self.hidden_dim, device=self.up_base.device
        )

    def accumulate_write(self, weight_name, delta_W):
        """Add a differentiable delta. delta_W must be in the computational graph."""
        if weight_name == "down":
            self.delta_down = self.delta_down + delta_W
        else:
            self.delta_up = self.delta_up + delta_W

    def forward(self, hidden_states):
        """Forward with base + accumulated deltas. Operates in float32."""
        W_down = self.down_base + (self.delta_down if self.delta_down is not None else 0)
        W_up = self.up_base + (self.delta_up if self.delta_up is not None else 0)

        h = torch.nn.functional.linear(hidden_states, W_down)
        h = self.act(h)
        h = torch.nn.functional.linear(h, W_up)
        return h


class MACHPhase2(nn.Module):
    """
    Simplified universal module for Phase 2.
    No cerebellum. No critic. No planning loop. No surprise-gated GRU.
    Fixed firing cadence (once per problem).
    """

    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=256,
                 d_meta=128, n_basis=8, detach_obs=True):
        super().__init__()
        self.d_model = d_model
        self.d_meta = d_meta
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers
        self.detach_obs = detach_obs

        # Observation projection (Qwen -> d_meta)
        self.obs_proj = ObservationProjection(d_model, d_meta)

        # GRU (simplified, no surprise gating)
        self.gru = SimpleGRU(d_meta)

        # Basis vectors for patch writes
        self.basis = BasisVectors(d_model, hidden_dim, len(patch_layers), n_basis)

        # Meta-learner transformer
        self.transformer = MetaLearnerTransformer(d_meta, n_tokens=7)

        # Heads
        self.action_head = ActionHead(d_meta, len(patch_layers), n_basis)
        self.memory_head = MemoryHead(d_meta)

        # Reward projection (placeholder for critic)
        self.reward_proj = RewardProjection(n_signals=3, d_meta=d_meta)

        # Differentiable patches
        self.patches = nn.ModuleList([
            DifferentiablePatch(d_model, hidden_dim) for _ in patch_layers
        ])

        # Transformer memory state (reset per episode)
        self._tf_mem = None

    def reset_episode(self):
        """Call at the start of each episode."""
        self.gru.reset()
        for patch in self.patches:
            patch.reset_deltas()
        self._tf_mem = self.transformer.tf_mem_init.clone()

    def observe(self, base_model, input_ids):
        """
        Run Qwen forward (detached), extract last-token hidden state,
        project to d_meta, integrate through GRU.

        The observation path is detached — the meta-learner gets gradient
        only about what to write, not what to observe.
        """
        with torch.no_grad():
            hidden_state = None
            target_layer = self.patch_layers[len(self.patch_layers) // 2]

            def hook(module, input, output):
                nonlocal hidden_state
                if isinstance(output, tuple):
                    hidden_state = output[0][:, -1, :]
                else:
                    hidden_state = output[:, -1, :]

            h = base_model.model.layers[target_layer].register_forward_hook(hook)
            base_model(input_ids=input_ids)
            h.remove()

        # Project and integrate
        projected = self.obs_proj(hidden_state.float().unsqueeze(1)).squeeze(0)
        if self.detach_obs:
            projected = projected.detach()
        gru_memory = self.gru.integrate(projected)
        return gru_memory

    def fire(self, gru_memory, reward_signals):
        """
        One meta-learner firing. Produces patch writes and memory update.

        gru_memory: (d_meta,)
        reward_signals: (3,) — [last_reward, cumulative_reward, firing_index]
        Returns: writes (list of (patch_idx, weight_name, coefficients, gate))
        """
        # Assemble input tokens
        reward_token = self.reward_proj(reward_signals)
        zero_placeholder = torch.zeros(self.d_meta, device=gru_memory.device)

        tokens = torch.stack([
            gru_memory,                             # world state
            reward_token,                           # reward signals
            zero_placeholder,                       # cerebellar correction (Phase 2: zeros)
            self._tf_mem,                           # transformer memory
            self.transformer.think_0_init,          # action
            self.transformer.think_1_init,          # memory update
            self.transformer.think_2_init,          # upstream output (unused Phase 2)
        ])  # (7, d_meta)

        # Forward through meta-learner
        hidden = self.transformer(tokens)  # (7, d_meta)

        # Action from think_0 (position 4)
        writes = self.action_head(hidden[4])

        # Memory update from think_1 (position 5)
        self._tf_mem = self.memory_head(hidden[5], self._tf_mem)

        return writes

    def apply_writes(self, writes):
        """Apply differentiable patch weight modifications via basis vectors."""
        for (patch_idx, weight_name, coefficients, gate) in writes:
            delta_W = self.basis.compute_delta_W(
                patch_idx, weight_name, coefficients, gate
            )
            self.patches[patch_idx].accumulate_write(weight_name, delta_W)


class MACHPhase3(nn.Module):
    """
    Phase 3 universal module: Phase 2 + Critic (basal ganglia).

    Changes from Phase 2:
    - Adds Critic MLP (evaluates mean-pooled transformer hidden states)
    - Replaces RewardProjection with CriticSignalProjection
    - fire() takes value + td_error instead of raw reward_signals
    - Stores last transformer hidden states for critic evaluation
    """

    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=256,
                 d_meta=128, n_basis=8, detach_obs=True):
        super().__init__()
        self.d_model = d_model
        self.d_meta = d_meta
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers
        self.detach_obs = detach_obs

        # Same as Phase 2
        self.obs_proj = ObservationProjection(d_model, d_meta)
        self.gru = SimpleGRU(d_meta)
        self.basis = BasisVectors(d_model, hidden_dim, len(patch_layers), n_basis)
        self.transformer = MetaLearnerTransformer(d_meta, n_tokens=7)
        self.action_head = ActionHead(d_meta, len(patch_layers), n_basis)
        self.memory_head = MemoryHead(d_meta)

        # NEW: Critic (basal ganglia)
        self.critic = Critic(d_meta, hidden_dim=d_meta)

        # NEW: Critic signal projection (replaces reward_proj)
        self.critic_signal_proj = CriticSignalProjection(d_meta)

        # Differentiable patches
        self.patches = nn.ModuleList([
            DifferentiablePatch(d_model, hidden_dim) for _ in patch_layers
        ])

        # State
        self._tf_mem = None
        self._last_hidden = None

    def reset_episode(self):
        """Call at the start of each episode."""
        self.gru.reset()
        for patch in self.patches:
            patch.reset_deltas()
        self._tf_mem = self.transformer.tf_mem_init.clone()
        self._last_hidden = None

    def observe(self, base_model, input_ids):
        """
        Observation path. Qwen forward is always no_grad (frozen).
        If detach_obs=False, gradient flows through obs_proj and GRU.
        """
        with torch.no_grad():
            hidden_state = None
            target_layer = self.patch_layers[len(self.patch_layers) // 2]

            def hook(module, input, output):
                nonlocal hidden_state
                if isinstance(output, tuple):
                    hidden_state = output[0][:, -1, :]
                else:
                    hidden_state = output[:, -1, :]

            h = base_model.model.layers[target_layer].register_forward_hook(hook)
            base_model(input_ids=input_ids)
            h.remove()

        projected = self.obs_proj(hidden_state.float().unsqueeze(1)).squeeze(0)
        if self.detach_obs:
            projected = projected.detach()
        gru_memory = self.gru.integrate(projected)
        return gru_memory

    def fire(self, gru_memory, last_value, last_td_error):
        """
        One meta-learner firing. Produces patch writes and memory update.

        gru_memory: (d_meta,)
        last_value: scalar tensor — previous step's critic value estimate
        last_td_error: scalar tensor — previous step's TD error
        Returns: writes (list of (patch_idx, weight_name, coefficients, gate))
        """
        critic_token = self.critic_signal_proj(last_value, last_td_error)
        zero_placeholder = torch.zeros(self.d_meta, device=gru_memory.device)

        tokens = torch.stack([
            gru_memory,                             # world state
            critic_token,                           # critic signals (was reward)
            zero_placeholder,                       # cerebellar correction (still zeros)
            self._tf_mem,                           # transformer memory
            self.transformer.think_0_init,          # action
            self.transformer.think_1_init,          # memory update
            self.transformer.think_2_init,          # upstream output
        ])  # (7, d_meta)

        hidden = self.transformer(tokens)  # (7, d_meta)

        # Store hidden states for critic (stays in graph)
        self._last_hidden = hidden

        # Action from think_0 (position 4)
        writes = self.action_head(hidden[4])

        # Memory update from think_1 (position 5)
        self._tf_mem = self.memory_head(hidden[5], self._tf_mem)

        return writes

    def get_value(self):
        """
        Evaluate current transformer hidden states with the critic.
        Must be called AFTER fire(). Returns scalar value (in graph).
        """
        assert self._last_hidden is not None, "Must call fire() before get_value()"
        return self.critic(self._last_hidden)

    def apply_writes(self, writes):
        """Apply differentiable patch weight modifications via basis vectors."""
        for (patch_idx, weight_name, coefficients, gate) in writes:
            delta_W = self.basis.compute_delta_W(
                patch_idx, weight_name, coefficients, gate
            )
            self.patches[patch_idx].accumulate_write(weight_name, delta_W)

    def load_phase2_checkpoint(self, checkpoint_path, device='cpu'):
        """
        Load compatible weights from a Phase 2 checkpoint.
        Skips reward_proj (replaced by critic_signal_proj).
        """
        state_dict = torch.load(checkpoint_path, map_location=device)
        compatible = {
            k: v for k, v in state_dict.items()
            if not k.startswith('reward_proj')
        }
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        print(f"  Loaded Phase 2 checkpoint: {len(compatible)} keys loaded, "
              f"{len(missing)} missing (new), {len(unexpected)} unexpected")
        return missing, unexpected


class MACHPhase4(MACHPhase3):
    """
    Phase 4 universal module: Phase 3 + Cerebellum + Surprise-Gated GRU.

    Changes from Phase 3:
    - Replaces SimpleGRU with SurpriseGatedGRU
    - Adds Cerebellum (predictor + correction projection)
    - observe() computes surprise and correction from cerebellar prediction error
    - fire() uses accumulated correction at position 2 (was zeros)
    - Cerebellum predictor has its own optimizer (online supervised)
    - correction_proj is in meta_params (gets CE loss gradient)
    """

    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=256,
                 d_meta=128, n_basis=8, detach_obs=True, surprise_scale=2.0):
        super().__init__(
            d_model=d_model, n_layers=n_layers, patch_layers=patch_layers,
            hidden_dim=hidden_dim, d_meta=d_meta, n_basis=n_basis,
            detach_obs=detach_obs,
        )

        # Replace SimpleGRU with SurpriseGatedGRU
        self.gru = SurpriseGatedGRU(d_meta, surprise_scale=surprise_scale)

        # Cerebellum
        self.cerebellum = Cerebellum(d_meta, hidden_dim=d_meta)

        # Surprise normalization (running stats, persist across episodes)
        self._surprise_ema = 1.0   # exponential moving average of raw surprise
        self._surprise_ema_decay = 0.99

        # State (reset per episode)
        self._accumulated_correction = None
        self._last_predicted_obs = None
        self._cerebellum_losses = []
        self._surprises = []

    def reset_episode(self):
        """Call at the start of each episode."""
        super().reset_episode()
        self._accumulated_correction = None
        self._last_predicted_obs = None
        self._cerebellum_losses = []
        self._surprises = []

    def observe(self, base_model, input_ids):
        """
        Observation with cerebellar prediction error.

        1. Get hidden state from frozen Qwen + project to d_meta
        2. Compare to cerebellum's prediction -> surprise, correction
        3. Train cerebellum predictor (accumulate loss)
        4. Accumulate correction
        5. GRU integrate with surprise gating
        6. Predict next observation
        """
        with torch.no_grad():
            hidden_state = None
            target_layer = self.patch_layers[len(self.patch_layers) // 2]

            def hook(module, input, output):
                nonlocal hidden_state
                if isinstance(output, tuple):
                    hidden_state = output[0][:, -1, :]
                else:
                    hidden_state = output[:, -1, :]

            h = base_model.model.layers[target_layer].register_forward_hook(hook)
            base_model(input_ids=input_ids)
            h.remove()

        projected = self.obs_proj(hidden_state.float().unsqueeze(1)).squeeze(0)
        if self.detach_obs:
            projected = projected.detach()

        # Cerebellar prediction error
        surprise = None
        if self._last_predicted_obs is not None:
            surprise, correction, pred_loss = self.cerebellum.compute_error(
                self._last_predicted_obs, projected
            )
            self._cerebellum_losses.append(pred_loss)

            # Normalize surprise by running mean (relative, not absolute)
            raw_surprise = surprise.item()
            self._surprise_ema = (
                self._surprise_ema_decay * self._surprise_ema
                + (1 - self._surprise_ema_decay) * raw_surprise
            )
            # Relative surprise: 1.0 = average, >1 = more surprising
            surprise = raw_surprise / max(self._surprise_ema, 1e-6)
            self._surprises.append(surprise)

            if self._accumulated_correction is None:
                self._accumulated_correction = correction
            else:
                self._accumulated_correction = (
                    self._accumulated_correction + correction
                )

        # GRU integrate with surprise gating
        gru_memory = self.gru.integrate(projected, surprise=surprise)

        # Predict next observation
        self._last_predicted_obs = self.cerebellum.predict(gru_memory)

        return gru_memory

    def fire(self, gru_memory, last_value, last_td_error):
        """
        Meta-learner firing with accumulated correction at position 2.
        Resets accumulated correction after use.
        """
        critic_token = self.critic_signal_proj(last_value, last_td_error)

        if self._accumulated_correction is not None:
            correction_token = self._accumulated_correction
        else:
            correction_token = torch.zeros(
                self.d_meta, device=gru_memory.device
            )

        tokens = torch.stack([
            gru_memory,                             # 0: world state
            critic_token,                           # 1: critic signals
            correction_token,                       # 2: cerebellar correction
            self._tf_mem,                           # 3: transformer memory
            self.transformer.think_0_init,          # 4: action
            self.transformer.think_1_init,          # 5: memory update
            self.transformer.think_2_init,          # 6: upstream output
        ])  # (7, d_meta)

        hidden = self.transformer(tokens)  # (7, d_meta)
        self._last_hidden = hidden

        writes = self.action_head(hidden[4])
        self._tf_mem = self.memory_head(hidden[5], self._tf_mem)

        # Reset accumulated correction after firing
        self._accumulated_correction = None

        return writes

    def get_cerebellum_loss(self):
        """Sum of predictor losses for the cerebellum optimizer."""
        if not self._cerebellum_losses:
            return torch.tensor(0.0)
        return sum(self._cerebellum_losses)

    def get_cerebellum_diagnostics(self):
        """Cerebellum-specific diagnostics."""
        diag = {}
        if self._surprises:
            diag["avg_surprise"] = sum(self._surprises) / len(self._surprises)
            diag["max_surprise"] = max(self._surprises)
            diag["min_surprise"] = min(self._surprises)
        if self._cerebellum_losses:
            diag["cerebellum_loss"] = sum(
                l.item() for l in self._cerebellum_losses
            ) / len(self._cerebellum_losses)
        return diag

    def load_phase3_checkpoint(self, checkpoint_path, device='cpu'):
        """
        Load Phase 3 checkpoint. GRU cell weights transfer directly.
        Cerebellum components start fresh.
        """
        state_dict = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"  Loaded Phase 3 checkpoint: {len(state_dict)} keys loaded, "
              f"{len(missing)} missing (new), {len(unexpected)} unexpected")
        return missing, unexpected


class MACHPhase6(MACHPhase3):
    """
    Phase 6 universal module: Phase 3 + Planning Loop.

    Overrides fire() to run 1-N transformer iterations per problem.
    The critic evaluates each iteration's proposal, and only the
    best-valued one is committed (writes applied). Zero new parameters.

    Gradient flow:
    - CE loss flows through committed iteration's writes (and any earlier
      iterations that shaped tf_mem for the committed one).
    - Critic loss flows through ALL iterations (every proposal should
      predict value accurately).
    """

    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=256,
                 d_meta=128, n_basis=8, detach_obs=True, max_planning_iters=3):
        super().__init__(
            d_model=d_model, n_layers=n_layers, patch_layers=patch_layers,
            hidden_dim=hidden_dim, d_meta=d_meta, n_basis=n_basis,
            detach_obs=detach_obs,
        )
        self.max_planning_iters = max_planning_iters

        # Planning diagnostics (not parameters, just tracking)
        self._iteration_values = []     # all values across iterations (in graph)
        self._committed_iteration = 0   # which iteration was committed
        self._committed_writes = None   # writes from committed iteration
        self._committed_value = None    # value from committed iteration

    def reset_episode(self):
        """Call at the start of each episode."""
        super().reset_episode()
        self._iteration_values = []
        self._committed_iteration = 0
        self._committed_writes = None
        self._committed_value = None

    def fire(self, gru_memory, last_value, last_td_error):
        """
        Planning loop: run transformer up to max_planning_iters times.
        Each iteration gets fresh think tokens but evolving tf_mem.
        Commit the iteration with highest critic value.

        gru_memory: (d_meta,) — fixed across iterations
        last_value: scalar tensor — previous step's critic value
        last_td_error: scalar tensor — previous step's TD error
        Returns: writes from the best-valued iteration
        """
        critic_token = self.critic_signal_proj(last_value, last_td_error)
        zero_placeholder = torch.zeros(self.d_meta, device=gru_memory.device)

        iteration_data = []  # (value, writes, hidden) per iteration
        tf_mem = self._tf_mem  # evolves across iterations

        for iteration in range(self.max_planning_iters):
            # Fresh think tokens each iteration (new proposal)
            tokens = torch.stack([
                gru_memory,                             # world state (fixed)
                critic_token,                           # critic signals
                zero_placeholder,                       # cerebellar correction
                tf_mem,                                 # evolving memory
                self.transformer.think_0_init,          # fresh action
                self.transformer.think_1_init,          # fresh memory update
                self.transformer.think_2_init,          # fresh upstream
            ])  # (7, d_meta)

            hidden = self.transformer(tokens)  # (7, d_meta)

            # Critic evaluates this proposal
            value = self.critic(hidden)

            # Action head produces writes from this proposal
            writes = self.action_head(hidden[4])

            iteration_data.append((value, writes, hidden))

            # Evolve state for next iteration (if not last)
            if iteration < self.max_planning_iters - 1:
                # Update tf_mem via memory head (stays in graph)
                tf_mem = self.memory_head(hidden[5], tf_mem)
                # Feed critic value back as updated signal
                # Use current value and approximate td_error from value change
                td_approx = value.detach() - last_value.detach()
                critic_token = self.critic_signal_proj(value.detach(), td_approx)

        # Select iteration with highest critic value
        values = [d[0] for d in iteration_data]
        best_idx = max(range(len(values)), key=lambda i: values[i].item())

        # Store all iteration values for multi-iteration critic loss
        self._iteration_values = values
        self._committed_iteration = best_idx
        self._committed_writes = iteration_data[best_idx][1]
        self._committed_value = iteration_data[best_idx][0]

        # Store hidden from committed iteration for get_value()
        self._last_hidden = iteration_data[best_idx][2]

        # Update tf_mem: last iteration's think_1 updates the evolved memory.
        # tf_mem chains through all iterations' memory_head calls (gradient flows).
        self._tf_mem = self.memory_head(iteration_data[-1][2][5], tf_mem)

        return self._committed_writes

    def get_value(self):
        """Returns committed iteration's value (in graph)."""
        assert self._committed_value is not None, "Must call fire() before get_value()"
        return self._committed_value

    def get_all_iteration_values(self):
        """Returns all iteration values for multi-iteration critic loss."""
        return self._iteration_values

    def get_committed_iteration(self):
        """Returns which iteration index was committed."""
        return self._committed_iteration

    def load_phase3_checkpoint(self, checkpoint_path, device='cpu'):
        """
        Load Phase 3 checkpoint directly — identical parameter structure.
        """
        state_dict = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"  Loaded Phase 3 checkpoint: {len(state_dict)} keys loaded, "
              f"{len(missing)} missing (new), {len(unexpected)} unexpected")
        return missing, unexpected


class MACHPatchedModel(nn.Module):
    """Wraps Qwen with MACH differentiable patches hooked into residual stream."""

    def __init__(self, base_model, mach):
        super().__init__()
        self.base_model = base_model
        self.mach = mach
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for i, layer_idx in enumerate(self.mach.patch_layers):
            layer = self.base_model.model.layers[layer_idx]

            def make_hook(patch_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        # Patches operate in float32 for stable gradients
                        patch_out = self.mach.patches[patch_idx](h.float())
                        return (h + patch_out.to(h.dtype),) + output[1:]
                    else:
                        patch_out = self.mach.patches[patch_idx](output.float())
                        return output + patch_out.to(output.dtype)
                return hook

            handle = layer.register_forward_hook(make_hook(i))
            self._hooks.append(handle)

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
