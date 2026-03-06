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
from models.error_observation import ErrorProjection


class DifferentiablePatch(nn.Module):
    """
    Cortical patch with differentiable accumulated writes.

    All computation comes from delta tensors that remain in the
    computational graph. Metalearned decay controls memory timescale:
    high decay (0.99) = slow memory, low decay (0.8) = fast adaptation.
    No separate slow/fast buffers — just one set of deltas with learned decay.
    """

    def __init__(self, d_model, hidden_dim=256):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.act = nn.GELU()

        # Device tracker (no parameters/buffers left after removing slow memory)
        self.register_buffer('_device_tracker', torch.zeros(1))

        # Accumulated deltas — set externally, part of computational graph
        self.delta_down = None
        self.delta_up = None
        self.delta_gain = None  # multiplicative gain vector (d_model,)

    def reset_deltas(self, use_slow=False):
        """Zero all deltas."""
        device = self._device_tracker.device
        self.delta_down = torch.zeros(
            self.hidden_dim, self.d_model, device=device
        )
        self.delta_up = torch.zeros(
            self.d_model, self.hidden_dim, device=device
        )
        self.delta_gain = torch.zeros(
            self.d_model, device=device
        )

    def consolidate(self, decay=0.95):
        """No-op. Kept for API compatibility.
        Memory timescale is now controlled by metalearned decay in accumulate_write.
        """
        pass

    def accumulate_write(self, weight_name, delta_W, decay=1.0):
        """Add a differentiable delta with learned decay.
        decay controls memory timescale: 0.99 = slow, 0.8 = fast.
        Fully differentiable — gradient flows through decay to plasticity controller.
        """
        if weight_name == "down":
            self.delta_down = self.delta_down * decay + delta_W
        elif weight_name == "gain":
            self.delta_gain = self.delta_gain * decay + delta_W
        else:
            self.delta_up = self.delta_up * decay + delta_W

    def get_gain(self):
        """Return accumulated gain vector for multiplicative modulation."""
        return self.delta_gain if self.delta_gain is not None else 0

    def forward(self, hidden_states):
        """Forward with accumulated deltas. Operates in float32."""
        W_down = self.delta_down if self.delta_down is not None else 0
        W_up = self.delta_up if self.delta_up is not None else 0

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
    - Observation-conditioned action head (skip connection from GRU memory)
    - observe() computes surprise and correction from cerebellar prediction error
    - fire() uses accumulated correction at position 2 (was zeros)
    - fire() passes gru_memory directly to action head for obs-conditioning
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

        # Observation-conditioned action head (skip connection)
        # GRU memory goes directly to action head, bypassing transformer.
        # Zero-init obs weights so it starts identical to Phase 3.
        self.action_head = ActionHead(
            d_meta, len(patch_layers), n_basis, obs_conditioned=True,
        )

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

        writes = self.action_head(hidden[4], gru_memory)
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
        Cerebellum components start fresh. Action head fc1 is resized:
        Phase 3 has (64, 128), Phase 4 has (64, 256) — the first 128
        columns are loaded, the obs half stays zero-initialized.
        """
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Remap action_head keys: Sequential (head.0/head.2) -> fc1/fc2
        key_remap = {
            "action_head.head.0.weight": "action_head.fc1.weight",
            "action_head.head.0.bias": "action_head.fc1.bias",
            "action_head.head.2.weight": "action_head.fc2.weight",
            "action_head.head.2.bias": "action_head.fc2.bias",
        }
        for old_key, new_key in key_remap.items():
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

        # Handle action_head.fc1 shape mismatch (128 -> 256 input)
        key = "action_head.fc1.weight"
        if key in state_dict and self.action_head.obs_conditioned:
            old_weight = state_dict[key]  # (64, 128)
            new_weight = self.action_head.fc1.weight.data.clone()  # (64, 256)
            new_weight[:, :old_weight.shape[1]] = old_weight
            state_dict[key] = new_weight

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


class TaskStateModule(nn.Module):
    """
    PFC-like gated task representation.

    Maintains a persistent task state (d_task) that is updated by gated
    integration of GRU observations. The gate learns which observations
    are informative and how much to update the task representation.
    """

    def __init__(self, d_gru, d_task):
        super().__init__()
        self.d_task = d_task
        self.gate_net = nn.Linear(d_gru + d_task, d_task)
        self.candidate_net = nn.Linear(d_gru + d_task, d_task)

    def forward(self, gru_memory, task_state):
        combined = torch.cat([gru_memory, task_state])
        gate = torch.sigmoid(self.gate_net(combined))
        candidate = torch.tanh(self.candidate_net(combined))
        new_task_state = gate * candidate + (1 - gate) * task_state
        return new_task_state


class DeliberationModule(nn.Module):
    """
    PFC iterative refinement — task state refines itself.

    After observation updates the task state, this module runs N steps
    of self-recurrence. Like PFC attractor dynamics: recurrent processing
    settles into a stable, structured representation.

    Gate bias initialized to -3.0 (sigmoid ≈ 0.05) so deliberation starts
    as near-identity — safe to add to existing checkpoints.
    """

    def __init__(self, d_task):
        super().__init__()
        self.gate_net = nn.Linear(d_task, d_task)
        self.candidate_net = nn.Linear(d_task, d_task)
        # Start as near-identity so loading existing checkpoints works
        nn.init.constant_(self.gate_net.bias, -3.0)

    def forward(self, task_state):
        gate = torch.sigmoid(self.gate_net(task_state))
        candidate = torch.tanh(self.candidate_net(task_state))
        return gate * candidate + (1 - gate) * task_state


class ActionCompiler(nn.Module):
    """
    Compiles task state into patch write coefficients.
    Outputs additive (down/up) and multiplicative (gain) writes.
    """

    def __init__(self, d_task, n_patches=4, n_basis=8, n_gain_basis=4):
        super().__init__()
        self.n_patches = n_patches
        self.n_basis = n_basis
        self.n_gain_basis = n_gain_basis
        # Additive: 2 writes per patch (down + up), each n_basis + 1
        # Gain: 1 write per patch, n_gain_basis + 1
        n_additive = n_patches * 2 * (n_basis + 1)
        n_gain = n_patches * (n_gain_basis + 1)
        n_outputs = n_additive + n_gain
        # Scale hidden dim with output size, min 64
        hidden = max(64, n_outputs // 3)

        self.head = nn.Sequential(
            nn.Linear(d_task, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, task_state):
        from config import GATE_SCALE
        raw = self.head(task_state)

        writes = []
        idx = 0
        for patch_i in range(self.n_patches):
            # Additive writes: down + up
            for weight_name in ["down", "up"]:
                coefficients = raw[idx:idx + self.n_basis]
                gate = torch.sigmoid(raw[idx + self.n_basis]) * GATE_SCALE
                idx += self.n_basis + 1
                writes.append((patch_i, weight_name, coefficients, gate))
            # Gain write: multiplicative modulation
            coefficients = raw[idx:idx + self.n_gain_basis]
            gate = torch.sigmoid(raw[idx + self.n_gain_basis]) * GATE_SCALE
            idx += self.n_gain_basis + 1
            writes.append((patch_i, "gain", coefficients, gate))

        return writes


class DemoSelector(nn.Module):
    """
    Attentional selection: task state queries over demo embeddings
    to choose which demo is most informative to re-examine.

    Like PFC directing attention to task-relevant stimuli.
    """

    def __init__(self, d_task, d_obs):
        super().__init__()
        self.query = nn.Linear(d_task, d_obs)

    def forward(self, task_state, demo_embeddings):
        """
        task_state: (d_task,)
        demo_embeddings: (n_demos, d_obs)
        Returns: scores (n_demos,)
        """
        q = self.query(task_state)  # (d_obs,)
        scores = torch.mv(demo_embeddings, q)  # (n_demos,)
        return scores


class TaskStateCritic(nn.Module):
    """
    Basal ganglia: value estimator over task state.

    Predicts how well the current task representation will perform.
    Used for TD-weighted CE loss (plasticity modulation) and
    inference-time satisfaction signal (self-eval loop).

    Never input to fire() — only shapes gradient magnitude.
    """

    def __init__(self, d_task, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_task, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Start pessimistic: default to "not satisfied"
        # Must learn to be confident — like basal ganglia defaulting to low dopamine
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, task_state):
        """Returns scalar value estimate in [0, 1]."""
        return torch.sigmoid(self.net(task_state).squeeze(-1))


class SlowMemory(nn.Module):
    """
    Cross-episode consolidation: neocortical long-term storage.

    EMA of successful task states provides warm start for new episodes.
    Retrieval gate (trainable) learns which dimensions are useful.
    Memory buffer itself is detached — no gradient through EMA.
    """

    def __init__(self, d_task, ema_decay=0.95, consolidation_threshold=0.3):
        super().__init__()
        self.register_buffer('memory', torch.zeros(d_task))
        self.retrieval_gate = nn.Linear(d_task, d_task)
        self.ema_decay = ema_decay
        self.consolidation_threshold = consolidation_threshold
        # Initialize gate to near-zero so early episodes start clean
        nn.init.zeros_(self.retrieval_gate.weight)
        nn.init.constant_(self.retrieval_gate.bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def consolidate(self, task_state, success_rate):
        """Called after episode. Blend successful task states into slow memory."""
        with torch.no_grad():
            if success_rate > self.consolidation_threshold:
                weight = min(success_rate, 1.0) * (1 - self.ema_decay)
                self.memory = self.ema_decay * self.memory + weight * task_state.detach()

    def retrieve(self):
        """Called at episode start. Returns gated slow memory as warm start."""
        gate = torch.sigmoid(self.retrieval_gate(self.memory))
        return gate * self.memory


class WorkingMemoryCrossAttention(nn.Module):
    """
    Token-level cross-attention for working memory.

    Task state queries over stored demo token representations to extract
    fine-grained information (e.g., numerical relationships between tokens).
    Single-head attention with gated residual connection.

    Gate bias=-3.0 → starts near-identity. Safe for existing checkpoints.
    """

    def __init__(self, d_task, d_obs):
        super().__init__()
        self.q_proj = nn.Linear(d_task, d_obs)       # task state → query
        self.out_proj = nn.Linear(d_obs, d_task)      # context → task update
        self.gate = nn.Linear(d_obs + d_task, d_task)  # gated residual
        self.scale = d_obs ** -0.5
        # Start near-identity so existing checkpoints aren't disrupted
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(self, task_state, memory):
        """
        task_state: (d_task,)
        memory: (n_tokens, d_obs)
        Returns: updated task_state (d_task,)
        """
        q = self.q_proj(task_state)                   # (d_obs,)
        attn = torch.softmax(memory @ q * self.scale, dim=0)  # (n_tokens,)
        context = attn @ memory                       # (d_obs,)
        out = self.out_proj(context)                   # (d_task,)
        g = torch.sigmoid(self.gate(torch.cat([context, task_state])))
        return g * out + (1 - g) * task_state


class MACHPhase5(nn.Module):
    """
    Phase 5: Brain-like meta-learner with structured task bottleneck.

    Key changes from Phase 2:
    - Smaller obs projection (d_model -> d_obs instead of d_meta)
    - Smaller GRU (d_obs -> d_gru)
    - Gated task state (d_task) replaces transformer + memory head
    - ActionCompiler takes d_task input (the bottleneck)
    - No reward input to fire()
    - L1 sparsity penalty on task_state
    - ~540K params (down from ~5M)

    Brain mapping:
    - obs_proj: sensory cortex
    - GRU: hippocampus (episodic memory)
    - task_state: PFC (working memory)
    - action_compiler: premotor cortex
    - basis + patches: motor execution
    """

    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=256,
                 d_obs=64, d_gru=64, d_task=32, n_basis=8,
                 n_deliberation_steps=0, task_noise=0.0,
                 multi_layer_obs=False, consolidation=False,
                 ema_decay=0.95, n_planning_steps=0,
                 planning_temperature=1.0, n_thinking_steps=0):
        super().__init__()
        self.d_model = d_model
        self.d_obs = d_obs
        self.d_gru = d_gru
        self.d_task = d_task
        self.n_basis = n_basis
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers
        self.n_deliberation_steps = n_deliberation_steps
        self.n_planning_steps = n_planning_steps
        self.planning_temperature = planning_temperature
        self.task_noise = task_noise  # noise std on task state (forgetting)
        self.multi_layer_obs = multi_layer_obs

        # Sensory cortex: project Qwen hidden states
        if multi_layer_obs:
            # Multi-layer: each patch layer gets its own projection
            # Like distinct sensory areas (V1, V4, IT, PFC)
            d_per_layer = d_obs // len(patch_layers)
            self.layer_projs = nn.ModuleList([
                nn.Linear(d_model, d_per_layer, bias=False)
                for _ in patch_layers
            ])
            # Top-down attention: task state controls which layers to observe
            # PFC tells sensory cortex what to look for — per-task, per-episode
            # When task_state is zero (episode start), all gates = 0.5 (uniform)
            self.obs_gate_net = nn.Linear(d_task, len(patch_layers))
            # Initialize bias to 0 → sigmoid = 0.5 at episode start
            nn.init.zeros_(self.obs_gate_net.bias)
            nn.init.zeros_(self.obs_gate_net.weight)
        else:
            self.obs_proj = ObservationProjection(d_model, d_obs)

        # Hippocampus: episodic sequential memory
        self.gru = SimpleGRU(d_obs)
        # Override GRU internal dim: input is d_obs, hidden is d_gru
        self.gru.gru_cell = nn.GRUCell(d_obs, d_gru)

        # PFC: gated task state
        self.task_state_module = TaskStateModule(d_gru, d_task)

        # PFC deliberation: iterative self-refinement
        if n_deliberation_steps > 0:
            self.deliberation = DeliberationModule(d_task)

        # Premotor: compile task state into patch writes
        n_gain_basis = 4  # fewer gain basis vectors than additive (cheaper)
        self.action_compiler = ActionCompiler(
            d_task, len(patch_layers), n_basis, n_gain_basis
        )

        # Motor execution: basis vectors + patches
        self.basis = BasisVectors(
            d_model, hidden_dim, len(patch_layers), n_basis, n_gain_basis
        )
        self.patches = nn.ModuleList([
            DifferentiablePatch(d_model, hidden_dim) for _ in patch_layers
        ])

        # Basal ganglia: value estimator (shapes gradient, not input to fire)
        self.critic = TaskStateCritic(d_task)

        # Attentional selection: choose which demo to re-examine
        self.demo_selector = DemoSelector(d_task, d_obs)

        # Working memory: token-level cross-attention over demo representations
        self.n_thinking_steps = n_thinking_steps
        if n_thinking_steps > 0:
            self.wm_proj = nn.Linear(d_model, d_obs, bias=False)
            self.wm_cross_attn = WorkingMemoryCrossAttention(d_task, d_obs)
        self._working_memory_buffer = []

        # Neocortical consolidation: cross-episode slow memory
        self.consolidation = consolidation
        if consolidation:
            self.slow_memory = SlowMemory(d_task, ema_decay=ema_decay)

        # Persistent state (reset per episode)
        self._task_state = None

    def reset_episode(self):
        """Call at the start of each episode."""
        self.gru.reset()
        for patch in self.patches:
            patch.reset_deltas()
        self._working_memory_buffer = []
        if self.consolidation:
            # Warm start from consolidated slow memory
            self._task_state = self.slow_memory.retrieve()
        else:
            self._task_state = torch.zeros(
                self.d_task, device=next(self.parameters()).device
            )

    def _extract_hidden_states(self, model_layers, input_ids, run_model,
                               capture_all_tokens=False):
        """
        Extract hidden states from Qwen layers (frozen forward pass).
        Returns dict of {layer_idx: hidden_state} if multi_layer_obs,
        or single hidden_state tensor if not.

        If capture_all_tokens=True, additionally captures ALL token positions
        from the middle layer (for working memory). Returns (hidden, all_tokens)
        where all_tokens is (seq_len, d_model) or None.
        """
        all_tokens = None

        with torch.no_grad():
            if self.multi_layer_obs:
                hidden_states = {}
                hooks = []
                middle_layer = self.patch_layers[len(self.patch_layers) // 2]

                for layer_idx in self.patch_layers:
                    def make_hook(idx):
                        def hook(module, input, output):
                            nonlocal all_tokens
                            t = output[0] if isinstance(output, tuple) else output
                            hidden_states[idx] = t[:, -1, :]
                            if capture_all_tokens and idx == middle_layer:
                                all_tokens = t[0].detach()  # (seq_len, d_model)
                        return hook
                    h = model_layers[layer_idx].register_forward_hook(
                        make_hook(layer_idx)
                    )
                    hooks.append(h)
                run_model(input_ids=input_ids)
                for h in hooks:
                    h.remove()
                if capture_all_tokens:
                    return hidden_states, all_tokens
                return hidden_states
            else:
                hidden_state = None
                target_layer = self.patch_layers[len(self.patch_layers) // 2]

                def hook(module, input, output):
                    nonlocal hidden_state, all_tokens
                    t = output[0] if isinstance(output, tuple) else output
                    hidden_state = t[:, -1, :]
                    if capture_all_tokens:
                        all_tokens = t[0].detach()  # (seq_len, d_model)

                h = model_layers[target_layer].register_forward_hook(hook)
                run_model(input_ids=input_ids)
                h.remove()
                if capture_all_tokens:
                    return hidden_state, all_tokens
                return hidden_state

    def _project_hidden(self, hidden):
        """Project extracted hidden states to d_obs. Returns (d_obs,) 1D."""
        if self.multi_layer_obs:
            # Top-down attention: task state modulates observation gates
            # At episode start (task_state=zeros) → all gates 0.5 (observe everything)
            # After first fire() → gates become task-specific
            gates = torch.sigmoid(self.obs_gate_net(self._task_state))
            layer_obs = []
            for i, layer_idx in enumerate(self.patch_layers):
                proj = gates[i] * self.layer_projs[i](hidden[layer_idx].float())
                layer_obs.append(proj)
            projected = torch.cat(layer_obs, dim=-1)  # (batch, d_obs)
            return projected.squeeze(0)  # (d_obs,)
        else:
            return self.obs_proj(hidden.float().unsqueeze(1)).squeeze(0)

    def observe(self, base_model, input_ids, return_embedding=False,
                store_working_memory=False):
        """
        Sensory processing: Qwen forward (frozen) -> project -> GRU.
        Gradient flows through obs_proj/layer_projs and GRU (always undetached).

        If return_embedding=True, also returns the projected observation
        (for active learning demo selection).
        If store_working_memory=True, captures all token positions from middle
        layer and stores projected tokens in working memory buffer.
        """
        capture = store_working_memory and self.n_thinking_steps > 0
        result = self._extract_hidden_states(
            base_model.model.layers, input_ids, base_model,
            capture_all_tokens=capture,
        )
        if capture:
            hidden, all_tokens = result
            # Project all tokens with dedicated wm_proj and store
            wm_tokens = self.wm_proj(all_tokens.float())  # (seq_len, d_obs)
            self._working_memory_buffer.append(wm_tokens.detach())
        else:
            hidden = result
        projected = self._project_hidden(hidden)
        gru_memory = self.gru.integrate(projected)
        if return_embedding:
            return gru_memory, projected.detach()
        return gru_memory

    def fire(self, gru_memory):
        """
        PFC update + action compilation. No reward input.

        gru_memory: (d_gru,) from GRU
        Returns: writes (list of (patch_idx, weight_name, coefficients, gate))
        """
        # PFC: gated task state update
        self._task_state = self.task_state_module(gru_memory, self._task_state)

        # Forgetting: noise on task state forces robust representations
        if self.task_noise > 0 and self.training:
            self._task_state = self._task_state + \
                self.task_noise * torch.randn_like(self._task_state)

        # PFC deliberation + planning
        if self.n_deliberation_steps > 0:
            if self.n_planning_steps > 0:
                # Critic-gated planning: generate candidates, soft-select best
                candidates = [self._task_state]
                for _ in range(self.n_planning_steps):
                    candidate = self.deliberation(self._task_state)
                    candidates.append(candidate)
                    self._task_state = candidate

                # Evaluate each candidate with critic (basal ganglia)
                values = torch.stack([self.critic(c) for c in candidates])

                if self.training:
                    # Soft selection: differentiable weighted average
                    weights = torch.softmax(
                        values * self.planning_temperature, dim=0
                    )
                    self._task_state = sum(
                        w * c for w, c in zip(weights, candidates)
                    )
                else:
                    # Hard selection: pick the best at inference
                    best_idx = values.argmax()
                    self._task_state = candidates[best_idx]
            else:
                # Blind deliberation (backward compatible)
                for _ in range(self.n_deliberation_steps):
                    self._task_state = self.deliberation(self._task_state)

        # Premotor: compile task state into patch writes
        writes = self.action_compiler(self._task_state)
        return writes

    def get_task_state(self):
        """Return current task state for sparsity loss."""
        return self._task_state

    def get_value(self):
        """Critic value estimate of current task state (basal ganglia)."""
        if self._task_state is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.critic(self._task_state)

    def select_demo(self, demo_embeddings):
        """
        Active learning: task state attends over stored demo observation
        embeddings to select the most informative demo for self-eval.

        demo_embeddings: (n_demos, d_obs) — detached projected observations
        Returns: scores (n_demos,) — higher = more informative
        """
        return self.demo_selector(self._task_state, demo_embeddings)

    def think(self, n_steps=None):
        """
        Working memory thinking phase: cross-attend over stored demo tokens,
        then compile writes. Called after all demos, before test problems.

        No new Qwen forward pass — purely re-reads stored representations.
        Bypasses fire() (no GRU update, no task_state_module).
        """
        if n_steps is None:
            n_steps = self.n_thinking_steps
        if n_steps <= 0 or not self._working_memory_buffer:
            return
        # Concatenate all stored demo tokens
        memory = torch.cat(self._working_memory_buffer, dim=0)  # (total_tokens, d_obs)
        for _ in range(n_steps):
            self._task_state = self.wm_cross_attn(self._task_state, memory)
            writes = self.action_compiler(self._task_state)
            self.apply_writes(writes)

    def consolidate(self, success_rate):
        """Consolidate current task state into slow memory (after episode)."""
        if self.consolidation and self._task_state is not None:
            self.slow_memory.consolidate(self._task_state, success_rate)

    def get_slow_memory_stats(self):
        """Return slow memory diagnostics. None if consolidation is off."""
        if not self.consolidation:
            return None
        mem = self.slow_memory.memory
        gate = torch.sigmoid(
            self.slow_memory.retrieval_gate(mem)
        ).detach()
        return {
            "norm": mem.norm().item(),
            "max": mem.abs().max().item(),
            "gate_mean": gate.mean().item(),
            "gate_active": (gate > 0.3).sum().item(),
        }

    def apply_writes(self, writes):
        """Apply differentiable patch weight modifications via basis vectors."""
        for (patch_idx, weight_name, coefficients, gate) in writes:
            delta_W = self.basis.compute_delta_W(
                patch_idx, weight_name, coefficients, gate
            )
            self.patches[patch_idx].accumulate_write(weight_name, delta_W)

    def observe_patched(self, patched_model, input_ids):
        """
        Self-evaluation: observe own output through patched model.

        Same observation pathway (layer_projs/obs_proj → GRU) but runs
        through patched_model instead of base_model, so hidden states
        reflect current patches.

        Task-agnostic: no explicit error computation, the model learns
        to detect its own errors through the same pathway it uses for
        everything else.
        """
        hidden = self._extract_hidden_states(
            patched_model.base_model.model.layers, input_ids, patched_model
        )
        projected = self._project_hidden(hidden)
        gru_memory = self.gru.integrate(projected)
        return gru_memory

    def metabolic_cost(self):
        """
        Free energy principle: total activation cost across the meta-learner.

        Replaces separate sparsity + decorrelation losses with a single
        cost: how much energy is the system expending? Less is better,
        subject to being accurate.

        Components:
        - Task state activation (PFC firing cost)
        - Patch delta magnitude (motor execution cost)
        - GRU hidden state (memory maintenance cost)

        Sparsity, decorrelation, and robustness emerge naturally from
        minimizing this single cost under prediction error pressure.
        """
        cost = torch.tensor(0.0, device=self._task_state.device)

        # PFC firing cost: activation magnitude
        cost = cost + self._task_state.abs().mean()

        # Motor execution cost: patch modification magnitude
        for patch in self.patches:
            if patch.delta_down is not None:
                cost = cost + patch.delta_down.abs().mean()
            if patch.delta_up is not None:
                cost = cost + patch.delta_up.abs().mean()
            if patch.delta_gain is not None:
                cost = cost + patch.delta_gain.abs().mean()

        # Memory maintenance cost: GRU hidden state
        if self.gru.memory is not None:
            cost = cost + self.gru.memory.abs().mean()

        # Observation gate cost: push unused layers toward zero
        if self.multi_layer_obs and self._task_state is not None:
            gates = torch.sigmoid(self.obs_gate_net(self._task_state))
            cost = cost + gates.mean()

        return cost

    def get_obs_gates(self):
        """Return current obs gate values for diagnostics. None if single-layer."""
        if self.multi_layer_obs and self._task_state is not None:
            return torch.sigmoid(
                self.obs_gate_net(self._task_state)
            ).detach()
        return None


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
                    patch = self.mach.patches[patch_idx]
                    if isinstance(output, tuple):
                        h = output[0]
                        # Additive: down → GELU → up
                        patch_out = patch(h.float())
                        # Multiplicative: gain modulation
                        gain = patch.get_gain()
                        h_mod = h * (1 + gain).to(h.dtype) + patch_out.to(h.dtype)
                        return (h_mod,) + output[1:]
                    else:
                        patch_out = patch(output.float())
                        gain = patch.get_gain()
                        return output * (1 + gain).to(output.dtype) \
                            + patch_out.to(output.dtype)
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


class DemoProjection(nn.Module):
    """
    Replaces GRU + TaskStateModule for concat architecture.
    Projects concatenated multi-layer observation to task_state in one shot.
    LayerNorm controls magnitude without killing gradients (unlike tanh).
    """

    def __init__(self, d_obs, d_task):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_obs, d_obs),
            nn.GELU(),
            nn.Linear(d_obs, d_task),
            nn.LayerNorm(d_task),
        )

    def forward(self, obs):
        return self.net(obs)


class MACHPhase5Concat(nn.Module):
    """
    Concatenated demo architecture: all demos processed in one Qwen forward pass.

    Qwen's attention does cross-demo integration (free — frozen 4B params).
    Meta-learner is just: layer_projs → DemoProjection → ActionCompiler → writes.
    ~410K params total.

    Episode flow:
    1. Concatenate demo strings with newline
    2. One Qwen forward pass → hook hidden states at patch layers
    3. Last-token hidden from each layer → layer_projs → concat → (d_obs,)
    4. DemoProjection → task_state (d_task,)
    5. ActionCompiler → writes → patches get delta_W
    6. Evaluate test problems through patched model (patches fixed)
    """

    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=256,
                 d_obs=96, d_task=32, n_basis=8):
        super().__init__()
        self.d_model = d_model
        self.d_obs = d_obs
        self.d_task = d_task
        self.n_basis = n_basis
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers

        # Sensory cortex: per-layer projection (multi-layer always)
        d_per_layer = d_obs // len(patch_layers)
        self.d_per_layer = d_per_layer
        self.layer_projs = nn.ModuleList([
            nn.Linear(d_model, d_per_layer, bias=False)
            for _ in patch_layers
        ])

        # DemoProjection: replaces GRU + TaskStateModule
        self.demo_projection = DemoProjection(d_obs, d_task)

        # Premotor: compile task state into patch writes
        n_gain_basis = 4
        self.action_compiler = ActionCompiler(
            d_task, len(patch_layers), n_basis, n_gain_basis
        )

        # Motor execution: basis vectors + patches
        self.basis = BasisVectors(
            d_model, hidden_dim, len(patch_layers), n_basis, n_gain_basis
        )
        self.patches = nn.ModuleList([
            DifferentiablePatch(d_model, hidden_dim) for _ in patch_layers
        ])

        # Basal ganglia: value estimator
        self.critic = TaskStateCritic(d_task)

        # Persistent state
        self._task_state = None

    def reset_episode(self):
        """Call at the start of each episode."""
        for patch in self.patches:
            patch.reset_deltas()
        self._task_state = torch.zeros(
            self.d_task, device=next(self.parameters()).device
        )

    def process_demos(self, base_model, demo_input_ids):
        """
        One-shot demo processing: single Qwen forward pass → project → write.

        demo_input_ids: tokenized concatenated demo string (1, seq_len)
        """
        # Extract hidden states from all patch layers (frozen)
        hidden_states = {}
        hooks = []

        with torch.no_grad():
            for layer_idx in self.patch_layers:
                def make_hook(idx):
                    def hook(module, input, output):
                        t = output[0] if isinstance(output, tuple) else output
                        hidden_states[idx] = t[:, -1, :]  # last token
                    return hook
                h = base_model.model.layers[layer_idx].register_forward_hook(
                    make_hook(layer_idx)
                )
                hooks.append(h)
            base_model(input_ids=demo_input_ids)
            for h in hooks:
                h.remove()

        # Project each layer and concatenate
        layer_obs = []
        for i, layer_idx in enumerate(self.patch_layers):
            proj = self.layer_projs[i](hidden_states[layer_idx].float())
            layer_obs.append(proj)
        obs = torch.cat(layer_obs, dim=-1).squeeze(0)  # (d_obs,)

        # Project to task state
        self._task_state = self.demo_projection(obs)

        # Compile writes and apply
        writes = self.action_compiler(self._task_state)
        self.apply_writes(writes)

    def get_task_state(self):
        """Return current task state for sparsity/energy loss."""
        return self._task_state

    def get_value(self):
        """Critic value estimate of current task state."""
        if self._task_state is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.critic(self._task_state)

    def apply_writes(self, writes):
        """Apply differentiable patch weight modifications via basis vectors."""
        for (patch_idx, weight_name, coefficients, gate) in writes:
            delta_W = self.basis.compute_delta_W(
                patch_idx, weight_name, coefficients, gate
            )
            self.patches[patch_idx].accumulate_write(weight_name, delta_W)

    def metabolic_cost(self):
        """
        Free energy: total activation cost across the meta-learner.
        Task state + patch deltas.
        """
        cost = torch.tensor(0.0, device=self._task_state.device)

        # PFC firing cost
        cost = cost + self._task_state.abs().mean()

        # Motor execution cost
        for patch in self.patches:
            if patch.delta_down is not None:
                cost = cost + patch.delta_down.abs().mean()
            if patch.delta_up is not None:
                cost = cost + patch.delta_up.abs().mean()
            if patch.delta_gain is not None:
                cost = cost + patch.delta_gain.abs().mean()

        return cost


class ActivationPrimitives(nn.Module):
    """
    Learned activation-space primitives for fast modulation (Channel 1).

    Each primitive is a pair of (bias, gain) vectors per patch layer.
    The ModulationHead produces scalar weights; modulation is:
        h += sum_k(w_k * bias[layer, k])        (additive steering)
        h *= (1 + sum_k(w_k * gain[layer, k]))  (multiplicative gating)

    Interpolation is natural: w=3.0 means "3x this primitive".
    """

    def __init__(self, n_layers, n_prims, d_model, modulation_scale=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.n_prims = n_prims
        self.d_model = d_model
        self.modulation_scale = modulation_scale

        # (n_layers, n_prims, d_model) — learned end-to-end across episodes
        self.bias = nn.Parameter(torch.randn(n_layers, n_prims, d_model) * 0.01)
        self.gain = nn.Parameter(torch.randn(n_layers, n_prims, d_model) * 0.01)

        # Scalar weights set per-episode by ModulationHead
        # Shape: (n_layers * n_prims,) — reshaped to (n_layers, n_prims) for use
        self._weights = None

    def set_weights(self, weights):
        """Set primitive weights from ModulationHead output. weights: (n_layers * n_prims,)"""
        self._weights = weights.view(self.n_layers, self.n_prims)

    def reset(self):
        """Clear weights at episode start."""
        self._weights = None

    def modulate(self, layer_idx, hidden_states):
        """
        Apply weighted primitives at a specific layer.

        layer_idx: index into self's layer dimension (0..n_layers-1)
        hidden_states: (batch, seq, d_model) in bfloat16
        Returns: modulated hidden_states
        """
        if self._weights is None:
            return hidden_states

        w = self._weights[layer_idx]  # (n_prims,)
        b = self.bias[layer_idx]      # (n_prims, d_model)
        g = self.gain[layer_idx]      # (n_prims, d_model)

        # Weighted sum of primitives, scaled to keep initial modulation small
        bias_mod = self.modulation_scale * torch.einsum('k,kd->d', w, b)   # (d_model,)
        gain_mod = self.modulation_scale * torch.einsum('k,kd->d', w, g)    # (d_model,)

        # Apply: gain first (multiplicative context), then bias (additive steering)
        h = hidden_states * (1 + gain_mod).to(hidden_states.dtype)
        h = h + bias_mod.to(hidden_states.dtype)
        return h


class ModulationHead(nn.Module):
    """
    Maps task_state to scalar weights for activation primitives.

    Output is unconstrained (no sigmoid) — allows negative and large weights
    for continuous interpolation. Initialized to zero output so modulation
    starts inactive.
    """

    def __init__(self, d_task, n_layers, n_prims, hidden_dim=32):
        super().__init__()
        n_outputs = n_layers * n_prims
        self.head = nn.Sequential(
            nn.Linear(d_task, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_outputs),
        )
        # Initialize to zero output: modulation starts inactive
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, task_state):
        """Returns (n_layers * n_prims,) unconstrained scalar weights."""
        return self.head(task_state)


class DemoEncoder(nn.Module):
    """
    Learned encoder on frozen token embeddings.
    Replaces Qwen forward pass for demo observation.

    embed_tokens(demo_ids) → proj → positional encoding → transformer → pool → task_state

    Why: Qwen's hidden states don't encode arithmetic relationships (verified
    up to 14B-Instruct). This encoder learns end-to-end what to extract from
    raw token embeddings via the meta-learning loss.
    """

    def __init__(self, d_embed, d_enc=128, d_task=32, n_layers=2, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(d_embed, d_enc)
        self.pos_enc = nn.Embedding(512, d_enc)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_enc, nhead=n_heads, dim_feedforward=d_enc * 2,
                batch_first=True, norm_first=True
            ) for _ in range(n_layers)
        ])
        self.out_proj = nn.Sequential(
            nn.Linear(d_enc, d_task),
            nn.LayerNorm(d_task),
        )

    def forward(self, token_embeddings):
        """
        token_embeddings: (seq_len, d_embed) — from frozen embed_tokens
        Returns: task_state (d_task,)
        """
        x = self.input_proj(token_embeddings)
        pos = self.pos_enc(torch.arange(x.shape[0], device=x.device))
        x = x + pos
        x = x.unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        x = x.squeeze(0).mean(dim=0)
        return self.out_proj(x)


class MACHTwoChannel(nn.Module):
    """
    Two-channel universal adapter: modulation (Channel 1) + writing (Channel 2).

    Channel 1 (fast, generalizable): Learned activation primitives scaled by
    continuous scalar weights. Natural interpolation.

    Channel 2 (slow, novel capabilities): Basis vector outer products →
    DifferentiablePatch weight deltas. Penalized by higher metabolic cost.

    Both channels driven by task_state from DemoEncoder (learned on token embeddings).
    """

    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=256,
                 d_obs=96, d_task=32, n_basis=8, n_prims=8,
                 write_cost_scale=10.0):
        super().__init__()
        self.d_model = d_model
        self.d_obs = d_obs
        self.d_task = d_task
        self.n_basis = n_basis
        self.n_prims = n_prims
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers
        self.write_cost_scale = write_cost_scale

        # Demo encoder: token embeddings → task_state
        self.demo_encoder = DemoEncoder(d_model, d_enc=128, d_task=d_task)

        # Channel 1: Modulation (fast, generalizable)
        self.primitives = ActivationPrimitives(
            len(patch_layers), n_prims, d_model
        )
        self.modulation_head = ModulationHead(
            d_task, len(patch_layers), n_prims
        )

        # Channel 2: Writing (slow, novel capabilities)
        n_gain_basis = 4
        self.action_compiler = ActionCompiler(
            d_task, len(patch_layers), n_basis, n_gain_basis
        )
        self.basis = BasisVectors(
            d_model, hidden_dim, len(patch_layers), n_basis, n_gain_basis
        )
        self.patches = nn.ModuleList([
            DifferentiablePatch(d_model, hidden_dim) for _ in patch_layers
        ])

        # Basal ganglia: value estimator
        self.critic = TaskStateCritic(d_task)

        # Persistent state
        self._task_state = None

    def reset_episode(self):
        """Call at the start of each episode."""
        for patch in self.patches:
            patch.reset_deltas()
        self.primitives.reset()
        self._task_state = torch.zeros(
            self.d_task, device=next(self.parameters()).device
        )

    def process_demos(self, base_model, demo_input_ids):
        """
        Demo processing via learned encoder on token embeddings.
        No Qwen forward pass — just embedding lookup + encoder.
        """
        # Get token embeddings (frozen lookup, no transformer)
        with torch.no_grad():
            token_embeds = base_model.model.embed_tokens(
                demo_input_ids.squeeze(0)
            )

        # Learned encoder extracts task state
        self._task_state = self.demo_encoder(token_embeds.float())

        # Channel 1: Modulation
        prim_weights = self.modulation_head(self._task_state)
        self.primitives.set_weights(prim_weights)

        # Channel 2: Writing
        writes = self.action_compiler(self._task_state)
        self.apply_writes(writes)

    def get_task_state(self):
        return self._task_state

    def get_value(self):
        if self._task_state is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.critic(self._task_state)

    def apply_writes(self, writes):
        for (patch_idx, weight_name, coefficients, gate) in writes:
            delta_W = self.basis.compute_delta_W(
                patch_idx, weight_name, coefficients, gate
            )
            self.patches[patch_idx].accumulate_write(weight_name, delta_W)

    def metabolic_cost(self):
        """
        Free energy with asymmetric channel costs.
        Channel 2 (writing) costs write_cost_scale more than Channel 1 (modulation).
        """
        cost = torch.tensor(0.0, device=self._task_state.device)

        # PFC firing cost
        cost = cost + self._task_state.abs().mean()

        # Channel 1 cost: primitive weight magnitude (cheap)
        if self.primitives._weights is not None:
            cost = cost + self.primitives._weights.abs().mean()

        # Channel 2 cost: patch delta magnitude (expensive)
        for patch in self.patches:
            if patch.delta_down is not None:
                cost = cost + self.write_cost_scale * patch.delta_down.abs().mean()
            if patch.delta_up is not None:
                cost = cost + self.write_cost_scale * patch.delta_up.abs().mean()
            if patch.delta_gain is not None:
                cost = cost + self.write_cost_scale * patch.delta_gain.abs().mean()

        return cost


class TwoChannelPatchedModel(nn.Module):
    """
    Wraps Qwen with both modulation (Channel 1) and patches (Channel 2).

    Hook order at each patch layer:
    1. Channel 1: h *= (1 + gain_mod), h += bias_mod  (modulation context)
    2. Channel 2: h += patch(h), h *= (1 + patch_gain)  (patch computation)
    """

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
                    mach = self.mach
                    patch = mach.patches[patch_idx]

                    if isinstance(output, tuple):
                        h = output[0]
                        # Channel 1: Modulation
                        h = mach.primitives.modulate(patch_idx, h)
                        # Channel 2: Patch
                        patch_out = patch(h.float())
                        gain = patch.get_gain()
                        h = h * (1 + gain).to(h.dtype) + patch_out.to(h.dtype)
                        return (h,) + output[1:]
                    else:
                        # Channel 1: Modulation
                        h = mach.primitives.modulate(patch_idx, output)
                        # Channel 2: Patch
                        patch_out = patch(h.float())
                        gain = patch.get_gain()
                        return h * (1 + gain).to(h.dtype) + patch_out.to(h.dtype)
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


class MACHDemoRead(nn.Module):
    """
    Demo-reading meta-learner with skip connection for gradient stability.

    Architecture:
      Demo tokens → frozen embed_tokens → DemoEncoder (2-layer transformer)
        → task_state (d_meta)
        → action_head (SKIP connection, short gradient path)
        → also: GRU → MetaLearnerTransformer → action_head (long path)
        → basis vectors → patch writes

    Key difference from MACHTwoChannel: no modulation channel, no primitives,
    no critic. Just encode demos → write patches. The skip connection
    (task_state directly to action_head as gru_memory) ensures DemoEncoder
    always gets gradient, preventing the gradient death seen in previous
    DemoEncoder attempts.

    Gradient path (short): test_loss → patches → basis → action_head → task_state → DemoEncoder
    Gradient path (long):  test_loss → patches → basis → action_head → transformer → GRU → task_state → DemoEncoder
    """

    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=256,
                 d_meta=128, n_basis=8, oracle=False):
        super().__init__()
        self.d_model = d_model
        self.d_meta = d_meta
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers
        self.oracle = oracle

        if not oracle:
            # Demo encoder: token embeddings → task_state
            self.demo_encoder = DemoEncoder(
                d_embed=d_model, d_enc=d_meta, d_task=d_meta,
                n_layers=2, n_heads=4,
            )
            # Diagnostic: supervised coefficient predictor (training only)
            self.coeff_head = nn.Linear(d_meta, 2)
        else:
            # Oracle mode: embed [c1, c2] directly → task_state
            self.coeff_embed = nn.Sequential(
                nn.Linear(2, d_meta),
                nn.GELU(),
                nn.Linear(d_meta, d_meta),
            )

        # Working memory (integrates task_state, provides context)
        self.gru = SimpleGRU(d_meta)

        # Meta-learner: [gru_mem, task_state, tf_mem, think_0, think_1]
        self.transformer = MetaLearnerTransformer(d_meta, n_tokens=5)

        # Patch writing — obs_conditioned for skip connection from task_state
        self.action_head = ActionHead(
            d_meta, len(patch_layers), n_basis, obs_conditioned=True
        )
        self.memory_head = MemoryHead(d_meta)
        self.basis = BasisVectors(d_model, hidden_dim, len(patch_layers), n_basis)
        self.patches = nn.ModuleList([
            DifferentiablePatch(d_model, hidden_dim) for _ in patch_layers
        ])

        # State
        self._tf_mem = None
        self._task_state = None

    def reset_episode(self):
        """Call at the start of each episode."""
        self.gru.reset()
        for patch in self.patches:
            patch.reset_deltas()
        self._tf_mem = self.transformer.tf_mem_init.clone()
        self._task_state = None

    def process_demos(self, base_model, demo_input_ids, device):
        """
        Encode demo tokens → task_state → write patches.

        demo_input_ids: (1, seq_len) or (seq_len,) token IDs
        No Qwen forward pass — just frozen embedding lookup + learned encoder.
        """
        # Frozen token embedding lookup
        with torch.no_grad():
            token_embeds = base_model.model.embed_tokens(
                demo_input_ids.squeeze(0)
            )

        # Learned encoder: token_embeds → task_state (NOT detached)
        self._task_state = self.demo_encoder(token_embeds.float())
        self._write_from_task_state()

    def process_oracle(self, c1, c2, device):
        """Oracle mode: embed true [c1, c2] directly as task_state."""
        coeffs = torch.tensor([float(c1), float(c2)], device=device)
        self._task_state = self.coeff_embed(coeffs)
        self._write_from_task_state()

    def _write_from_task_state(self):
        """Shared: task_state → GRU → transformer → action_head → patches."""
        # GRU integration (provides memory context)
        gru_mem = self.gru.integrate(self._task_state)

        # Meta-learner transformer
        tokens = torch.stack([
            gru_mem,
            self._task_state,
            self._tf_mem,
            self.transformer.think_0_init,
            self.transformer.think_1_init,
        ])  # (5, d_meta)
        hidden = self.transformer(tokens)  # (5, d_meta)

        # Write patches — skip connection: task_state → action_head directly
        writes = self.action_head(hidden[3], gru_memory=self._task_state)
        self.apply_writes(writes)

        # Update transformer memory
        self._tf_mem = self.memory_head(hidden[4], self._tf_mem)

    def apply_writes(self, writes):
        """Apply differentiable patch weight modifications via basis vectors."""
        for (patch_idx, weight_name, coefficients, gate) in writes:
            delta_W = self.basis.compute_delta_W(
                patch_idx, weight_name, coefficients, gate
            )
            self.patches[patch_idx].accumulate_write(weight_name, delta_W)

    def predict_coeffs(self):
        """Predict c1, c2 from task_state. Training diagnostic only."""
        if self._task_state is None or self.oracle:
            return None
        return self.coeff_head(self._task_state)

    def get_task_state(self):
        return self._task_state


class MACHOracleMinimal(nn.Module):
    """
    Minimal oracle: Linear(2) → patch write coefficients via basis vectors.
    No GRU, no transformer, no memory. Tests whether the basis vector
    write mechanism can express continuous arithmetic at all.
    """

    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=256,
                 n_basis=8):
        super().__init__()
        self.d_model = d_model
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers
        self.oracle = True  # for compatibility with training loop

        n_writes = self.n_patches * 2  # down + up per patch
        n_outputs = n_writes * (n_basis + 1)  # coefficients + gate per write
        self.n_basis = n_basis

        # [c1, c2] → patch write coefficients directly
        # Linear head: guarantees output([1,1]) = output([1,0]) + output([0,1])
        self.head = nn.Linear(2, n_outputs)

        self.basis = BasisVectors(d_model, hidden_dim, len(patch_layers), n_basis)
        self.patches = nn.ModuleList([
            DifferentiablePatch(d_model, hidden_dim) for _ in patch_layers
        ])

        self._task_state = None

    def reset_episode(self):
        for patch in self.patches:
            patch.reset_deltas()
        self._task_state = None

    def process_oracle(self, c1, c2, device):
        from config import GATE_SCALE
        coeffs = torch.tensor([float(c1), float(c2)], device=device)
        self._task_state = coeffs  # just store for diagnostics
        raw = self.head(coeffs)

        # Parse into writes (same format as ActionHead)
        idx = 0
        for patch_i in range(self.n_patches):
            for weight_name in ["down", "up"]:
                coefficients = raw[idx:idx + self.n_basis]
                gate = torch.sigmoid(raw[idx + self.n_basis]) * GATE_SCALE
                idx += self.n_basis + 1
                delta_W = self.basis.compute_delta_W(
                    patch_i, weight_name, coefficients, gate
                )
                self.patches[patch_i].accumulate_write(weight_name, delta_W)

    def predict_coeffs(self):
        return None

    def get_task_state(self):
        return self._task_state


class IterativePatchedModel(nn.Module):
    """
    Wraps Qwen with DifferentiablePatch hooks for MACHIterative.
    Simpler than TwoChannelPatchedModel — no modulation channel.
    """

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
                    patch = self.mach.patches[patch_idx]

                    if isinstance(output, tuple):
                        h = output[0]
                        patch_out = patch(h.float())
                        gain = patch.get_gain()
                        h = h * (1 + gain).to(h.dtype) + patch_out.to(h.dtype)
                        return (h,) + output[1:]
                    else:
                        patch_out = patch(output.float())
                        gain = patch.get_gain()
                        return output * (1 + gain).to(output.dtype) + patch_out.to(output.dtype)
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


class MACHHebbian(nn.Module):
    """
    Three-factor Hebbian learning with critic modulation.

    Instead of a meta-learner computing write coefficients, patches learn via
    local Hebbian updates: Δw = η × δ × pre_h × post_h, where δ is the critic's
    TD error (dopamine analogue).

    Meta-training learns the FORM of the rule (projections, basis vectors,
    critic, learning rate). At inference: forward passes + local updates only.

    Gradient path (meta-training):
      loss → patches → delta_W → basis U,V + coefficients → pre_projs, post_projs
      Activations from Qwen are DETACHED — no backprop through frozen model.
    """

    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=256,
                 n_basis=8, exploration_noise=0.3, init_std=0.001,
                 delta_decay=1.0):
        super().__init__()
        from config import GATE_SCALE
        self.d_model = d_model
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers
        self.n_basis = n_basis
        self.gate_scale = GATE_SCALE
        self.exploration_noise = exploration_noise
        self.init_std = init_std
        self.delta_decay = delta_decay

        # Patches
        self.patches = nn.ModuleList([
            DifferentiablePatch(d_model, hidden_dim) for _ in patch_layers
        ])

        # Basis vectors (reuse existing infrastructure)
        self.basis = BasisVectors(d_model, hidden_dim, len(patch_layers), n_basis)

        # Hebbian projections: compress activations to correlation signals
        # d_model → n_basis, so correlation signal directly drives basis coefficients
        self.pre_projs = nn.ModuleList([
            nn.Linear(d_model, n_basis, bias=False) for _ in patch_layers
        ])
        self.post_projs = nn.ModuleList([
            nn.Linear(d_model, n_basis, bias=False) for _ in patch_layers
        ])

        # Critic (basal ganglia): patch activation summary → value
        self.critic_proj = nn.Linear(len(patch_layers) * n_basis, 64)
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Adaptive learning rate: scalar features → per-patch eta
        self.eta_head = nn.Sequential(
            nn.Linear(3, 32),  # [td_error, reward, step_frac]
            nn.GELU(),
            nn.Linear(32, len(patch_layers)),
            nn.Softplus(),
        )

        # State (set by hooks during forward pass)
        self._pre_activations = {}
        self._post_activations = {}
        self._last_value = None

    def reset_episode(self):
        """Call at the start of each episode.
        Initializes patches with small random weights for exploration diversity.
        """
        for patch in self.patches:
            patch.reset_deltas()
            # Non-zero init: give the model something to work with
            if self.init_std > 0:
                patch.delta_down = torch.randn_like(patch.delta_down) * self.init_std
                patch.delta_up = torch.randn_like(patch.delta_up) * self.init_std
        self._pre_activations.clear()
        self._post_activations.clear()
        self._last_value = None

    def get_activation_summary(self):
        """Summarize captured activations for critic input."""
        summaries = []
        for i in range(self.n_patches):
            pre = self._pre_activations.get(i)
            if pre is not None:
                pre_h = self.pre_projs[i](pre.float()).mean(dim=-2)
                if pre_h.dim() > 1:
                    pre_h = pre_h.squeeze(0)
                summaries.append(pre_h)
            else:
                summaries.append(torch.zeros(
                    self.n_basis, device=self.basis.down_U[0].device
                ))
        return torch.cat(summaries)  # (n_patches * n_basis,)

    def compute_hebbian_update(self, patch_idx, td_error, eta):
        """
        Three-factor Hebbian: coefficients = pre_h * post_h, gate = η × δ.
        Uses basis vectors for low-rank structure.
        """
        pre = self._pre_activations[patch_idx]
        post = self._post_activations[patch_idx]

        # Project to n_basis dims and pool
        pre_h = self.pre_projs[patch_idx](pre.float())
        post_h = self.post_projs[patch_idx](post.float())

        while pre_h.dim() > 1:
            pre_h = pre_h.mean(dim=0)
        while post_h.dim() > 1:
            post_h = post_h.mean(dim=0)

        # Correlation signal: element-wise product (Hebbian), normalized
        coefficients = pre_h * post_h  # (n_basis,)
        coefficients = coefficients / (coefficients.norm() + 1e-8)

        # Stochastic exploration: add noise so reward can select good directions
        if self.training and self.exploration_noise > 0:
            noise = torch.randn_like(coefficients) * self.exploration_noise
            coefficients = coefficients + noise

        # Gate = learning rate × TD error (dopamine modulation), clamped
        gate = eta * td_error * self.gate_scale
        gate = gate.clamp(-1.0, 1.0)

        delta_down = self.basis.compute_delta_W(
            patch_idx, "down", coefficients, gate
        )
        delta_up = self.basis.compute_delta_W(
            patch_idx, "up", coefficients, gate
        )
        return delta_down, delta_up

    def hebbian_step(self, reward, step_idx, n_steps, device):
        """
        Compute Hebbian updates modulated by reward.
        Uses raw reward as modulator (not TD error) to avoid critic collapse.
        Critic still trained for future use but doesn't gate updates.
        """
        act_summary = self.get_activation_summary()
        act_summary = act_summary / (act_summary.norm() + 1e-8)
        value = self.critic(self.critic_proj(act_summary)).squeeze(-1)

        # Use raw reward as modulator (can't collapse to zero)
        modulator = torch.tensor(reward, device=device, dtype=torch.float32)

        step_frac = step_idx / max(n_steps, 1)
        eta_input = torch.tensor(
            [reward, reward, step_frac], device=device,
        )
        etas = self.eta_head(eta_input).clamp(max=1.0)

        for patch_idx in range(self.n_patches):
            if patch_idx in self._pre_activations:
                delta_down, delta_up = self.compute_hebbian_update(
                    patch_idx, modulator, etas[patch_idx]
                )
                self.patches[patch_idx].accumulate_write("down", delta_down, decay=self.delta_decay)
                self.patches[patch_idx].accumulate_write("up", delta_up, decay=self.delta_decay)

        return value, modulator


class HebbianPatchedModel(nn.Module):
    """Wraps Qwen with patches that capture pre/post activations for Hebbian learning."""

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
                    patch = self.mach.patches[patch_idx]

                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output

                    # Capture pre-activation (detached)
                    self.mach._pre_activations[patch_idx] = h.detach()

                    # Apply patch
                    patch_out = patch(h.float())
                    gain = patch.get_gain()
                    h_new = h * (1 + gain).to(h.dtype) + patch_out.to(h.dtype)

                    # Capture post-activation (detached)
                    self.mach._post_activations[patch_idx] = h_new.detach()

                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new
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

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---- Activation-Derived Hebbian (no basis vectors) ----

class ActivationHebbianRule(nn.Module):
    """
    Activation-derived Hebbian updates. No basis vectors.

    Instead of selecting from fixed basis directions, computes rank-1 outer
    products where one factor comes from a learned projection and the other
    from the activation itself (scaled by a learned diagonal).

    This means update DIRECTIONS depend on current input, not on memorized
    templates. Meta-training learns WHAT features to extract, not WHAT
    directions to write.
    """

    def __init__(self, d_model, hidden_dim=256, n_patches=4, n_rank=2, d_proj=32):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.n_patches = n_patches
        self.n_rank = n_rank
        self.d_proj = d_proj

        # Shared compression: d_model -> d_proj (per patch)
        self.compress = nn.ModuleList([
            nn.Linear(d_model, d_proj, bias=False) for _ in range(n_patches)
        ])

        # Per-rank, per-patch projections for delta_down (hidden_dim x d_model)
        self.row_heads_down = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(d_proj, hidden_dim) for _ in range(n_rank)
            ]) for _ in range(n_patches)
        ])
        self.col_diags_down = nn.ParameterList([
            nn.ParameterList([
                nn.Parameter(torch.ones(d_model) * 0.01) for _ in range(n_rank)
            ]) for _ in range(n_patches)
        ])

        # Per-rank, per-patch projections for delta_up (d_model x hidden_dim)
        self.row_diags_up = nn.ParameterList([
            nn.ParameterList([
                nn.Parameter(torch.ones(d_model) * 0.01) for _ in range(n_rank)
            ]) for _ in range(n_patches)
        ])
        self.col_heads_up = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(d_proj, hidden_dim) for _ in range(n_rank)
            ]) for _ in range(n_patches)
        ])

        # Hebbian correlation: pre * post -> scalar coefficient per rank
        self.hebb_pre = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(d_proj, 1, bias=False) for _ in range(n_rank)
            ]) for _ in range(n_patches)
        ])
        self.hebb_post = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(d_proj, 1, bias=False) for _ in range(n_rank)
            ]) for _ in range(n_patches)
        ])

        # Small init for row/col heads
        for patch_heads in self.row_heads_down:
            for head in patch_heads:
                nn.init.normal_(head.weight, std=0.01)
                nn.init.zeros_(head.bias)
        for patch_heads in self.col_heads_up:
            for head in patch_heads:
                nn.init.normal_(head.weight, std=0.01)
                nn.init.zeros_(head.bias)

    def compute_update(self, patch_idx, pre_act, post_act, gate,
                       exploration_noise=0.0, training=False):
        """
        Compute activation-derived Hebbian update.

        pre_act: (seq_len, d_model) or (batch, seq_len, d_model) -- detached
        post_act: same shape -- detached
        gate: scalar (reward * eta * gate_scale)

        Returns: delta_down (hidden_dim, d_model), delta_up (d_model, hidden_dim)
        """
        # Pool to (d_model,)
        pre = pre_act.float()
        while pre.dim() > 1:
            pre = pre.mean(dim=0)
        post = post_act.float()
        while post.dim() > 1:
            post = post.mean(dim=0)

        # Shared compression
        pre_c = self.compress[patch_idx](pre)    # (d_proj,)
        post_c = self.compress[patch_idx](post)  # (d_proj,)

        # Normalize activations for stable outer products
        pre_norm = pre / (pre.norm() + 1e-8)
        post_norm = post / (post.norm() + 1e-8)

        delta_down = torch.zeros(
            self.hidden_dim, self.d_model, device=pre.device
        )
        delta_up = torch.zeros(
            self.d_model, self.hidden_dim, device=pre.device
        )

        for r in range(self.n_rank):
            # Hebbian coefficient: scalar correlation
            coeff = (
                self.hebb_pre[patch_idx][r](pre_c)
                * self.hebb_post[patch_idx][r](post_c)
            ).squeeze()
            coeff = torch.tanh(coeff)

            # Exploration noise on coefficient
            if training and exploration_noise > 0:
                coeff = coeff + torch.randn_like(coeff) * exploration_noise

            # delta_down: learned row, activation-derived col
            u_down = self.row_heads_down[patch_idx][r](pre_c)      # (hidden_dim,)
            v_down = self.col_diags_down[patch_idx][r] * pre_norm   # (d_model,)
            # Normalize both vectors to bound outer product norm
            u_down = u_down / (u_down.norm() + 1e-8)
            v_down = v_down / (v_down.norm() + 1e-8)
            delta_down = delta_down + coeff * torch.outer(u_down, v_down)

            # delta_up: activation-derived row, learned col
            u_up = self.row_diags_up[patch_idx][r] * post_norm      # (d_model,)
            v_up = self.col_heads_up[patch_idx][r](post_c)          # (hidden_dim,)
            u_up = u_up / (u_up.norm() + 1e-8)
            v_up = v_up / (v_up.norm() + 1e-8)
            delta_up = delta_up + coeff * torch.outer(u_up, v_up)

        return gate * delta_down, gate * delta_up


class MACHActivationHebbian(nn.Module):
    """
    Hebbian learning with activation-derived updates (no basis vectors).

    Same structure as MACHHebbian but uses ActivationHebbianRule instead
    of BasisVectors + pre_projs + post_projs.
    """

    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=256,
                 n_rank=2, d_proj=32, exploration_noise=0.3, init_std=0.001,
                 frozen_projections=False, consolidation=False, ema_decay=0.95,
                 delta_decay=1.0, consolidation_interval=0):
        super().__init__()
        from config import GATE_SCALE
        self.d_model = d_model
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers
        self.n_rank = n_rank
        self.gate_scale = GATE_SCALE
        self.exploration_noise = exploration_noise
        self.init_std = init_std
        self.consolidation = consolidation
        self.ema_decay = ema_decay
        self.delta_decay = delta_decay
        self.consolidation_interval = consolidation_interval
        self._step_count = 0
        self._reward_ema = 0.0

        # Patches
        self.patches = nn.ModuleList([
            DifferentiablePatch(d_model, hidden_dim) for _ in patch_layers
        ])

        # Activation-derived Hebbian rule (replaces BasisVectors + projs)
        self.hebb_rule = ActivationHebbianRule(
            d_model, hidden_dim, len(patch_layers), n_rank, d_proj
        )

        # Freeze projections for reservoir-style learning
        if frozen_projections:
            for param in self.hebb_rule.parameters():
                param.requires_grad = False

        # Critic (basal ganglia): patch activation summary -> value
        d_critic_in = len(patch_layers) * d_proj
        self.critic_proj = nn.Linear(d_critic_in, 64)
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Plasticity controller: driven by TD error (like dopamine)
        # Input: activation summary + td_error + reward_ema + per-patch delta norms
        # Trained by its own loss, not backprop through Hebbian chain
        n_patches = len(patch_layers)
        plasticity_in = d_critic_in + 2 + n_patches
        self.plasticity_head = nn.Sequential(
            nn.Linear(plasticity_in, 64),
            nn.GELU(),
            nn.Linear(64, n_patches * 2),  # 2 outputs per patch: eta, decay
        )

        # State
        self._pre_activations = {}
        self._post_activations = {}

    def reset_episode(self):
        """Call at the start of each episode."""
        for patch in self.patches:
            patch.reset_deltas(use_slow=self.consolidation)
            if self.init_std > 0:
                patch.delta_down = patch.delta_down + torch.randn_like(patch.delta_down) * self.init_std
                patch.delta_up = patch.delta_up + torch.randn_like(patch.delta_up) * self.init_std
        self._pre_activations.clear()
        self._post_activations.clear()

    def consolidate(self, avg_reward=None, threshold=0.0):
        """Consolidate deltas into slow memory.
        Called explicitly (episode-based) or automatically (step-based).
        Only consolidates if reward signal is positive.
        """
        if not self.consolidation:
            return
        if avg_reward is not None and avg_reward <= threshold:
            return
        for patch in self.patches:
            patch.consolidate(decay=self.ema_decay)

    def _maybe_consolidate_step(self, reward):
        """Step-based consolidation: consolidate every N steps if reward EMA is positive."""
        if self.consolidation_interval <= 0:
            return
        self._reward_ema = 0.9 * self._reward_ema + 0.1 * reward
        self._step_count += 1
        if self._step_count % self.consolidation_interval == 0 and self._reward_ema > 0:
            for patch in self.patches:
                patch.consolidate(decay=self.ema_decay)

    def get_activation_summary(self):
        """Summarize captured activations for critic input."""
        summaries = []
        for i in range(self.n_patches):
            pre = self._pre_activations.get(i)
            if pre is not None:
                compressed = self.hebb_rule.compress[i](pre.float())
                while compressed.dim() > 1:
                    compressed = compressed.mean(dim=0)
                summaries.append(compressed)
            else:
                summaries.append(torch.zeros(
                    self.hebb_rule.d_proj,
                    device=self.patches[0].down_base.device,
                ))
        return torch.cat(summaries)

    def hebbian_step(self, reward, step_idx, n_steps, device):
        """Compute activation-derived Hebbian updates with dopamine-like plasticity.

        Critic predicts value, TD error drives plasticity controller.
        Both get direct learning signals — no backprop through Hebbian chain.
        """
        act_summary = self.get_activation_summary()
        act_summary = act_summary / (act_summary.norm() + 1e-8)
        value = self.critic(self.critic_proj(act_summary)).squeeze(-1)

        reward_t = torch.tensor(reward, device=device, dtype=torch.float32)

        # TD error = surprise = dopamine
        td_error = (reward_t - value).detach()

        # Update reward EMA (continuous, no episode boundary)
        self._reward_ema = 0.9 * self._reward_ema + 0.1 * reward

        # Compute per-patch delta norms
        delta_norms = torch.tensor(
            [p.delta_down.norm().item() + p.delta_up.norm().item()
             for p in self.patches],
            device=device,
        )

        # Plasticity controller input: TD error replaces raw reward
        plasticity_input = torch.cat([
            act_summary.detach(),                                           # what patches are doing
            torch.tensor([td_error.item(), self._reward_ema], device=device),  # surprise + running reward
            delta_norms,                                                    # how modified patches are
        ])

        # Output: eta (learning rate), decay (memory timescale) per patch
        plasticity_out = self.plasticity_head(plasticity_input)
        plasticity_out = plasticity_out.view(self.n_patches, 2)

        etas = torch.sigmoid(plasticity_out[:, 0])           # [0, 1] learning rate
        decays = torch.sigmoid(plasticity_out[:, 1])         # [0, 1] memory retention

        # Store for diagnostics
        self._last_etas = etas.detach()
        self._last_decays = decays.detach()
        self._last_td_error = td_error.item()

        for patch_idx in range(self.n_patches):
            if patch_idx in self._pre_activations:
                gate = (etas[patch_idx] * td_error * self.gate_scale).clamp(-1.0, 1.0)
                delta_down, delta_up = self.hebb_rule.compute_update(
                    patch_idx,
                    self._pre_activations[patch_idx],
                    self._post_activations[patch_idx],
                    gate,
                    exploration_noise=self.exploration_noise,
                    training=self.training,
                )
                decay = decays[patch_idx]
                self.patches[patch_idx].accumulate_write("down", delta_down, decay=decay)
                self.patches[patch_idx].accumulate_write("up", delta_up, decay=decay)

        # Plasticity controller loss: maximize reward through eta/decay choices
        # Positive TD error → reinforce current eta/decay (do more of this)
        # Negative TD error → push eta/decay away from current values
        plasticity_loss = -(td_error * plasticity_out.sum())
        self._plasticity_loss = plasticity_loss

        return value, reward_t


class ActivationHebbianPatchedModel(nn.Module):
    """Wraps Qwen with patches for activation-derived Hebbian learning."""

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
                    patch = self.mach.patches[patch_idx]

                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output

                    self.mach._pre_activations[patch_idx] = h.detach()

                    patch_out = patch(h.float())
                    gain = patch.get_gain()
                    h_new = h * (1 + gain).to(h.dtype) + patch_out.to(h.dtype)

                    self.mach._post_activations[patch_idx] = h_new.detach()

                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new
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

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================
# Dense Hebbian: many layers + top-down gain modulation
# ============================================================

class MACHDenseHebbian(nn.Module):
    """
    Dense Hebbian steering: patches at 12+ layers with top-down gain modulation.

    Brain analogy:
    - Many small synaptic changes across the whole pathway (dense patches)
    - Neuromodulatory gain control (PFC/thalamus biases which patches activate)
    - Cascading small changes create new computational paths through existing circuits

    Key differences from MACHActivationHebbian:
    - 12 layers instead of 4 (every 3rd Qwen layer)
    - Smaller patches (64 hidden dim) to keep total params reasonable
    - Top-down gain modulator: activation summary -> per-layer gain scalar
    - Gain multiplies patch output BEFORE adding to residual stream
    """

    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=64,
                 n_rank=2, d_proj=32, exploration_noise=0.3, init_std=0.001,
                 consolidation=False, ema_decay=0.95, delta_decay=1.0,
                 consolidation_interval=0):
        super().__init__()
        from config import GATE_SCALE
        self.d_model = d_model
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers
        self.n_rank = n_rank
        self.gate_scale = GATE_SCALE
        self.exploration_noise = exploration_noise
        self.init_std = init_std
        self.consolidation = consolidation
        self.ema_decay = ema_decay
        self.delta_decay = delta_decay
        self.consolidation_interval = consolidation_interval
        self._step_count = 0
        self._reward_ema = 0.0

        # Dense patches — smaller hidden dim, more layers
        self.patches = nn.ModuleList([
            DifferentiablePatch(d_model, hidden_dim) for _ in patch_layers
        ])

        # Hebbian rule for all patches
        self.hebb_rule = ActivationHebbianRule(
            d_model, hidden_dim, len(patch_layers), n_rank, d_proj
        )

        # Top-down gain modulator: activation summary -> per-layer gain
        # Like PFC/thalamus sending gain signals to cortical areas
        d_critic_in = len(patch_layers) * d_proj
        self.gain_net = nn.Sequential(
            nn.Linear(d_critic_in, 64),
            nn.GELU(),
            nn.Linear(64, len(patch_layers)),
            nn.Sigmoid(),  # gain between 0 and 1
        )

        # Critic
        self.critic_proj = nn.Linear(d_critic_in, 64)
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Adaptive learning rate
        self.eta_head = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, len(patch_layers)),
            nn.Softplus(),
        )

        # State
        self._pre_activations = {}
        self._post_activations = {}
        self._gains = None  # computed per forward pass

    def reset_episode(self):
        for patch in self.patches:
            patch.reset_deltas(use_slow=self.consolidation)
            if self.init_std > 0:
                patch.delta_down = patch.delta_down + torch.randn_like(patch.delta_down) * self.init_std
                patch.delta_up = patch.delta_up + torch.randn_like(patch.delta_up) * self.init_std
        self._pre_activations.clear()
        self._post_activations.clear()
        self._gains = None

    def consolidate(self, avg_reward=None, threshold=0.0):
        """Consolidate deltas into slow memory."""
        if not self.consolidation:
            return
        if avg_reward is not None and avg_reward <= threshold:
            return
        for patch in self.patches:
            patch.consolidate(decay=self.ema_decay)

    def _maybe_consolidate_step(self, reward):
        """Step-based consolidation: consolidate every N steps if reward EMA is positive."""
        if self.consolidation_interval <= 0:
            return
        self._reward_ema = 0.9 * self._reward_ema + 0.1 * reward
        self._step_count += 1
        if self._step_count % self.consolidation_interval == 0 and self._reward_ema > 0:
            for patch in self.patches:
                patch.consolidate(decay=self.ema_decay)

    def get_activation_summary(self):
        summaries = []
        for i in range(self.n_patches):
            pre = self._pre_activations.get(i)
            if pre is not None:
                compressed = self.hebb_rule.compress[i](pre.float())
                while compressed.dim() > 1:
                    compressed = compressed.mean(dim=0)
                summaries.append(compressed)
            else:
                summaries.append(torch.zeros(
                    self.hebb_rule.d_proj,
                    device=self.patches[0].down_base.device,
                ))
        return torch.cat(summaries)

    def compute_gains(self):
        """Compute top-down gain modulation from activation summary."""
        act_summary = self.get_activation_summary()
        act_summary = act_summary / (act_summary.norm() + 1e-8)
        self._gains = self.gain_net(act_summary)  # (n_patches,)
        return self._gains

    def get_gain_for_patch(self, patch_idx):
        """Get top-down gain for a specific patch. Returns 1.0 if not computed yet."""
        if self._gains is not None:
            return self._gains[patch_idx]
        return torch.tensor(1.0, device=self.patches[0].down_base.device)

    def hebbian_step(self, reward, step_idx, n_steps, device):
        act_summary = self.get_activation_summary()
        act_summary = act_summary / (act_summary.norm() + 1e-8)
        value = self.critic(self.critic_proj(act_summary)).squeeze(-1)

        modulator = torch.tensor(reward, device=device, dtype=torch.float32)
        step_frac = step_idx / max(n_steps, 1)
        eta_input = torch.tensor([reward, reward, step_frac], device=device)
        etas = self.eta_head(eta_input).clamp(max=1.0)

        for patch_idx in range(self.n_patches):
            if patch_idx in self._pre_activations:
                gate = (etas[patch_idx] * modulator * self.gate_scale).clamp(-1.0, 1.0)
                delta_down, delta_up = self.hebb_rule.compute_update(
                    patch_idx,
                    self._pre_activations[patch_idx],
                    self._post_activations[patch_idx],
                    gate,
                    exploration_noise=self.exploration_noise,
                    training=self.training,
                )
                self.patches[patch_idx].accumulate_write("down", delta_down, decay=self.delta_decay)
                self.patches[patch_idx].accumulate_write("up", delta_up, decay=self.delta_decay)

        self._maybe_consolidate_step(reward)
        return value, modulator


class DenseHebbianPatchedModel(nn.Module):
    """Wraps Qwen with dense patches + top-down gain modulation."""

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
                    patch = self.mach.patches[patch_idx]

                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output

                    self.mach._pre_activations[patch_idx] = h.detach()

                    patch_out = patch(h.float())

                    # Top-down gain modulation
                    top_down_gain = self.mach.get_gain_for_patch(patch_idx)
                    patch_out = patch_out * top_down_gain

                    gain = patch.get_gain()
                    h_new = h * (1 + gain).to(h.dtype) + patch_out.to(h.dtype)

                    self.mach._post_activations[patch_idx] = h_new.detach()

                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new
                return hook

            handle = layer.register_forward_hook(make_hook(i))
            self._hooks.append(handle)

    @property
    def device(self):
        return self.base_model.device

    def forward(self, input_ids, labels=None, attention_mask=None):
        # Compute top-down gains from previous forward pass activations
        # (first forward pass uses default gain=1.0)
        if self.mach._pre_activations:
            self.mach.compute_gains()
        return self.base_model(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class AttentionPatch(nn.Module):
    """
    Small MLP that produces attention output corrections.
    Same structure as DifferentiablePatch but smaller hidden dim (64 vs 256).
    Base weights are zero buffers; accumulated deltas via Hebbian updates.
    """

    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.act = nn.GELU()

        # Base weights fixed at zero
        self.register_buffer('down_base', torch.zeros(hidden_dim, d_model))
        self.register_buffer('up_base', torch.zeros(d_model, hidden_dim))

        # Accumulated deltas
        self.delta_down = None
        self.delta_up = None

    def reset_deltas(self):
        self.delta_down = torch.zeros(
            self.hidden_dim, self.d_model, device=self.down_base.device
        )
        self.delta_up = torch.zeros(
            self.d_model, self.hidden_dim, device=self.up_base.device
        )

    def accumulate_write(self, weight_name, delta_W):
        if weight_name == "down":
            self.delta_down = self.delta_down + delta_W
        else:
            self.delta_up = self.delta_up + delta_W

    def forward(self, hidden_states):
        W_down = self.down_base + (self.delta_down if self.delta_down is not None else 0)
        W_up = self.up_base + (self.delta_up if self.delta_up is not None else 0)

        h = torch.nn.functional.linear(hidden_states, W_down)
        h = self.act(h)
        h = torch.nn.functional.linear(h, W_up)
        return h


class MACHDualHebbian(nn.Module):
    """
    Dual Hebbian: residual patches + attention output patches.

    Extends MACHActivationHebbian with AttentionPatch per layer that modifies
    attention output before the residual connection. Both patch types use
    independent ActivationHebbianRule instances.
    """

    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=256,
                 attn_hidden_dim=64, n_rank=2, d_proj=32,
                 exploration_noise=0.3, init_std=0.001, delta_decay=1.0):
        super().__init__()
        from config import GATE_SCALE
        self.d_model = d_model
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers
        self.n_rank = n_rank
        self.gate_scale = GATE_SCALE
        self.exploration_noise = exploration_noise
        self.init_std = init_std
        self.delta_decay = delta_decay

        # Residual patches (existing)
        self.patches = nn.ModuleList([
            DifferentiablePatch(d_model, hidden_dim) for _ in patch_layers
        ])

        # Attention patches (NEW)
        self.attn_patches = nn.ModuleList([
            AttentionPatch(d_model, attn_hidden_dim) for _ in patch_layers
        ])

        # Hebbian rules: one for residual, one for attention
        self.hebb_rule = ActivationHebbianRule(
            d_model, hidden_dim, len(patch_layers), n_rank, d_proj
        )
        self.attn_hebb_rule = ActivationHebbianRule(
            d_model, attn_hidden_dim, len(patch_layers), n_rank, d_proj
        )

        # Critic (basal ganglia)
        d_critic_in = len(patch_layers) * d_proj
        self.critic_proj = nn.Linear(d_critic_in, 64)
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Adaptive learning rate
        self.eta_head = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, len(patch_layers)),
            nn.Softplus(),
        )

        # State: separate pre/post for residual and attention hooks
        self._pre_activations = {}
        self._post_activations = {}
        self._attn_pre = {}
        self._attn_post = {}

    def reset_episode(self):
        for patch in self.patches:
            patch.reset_deltas()
            if self.init_std > 0:
                patch.delta_down = torch.randn_like(patch.delta_down) * self.init_std
                patch.delta_up = torch.randn_like(patch.delta_up) * self.init_std
        for patch in self.attn_patches:
            patch.reset_deltas()
            if self.init_std > 0:
                patch.delta_down = torch.randn_like(patch.delta_down) * self.init_std
                patch.delta_up = torch.randn_like(patch.delta_up) * self.init_std
        self._pre_activations.clear()
        self._post_activations.clear()
        self._attn_pre.clear()
        self._attn_post.clear()

    def get_activation_summary(self):
        summaries = []
        for i in range(self.n_patches):
            pre = self._pre_activations.get(i)
            if pre is not None:
                compressed = self.hebb_rule.compress[i](pre.float())
                while compressed.dim() > 1:
                    compressed = compressed.mean(dim=0)
                summaries.append(compressed)
            else:
                summaries.append(torch.zeros(
                    self.hebb_rule.d_proj,
                    device=self.patches[0].down_base.device,
                ))
        return torch.cat(summaries)

    def hebbian_step(self, reward, step_idx, n_steps, device):
        act_summary = self.get_activation_summary()
        act_summary = act_summary / (act_summary.norm() + 1e-8)
        value = self.critic(self.critic_proj(act_summary)).squeeze(-1)

        modulator = torch.tensor(reward, device=device, dtype=torch.float32)

        step_frac = step_idx / max(n_steps, 1)
        eta_input = torch.tensor(
            [reward, reward, step_frac], device=device,
        )
        etas = self.eta_head(eta_input).clamp(max=1.0)

        for patch_idx in range(self.n_patches):
            gate = (etas[patch_idx] * modulator * self.gate_scale).clamp(-1.0, 1.0)

            # Residual patch update
            if patch_idx in self._pre_activations:
                delta_down, delta_up = self.hebb_rule.compute_update(
                    patch_idx,
                    self._pre_activations[patch_idx],
                    self._post_activations[patch_idx],
                    gate,
                    exploration_noise=self.exploration_noise,
                    training=self.training,
                )
                self.patches[patch_idx].accumulate_write("down", delta_down, decay=self.delta_decay)
                self.patches[patch_idx].accumulate_write("up", delta_up, decay=self.delta_decay)

            # Attention patch update
            if patch_idx in self._attn_pre:
                delta_down_a, delta_up_a = self.attn_hebb_rule.compute_update(
                    patch_idx,
                    self._attn_pre[patch_idx],
                    self._attn_post[patch_idx],
                    gate,
                    exploration_noise=self.exploration_noise,
                    training=self.training,
                )
                self.attn_patches[patch_idx].accumulate_write("down", delta_down_a, decay=self.delta_decay)
                self.attn_patches[patch_idx].accumulate_write("up", delta_up_a, decay=self.delta_decay)

        return value, modulator


class DualHebbianPatchedModel(nn.Module):
    """Wraps Qwen with both residual patches and attention output patches."""

    def __init__(self, base_model, mach):
        super().__init__()
        self.base_model = base_model
        self.mach = mach
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for i, layer_idx in enumerate(self.mach.patch_layers):
            layer = self.base_model.model.layers[layer_idx]
            self_attn = layer.self_attn

            # Attention output hook (NEW)
            def make_attn_hook(patch_idx):
                def hook(module, args, kwargs, output):
                    h_in = kwargs['hidden_states']  # (batch, seq, d_model)
                    attn_out = output[0]             # (batch, seq, d_model)

                    self.mach._attn_pre[patch_idx] = h_in.detach()

                    attn_patch = self.mach.attn_patches[patch_idx]
                    correction = attn_patch(h_in.float())
                    attn_out_new = attn_out + correction.to(attn_out.dtype)

                    self.mach._attn_post[patch_idx] = attn_out_new.detach()

                    return (attn_out_new,) + output[1:]
                return hook

            handle_attn = self_attn.register_forward_hook(
                make_attn_hook(i), with_kwargs=True
            )
            self._hooks.append(handle_attn)

            # Residual stream hook (existing pattern)
            def make_layer_hook(patch_idx):
                def hook(module, input, output):
                    patch = self.mach.patches[patch_idx]

                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output

                    self.mach._pre_activations[patch_idx] = h.detach()

                    patch_out = patch(h.float())
                    gain = patch.get_gain()
                    h_new = h * (1 + gain).to(h.dtype) + patch_out.to(h.dtype)

                    self.mach._post_activations[patch_idx] = h_new.detach()

                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new
                return hook

            handle_layer = layer.register_forward_hook(make_layer_hook(i))
            self._hooks.append(handle_layer)

    @property
    def device(self):
        return self.base_model.device

    def forward(self, input_ids, labels=None, attention_mask=None):
        return self.base_model(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================
# Coprocessor v2: direct injection via hooks (no virtual tokens)
# ============================================================

class MACHCoprocessor(nn.Module):
    """
    Coprocessor as a "deep patch": reads from an early Qwen layer,
    processes through multiple DifferentiablePatch layers, and injects
    the result into a later layer. All via hooks — directly in the
    residual stream gradient path.

    Architecture:
    - Read hook at read_layer: captures hidden state
    - N DifferentiablePatch layers process it (with residual connections)
    - Write hook at write_layer: adds coprocessor output to hidden state
    - Plus standard residual patches at other layers for steering

    All coprocessor weights are buffers written by ActivationHebbianRule.
    """

    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=256,
                 copro_hidden_dim=256, n_copro_layers=3,
                 read_layer=None, write_layer=None,
                 n_rank=2, d_proj=32,
                 exploration_noise=0.3, init_std=0.001, delta_decay=1.0):
        super().__init__()
        from config import GATE_SCALE
        self.d_model = d_model
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers
        self.n_rank = n_rank
        self.gate_scale = GATE_SCALE
        self.exploration_noise = exploration_noise
        self.init_std = init_std
        self.delta_decay = delta_decay
        self.n_copro_layers = n_copro_layers

        # Read from early layer, write to later layer
        # Default: read at 25%, write at 75%
        self.read_layer = read_layer if read_layer is not None else n_layers // 4
        self.write_layer = write_layer if write_layer is not None else 3 * n_layers // 4

        # Coprocessor: stack of DifferentiablePatch layers
        self.copro_patches = nn.ModuleList([
            DifferentiablePatch(d_model, copro_hidden_dim)
            for _ in range(n_copro_layers)
        ])

        # Hebbian rule for coprocessor
        self.copro_hebb_rule = ActivationHebbianRule(
            d_model, copro_hidden_dim, n_copro_layers, n_rank, d_proj
        )

        # Residual patches (steering)
        self.patches = nn.ModuleList([
            DifferentiablePatch(d_model, hidden_dim) for _ in patch_layers
        ])

        # Hebbian rule for residual patches
        self.hebb_rule = ActivationHebbianRule(
            d_model, hidden_dim, len(patch_layers), n_rank, d_proj
        )

        # Critic
        n_total = len(patch_layers) + n_copro_layers
        d_critic_in = n_total * d_proj
        self.critic_proj = nn.Linear(d_critic_in, 64)
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Adaptive learning rate
        n_eta = len(patch_layers) + 1  # +1 shared for coprocessor
        self.eta_head = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, n_eta),
            nn.Softplus(),
        )

        # State
        self._pre_activations = {}
        self._post_activations = {}
        self._copro_pre = {}
        self._copro_post = {}
        self._captured_hidden = None  # hidden state from read_layer

    def reset_episode(self):
        for patch in self.patches:
            patch.reset_deltas()
            if self.init_std > 0:
                patch.delta_down = torch.randn_like(patch.delta_down) * self.init_std
                patch.delta_up = torch.randn_like(patch.delta_up) * self.init_std
        for patch in self.copro_patches:
            patch.reset_deltas()
            if self.init_std > 0:
                patch.delta_down = torch.randn_like(patch.delta_down) * self.init_std
                patch.delta_up = torch.randn_like(patch.delta_up) * self.init_std
        self._pre_activations.clear()
        self._post_activations.clear()
        self._copro_pre.clear()
        self._copro_post.clear()
        self._captured_hidden = None

    def run_coprocessor(self, hidden_state):
        """Process captured hidden state through coprocessor patches."""
        x = hidden_state.float()
        for i, patch in enumerate(self.copro_patches):
            self._copro_pre[i] = x.detach()
            x = x + patch(x)  # residual
            self._copro_post[i] = x.detach()
        # Return the delta (what coprocessor adds), not the full state
        return (x - hidden_state.float())

    def get_activation_summary(self):
        summaries = []
        # Residual patches
        for i in range(self.n_patches):
            pre = self._pre_activations.get(i)
            if pre is not None:
                compressed = self.hebb_rule.compress[i](pre.float())
                while compressed.dim() > 1:
                    compressed = compressed.mean(dim=0)
                summaries.append(compressed)
            else:
                summaries.append(torch.zeros(
                    self.hebb_rule.d_proj,
                    device=self.patches[0].down_base.device,
                ))
        # Coprocessor patches
        for i in range(self.n_copro_layers):
            pre = self._copro_pre.get(i)
            if pre is not None:
                compressed = self.copro_hebb_rule.compress[i](pre.float())
                while compressed.dim() > 1:
                    compressed = compressed.mean(dim=0)
                summaries.append(compressed)
            else:
                summaries.append(torch.zeros(
                    self.copro_hebb_rule.d_proj,
                    device=self.patches[0].down_base.device,
                ))
        return torch.cat(summaries)

    def hebbian_step(self, reward, step_idx, n_steps, device):
        act_summary = self.get_activation_summary()
        act_summary = act_summary / (act_summary.norm() + 1e-8)
        value = self.critic(self.critic_proj(act_summary)).squeeze(-1)

        modulator = torch.tensor(reward, device=device, dtype=torch.float32)
        step_frac = step_idx / max(n_steps, 1)
        eta_input = torch.tensor([reward, reward, step_frac], device=device)
        etas = self.eta_head(eta_input).clamp(max=1.0)

        # Update residual patches
        for patch_idx in range(self.n_patches):
            if patch_idx in self._pre_activations:
                gate = (etas[patch_idx] * modulator * self.gate_scale).clamp(-1.0, 1.0)
                delta_down, delta_up = self.hebb_rule.compute_update(
                    patch_idx,
                    self._pre_activations[patch_idx],
                    self._post_activations[patch_idx],
                    gate,
                    exploration_noise=self.exploration_noise,
                    training=self.training,
                )
                self.patches[patch_idx].accumulate_write("down", delta_down, decay=self.delta_decay)
                self.patches[patch_idx].accumulate_write("up", delta_up, decay=self.delta_decay)

        # Update coprocessor patches
        copro_eta = etas[self.n_patches]  # shared eta for coprocessor
        for copro_idx in range(self.n_copro_layers):
            if copro_idx in self._copro_pre:
                gate = (copro_eta * modulator * self.gate_scale).clamp(-1.0, 1.0)
                delta_down, delta_up = self.copro_hebb_rule.compute_update(
                    copro_idx,
                    self._copro_pre[copro_idx],
                    self._copro_post[copro_idx],
                    gate,
                    exploration_noise=self.exploration_noise,
                    training=self.training,
                )
                self.copro_patches[copro_idx].accumulate_write("down", delta_down, decay=self.delta_decay)
                self.copro_patches[copro_idx].accumulate_write("up", delta_up, decay=self.delta_decay)

        return value, modulator


class CoprocessorPatchedModel(nn.Module):
    """
    Wraps Qwen with direct-injection coprocessor + residual patches.

    Hooks:
    1. Read hook at read_layer: captures hidden state for coprocessor
    2. Write hook at write_layer: runs coprocessor, adds output to residual stream
    3. Residual patch hooks at patch_layers: steer existing computation
    """

    def __init__(self, base_model, mach):
        super().__init__()
        self.base_model = base_model
        self.mach = mach
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # Read hook: capture hidden state at read_layer
        read_layer = self.base_model.model.layers[self.mach.read_layer]

        def read_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            self.mach._captured_hidden = h  # keep in graph for gradient
            return output

        handle = read_layer.register_forward_hook(read_hook)
        self._hooks.append(handle)

        # Write hook: run coprocessor and inject at write_layer
        write_layer = self.base_model.model.layers[self.mach.write_layer]

        def write_hook(module, input, output):
            if self.mach._captured_hidden is None:
                return output

            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            # Run coprocessor on captured hidden state
            copro_delta = self.mach.run_coprocessor(self.mach._captured_hidden)
            h_new = h + copro_delta.to(h.dtype)

            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new

        handle = write_layer.register_forward_hook(write_hook)
        self._hooks.append(handle)

        # Residual patch hooks (same as before)
        for i, layer_idx in enumerate(self.mach.patch_layers):
            layer = self.base_model.model.layers[layer_idx]

            def make_hook(patch_idx):
                def hook(module, input, output):
                    patch = self.mach.patches[patch_idx]
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output

                    self.mach._pre_activations[patch_idx] = h.detach()

                    patch_out = patch(h.float())
                    gain = patch.get_gain()
                    h_new = h * (1 + gain).to(h.dtype) + patch_out.to(h.dtype)

                    self.mach._post_activations[patch_idx] = h_new.detach()

                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new
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

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
