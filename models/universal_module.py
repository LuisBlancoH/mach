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
    Same output format as ActionHead but takes d_task input.
    """

    def __init__(self, d_task, n_patches=4, n_basis=8):
        super().__init__()
        self.n_patches = n_patches
        self.n_basis = n_basis
        n_writes = n_patches * 2
        n_outputs = n_writes * (n_basis + 1)
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
            for weight_name in ["down", "up"]:
                coefficients = raw[idx:idx + self.n_basis]
                gate = torch.sigmoid(raw[idx + self.n_basis]) * GATE_SCALE
                idx += self.n_basis + 1
                writes.append((patch_i, weight_name, coefficients, gate))

        return writes


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
                 multi_layer_obs=False):
        super().__init__()
        self.d_model = d_model
        self.d_obs = d_obs
        self.d_gru = d_gru
        self.d_task = d_task
        self.n_basis = n_basis
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers
        self.n_deliberation_steps = n_deliberation_steps
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
        self.action_compiler = ActionCompiler(d_task, len(patch_layers), n_basis)

        # Motor execution: basis vectors + patches
        self.basis = BasisVectors(d_model, hidden_dim, len(patch_layers), n_basis)
        self.patches = nn.ModuleList([
            DifferentiablePatch(d_model, hidden_dim) for _ in patch_layers
        ])

        # Persistent state (reset per episode)
        self._task_state = None

    def reset_episode(self):
        """Call at the start of each episode."""
        self.gru.reset()
        for patch in self.patches:
            patch.reset_deltas()
        self._task_state = torch.zeros(
            self.d_task, device=next(self.parameters()).device
        )

    def _extract_hidden_states(self, model_layers, input_ids, run_model):
        """
        Extract hidden states from Qwen layers (frozen forward pass).
        Returns dict of {layer_idx: hidden_state} if multi_layer_obs,
        or single hidden_state tensor if not.
        """
        with torch.no_grad():
            if self.multi_layer_obs:
                hidden_states = {}
                hooks = []
                for layer_idx in self.patch_layers:
                    def make_hook(idx):
                        def hook(module, input, output):
                            t = output[0] if isinstance(output, tuple) else output
                            hidden_states[idx] = t[:, -1, :]
                        return hook
                    h = model_layers[layer_idx].register_forward_hook(
                        make_hook(layer_idx)
                    )
                    hooks.append(h)
                run_model(input_ids=input_ids)
                for h in hooks:
                    h.remove()
                return hidden_states
            else:
                hidden_state = None
                target_layer = self.patch_layers[len(self.patch_layers) // 2]

                def hook(module, input, output):
                    nonlocal hidden_state
                    if isinstance(output, tuple):
                        hidden_state = output[0][:, -1, :]
                    else:
                        hidden_state = output[:, -1, :]

                h = model_layers[target_layer].register_forward_hook(hook)
                run_model(input_ids=input_ids)
                h.remove()
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

    def observe(self, base_model, input_ids):
        """
        Sensory processing: Qwen forward (frozen) -> project -> GRU.
        Gradient flows through obs_proj/layer_projs and GRU (always undetached).
        """
        hidden = self._extract_hidden_states(
            base_model.model.layers, input_ids, base_model
        )
        projected = self._project_hidden(hidden)
        gru_memory = self.gru.integrate(projected)
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

        # PFC deliberation: iterative refinement
        if self.n_deliberation_steps > 0:
            for _ in range(self.n_deliberation_steps):
                self._task_state = self.deliberation(self._task_state)

        # Premotor: compile task state into patch writes
        writes = self.action_compiler(self._task_state)
        return writes

    def get_task_state(self):
        """Return current task state for sparsity loss."""
        return self._task_state

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
