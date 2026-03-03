import torch
import torch.nn as nn

from models.observation import ObservationProjection
from models.gru import SimpleGRU
from models.basis_vectors import BasisVectors
from models.meta_learner import MetaLearnerTransformer
from models.action_head import ActionHead
from models.memory_head import MemoryHead
from models.reward_projection import RewardProjection


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
                 d_meta=128, n_basis=8):
        super().__init__()
        self.d_model = d_model
        self.d_meta = d_meta
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers

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

        # Project and integrate (detached — no gradient through observation)
        projected = self.obs_proj(hidden_state.float().unsqueeze(1)).squeeze(0).detach()
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
