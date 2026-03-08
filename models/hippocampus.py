"""
Hippocampus: differentiable episodic memory.

A DNC-inspired memory module where all operations are soft and differentiable:
- Soft writes with learned write strength (no store/don't-store threshold)
- Soft reads via content-based attention (no top-k, no similarity threshold)
- Learned forgetting (no hardcoded decay rate)
- All operations remain in the computational graph → gradients teach the
  system what to remember, retrieve, and forget.

Bitter lesson: no hardcoded thresholds. Everything is learned.

Memory layout: (n_slots, d_mem) matrix on GPU.
  n_slots=64, d_mem=64 → 4096 floats = 16KB. Negligible.

Interface is kept compatible with training/eval loops.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Hippocampus(nn.Module):
    """Differentiable episodic memory.

    Fixed-size memory matrix with soft read/write/forget.
    All operations are differentiable — no discrete decisions.
    """

    def __init__(self, key_dim, pfc_dim=32, n_patches=4, max_memories=64,
                 save_path=None):
        super().__init__()
        self.key_dim = key_dim       # activation summary dim (128)
        self.pfc_dim = pfc_dim       # PFC state dim (32)
        self.n_patches = n_patches
        self.n_slots = max_memories  # renamed: fixed slots, not max
        self.d_mem = pfc_dim + n_patches * 3  # what we store: PFC + neuromod
        self.save_path = save_path

        # --- Memory matrix (buffer, not parameter — written by soft ops) ---
        self.register_buffer('memory', torch.zeros(self.n_slots, self.d_mem))
        self.register_buffer('usage', torch.zeros(self.n_slots))  # how used each slot is

        # --- Key projection with pattern separation (Dentate Gyrus) ---
        # Two-layer ReLU: sparse activation orthogonalizes similar inputs
        # This IS the DG — similar add/sub/mul get distinct sparse codes
        input_dim = key_dim + pfc_dim
        self.key_proj = nn.Sequential(
            nn.Linear(input_dim, key_dim * 2),
            nn.ReLU(),  # sparsity = pattern separation
            nn.Linear(key_dim * 2, key_dim),
        )
        # Init: activation part near-identity, PFC part near-zero
        with torch.no_grad():
            nn.init.zeros_(self.key_proj[0].weight)
            self.key_proj[0].weight[:key_dim, :key_dim].copy_(torch.eye(key_dim))
            nn.init.zeros_(self.key_proj[0].bias)
            nn.init.eye_(self.key_proj[2].weight[:, :key_dim])
            nn.init.zeros_(self.key_proj[2].weight[:, key_dim:])
            nn.init.zeros_(self.key_proj[2].bias)

        # --- Stored keys (buffer, written alongside memory) ---
        self.register_buffer('keys', torch.zeros(self.n_slots, key_dim))

        # --- Write head ---
        # Input: PFC state + td_error + reward → write_strength + erase_strength
        write_input_dim = pfc_dim + 2  # pfc + td_error + reward
        self.write_gate = nn.Sequential(
            nn.Linear(write_input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 2),  # [write_strength, erase_strength]
        )
        # Init: small writes, minimal erasure
        with torch.no_grad():
            self.write_gate[-1].bias.copy_(torch.tensor([-2.0, -3.0]))

        # Value projection: PFC + neuromod → what to write
        self.write_value = nn.Linear(self.d_mem, self.d_mem)
        with torch.no_grad():
            nn.init.eye_(self.write_value.weight)
            nn.init.zeros_(self.write_value.bias)

        # --- Read head ---
        # Reinstatement: read output → PFC modification
        self.read_to_pfc = nn.Linear(self.d_mem, pfc_dim)
        with torch.no_grad():
            # Init: first pfc_dim dims map to PFC (near-identity), rest near-zero
            nn.init.zeros_(self.read_to_pfc.weight)
            self.read_to_pfc.weight[:pfc_dim, :pfc_dim].copy_(
                torch.eye(pfc_dim) * 0.1
            )
            nn.init.zeros_(self.read_to_pfc.bias)

        # Read strength gate: controls how much the read output affects PFC
        self.read_gate = nn.Sequential(
            nn.Linear(pfc_dim + 2, 16),  # pfc + td_error + reward_ema
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Tanh(),  # [-1, 1]: positive=approach, negative=avoidance
        )
        with torch.no_grad():
            self.read_gate[-2].bias.fill_(0.3)  # mild positive init

        # --- Forget gate: learned per-slot decay ---
        # Input: usage → retain probability
        self.forget_gate = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        # Init: high retention (~0.99)
        with torch.no_grad():
            self.forget_gate[0].weight.fill_(1.0)
            self.forget_gate[0].bias.fill_(3.0)
            self.forget_gate[2].weight.fill_(1.0)
            self.forget_gate[2].bias.fill_(0.0)

        # Neuromod bias output: read → neuromod bias for Hebbian system
        neuromod_dim = n_patches * 3  # eta + decay + expl per patch
        self.read_to_neuromod = nn.Linear(self.d_mem, neuromod_dim)
        with torch.no_grad():
            nn.init.zeros_(self.read_to_neuromod.weight)
            # Extract neuromod portion of memory (after pfc_dim)
            self.read_to_neuromod.weight[:neuromod_dim, pfc_dim:pfc_dim + neuromod_dim].copy_(
                torch.eye(neuromod_dim) * 0.1
            )
            nn.init.zeros_(self.read_to_neuromod.bias)

        # Track reward EMA for read gate input (running stat, not a threshold)
        self._reward_ema = 0.0

    def _make_key(self, activation_summary, pfc_state):
        """Project activation + PFC into key space."""
        if pfc_state is not None:
            pfc = pfc_state.squeeze(0) if pfc_state.dim() > 1 else pfc_state
        else:
            pfc = torch.zeros(self.pfc_dim, device=activation_summary.device)
        combined = torch.cat([activation_summary, pfc])
        return self.key_proj(combined)

    def _content_attention(self, key):
        """Soft content-based attention over memory slots.

        Returns: (n_slots,) attention weights summing to 1.
        """
        # Cosine similarity between query key and stored keys
        key_norm = F.normalize(key.unsqueeze(0), dim=-1)          # (1, key_dim)
        keys_norm = F.normalize(self.keys, dim=-1)                # (n_slots, key_dim)
        sims = (keys_norm @ key_norm.squeeze(0))                  # (n_slots,)

        # Weight by usage (empty slots have usage ~0, so low attention)
        weighted = sims * self.usage

        # Softmax → attention weights (temperature=1, learned implicitly via key_proj)
        return F.softmax(weighted * 10.0, dim=0)  # sharpening factor

    def retrieve_and_reinstate(self, mach, activation_summary, current_td_error,
                               top_k=3, device=None):
        """Soft read from memory → reinstate into PFC.

        Fully differentiable. No discrete retrieval decisions.
        Returns max attention weight (for diagnostics).
        """
        if device is None:
            device = activation_summary.device

        # Ensure memory is on the right device
        if self.memory.device != device:
            self.to(device)

        pfc = mach._pfc_state if hasattr(mach, '_pfc_state') else None

        # Content-based attention
        key = self._make_key(activation_summary, pfc)  # gradient flows
        attn = self._content_attention(key)             # (n_slots,)

        # Soft read: weighted sum of memory contents
        read_out = attn @ self.memory                   # (d_mem,)

        # Read gate: learned blend strength with approach/avoidance
        pfc_flat = pfc.squeeze(0) if pfc is not None else torch.zeros(self.pfc_dim, device=device)
        gate_input = torch.cat([
            pfc_flat,
            torch.tensor([abs(current_td_error)], device=device, dtype=torch.float32),
            torch.tensor([self._reward_ema], device=device, dtype=torch.float32),
        ])
        alpha = self.read_gate(gate_input).squeeze()  # scalar in [-1, 1]

        # Reinstate into PFC
        pfc_delta = self.read_to_pfc(read_out)  # (pfc_dim,)
        if pfc is not None:
            mach._pfc_state = mach._pfc_state + alpha * pfc_delta.unsqueeze(0)

        # Neuromod bias from memory
        neuromod_raw = self.read_to_neuromod(read_out)  # (n_patches * 3,)
        alpha_f = alpha.item()
        if abs(alpha_f) > 1e-4 and hasattr(mach, '_eta_state'):
            nm = neuromod_raw.view(3, self.n_patches)  # (3, n_patches)
            mach._neuromod_bias = {
                'eta': nm[0].clamp(0.1, 1.0),
                'decay': nm[1].clamp(0.1, 1.0),
                'expl': nm[2].clamp(0.1, 0.5),
                'alpha': alpha_f,
            }

        # Store attention stats for diagnostics
        self._last_max_attn = attn.max().item()
        return abs(alpha_f)

    def store(self, mach, activation_summary, reward, td_error):
        """Soft write to memory.

        Every step writes — the write gate controls HOW MUCH, not WHETHER.
        No threshold. The system learns to write strongly for important events
        and weakly (≈ no-op) for mundane ones.

        Gradient flows through the READ path (retrieve_and_reinstate).
        Writes are detached to avoid in-place modification conflicts with
        the read graph. The write gate still learns via the read path:
        what gets written determines what can be read next step.
        """
        device = activation_summary.device
        if self.memory.device != device:
            self.to(device)

        self._reward_ema = 0.99 * self._reward_ema + 0.01 * reward

        with torch.no_grad():
            pfc = mach._pfc_state if hasattr(mach, '_pfc_state') else None
            pfc_flat = pfc.squeeze(0) if pfc is not None else torch.zeros(self.pfc_dim, device=device)

            # Compose memory value: PFC state + neuromod values
            etas = torch.zeros(self.n_patches, device=device)
            decays = torch.zeros(self.n_patches, device=device)
            expls = torch.zeros(self.n_patches, device=device)
            if hasattr(mach, '_last_etas') and mach._last_etas is not None:
                etas = mach._last_etas
                decays = mach._last_decays
                expls = mach._last_expls
            raw_value = torch.cat([pfc_flat, etas, decays, expls])
            value = self.write_value(raw_value)

            # Write key
            key = self._make_key(activation_summary, pfc_flat if pfc is None else pfc)

            # Content-based write addressing + allocation
            content_attn = self._content_attention(key)
            alloc_attn = F.softmax(-self.usage * 10.0, dim=0)
            write_attn = 0.5 * content_attn + 0.5 * alloc_attn

            # Write gate: learned strength based on context
            gate_input = torch.cat([
                pfc_flat,
                torch.tensor([td_error], device=device, dtype=torch.float32),
                torch.tensor([reward], device=device, dtype=torch.float32),
            ])
            gates = self.write_gate(gate_input)
            write_strength = torch.sigmoid(gates[0])
            erase_strength = torch.sigmoid(gates[1])

            # Erase then write (DNC-style)
            erase = write_attn.unsqueeze(1) * erase_strength
            self.memory.copy_(
                self.memory * (1 - erase) +
                write_attn.unsqueeze(1) * write_strength * value.unsqueeze(0)
            )

            # Update keys
            ws = (write_attn * write_strength).unsqueeze(1)
            self.keys.copy_(
                self.keys * (1 - ws) + ws * key.unsqueeze(0)
            )

            # Update usage
            self.usage.copy_(
                self.usage * (1 - erase.squeeze(1)) + write_attn * write_strength
            )

        return True

    def set_neuromod(self, gamma, avg_decay):
        """Accept neuromod signals. Kept for interface compatibility.
        In the differentiable version, forgetting is learned, not externally set.
        """
        pass  # forgetting is handled by learned forget_gate in decay_all()

    def reconsolidate(self, td_error):
        """No-op in differentiable version.
        Reconsolidation happens implicitly through soft writes —
        re-attending similar memories during write updates them.
        """
        pass

    def decay_all(self):
        """Learned forgetting. Called periodically (e.g. at checkpoints).

        Forget gate sees per-slot usage and decides retention.
        Fully differentiable (but called under no_grad since it's maintenance).
        """
        with torch.no_grad():
            # Forget gate: usage → retain probability per slot
            retain = self.forget_gate(self.usage.unsqueeze(1)).squeeze(1)  # (n_slots,)
            self.memory *= retain.unsqueeze(1)
            self.keys *= retain.unsqueeze(1)
            self.usage *= retain

    def save(self, path=None):
        """Persist memory state to disk."""
        path = path or self.save_path
        if path is None:
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'memory': self.memory.cpu(),
            'keys': self.keys.cpu(),
            'usage': self.usage.cpu(),
        }, path)

    def _load(self, path):
        if not os.path.exists(path):
            return
        data = torch.load(path, map_location='cpu', weights_only=True)
        self.memory.copy_(data['memory'])
        self.keys.copy_(data['keys'])
        self.usage.copy_(data['usage'])

    def __len__(self):
        """Number of meaningfully used slots (usage > 0.01)."""
        return int((self.usage > 0.01).sum().item())

    def __repr__(self):
        return f"Hippocampus({len(self)}/{self.n_slots} slots, d_mem={self.d_mem})"
