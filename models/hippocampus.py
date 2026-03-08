"""
Hippocampus: discrete slot memory with learned prototypes.

VQ-VAE inspired: a small codebook of learned prototype keys.
Each slot learns to represent one "mode" of operation.
Matching is discrete (argmax) but gradients flow via straight-through.

No hardcoded thresholds. The system learns:
- How many effective slots to use (prototypes specialize or merge)
- What each slot represents (via prototype key gradient)
- What to store per slot (PFC state + neuromod values)
- How strongly to reinstate (read gate with approach/avoidance)

Like thalamic nuclei: a small number of distinct modes the system
switches between, each with associated cortical/neuromod settings.

Interface is kept compatible with training/eval loops.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Hippocampus(nn.Module):
    """Discrete slot memory with learned prototypes.

    N_SLOTS learned prototype keys. Incoming state matches to the best
    prototype (argmax). That slot's contents are reinstated.
    Straight-through estimator: forward=argmax, backward=softmax.
    """

    def __init__(self, key_dim, pfc_dim=32, n_patches=4, max_memories=16,
                 save_path=None):
        super().__init__()
        self.key_dim = key_dim       # activation summary dim (128)
        self.pfc_dim = pfc_dim       # PFC state dim (32)
        self.n_patches = n_patches
        self.n_slots = max_memories  # number of discrete slots
        self.d_mem = pfc_dim + n_patches * 3  # PFC + neuromod per slot
        self.save_path = save_path

        # --- Learned prototype keys (the codebook) ---
        # These are nn.Parameters — they learn via straight-through gradient
        # Each prototype learns to represent one "type" of situation
        self.prototypes = nn.Parameter(torch.randn(self.n_slots, key_dim) * 0.1)

        # --- Key projection with pattern separation (Dentate Gyrus) ---
        # Two-layer ReLU: sparse activation orthogonalizes similar inputs
        input_dim = key_dim + pfc_dim
        self.key_proj = nn.Sequential(
            nn.Linear(input_dim, key_dim * 2),
            nn.ReLU(),
            nn.Linear(key_dim * 2, key_dim),
        )
        with torch.no_grad():
            nn.init.zeros_(self.key_proj[0].weight)
            self.key_proj[0].weight[:key_dim, :key_dim].copy_(torch.eye(key_dim))
            nn.init.zeros_(self.key_proj[0].bias)
            nn.init.eye_(self.key_proj[2].weight[:, :key_dim])
            nn.init.zeros_(self.key_proj[2].weight[:, key_dim:])
            nn.init.zeros_(self.key_proj[2].bias)

        # --- Slot contents (buffers — written by discrete ops) ---
        self.register_buffer('memory', torch.zeros(self.n_slots, self.d_mem))
        self.register_buffer('write_count', torch.zeros(self.n_slots))

        # --- Read gate: controls reinstatement strength ---
        # Input: match_similarity + td_error + reward_ema
        # Output: [-1, 1] — approach/avoidance
        self.read_gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        with torch.no_grad():
            self.read_gate[-2].bias.fill_(0.1)  # mild positive init

        # --- Reinstatement projections ---
        self.read_to_pfc = nn.Linear(self.d_mem, pfc_dim)
        with torch.no_grad():
            nn.init.zeros_(self.read_to_pfc.weight)
            self.read_to_pfc.weight[:pfc_dim, :pfc_dim].copy_(
                torch.eye(pfc_dim) * 0.1
            )
            nn.init.zeros_(self.read_to_pfc.bias)

        neuromod_dim = n_patches * 3
        self.read_to_neuromod = nn.Linear(self.d_mem, neuromod_dim)
        with torch.no_grad():
            nn.init.zeros_(self.read_to_neuromod.weight)
            self.read_to_neuromod.weight[:neuromod_dim, pfc_dim:pfc_dim + neuromod_dim].copy_(
                torch.eye(neuromod_dim) * 0.1
            )
            nn.init.zeros_(self.read_to_neuromod.bias)

        self._reward_ema = 0.0
        self._last_slot = -1  # for diagnostics

        if save_path and os.path.exists(save_path):
            self._load(save_path)

    def _make_key(self, activation_summary, pfc_state):
        """Project activation + PFC into key space (pattern separation)."""
        if pfc_state is not None:
            pfc = pfc_state.squeeze(0) if pfc_state.dim() > 1 else pfc_state
        else:
            pfc = torch.zeros(self.pfc_dim, device=activation_summary.device)
        combined = torch.cat([activation_summary, pfc])
        return self.key_proj(combined)

    def _match(self, key):
        """Find best matching prototype. Returns (slot_idx, similarities).

        Straight-through: forward uses argmax (discrete), backward uses
        softmax gradient (continuous). This lets prototype keys learn
        via gradient despite the discrete selection.
        """
        key_norm = F.normalize(key.unsqueeze(0), dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sims = (proto_norm @ key_norm.squeeze(0))  # (n_slots,)

        # Straight-through: argmax forward, softmax backward
        soft = F.softmax(sims * 10.0, dim=0)       # soft for gradient
        hard = torch.zeros_like(soft)
        hard[sims.argmax()] = 1.0                   # hard for forward
        one_hot = hard - soft.detach() + soft       # straight-through trick

        return sims.argmax().item(), sims, one_hot

    def retrieve_and_reinstate(self, mach, activation_summary, current_td_error,
                               top_k=3, device=None):
        """Match current state to best prototype, reinstate that slot.

        Discrete retrieval (one slot) with straight-through gradient.
        Returns reinstatement alpha for diagnostics.
        """
        if device is None:
            device = activation_summary.device
        if self.memory.device != device:
            self.to(device)

        pfc = mach._pfc_state if hasattr(mach, '_pfc_state') else None

        # Pattern separation + matching
        key = self._make_key(activation_summary, pfc)
        slot_idx, sims, one_hot = self._match(key)
        self._last_slot = slot_idx

        # Read from matched slot (straight-through one-hot)
        # Clone memory so store()'s in-place writes don't break autograd
        read_out = one_hot @ self.memory.clone()  # gradient flows through one_hot

        # Read gate: approach/avoidance
        best_sim = sims[slot_idx]
        gate_input = torch.stack([
            best_sim,
            torch.tensor(abs(current_td_error), device=device, dtype=torch.float32),
            torch.tensor(self._reward_ema, device=device, dtype=torch.float32),
        ])
        alpha = self.read_gate(gate_input).squeeze()

        # Reinstate PFC
        pfc_delta = self.read_to_pfc(read_out)
        if pfc is not None:
            mach._pfc_state = mach._pfc_state + alpha * pfc_delta.unsqueeze(0)

        # Neuromod bias
        alpha_f = alpha.item()
        if abs(alpha_f) > 1e-4 and hasattr(mach, '_eta_state'):
            neuromod_raw = self.read_to_neuromod(read_out)
            nm = neuromod_raw.view(3, self.n_patches)
            mach._neuromod_bias = {
                'eta': nm[0].clamp(0.1, 1.0),
                'decay': nm[1].clamp(0.1, 1.0),
                'expl': nm[2].clamp(0.1, 0.5),
                'alpha': alpha_f,
            }

        return abs(alpha_f)

    def store(self, mach, activation_summary, reward, td_error):
        """Write current state to the best-matching slot.

        Discrete write: only the matched slot is updated.
        Uses exponential moving average so the slot content tracks
        the running state for this "type" of situation.
        """
        device = activation_summary.device
        if self.memory.device != device:
            self.to(device)

        self._reward_ema = 0.99 * self._reward_ema + 0.01 * reward

        with torch.no_grad():
            pfc = mach._pfc_state if hasattr(mach, '_pfc_state') else None
            pfc_flat = pfc.squeeze(0) if pfc is not None else torch.zeros(self.pfc_dim, device=device)

            key = self._make_key(activation_summary, pfc_flat if pfc is None else pfc)
            slot_idx, _, _ = self._match(key)

            # Compose value: PFC + neuromod
            etas = torch.zeros(self.n_patches, device=device)
            decays = torch.zeros(self.n_patches, device=device)
            expls = torch.zeros(self.n_patches, device=device)
            if hasattr(mach, '_last_etas') and mach._last_etas is not None:
                etas = mach._last_etas
                decays = mach._last_decays
                expls = mach._last_expls
            value = torch.cat([pfc_flat, etas, decays, expls])

            # EMA update: slot tracks running average for this mode
            count = self.write_count[slot_idx]
            # Momentum: starts at 1.0 (first write overwrites), decays to 0.1
            momentum = max(0.1, 1.0 / (1.0 + count.item()))
            self.memory[slot_idx] = (1 - momentum) * self.memory[slot_idx] + momentum * value
            self.write_count[slot_idx] += 1

        return True

    def set_neuromod(self, gamma, avg_decay):
        """Interface compatibility. No-op for discrete slots."""
        pass

    def reconsolidate(self, td_error):
        """Interface compatibility. No-op — slots are updated via EMA in store()."""
        pass

    def decay_all(self):
        """No-op. Discrete slots don't decay — they're overwritten by EMA.
        Unused slots naturally get stale but that's fine.
        """
        pass

    def save(self, path=None):
        """Persist memory state to disk."""
        path = path or self.save_path
        if path is None:
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'memory': self.memory.cpu(),
            'write_count': self.write_count.cpu(),
        }, path)

    def _load(self, path):
        if not os.path.exists(path):
            return
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['memory'].shape == self.memory.shape:
            self.memory.copy_(data['memory'])
            self.write_count.copy_(data['write_count'])

    def __len__(self):
        """Number of slots that have been written to."""
        return int((self.write_count > 0).sum().item())

    def __repr__(self):
        return f"Hippocampus({len(self)}/{self.n_slots} slots, d_mem={self.d_mem})"
