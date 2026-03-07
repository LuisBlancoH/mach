"""
Hippocampus: compressed episodic memory with neural state reinstatement.

Real hippocampus circuit:
  Cortex → Entorhinal (compression) → Dentate Gyrus (pattern separation) →
  CA3 (pattern completion, recurrent attractors) →
  CA1 (novelty/comparison) → back to Cortex (reinstatement)

Key design decisions (brain-faithful):
- Stores COMPRESSED state: PFC state (32d) + neuromod values (12d),
  NOT raw patch deltas. The hippocampus stores the index, cortex
  reconstructs the details via plasticity.
- No storage threshold: everything gets encoded, with initial strength
  proportional to |TD error|. Weak memories decay away naturally.
- Reconsolidation: retrieved memories are updated by subsequent TD error.
  Positive surprise strengthens, negative surprise weakens. No threshold.
- Sequential structure: PFC GRU state drifts slowly, so memories from
  the same task block naturally cluster. Temporal context is implicit.
- Pattern separation (DG): orthogonalizes similar keys so add/sub
  don't collide.
- Reinstatement is PARTIAL: blend stored PFC + neuromod into current
  state. The system reconstructs appropriate patches from the bias.

Storage per memory: ~170 floats = 680 bytes.
500 memories = 340KB (vs 6.5GB for raw patch deltas).
"""

import os
import torch
import torch.nn as nn
import numpy as np


class Hippocampus(nn.Module):
    """Compressed episodic memory with neural state reinstatement.

    Stores: (key, pfc_state, neuromod_values, reward)
    Retrieves: blends stored PFC + neuromod into current state
    Cortex (plasticity system) reconstructs patch deltas from the bias.
    """

    def __init__(self, key_dim, pfc_dim=32, n_patches=4, max_memories=500,
                 decay_rate=0.999, save_path=None):
        super().__init__()
        self.key_dim = key_dim
        self.pfc_dim = pfc_dim
        self.n_patches = n_patches
        self.max_memories = max_memories
        self.decay_rate = decay_rate
        self.save_path = save_path

        # Pattern separation: key_dim → key_dim (like dentate gyrus)
        # Sparse ReLU = DG's extremely sparse firing (~2% active)
        # Orthogonalizes similar inputs so add/sub get distinct codes
        self.pattern_sep = nn.Sequential(
            nn.Linear(key_dim, key_dim * 2),
            nn.ReLU(),
            nn.Linear(key_dim * 2, key_dim),
        )
        # Init near-identity so it works before training
        with torch.no_grad():
            nn.init.eye_(self.pattern_sep[0].weight[:key_dim])
            nn.init.zeros_(self.pattern_sep[0].weight[key_dim:])
            nn.init.zeros_(self.pattern_sep[0].bias)
            nn.init.eye_(self.pattern_sep[2].weight[:, :key_dim])
            nn.init.zeros_(self.pattern_sep[2].weight[:, key_dim:])
            nn.init.zeros_(self.pattern_sep[2].bias)

        # Reinstatement gate: learned blend strength
        # Input: similarity + |current_td_error| + stored_reward
        self.reinstatement_gate = nn.Sequential(
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        # Init conservative — low reinstatement by default
        with torch.no_grad():
            self.reinstatement_gate[-2].bias.fill_(-1.0)

        # Storage: compressed neural states (~170 floats per memory)
        self._keys = []           # (key_dim,) — pattern-separated activation key
        self._strengths = []      # scalar — decays over time, updated by reconsolidation
        self._rewards = []        # scalar — reward at storage time
        self._pfc_states = []     # (pfc_dim,) — compressed task representation
        self._neuromod_etas = []  # (n_patches,) — per-patch learning rates
        self._neuromod_decays = []  # (n_patches,) — per-patch retention
        self._neuromod_expls = []   # (n_patches,) — per-patch exploration
        # Track which memories were retrieved this step (for reconsolidation)
        self._last_retrieved_indices = []

        if save_path and os.path.exists(save_path):
            self._load(save_path)

    def encode_key(self, activation_summary):
        """Entorhinal → DG: compress and pattern-separate."""
        return self.pattern_sep(activation_summary)

    def store(self, mach, activation_summary, reward, td_error):
        """Encode current state into memory. Strength proportional to surprise.

        No threshold — everything gets stored. Weak memories decay away.
        Near-duplicates reinforce existing memory instead of duplicating.
        """
        strength = abs(td_error)  # initial strength = how surprising
        if strength < 1e-6:
            strength = 1e-6  # floor so memory exists

        with torch.no_grad():
            key = self.encode_key(activation_summary)
            key_np = key.cpu().numpy().astype(np.float32)

        # Near-duplicate check (CA3 — don't store what you already know)
        if self._keys:
            sims = self._cosine_similarities(key_np)
            if sims.max() > 0.9:
                # Reinforce existing memory via reconsolidation
                idx = int(sims.argmax())
                self._strengths[idx] += strength
                return False

        # Extract compressed state
        pfc = mach._pfc_state.detach().squeeze(0).cpu().numpy().astype(np.float32)

        etas = np.zeros(self.n_patches, dtype=np.float32)
        decays = np.zeros(self.n_patches, dtype=np.float32)
        expls = np.zeros(self.n_patches, dtype=np.float32)
        if hasattr(mach, '_last_etas') and mach._last_etas is not None:
            etas = mach._last_etas.cpu().numpy().astype(np.float32)
            decays = mach._last_decays.cpu().numpy().astype(np.float32)
            expls = mach._last_expls.cpu().numpy().astype(np.float32)

        self._keys.append(key_np)
        self._strengths.append(float(strength))
        self._rewards.append(float(reward))
        self._pfc_states.append(pfc)
        self._neuromod_etas.append(etas)
        self._neuromod_decays.append(decays)
        self._neuromod_expls.append(expls)

        # Evict weakest if over capacity
        if len(self._keys) > self.max_memories:
            weakest = int(np.argmin(self._strengths))
            self._evict(weakest)

        return True

    def retrieve_and_reinstate(self, mach, activation_summary, current_td_error,
                               top_k=3, device=None):
        """Retrieve similar memories and partially reinstate their state.

        Reinstatement biases PFC state and neuromod values toward stored state.
        The plasticity system then reconstructs appropriate patches from the bias.
        Blend strength = learned gate × similarity (no arbitrary thresholds).

        Returns:
            float: max reinstatement alpha used (0 = no retrieval)
        """
        if not self._keys:
            return 0.0

        if device is None:
            device = activation_summary.device

        with torch.no_grad():
            key = self.encode_key(activation_summary)
            key_np = key.cpu().numpy().astype(np.float32)

        sims = self._cosine_similarities(key_np)
        weighted_sims = sims * np.array(self._strengths)

        k = min(top_k, len(self._keys))
        top_indices = np.argsort(weighted_sims)[-k:][::-1]

        max_alpha = 0.0
        self._last_retrieved_indices = []

        for idx in top_indices:
            sim = float(sims[idx])
            if sim < 0.1:
                continue

            self._last_retrieved_indices.append(int(idx))

            # Retrieval strengthens memory (use it or lose it)
            # Counteracts passive decay — frequently useful memories persist
            self._strengths[idx] += 0.01

            # Learned reinstatement gate
            gate_input = torch.tensor([
                sim,
                abs(current_td_error),
                self._rewards[idx],
            ], device=device, dtype=torch.float32)
            alpha = self.reinstatement_gate(gate_input).item()
            alpha = alpha * sim  # similarity-scaled

            if alpha < 0.001:
                continue

            max_alpha = max(max_alpha, alpha)

            # Partial reinstatement: blend stored PFC state into current
            stored_pfc = torch.from_numpy(self._pfc_states[idx]).to(device).unsqueeze(0)
            mach._pfc_state = (1 - alpha) * mach._pfc_state + alpha * stored_pfc

            # Bias neuromod nuclei toward stored values
            # We bias the GRU states that produce eta/decay/expl
            # This is indirect — the nuclei will drift from this bias,
            # but it gives them a starting point
            if hasattr(mach, '_eta_state') and len(self._neuromod_etas[idx]) > 0:
                # We can't directly set eta/decay/expl (they're computed from GRU states)
                # Instead, store a bias that gets applied at the next hebbian_step
                if not hasattr(mach, '_neuromod_bias'):
                    mach._neuromod_bias = None
                mach._neuromod_bias = {
                    'eta': torch.from_numpy(self._neuromod_etas[idx]).to(device),
                    'decay': torch.from_numpy(self._neuromod_decays[idx]).to(device),
                    'expl': torch.from_numpy(self._neuromod_expls[idx]).to(device),
                    'alpha': alpha,
                }

        return max_alpha

    def reconsolidate(self, td_error):
        """Update retrieved memories based on outcome (reconsolidation).

        Called after the step that used retrieved memories.
        Positive TD error (better than expected) → strengthen.
        Negative TD error (worse than expected) → weaken.
        No threshold — the TD error IS the update signal.
        """
        for idx in self._last_retrieved_indices:
            if idx < len(self._strengths):
                self._strengths[idx] += td_error
                # Clamp: strength can't go negative (memory dies at 0)
                if self._strengths[idx] < 0:
                    self._strengths[idx] = 0.0
        self._last_retrieved_indices = []

    def decay_all(self):
        """Passive decay. Unretrieved memories fade (use it or lose it)."""
        alive = []
        for i in range(len(self._strengths)):
            self._strengths[i] *= self.decay_rate
            if self._strengths[i] >= 0.01:
                alive.append(i)

        if len(alive) < len(self._strengths):
            self._filter_indices(alive)

    def _cosine_similarities(self, query):
        if not self._keys:
            return np.array([])
        matrix = np.stack(self._keys)
        q_norm = query / (np.linalg.norm(query) + 1e-8)
        m_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return m_norms @ q_norm

    def _evict(self, idx):
        for lst in (self._keys, self._strengths, self._rewards,
                    self._pfc_states, self._neuromod_etas,
                    self._neuromod_decays, self._neuromod_expls):
            lst.pop(idx)

    def _filter_indices(self, alive):
        for attr in ('_keys', '_strengths', '_rewards',
                     '_pfc_states', '_neuromod_etas',
                     '_neuromod_decays', '_neuromod_expls'):
            lst = getattr(self, attr)
            setattr(self, attr, [lst[i] for i in alive])

    def save(self, path=None):
        """Persist to disk."""
        path = path or self.save_path
        if path is None:
            return
        data = {
            'keys': [k.tolist() for k in self._keys],
            'strengths': self._strengths,
            'rewards': self._rewards,
            'pfc_states': [p.tolist() for p in self._pfc_states],
            'neuromod_etas': [e.tolist() for e in self._neuromod_etas],
            'neuromod_decays': [d.tolist() for d in self._neuromod_decays],
            'neuromod_expls': [x.tolist() for x in self._neuromod_expls],
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(data, path)

    def _load(self, path):
        if not os.path.exists(path):
            return
        data = torch.load(path, map_location='cpu', weights_only=False)
        self._keys = [np.array(k, dtype=np.float32) for k in data['keys']]
        self._strengths = data['strengths']
        self._rewards = data['rewards']
        self._pfc_states = [np.array(p, dtype=np.float32) for p in data['pfc_states']]
        self._neuromod_etas = [np.array(e, dtype=np.float32) for e in data['neuromod_etas']]
        self._neuromod_decays = [np.array(d, dtype=np.float32) for d in data['neuromod_decays']]
        self._neuromod_expls = [np.array(x, dtype=np.float32) for x in data['neuromod_expls']]

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return f"Hippocampus({len(self)} memories, key_dim={self.key_dim})"
