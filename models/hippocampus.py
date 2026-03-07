"""
Hippocampus: episodic memory that stores and reinstates neural states.

Real hippocampus circuit:
  Cortex → Entorhinal → Dentate Gyrus (pattern separation) →
  CA3 (pattern completion, recurrent attractors) →
  CA1 (novelty detection, comparison) → back to Cortex

Key insight: the hippocampus stores and retrieves NEURAL STATES,
not text. Retrieval reinstates the cortical activation pattern —
the patch deltas, PFC state, neuromod context that worked before.

Reinstatement is PARTIAL (blended with current state), not complete.
Memories are reconstructive — they bias current processing, not replace it.
Stronger matches reinstate more strongly.

Brain mapping:
- Encoding: patch_deltas + pfc_state + neuromod → compressed episode (entorhinal → hippocampus)
- Pattern separation: DG-like projection makes similar inputs distinct
- Storage: content-addressable (CA3 pattern completion)
- Retrieval: similarity search → partial reinstatement of stored neural state
- Novelty detection: CA1 comparator (is this new? → store it)
- Consolidation: replay → gradient updates (hippocampus → cortex during sleep)
- Forgetting: decay without reinforcement
"""

import os
import torch
import torch.nn as nn
import numpy as np


class Hippocampus(nn.Module):
    """Episodic memory that stores and reinstates neural states.

    Stores: (key_embedding, patch_deltas, pfc_state, neuromod_values, reward)
    Retrieves: blends stored state into current state (partial reinstatement)
    """

    def __init__(self, key_dim, n_patches, d_model, patch_hidden_dim,
                 attn_hidden_dim, pfc_dim=32, max_memories=500,
                 surprise_threshold=0.3, decay_rate=0.999,
                 save_path=None):
        super().__init__()
        self.key_dim = key_dim
        self.n_patches = n_patches
        self.d_model = d_model
        self.patch_hidden_dim = patch_hidden_dim
        self.attn_hidden_dim = attn_hidden_dim
        self.pfc_dim = pfc_dim
        self.max_memories = max_memories
        self.surprise_threshold = surprise_threshold
        self.decay_rate = decay_rate
        self.save_path = save_path

        # Pattern separation: key_dim → key_dim (like dentate gyrus)
        # Orthogonalizes similar inputs so add/sub don't collide
        self.pattern_sep = nn.Sequential(
            nn.Linear(key_dim, key_dim * 2),
            nn.ReLU(),  # sparse activation (DG has very sparse firing)
            nn.Linear(key_dim * 2, key_dim),
        )
        # Init to near-identity so it works before training
        with torch.no_grad():
            nn.init.eye_(self.pattern_sep[0].weight[:key_dim])
            nn.init.zeros_(self.pattern_sep[0].weight[key_dim:])
            nn.init.zeros_(self.pattern_sep[0].bias)
            nn.init.eye_(self.pattern_sep[2].weight[:, :key_dim])
            nn.init.zeros_(self.pattern_sep[2].weight[:, key_dim:])
            nn.init.zeros_(self.pattern_sep[2].bias)

        # Reinstatement gate: controls blend strength based on similarity
        # Input: similarity scalar + current td_error + stored reward
        self.reinstatement_gate = nn.Sequential(
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        # Init conservative — low reinstatement by default
        with torch.no_grad():
            self.reinstatement_gate[-2].bias.fill_(-1.0)

        # Storage (not nn.Parameters — these are memory contents, not weights)
        self._keys = []          # list of numpy arrays (key_dim,)
        self._strengths = []     # list of floats (decay over time)
        self._rewards = []       # list of floats (reward at storage time)
        self._td_errors = []     # list of floats (surprise at storage)

        # Stored neural states (numpy for disk persistence)
        self._patch_deltas_down = []   # list of list of numpy (hidden, d_model)
        self._patch_deltas_up = []     # list of list of numpy (d_model, hidden)
        self._attn_deltas_down = []    # list of list of numpy (attn_hidden, d_model)
        self._attn_deltas_up = []      # list of list of numpy (d_model, attn_hidden)
        self._pfc_states = []          # list of numpy (pfc_dim,)
        self._neuromod_values = []     # list of dict {eta, decay, expl}

        if save_path and os.path.exists(save_path):
            self._load(save_path)

    def encode_key(self, activation_summary):
        """Entorhinal → DG: encode activation summary into memory key.

        Pattern separation makes similar inputs more distinct.
        """
        return self.pattern_sep(activation_summary)

    def store(self, mach, activation_summary, reward, td_error):
        """Store current neural state if surprising enough (CA1 novelty detection).

        Args:
            mach: MACHActivationHebbian instance (we read its state)
            activation_summary: compressed activations (key_dim,)
            reward: scalar reward received
            td_error: scalar TD error (surprise)
        """
        if abs(td_error) < self.surprise_threshold:
            return False

        # Pattern-separated key
        with torch.no_grad():
            key = self.encode_key(activation_summary)
            key_np = key.cpu().numpy().astype(np.float32)

        # Check for near-duplicate (CA3 — don't store if already known)
        if self._keys:
            sims = self._cosine_similarities(key_np)
            if sims.max() > 0.9:
                # Reinforce existing memory instead of duplicating
                idx = int(sims.argmax())
                self._strengths[idx] = min(self._strengths[idx] + 0.1, 1.0)
                return False

        # Extract neural state to store
        patch_dd = []
        patch_du = []
        attn_dd = []
        attn_du = []
        for patch in mach.patches:
            dd = patch.delta_down.detach().cpu().numpy().astype(np.float16) if patch.delta_down is not None else None
            du = patch.delta_up.detach().cpu().numpy().astype(np.float16) if patch.delta_up is not None else None
            patch_dd.append(dd)
            patch_du.append(du)
        for patch in mach.attn_patches:
            dd = patch.delta_down.detach().cpu().numpy().astype(np.float16) if patch.delta_down is not None else None
            du = patch.delta_up.detach().cpu().numpy().astype(np.float16) if patch.delta_up is not None else None
            attn_dd.append(dd)
            attn_du.append(du)

        pfc = mach._pfc_state.detach().squeeze(0).cpu().numpy().astype(np.float32)

        neuromod = {}
        if hasattr(mach, '_last_etas'):
            neuromod['eta'] = mach._last_etas.cpu().numpy().astype(np.float32).tolist()
            neuromod['decay'] = mach._last_decays.cpu().numpy().astype(np.float32).tolist()
            neuromod['expl'] = mach._last_expls.cpu().numpy().astype(np.float32).tolist()

        # Store
        self._keys.append(key_np)
        self._strengths.append(1.0)
        self._rewards.append(float(reward))
        self._td_errors.append(float(abs(td_error)))
        self._patch_deltas_down.append(patch_dd)
        self._patch_deltas_up.append(patch_du)
        self._attn_deltas_down.append(attn_dd)
        self._attn_deltas_up.append(attn_du)
        self._pfc_states.append(pfc)
        self._neuromod_values.append(neuromod)

        # Evict weakest if over capacity
        if len(self._keys) > self.max_memories:
            weakest = int(np.argmin(self._strengths))
            self._evict(weakest)

        return True

    def retrieve_and_reinstate(self, mach, activation_summary, current_td_error,
                               top_k=3, device=None):
        """Retrieve similar memories and partially reinstate their neural states.

        Partial reinstatement: blend stored state into current state.
        Blend strength depends on similarity, surprise, and learned gate.

        Args:
            mach: MACHActivationHebbian instance (we modify its state)
            activation_summary: current compressed activations
            current_td_error: how surprised we are right now
            top_k: number of memories to consider
            device: torch device

        Returns:
            float: max reinstatement alpha used (0 = no reinstatement)
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

        for idx in top_indices:
            sim = float(sims[idx])
            if sim < 0.3:  # too dissimilar, skip
                continue

            # Reinforce retrieved memory (retrieval strengthens memory)
            self._strengths[idx] = min(self._strengths[idx] + 0.05, 1.0)

            # Compute reinstatement strength via learned gate
            # Input: similarity, current surprise, stored reward
            gate_input = torch.tensor([
                sim,
                abs(current_td_error),
                self._rewards[idx],
            ], device=device, dtype=torch.float32)
            alpha = self.reinstatement_gate(gate_input).item()
            alpha = alpha * sim  # scale by similarity (very different → almost no reinstatement)

            if alpha < 0.01:
                continue

            max_alpha = max(max_alpha, alpha)

            # Partial reinstatement: blend stored neural state into current
            # patch_deltas = (1 - alpha) * current + alpha * stored
            for i, patch in enumerate(mach.patches):
                stored_dd = self._patch_deltas_down[idx][i]
                stored_du = self._patch_deltas_up[idx][i]
                if stored_dd is not None and patch.delta_down is not None:
                    stored_t = torch.from_numpy(stored_dd.astype(np.float32)).to(device)
                    patch.delta_down = (1 - alpha) * patch.delta_down + alpha * stored_t
                if stored_du is not None and patch.delta_up is not None:
                    stored_t = torch.from_numpy(stored_du.astype(np.float32)).to(device)
                    patch.delta_up = (1 - alpha) * patch.delta_up + alpha * stored_t

            for i, patch in enumerate(mach.attn_patches):
                stored_dd = self._attn_deltas_down[idx][i]
                stored_du = self._attn_deltas_up[idx][i]
                if stored_dd is not None and patch.delta_down is not None:
                    stored_t = torch.from_numpy(stored_dd.astype(np.float32)).to(device)
                    patch.delta_down = (1 - alpha) * patch.delta_down + alpha * stored_t
                if stored_du is not None and patch.delta_up is not None:
                    stored_t = torch.from_numpy(stored_du.astype(np.float32)).to(device)
                    patch.delta_up = (1 - alpha) * patch.delta_up + alpha * stored_t

            # Blend PFC state
            stored_pfc = torch.from_numpy(self._pfc_states[idx]).to(device).unsqueeze(0)
            mach._pfc_state = (1 - alpha) * mach._pfc_state + alpha * stored_pfc

        return max_alpha

    def decay_all(self):
        """Apply passive decay. Unretrieved memories fade."""
        alive = []
        for i in range(len(self._strengths)):
            self._strengths[i] *= self.decay_rate
            if self._strengths[i] >= 0.01:
                alive.append(i)

        if len(alive) < len(self._strengths):
            self._filter_indices(alive)

    def get_replay_batch(self, batch_size=10):
        """Sample stored neural states for sleep replay, weighted by strength.

        Returns list of dicts with full neural state for each memory.
        """
        if not self._keys:
            return []

        strengths = np.array(self._strengths)
        probs = strengths / strengths.sum()
        k = min(batch_size, len(self._keys))
        indices = np.random.choice(len(self._keys), size=k, replace=False, p=probs)

        results = []
        for idx in indices:
            results.append({
                'patch_deltas_down': self._patch_deltas_down[idx],
                'patch_deltas_up': self._patch_deltas_up[idx],
                'attn_deltas_down': self._attn_deltas_down[idx],
                'attn_deltas_up': self._attn_deltas_up[idx],
                'pfc_state': self._pfc_states[idx],
                'neuromod': self._neuromod_values[idx],
                'reward': self._rewards[idx],
                'td_error': self._td_errors[idx],
            })
        return results

    def _cosine_similarities(self, query):
        if not self._keys:
            return np.array([])
        matrix = np.stack(self._keys)
        q_norm = query / (np.linalg.norm(query) + 1e-8)
        m_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return m_norms @ q_norm

    def _evict(self, idx):
        for lst in (self._keys, self._strengths, self._rewards, self._td_errors,
                    self._patch_deltas_down, self._patch_deltas_up,
                    self._attn_deltas_down, self._attn_deltas_up,
                    self._pfc_states, self._neuromod_values):
            lst.pop(idx)

    def _filter_indices(self, alive):
        for attr in ('_keys', '_strengths', '_rewards', '_td_errors',
                     '_patch_deltas_down', '_patch_deltas_up',
                     '_attn_deltas_down', '_attn_deltas_up',
                     '_pfc_states', '_neuromod_values'):
            lst = getattr(self, attr)
            setattr(self, attr, [lst[i] for i in alive])

    def save(self, path=None):
        """Persist memories to disk."""
        path = path or self.save_path
        if path is None:
            return

        data = {
            'keys': [k.tolist() for k in self._keys],
            'strengths': self._strengths,
            'rewards': self._rewards,
            'td_errors': self._td_errors,
            'patch_deltas_down': [
                [d.tolist() if d is not None else None for d in pdd]
                for pdd in self._patch_deltas_down
            ],
            'patch_deltas_up': [
                [d.tolist() if d is not None else None for d in pdu]
                for pdu in self._patch_deltas_up
            ],
            'attn_deltas_down': [
                [d.tolist() if d is not None else None for d in add]
                for add in self._attn_deltas_down
            ],
            'attn_deltas_up': [
                [d.tolist() if d is not None else None for d in adu]
                for adu in self._attn_deltas_up
            ],
            'pfc_states': [p.tolist() for p in self._pfc_states],
            'neuromod_values': self._neuromod_values,
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
        self._td_errors = data['td_errors']
        self._patch_deltas_down = [
            [np.array(d, dtype=np.float16) if d is not None else None for d in pdd]
            for pdd in data['patch_deltas_down']
        ]
        self._patch_deltas_up = [
            [np.array(d, dtype=np.float16) if d is not None else None for d in pdu]
            for pdu in data['patch_deltas_up']
        ]
        self._attn_deltas_down = [
            [np.array(d, dtype=np.float16) if d is not None else None for d in add]
            for add in data['attn_deltas_down']
        ]
        self._attn_deltas_up = [
            [np.array(d, dtype=np.float16) if d is not None else None for d in adu]
            for adu in data['attn_deltas_up']
        ]
        self._pfc_states = [np.array(p, dtype=np.float32) for p in data['pfc_states']]
        self._neuromod_values = data['neuromod_values']

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return f"Hippocampus({len(self)} memories, key_dim={self.key_dim})"
