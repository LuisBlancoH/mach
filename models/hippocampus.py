"""
Hippocampus: two-tier memory with learned mode prototypes + episodic buffers.

Tier 1 — VQ Codebook (basal ganglia / thalamus):
  16 learned prototype keys. Discrete matching via straight-through estimator.
  Each prototype learns to represent one "mode" (like thalamic nuclei).
  The EMA slot content tracks the running average for that mode.

Tier 2 — Per-slot episodic buffer (hippocampus proper):
  Each slot has a ring buffer of the last 8 specific experiences.
  Stores raw (PFC, neuromod, reward) snapshots — real episodic memory.
  On retrieval: within the matched slot, find the episode with the
  most similar PFC state (argmax over 8, not 500).

This mirrors the brain:
  - Prefrontal/thalamic context gates hippocampal retrieval
  - You don't search ALL memories — context narrows the search space
  - Complementary learning: fast episodic (buffer) + slow cortical (EMA)

No hardcoded thresholds. No storage gates. No similarity thresholds.

Interface is kept compatible with training/eval loops.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Hippocampus(nn.Module):
    """Two-tier memory: VQ codebook (mode switching) + episodic buffers.

    Tier 1: N_SLOTS learned prototypes select a mode (straight-through).
    Tier 2: per-slot ring buffer of recent episodes for specific recall.
    """

    def __init__(self, key_dim, pfc_dim=32, n_patches=4, max_memories=16,
                 save_path=None):
        super().__init__()
        self.key_dim = key_dim
        self.pfc_dim = pfc_dim
        self.n_patches = n_patches
        self.n_slots = max_memories
        self.episodes_per_slot = 8
        self.d_mem = pfc_dim + n_patches * 3  # PFC + neuromod per entry
        self.save_path = save_path

        # === Tier 1: VQ Codebook ===

        # Learned prototype keys
        self.prototypes = nn.Parameter(torch.randn(self.n_slots, key_dim) * 0.1)

        # Learned softmax temperature for matching sharpness
        # log-space so it's always positive; init at ln(10) ≈ 2.3
        self.log_temperature = nn.Parameter(torch.tensor(2.302585))

        # EMA momentum driven by neuromod (set externally each step)
        # Default: moderate momentum. Decay nucleus overrides via set_neuromod().
        self._ema_momentum = 0.1

        # Key projection with pattern separation (Dentate Gyrus)
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

        # EMA slot contents (slow / cortical)
        self.register_buffer('memory', torch.zeros(self.n_slots, self.d_mem))
        self.register_buffer('write_count', torch.zeros(self.n_slots))

        # === Tier 2: Per-slot episodic buffers ===

        # Ring buffer: (n_slots, episodes_per_slot, d_mem)
        self.register_buffer('episodes',
            torch.zeros(self.n_slots, self.episodes_per_slot, self.d_mem))
        # Per-slot write head position (circular index)
        self.register_buffer('ep_head',
            torch.zeros(self.n_slots, dtype=torch.long))
        # Per-slot episode count (for knowing if buffer is full)
        self.register_buffer('ep_count',
            torch.zeros(self.n_slots, dtype=torch.long))
        # Per-episode reward (for valence-weighted retrieval)
        self.register_buffer('ep_rewards',
            torch.zeros(self.n_slots, self.episodes_per_slot))

        # === Read head ===

        # Read gate: controls reinstatement strength
        # Input: match_similarity + td_error + reward_ema
        # Output: [-1, 1] — approach/avoidance
        self.read_gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        with torch.no_grad():
            self.read_gate[-2].bias.fill_(0.1)

        # Reinstatement projections
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

        # Learned blend between EMA (tier 1) and best episode (tier 2)
        # Input: ema_sim + episode_sim + td_error
        # Output: [0, 1] — 0 = use EMA only, 1 = use episode only
        self.blend_gate = nn.Sequential(
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            self.blend_gate[-2].bias.fill_(0.0)  # start at 0.5 blend

        self._reward_ema = 0.0
        self._last_slot = -1
        self._last_ep_idx = -1

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
        """Find best matching prototype. Straight-through gradient."""
        key_norm = F.normalize(key.unsqueeze(0), dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sims = (proto_norm @ key_norm.squeeze(0))  # (n_slots,)

        temperature = self.log_temperature.exp()
        soft = F.softmax(sims * temperature, dim=0)
        hard = torch.zeros_like(soft)
        hard[sims.argmax()] = 1.0
        one_hot = hard - soft.detach() + soft

        return sims.argmax().item(), sims, one_hot

    def _best_episode(self, slot_idx, pfc_state, device):
        """Find most relevant episode within a slot by PFC similarity.

        No threshold — just argmax over the buffer. Always returns something.
        Returns (episode_content, similarity, episode_index).
        """
        n_eps = min(self.ep_count[slot_idx].item(), self.episodes_per_slot)
        if n_eps == 0:
            return None, 0.0, -1

        slot_eps = self.episodes[slot_idx, :n_eps].clone()  # (n_eps, d_mem)
        # Compare PFC portion of episodes to current PFC
        pfc_flat = pfc_state.squeeze(0) if pfc_state.dim() > 1 else pfc_state
        stored_pfcs = slot_eps[:, :self.pfc_dim]  # (n_eps, pfc_dim)

        pfc_norm = F.normalize(pfc_flat.unsqueeze(0), dim=-1)
        stored_norm = F.normalize(stored_pfcs, dim=-1)
        sims = (stored_norm @ pfc_norm.squeeze(0))  # (n_eps,)

        best_idx = sims.argmax().item()
        return slot_eps[best_idx], sims[best_idx].item(), best_idx

    def retrieve_and_reinstate(self, mach, activation_summary, current_td_error,
                               top_k=3, device=None):
        """Two-tier retrieval:
        1. VQ match → select slot (mode)
        2. Within slot: blend EMA (slow) + best episode (fast)
        3. Reinstate into PFC + neuromod

        Returns reinstatement alpha for diagnostics.
        """
        if device is None:
            device = activation_summary.device
        if self.memory.device != device:
            self.to(device)

        pfc = mach._pfc_state if hasattr(mach, '_pfc_state') else None

        # Tier 1: VQ matching
        key = self._make_key(activation_summary, pfc)
        slot_idx, sims, one_hot = self._match(key)
        self._last_slot = slot_idx

        # Tier 1 read: EMA content (straight-through)
        ema_out = one_hot @ self.memory.clone()  # (d_mem,)

        # Tier 2 read: best episode within slot
        ep_content, ep_sim, ep_idx = self._best_episode(slot_idx, pfc, device)
        self._last_ep_idx = ep_idx

        # Blend EMA + episode (learned gate)
        if ep_content is not None:
            blend_input = torch.stack([
                sims[slot_idx],
                torch.tensor(ep_sim, device=device, dtype=torch.float32),
                torch.tensor(abs(current_td_error), device=device, dtype=torch.float32),
            ])
            ep_weight = self.blend_gate(blend_input).squeeze()  # [0, 1]
            read_out = (1 - ep_weight) * ema_out + ep_weight * ep_content
        else:
            read_out = ema_out

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
        """Two-tier write:
        1. Update EMA slot (slow cortical learning)
        2. Append to episodic ring buffer (fast hippocampal encoding)
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

            # Tier 1: EMA update (slow)
            # First write overwrites; after that, momentum from neuromod
            count = self.write_count[slot_idx]
            momentum = 1.0 if count.item() == 0 else self._ema_momentum
            self.memory[slot_idx] = (1 - momentum) * self.memory[slot_idx] + momentum * value
            self.write_count[slot_idx] += 1

            # Tier 2: episodic ring buffer (fast, always stores)
            head = self.ep_head[slot_idx].item()
            self.episodes[slot_idx, head] = value
            self.ep_rewards[slot_idx, head] = reward
            self.ep_head[slot_idx] = (head + 1) % self.episodes_per_slot
            self.ep_count[slot_idx] += 1

        return True

    def set_neuromod(self, gamma, avg_decay):
        """Neuromod drives EMA momentum for slot updates.

        High decay (stable patches) → low momentum (stable memory, slow update)
        Low decay (volatile patches) → high momentum (fast adaptation)
        avg_decay ∈ [0.1, 1.0] → momentum ∈ [0.5, 0.05]
        """
        self._ema_momentum = 0.5 - 0.45 * avg_decay

    def reconsolidate(self, td_error):
        """Interface compatibility."""
        pass

    def decay_all(self):
        """No-op. EMA slots are overwritten, episodes cycle via ring buffer."""
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
            'episodes': self.episodes.cpu(),
            'ep_head': self.ep_head.cpu(),
            'ep_count': self.ep_count.cpu(),
            'ep_rewards': self.ep_rewards.cpu(),
        }, path)

    def _load(self, path):
        if not os.path.exists(path):
            return
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['memory'].shape == self.memory.shape:
            self.memory.copy_(data['memory'])
            self.write_count.copy_(data['write_count'])
            self.episodes.copy_(data['episodes'])
            self.ep_head.copy_(data['ep_head'])
            self.ep_count.copy_(data['ep_count'])
            self.ep_rewards.copy_(data['ep_rewards'])

    def __len__(self):
        """Number of slots that have been written to."""
        return int((self.write_count > 0).sum().item())

    def total_episodes(self):
        """Total episodes stored across all slots."""
        return int(self.ep_count.clamp(max=self.episodes_per_slot).sum().item())

    def __repr__(self):
        n_eps = self.total_episodes()
        return f"Hippocampus({len(self)}/{self.n_slots} slots, {n_eps} episodes)"
