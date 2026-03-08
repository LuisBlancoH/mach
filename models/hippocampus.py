"""
Hippocampus: VQ mode selection + per-slot episodic memory + replay.

VQ Codebook (thalamus/basal ganglia):
  16 learned prototype keys. Discrete matching via straight-through.
  Each prototype learns to represent one "mode" of operation.

Per-slot episodic buffer (hippocampus proper):
  Each slot has a buffer of 8 experiences with TD-aware eviction.
  Stores raw (PFC, neuromod, TD error, timestamp) snapshots.
  Also stores compressed per-layer activations for NREM replay.
  On retrieval: learned scorer combines similarity, salience, recency.

Replay (memory consolidation):
  NREM replay: reactivate stored compressed activations → drive Hebbian
    updates without running Qwen. Like sharp-wave ripples during sleep.
    Prioritized by |TD error| — surprising memories replayed more.
  REM replay (dreaming): run Qwen forward with generated prompts while
    replayed PFC/neuromod influence patches. For generalization testing.

This mirrors the brain:
  - Prefrontal/thalamic context gates hippocampal retrieval
  - Hippocampus stores specific episodes, not averages
  - Dopamine (TD error) modulates memory salience
  - Surprising memories persist, mundane ones get overwritten
  - Temporal context: recency signal via timestamps
  - NREM: compressed replay → cortical consolidation
  - REM: generative replay → novel recombinations
  - Cortex does the slow learning (Hebbian patches handle this)

No hardcoded thresholds. Retrieval priority is learned, not formula'd.

Interface is kept compatible with training/eval loops.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Hippocampus(nn.Module):
    """VQ mode selection + per-slot episodic buffers.

    VQ codebook selects a mode (straight-through gradient).
    Per-slot ring buffer provides specific episodic recall.
    Cortex (Hebbian patches) handles slow learning — no EMA needed.
    """

    def __init__(self, key_dim, pfc_dim=32, n_patches=4, max_memories=16,
                 d_proj=32, save_path=None):
        super().__init__()
        self.key_dim = key_dim
        self.pfc_dim = pfc_dim
        self.n_patches = n_patches
        self.n_slots = max_memories
        self.episodes_per_slot = 8
        self.d_proj = d_proj
        self.d_mem = pfc_dim + n_patches * 3  # PFC + neuromod per entry
        # Per-layer compressed activations: pre + post per patch
        self.d_act = d_proj * n_patches * 2  # pre(d_proj) + post(d_proj) per layer
        self.save_path = save_path

        # === VQ Codebook (mode selection) ===

        # Learned prototype keys
        self.prototypes = nn.Parameter(torch.randn(self.n_slots, key_dim) * 0.1)

        # Learned softmax temperature for matching sharpness
        self.log_temperature = nn.Parameter(torch.tensor(2.302585))  # init ~10

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

        # === Per-slot episodic buffers ===

        # Ring buffer: (n_slots, episodes_per_slot, d_mem)
        self.register_buffer('episodes',
            torch.zeros(self.n_slots, self.episodes_per_slot, self.d_mem))
        self.register_buffer('ep_head',
            torch.zeros(self.n_slots, dtype=torch.long))
        self.register_buffer('ep_count',
            torch.zeros(self.n_slots, dtype=torch.long))
        # Per-episode TD error (surprise/salience signal)
        self.register_buffer('ep_td_errors',
            torch.zeros(self.n_slots, self.episodes_per_slot))
        # Per-episode timestamp (global step when stored)
        self.register_buffer('ep_timestamps',
            torch.zeros(self.n_slots, self.episodes_per_slot))
        # Per-episode compressed activations for NREM replay
        # Layout: [pre_0, post_0, pre_1, post_1, ...] each d_proj
        self.register_buffer('ep_activations',
            torch.zeros(self.n_slots, self.episodes_per_slot, self.d_act))
        self._global_step = 0

        # === Read head ===

        # Episode scorer: learns how to combine similarity, salience, recency
        # Input: [pfc_similarity, |td_error|, normalized_recency]
        # Output: scalar relevance score per episode
        self.episode_scorer = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        # Init: default to similarity × salience (ignore recency until learned)
        with torch.no_grad():
            nn.init.zeros_(self.episode_scorer[0].weight)
            self.episode_scorer[0].weight[0, 0] = 1.0  # sim
            self.episode_scorer[0].weight[1, 1] = 1.0  # |td|
            nn.init.zeros_(self.episode_scorer[0].bias)
            nn.init.ones_(self.episode_scorer[2].weight)
            nn.init.zeros_(self.episode_scorer[2].bias)

        # Read gate: approach/avoidance
        # Input: match_similarity + current_td_error + episode_td_error
        # Output: [-1, 1]
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
        """Find most relevant episode within a slot.

        Learned scorer combines [similarity, |TD error|, recency].
        Network decides how to weight these — no hardcoded formula.
        Returns (episode_content, episode_td_error, episode_index).
        """
        n_eps = min(self.ep_count[slot_idx].item(), self.episodes_per_slot)
        if n_eps == 0:
            return None, 0.0, -1

        slot_eps = self.episodes[slot_idx, :n_eps].clone()  # (n_eps, d_mem)
        pfc_flat = pfc_state.squeeze(0) if pfc_state.dim() > 1 else pfc_state
        stored_pfcs = slot_eps[:, :self.pfc_dim]  # (n_eps, pfc_dim)

        pfc_norm = F.normalize(pfc_flat.unsqueeze(0), dim=-1)
        stored_norm = F.normalize(stored_pfcs, dim=-1)
        sims = (stored_norm @ pfc_norm.squeeze(0))  # (n_eps,)

        td_errors = self.ep_td_errors[slot_idx, :n_eps]

        # Compute recency: how recently each episode was stored
        # Normalized to [0, 1] where 1 = most recent
        timestamps = self.ep_timestamps[slot_idx, :n_eps]
        ages = self._global_step - timestamps
        max_age = ages.max().clamp(min=1.0)
        recency = 1.0 - (ages / max_age)  # (n_eps,) in [0, 1]

        # Learned scoring: network decides how to combine signals
        scorer_input = torch.stack([
            sims,                           # PFC similarity
            td_errors.abs().clamp(min=1e-6), # salience
            recency,                         # temporal recency
        ], dim=-1)  # (n_eps, 3)

        relevance = self.episode_scorer(scorer_input).squeeze(-1)  # (n_eps,)

        best_idx = relevance.argmax().item()
        ep_td = td_errors[best_idx].item()
        return slot_eps[best_idx], ep_td, best_idx

    def retrieve_and_reinstate(self, mach, activation_summary, current_td_error,
                               top_k=3, device=None):
        """VQ match → select slot → retrieve best episode → reinstate.

        Returns reinstatement alpha for diagnostics.
        """
        if device is None:
            device = activation_summary.device
        if self.episodes.device != device:
            self.to(device)

        pfc = mach._pfc_state if hasattr(mach, '_pfc_state') else None

        # VQ matching
        key = self._make_key(activation_summary, pfc)
        slot_idx, sims, one_hot = self._match(key)
        self._last_slot = slot_idx

        # Episodic retrieval within matched slot
        ep_content, ep_td, ep_idx = self._best_episode(slot_idx, pfc, device)
        self._last_ep_idx = ep_idx

        if ep_content is None:
            return 0.0  # no episodes stored yet

        # Read gate: approach/avoidance
        # Sees: prototype match quality + current surprise + episode's surprise
        # Episode TD tells the gate whether this memory was from a surprising
        # moment (positive or negative) — informs approach vs avoidance
        best_sim = sims[slot_idx]
        gate_input = torch.stack([
            best_sim,
            torch.tensor(abs(current_td_error), device=device, dtype=torch.float32),
            torch.tensor(ep_td, device=device, dtype=torch.float32),
        ])
        alpha = self.read_gate(gate_input).squeeze()

        # Reinstate PFC from episode
        pfc_delta = self.read_to_pfc(ep_content)
        if pfc is not None:
            mach._pfc_state = mach._pfc_state + alpha * pfc_delta.unsqueeze(0)

        # Neuromod bias from episode
        alpha_f = alpha.item()
        if abs(alpha_f) > 1e-4 and hasattr(mach, '_eta_state'):
            neuromod_raw = self.read_to_neuromod(ep_content)
            nm = neuromod_raw.view(3, self.n_patches)
            mach._neuromod_bias = {
                'eta': nm[0].clamp(0.1, 1.0),
                'decay': nm[1].clamp(0.1, 1.0),
                'expl': nm[2].clamp(0.1, 0.5),
                'alpha': alpha_f,
            }

        return abs(alpha_f)

    def store(self, mach, activation_summary, reward, td_error, global_step=None):
        """VQ match → store episode in matched slot.

        Eviction: overwrite least-salient (lowest |TD error|) episode.
        Brain-like: surprising memories persist, mundane ones get overwritten.
        """
        device = activation_summary.device
        if self.episodes.device != device:
            self.to(device)

        if global_step is not None:
            self._global_step = global_step

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

            # Eviction: if buffer not full, use next empty slot (ring buffer)
            # If full, overwrite least-salient episode (lowest |TD error|)
            n_eps = min(self.ep_count[slot_idx].item(), self.episodes_per_slot)
            if n_eps < self.episodes_per_slot:
                # Buffer not full — use ring buffer head
                write_idx = self.ep_head[slot_idx].item()
                self.ep_head[slot_idx] = (write_idx + 1) % self.episodes_per_slot
            else:
                # Buffer full — evict least surprising episode
                write_idx = self.ep_td_errors[slot_idx].abs().argmin().item()

            self.episodes[slot_idx, write_idx] = value
            self.ep_td_errors[slot_idx, write_idx] = td_error
            self.ep_timestamps[slot_idx, write_idx] = self._global_step

            # Store compressed activations for NREM replay
            act_parts = []
            for i in range(self.n_patches):
                pre = mach._pre_activations.get(i)
                post = mach._post_activations.get(i)
                if pre is not None and post is not None and hasattr(mach, 'hebb_rule'):
                    pre_f = pre.float()
                    while pre_f.dim() > 1:
                        pre_f = pre_f.mean(dim=0)
                    post_f = post.float()
                    while post_f.dim() > 1:
                        post_f = post_f.mean(dim=0)
                    pre_c = mach.hebb_rule.compress[i](pre_f)   # (d_proj,)
                    post_c = mach.hebb_rule.compress[i](post_f)  # (d_proj,)
                    act_parts.extend([pre_c, post_c])
                else:
                    act_parts.extend([
                        torch.zeros(self.d_proj, device=device),
                        torch.zeros(self.d_proj, device=device),
                    ])
            self.ep_activations[slot_idx, write_idx] = torch.cat(act_parts)

            self.ep_count[slot_idx] += 1

        return True

    def replay_nrem(self, mach, n_replays=4, device=None):
        """NREM replay: reactivate stored activations → drive Hebbian updates.

        Like sharp-wave ripples during deep sleep:
        1. Sample episodes prioritized by |TD error| (surprising = more replayed)
        2. Reinstate PFC + neuromod from episode
        3. Feed stored compressed activations to Hebbian rule
        4. Patches learn from memory — no Qwen forward pass needed

        Returns number of episodes actually replayed.
        """
        if device is None:
            device = self.episodes.device

        # Collect all stored episodes across all slots
        all_episodes = []
        for slot_idx in range(self.n_slots):
            n_eps = min(self.ep_count[slot_idx].item(), self.episodes_per_slot)
            for ep_idx in range(n_eps):
                all_episodes.append((slot_idx, ep_idx))

        if not all_episodes:
            return 0

        # Prioritized sampling: probability ∝ |TD error|
        td_abs = torch.tensor([
            abs(self.ep_td_errors[s, e].item())
            for s, e in all_episodes
        ], device=device)
        td_abs = td_abs.clamp(min=1e-6)
        probs = td_abs / td_abs.sum()

        n_replays = min(n_replays, len(all_episodes))
        indices = torch.multinomial(probs, n_replays, replacement=False)

        replayed = 0
        for idx in indices:
            slot_idx, ep_idx = all_episodes[idx.item()]
            ep_content = self.episodes[slot_idx, ep_idx]
            ep_td = self.ep_td_errors[slot_idx, ep_idx].item()
            ep_acts = self.ep_activations[slot_idx, ep_idx]

            # Skip episodes with no stored activations
            if ep_acts.abs().sum() < 1e-8:
                continue

            # Reinstate PFC state
            pfc_stored = ep_content[:self.pfc_dim]
            if hasattr(mach, '_pfc_state'):
                mach._pfc_state = pfc_stored.unsqueeze(0).clone()

            # Reinstate neuromod from episode
            neuromod_stored = ep_content[self.pfc_dim:]
            if hasattr(mach, '_last_etas') and neuromod_stored.shape[0] >= self.n_patches * 3:
                etas = neuromod_stored[:self.n_patches].clamp(0.1, 1.0)
                decays = neuromod_stored[self.n_patches:self.n_patches*2].clamp(0.1, 1.0)
                expls = neuromod_stored[self.n_patches*2:].clamp(0.1, 0.5)
            else:
                etas = torch.full((self.n_patches,), 0.5, device=device)
                decays = torch.full((self.n_patches,), 0.5, device=device)
                expls = torch.full((self.n_patches,), 0.2, device=device)

            # Replay: feed stored compressed activations to Hebbian rule
            td_error_t = torch.tensor(ep_td, device=device, dtype=torch.float32)

            for patch_idx in range(self.n_patches):
                if not hasattr(mach, 'hebb_rule'):
                    break
                offset = patch_idx * self.d_proj * 2
                pre_c = ep_acts[offset:offset + self.d_proj]
                post_c = ep_acts[offset + self.d_proj:offset + self.d_proj * 2]

                if pre_c.abs().sum() < 1e-8:
                    continue

                # Drive Hebbian update with stored compressed activations
                # Uses replay_update which accepts pre-compressed inputs
                delta_down, delta_up = mach.hebb_rule.replay_update(
                    patch_idx, pre_c, post_c,
                    td_error=td_error_t,
                    eta=etas[patch_idx],
                    decay=decays[patch_idx],
                )
                scale = (etas[patch_idx] * mach.gate_scale).clamp(0, 1.0)
                mach.patches[patch_idx].accumulate_write(
                    "down", scale * delta_down, decay=decays[patch_idx]
                )
                mach.patches[patch_idx].accumulate_write(
                    "up", scale * delta_up, decay=decays[patch_idx]
                )

            replayed += 1

        return replayed

    def replay_rem(self, mach, patched_model, tokenizer, n_dreams=2, device=None):
        """REM replay (dreaming): run Qwen forward with replayed context.

        Like REM sleep — full generative replay:
        1. Sample salient episodes
        2. Reinstate PFC + neuromod from episode
        3. Generate a simple arithmetic prompt (novel combination)
        4. Run full Qwen forward → patches fire → normal Hebbian update

        This enables generalization: patches learn from novel input combinations
        while guided by stored memory states. Not wired into training loop —
        call manually for experimentation.

        Returns list of (prompt, replay_td_error) for each dream.
        """
        if device is None:
            device = self.episodes.device

        from data.arithmetic import generate_few_shot_episode

        # Collect salient episodes
        all_episodes = []
        for slot_idx in range(self.n_slots):
            n_eps = min(self.ep_count[slot_idx].item(), self.episodes_per_slot)
            for ep_idx in range(n_eps):
                all_episodes.append((slot_idx, ep_idx))

        if not all_episodes:
            return []

        td_abs = torch.tensor([
            abs(self.ep_td_errors[s, e].item())
            for s, e in all_episodes
        ], device=device)
        td_abs = td_abs.clamp(min=1e-6)
        probs = td_abs / td_abs.sum()

        n_dreams = min(n_dreams, len(all_episodes))
        indices = torch.multinomial(probs, n_dreams, replacement=False)

        dreams = []
        for idx in indices:
            slot_idx, ep_idx = all_episodes[idx.item()]
            ep_content = self.episodes[slot_idx, ep_idx]
            ep_td = self.ep_td_errors[slot_idx, ep_idx].item()

            # Reinstate PFC + neuromod
            pfc_stored = ep_content[:self.pfc_dim]
            if hasattr(mach, '_pfc_state'):
                mach._pfc_state = pfc_stored.unsqueeze(0).clone()

            neuromod_stored = ep_content[self.pfc_dim:]
            if hasattr(mach, '_neuromod_bias') and neuromod_stored.shape[0] >= self.n_patches * 3:
                mach._neuromod_bias = {
                    'eta': neuromod_stored[:self.n_patches].clamp(0.1, 1.0),
                    'decay': neuromod_stored[self.n_patches:self.n_patches*2].clamp(0.1, 1.0),
                    'expl': neuromod_stored[self.n_patches*2:].clamp(0.1, 0.5),
                    'alpha': 0.5,
                }

            # Generate a novel prompt (dream content)
            import random
            ops = ['add', 'sub', 'mul', 'div', 'mod', 'gcd', 'abs_diff']
            dream_op = random.choice(ops)
            problems = generate_few_shot_episode(1, n_demos=0, op_type=dream_op)
            prompt = problems[0]['text']

            # Run full forward pass through Qwen with patched model
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = patched_model(**inputs)

            # Normal Hebbian step fires on the dream activations
            # (caller handles this — we just set up the state)
            dreams.append((prompt, ep_td, dream_op))

        return dreams

    def set_neuromod(self, gamma, avg_decay):
        """Interface compatibility. No EMA to drive."""
        pass

    def reconsolidate(self, td_error):
        """Interface compatibility."""
        pass

    def decay_all(self):
        """No-op. Ring buffer cycles naturally."""
        pass

    def save(self, path=None):
        """Persist memory state to disk."""
        path = path or self.save_path
        if path is None:
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'episodes': self.episodes.cpu(),
            'ep_head': self.ep_head.cpu(),
            'ep_count': self.ep_count.cpu(),
            'ep_td_errors': self.ep_td_errors.cpu(),
            'ep_timestamps': self.ep_timestamps.cpu(),
            'ep_activations': self.ep_activations.cpu(),
            'global_step': self._global_step,
        }, path)

    def _load(self, path):
        if not os.path.exists(path):
            return
        data = torch.load(path, map_location='cpu', weights_only=True)
        if 'episodes' in data and data['episodes'].shape == self.episodes.shape:
            self.episodes.copy_(data['episodes'])
            self.ep_head.copy_(data['ep_head'])
            self.ep_count.copy_(data['ep_count'])
            if 'ep_td_errors' in data:
                self.ep_td_errors.copy_(data['ep_td_errors'])
            elif 'ep_rewards' in data:
                self.ep_td_errors.copy_(data['ep_rewards'])  # legacy compat
            if 'ep_timestamps' in data:
                self.ep_timestamps.copy_(data['ep_timestamps'])
            if 'ep_activations' in data and data['ep_activations'].shape == self.ep_activations.shape:
                self.ep_activations.copy_(data['ep_activations'])
            if 'global_step' in data:
                self._global_step = data['global_step']

    def __len__(self):
        """Number of slots with at least one episode."""
        return int((self.ep_count > 0).sum().item())

    def total_episodes(self):
        """Total episodes stored across all slots."""
        return int(self.ep_count.clamp(max=self.episodes_per_slot).sum().item())

    def __repr__(self):
        n_eps = self.total_episodes()
        return f"Hippocampus({len(self)}/{self.n_slots} slots, {n_eps} episodes)"
