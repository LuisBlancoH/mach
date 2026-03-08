"""
Hippocampus: distributed episodic memory + replay.

Global episode pool (CA3/CA1):
  Flat buffer of N episodes. No discrete slots or categories.
  Storage and retrieval use continuous similarity — like the brain's
  distributed population code, not a codebook lookup.
  A partial cue activates the most similar stored pattern via
  pattern completion (winner-take-all, not blending).

Pattern separation (Dentate Gyrus):
  2-layer ReLU key projection orthogonalizes similar inputs,
  preventing catastrophic interference between similar memories.

Retrieval:
  Query key → cosine similarity against all stored episode keys →
  learned scorer ranks by [similarity, salience, recency] →
  argmax selects ONE specific episode (no blending) →
  approach/avoidance gate controls reinstatement strength.

Eviction:
  When buffer full, evict globally least-relevant episode
  (learned scorer decides, not hardcoded rule).

Replay (memory consolidation):
  NREM: reactivate stored compressed activations → drive Hebbian
    updates without running Qwen. Like sharp-wave ripples.
  REM: run Qwen forward with generated prompts while replayed
    PFC/neuromod influence patches. For generalization testing.

No hardcoded thresholds. No discrete slots. No per-category limits.
Retrieval priority and eviction are both learned.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Hippocampus(nn.Module):
    """Distributed episodic memory with learned retrieval and replay.

    Global flat buffer — no VQ slots, no per-category limits.
    Continuous similarity retrieval with winner-take-all selection.
    Brain-like: DG pattern separation, CA3 pattern completion,
    dopamine-modulated salience, temporal context, replay.
    """

    def __init__(self, key_dim, pfc_dim=32, n_patches=4, capacity=128,
                 d_proj=32, save_path=None):
        super().__init__()
        self.key_dim = key_dim
        self.pfc_dim = pfc_dim
        self.n_patches = n_patches
        self.capacity = capacity
        self.d_proj = d_proj
        self.d_mem = pfc_dim + n_patches * 3  # PFC + neuromod per entry
        self.d_act = d_proj * n_patches * 2   # compressed activations for replay
        self.save_path = save_path

        # === Pattern separation (Dentate Gyrus) ===
        # Orthogonalizes similar inputs to reduce interference
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

        # === Global episode buffer ===
        # Flat storage — no slots, no categories
        self.register_buffer('episodes',
            torch.zeros(capacity, self.d_mem))
        self.register_buffer('ep_keys',
            torch.zeros(capacity, key_dim))
        self.register_buffer('ep_td_errors',
            torch.zeros(capacity))
        self.register_buffer('ep_timestamps',
            torch.zeros(capacity))
        self.register_buffer('ep_activations',
            torch.zeros(capacity, self.d_act))
        self.register_buffer('ep_valid',
            torch.zeros(capacity, dtype=torch.bool))
        self._write_head = 0
        self._global_step = 0

        # === Learned retrieval scoring ===
        # Input: [key_similarity, |td_error|, recency]
        # Network learns how to weight these — no hardcoded formula
        self.episode_scorer = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        with torch.no_grad():
            nn.init.zeros_(self.episode_scorer[0].weight)
            self.episode_scorer[0].weight[0, 0] = 1.0  # key similarity
            self.episode_scorer[0].weight[1, 1] = 1.0  # salience
            nn.init.zeros_(self.episode_scorer[0].bias)
            nn.init.ones_(self.episode_scorer[2].weight)
            nn.init.zeros_(self.episode_scorer[2].bias)

        # === Read gate: approach/avoidance ===
        # Input: [key_similarity, current_td_error, episode_td_error]
        # Output: [-1, 1] — positive=reinstate, negative=repel
        self.read_gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        with torch.no_grad():
            self.read_gate[-2].bias.fill_(0.1)

        # === Reinstatement projections ===
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

        self._last_ep_idx = -1
        self._last_alpha = 0.0
        self._last_key = None       # stored for local loss (before detach)
        self._last_best_sim = None  # similarity of retrieved episode

        if save_path and os.path.exists(save_path):
            self._load(save_path)

    def _make_key(self, activation_summary, pfc_state):
        """Project activation + PFC into key space (pattern separation / DG).

        Key is DETACHED from the main loss graph. key_proj learns via a local
        REINFORCE signal (compute_local_loss), not through the critic chain.
        Brain-faithful: DG learns pattern separation from local prediction
        error, not from cortical backprop.
        """
        if pfc_state is not None:
            pfc = pfc_state.squeeze(0) if pfc_state.dim() > 1 else pfc_state
        else:
            pfc = torch.zeros(self.pfc_dim, device=activation_summary.device)
        combined = torch.cat([activation_summary, pfc])
        key = self.key_proj(combined)
        # Store for local loss computation (before detach)
        self._last_key = key
        # Detach from main graph — key_proj learns via local loss only
        return key.detach()

    def _find_best(self, query_key, pfc_state, device):
        """Pattern completion: find most relevant episode in the global pool.

        Like CA3 recurrent attractor dynamics:
        - Partial cue (query key) activates stored patterns by similarity
        - Learned scorer ranks by [similarity, salience, recency]
        - Winner-take-all: argmax selects ONE specific episode
        - No blending — you recall a specific memory, not an average

        Returns (episode_idx, episode_content, episode_td, key_similarity).
        """
        n_valid = self.ep_valid.sum().item()
        if n_valid == 0:
            return -1, None, 0.0, 0.0

        valid_mask = self.ep_valid  # (capacity,)
        valid_indices = torch.where(valid_mask)[0]  # indices of valid episodes

        # Key similarity: cosine between query and all stored keys
        query_norm = F.normalize(query_key.unsqueeze(0), dim=-1)
        stored_keys = self.ep_keys[valid_indices]  # (n_valid, key_dim)
        stored_norm = F.normalize(stored_keys, dim=-1)
        key_sims = (stored_norm @ query_norm.squeeze(0))  # (n_valid,)

        # TD error salience
        td_errors = self.ep_td_errors[valid_indices]

        # Temporal recency
        timestamps = self.ep_timestamps[valid_indices]
        ages = self._global_step - timestamps
        max_age = ages.max().clamp(min=1.0)
        recency = 1.0 - (ages / max_age)  # [0, 1]

        # Learned scoring: network decides how to rank
        scorer_input = torch.stack([
            key_sims,
            td_errors.abs().clamp(min=1e-6),
            recency,
        ], dim=-1)  # (n_valid, 3)
        relevance = self.episode_scorer(scorer_input).squeeze(-1)  # (n_valid,)

        # Winner-take-all: one specific memory
        best_local = relevance.argmax().item()
        best_global = valid_indices[best_local].item()
        best_sim = key_sims[best_local]
        ep_td = td_errors[best_local].item()

        return best_global, self.episodes[best_global].clone(), ep_td, best_sim

    def _find_eviction_target(self, query_key, device):
        """Find which episode to evict when buffer is full.

        Uses the same learned scorer but in reverse: evict the episode
        with the LOWEST relevance to the current context.
        This is adaptive — the network learns what to forget.
        """
        valid_indices = torch.where(self.ep_valid)[0]

        query_norm = F.normalize(query_key.unsqueeze(0), dim=-1)
        stored_keys = self.ep_keys[valid_indices]
        stored_norm = F.normalize(stored_keys, dim=-1)
        key_sims = (stored_norm @ query_norm.squeeze(0))

        td_errors = self.ep_td_errors[valid_indices]
        timestamps = self.ep_timestamps[valid_indices]
        ages = self._global_step - timestamps
        max_age = ages.max().clamp(min=1.0)
        recency = 1.0 - (ages / max_age)

        scorer_input = torch.stack([
            key_sims,
            td_errors.abs().clamp(min=1e-6),
            recency,
        ], dim=-1)
        relevance = self.episode_scorer(scorer_input).squeeze(-1)

        # Evict least relevant
        worst_local = relevance.argmin().item()
        return valid_indices[worst_local].item()

    def retrieve_and_reinstate(self, mach, activation_summary, current_td_error,
                               top_k=3, device=None):
        """Pattern completion → retrieve one episode → reinstate.

        Returns reinstatement alpha for diagnostics.
        """
        if device is None:
            device = activation_summary.device
        if self.episodes.device != device:
            self.to(device)

        pfc = mach._pfc_state if hasattr(mach, '_pfc_state') else None

        # Pattern separation (DG) then pattern completion (CA3)
        key = self._make_key(activation_summary, pfc)
        ep_idx, ep_content, ep_td, best_sim = self._find_best(key, pfc, device)
        self._last_ep_idx = ep_idx

        if ep_content is None:
            self._last_best_sim = None
            return 0.0

        # Compute similarity using the NON-detached key for local loss
        # (the detached key was used for retrieval; this one tracks through key_proj)
        if self._last_key is not None:
            stored_key = self.ep_keys[ep_idx]  # buffer, no grad
            self._last_best_sim = F.cosine_similarity(
                self._last_key.unsqueeze(0), stored_key.unsqueeze(0)
            ).squeeze()
        else:
            self._last_best_sim = None

        # Read gate: approach/avoidance
        gate_input = torch.stack([
            best_sim,
            torch.tensor(abs(current_td_error), device=device, dtype=torch.float32),
            torch.tensor(ep_td, device=device, dtype=torch.float32),
        ])
        alpha = self.read_gate(gate_input).squeeze()

        # Reinstate PFC from episode
        # Detach old PFC: gradient flows through alpha and pfc_delta (learned),
        # not through the prior PFC chain (which has its own gradient path via GRU)
        pfc_delta = self.read_to_pfc(ep_content)
        if pfc is not None:
            mach._pfc_state = mach._pfc_state.detach() + alpha * pfc_delta.unsqueeze(0)

        # Neuromod bias from episode — keep as TENSORS for gradient flow
        # alpha stays as tensor → read_gate gets gradient through neuromod blending
        # neuromod values stay as tensors → read_to_neuromod gets gradient
        alpha_abs = alpha.abs()
        if alpha_abs.item() > 1e-4 and hasattr(mach, '_eta_state'):
            neuromod_raw = self.read_to_neuromod(ep_content)
            nm = neuromod_raw.view(3, self.n_patches)
            mach._neuromod_bias = {
                'eta': nm[0].clamp(0.1, 1.0),
                'decay': nm[1].clamp(0.1, 1.0),
                'expl': nm[2].clamp(0.1, 0.5),
                'alpha': alpha_abs,  # tensor, not float — gradient flows to read_gate
            }

        self._last_alpha = alpha_abs.item()
        return self._last_alpha

    def compute_local_loss(self, td_error):
        """Local REINFORCE loss for key_proj (pattern separation).

        Brain-faithful: DG learns from local prediction error signals,
        not from backprop through the cortical chain. When retrieval was
        followed by positive surprise (td_error > 0), reinforce the key
        mapping that found this memory. When negative, weaken it.

        loss = -td_error × similarity(key_proj(input), stored_key)

        This trains key_proj to map inputs to keys that retrieve helpful
        memories. Only key_proj gets gradient (similarity uses _last_key
        which was computed before detach).
        """
        if self._last_best_sim is None or self._last_key is None:
            return torch.tensor(0.0)

        # REINFORCE: reward good retrievals, punish bad ones
        td = torch.tensor(td_error, dtype=torch.float32, device=self._last_best_sim.device) \
            if not isinstance(td_error, torch.Tensor) else td_error.detach()
        loss = -td * self._last_best_sim
        return loss

    def store(self, mach, activation_summary, reward, td_error, global_step=None):
        """Store episode in global buffer. Evict least relevant if full.

        No slots, no categories — episodes are stored with their keys
        and retrieved by continuous similarity.
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

            # Compose value: PFC + neuromod
            etas = torch.zeros(self.n_patches, device=device)
            decays = torch.zeros(self.n_patches, device=device)
            expls = torch.zeros(self.n_patches, device=device)
            if hasattr(mach, '_last_etas') and mach._last_etas is not None:
                etas = mach._last_etas
                decays = mach._last_decays
                expls = mach._last_expls
            value = torch.cat([pfc_flat, etas, decays, expls])

            # Find write position
            n_valid = self.ep_valid.sum().item()
            if n_valid < self.capacity:
                # Buffer not full — use next empty position
                empty = torch.where(~self.ep_valid)[0]
                write_idx = empty[0].item()
            else:
                # Buffer full — evict least relevant episode
                write_idx = self._find_eviction_target(key, device)

            self.episodes[write_idx] = value
            self.ep_keys[write_idx] = key.detach()
            self.ep_td_errors[write_idx] = td_error
            self.ep_timestamps[write_idx] = self._global_step
            self.ep_valid[write_idx] = True

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
                    pre_c = mach.hebb_rule.compress[i](pre_f)
                    post_c = mach.hebb_rule.compress[i](post_f)
                    act_parts.extend([pre_c, post_c])
                else:
                    act_parts.extend([
                        torch.zeros(self.d_proj, device=device),
                        torch.zeros(self.d_proj, device=device),
                    ])
            self.ep_activations[write_idx] = torch.cat(act_parts)

        return True

    def replay_nrem(self, mach, n_replays=4, device=None):
        """NREM replay: reactivate stored activations → drive Hebbian updates.

        Like sharp-wave ripples during deep sleep:
        1. Sample episodes prioritized by |TD error|
        2. Reinstate PFC + neuromod from episode
        3. Feed stored compressed activations to Hebbian rule
        4. Patches learn from memory — no Qwen forward pass needed

        Returns number of episodes actually replayed.
        """
        if device is None:
            device = self.episodes.device

        valid_indices = torch.where(self.ep_valid)[0]
        if len(valid_indices) == 0:
            return 0

        # Replay is offline consolidation — no gradient needed
        # Like sleep: the brain isn't learning from live experience during replay
        with torch.no_grad():
            # Prioritized sampling: probability ∝ |TD error|
            td_abs = self.ep_td_errors[valid_indices].abs().clamp(min=1e-6)
            probs = td_abs / td_abs.sum()

            n_replays = min(n_replays, len(valid_indices))
            sampled = torch.multinomial(probs, n_replays, replacement=False)

            replayed = 0
            for s in sampled:
                idx = valid_indices[s].item()
                ep_content = self.episodes[idx]
                ep_td = self.ep_td_errors[idx].item()
                ep_acts = self.ep_activations[idx]

                if ep_acts.abs().sum() < 1e-8:
                    continue

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

                td_error_t = torch.tensor(ep_td, device=device, dtype=torch.float32)

                for patch_idx in range(self.n_patches):
                    if not hasattr(mach, 'hebb_rule'):
                        break
                    offset = patch_idx * self.d_proj * 2
                    pre_c = ep_acts[offset:offset + self.d_proj]
                    post_c = ep_acts[offset + self.d_proj:offset + self.d_proj * 2]

                    if pre_c.abs().sum() < 1e-8:
                        continue

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
        1. Sample salient episodes (prioritized by |TD error|)
        2. Reinstate PFC + neuromod from episode
        3. Generate a novel arithmetic prompt (dream content)
        4. Run full Qwen forward → patches capture activations → Hebbian update

        Unlike NREM (which uses stored compressed activations), REM runs the
        full forward pass — this lets the system discover new activation patterns
        and generalize across operations.

        Returns list of (dream_op, reward, td_error) for each dream.
        """
        if device is None:
            device = self.episodes.device

        from data.arithmetic import generate_few_shot_episode, extract_number

        valid_indices = torch.where(self.ep_valid)[0]
        if len(valid_indices) == 0:
            return []

        td_abs = self.ep_td_errors[valid_indices].abs().clamp(min=1e-6)
        probs = td_abs / td_abs.sum()

        n_dreams = min(n_dreams, len(valid_indices))
        sampled = torch.multinomial(probs, n_dreams, replacement=False)

        dreams = []
        with torch.no_grad():
            for s in sampled:
                idx = valid_indices[s].item()
                ep_content = self.episodes[idx]
                ep_td = self.ep_td_errors[idx].item()

                # Reinstate PFC + neuromod from episode
                pfc_stored = ep_content[:self.pfc_dim]
                if hasattr(mach, '_pfc_state'):
                    mach._pfc_state = pfc_stored.unsqueeze(0).clone()

                neuromod_stored = ep_content[self.pfc_dim:]
                if hasattr(mach, '_neuromod_bias') and neuromod_stored.shape[0] >= self.n_patches * 3:
                    mach._neuromod_bias = {
                        'eta': neuromod_stored[:self.n_patches].clamp(0.1, 1.0),
                        'decay': neuromod_stored[self.n_patches:self.n_patches*2].clamp(0.1, 1.0),
                        'expl': neuromod_stored[self.n_patches*2:].clamp(0.1, 0.5),
                        'alpha': torch.tensor(0.5, device=device),
                    }

                # Generate a novel prompt (dream content)
                import random
                ops = ['add', 'sub', 'mul', 'div', 'mod', 'gcd', 'abs_diff']
                dream_op = random.choice(ops)
                problems = generate_few_shot_episode(1, n_demos=0, op_type=dream_op)
                prompt = problems[0]['text']
                answer = problems[0]['answer']

                # Forward pass — hooks capture pre/post activations
                inputs = tokenizer(prompt, return_tensors='pt').to(device)
                outputs = patched_model(**inputs)

                # Score the dream: did patches help produce the right answer?
                logits = outputs.logits[0, -1]
                predicted_token = tokenizer.decode(logits.argmax(-1).item()).strip()
                predicted = extract_number(predicted_token)
                from training.two_channel_train import graded_reward
                reward = graded_reward(predicted, answer)

                # Hebbian update from dream experience
                # This drives plasticity using the LIVE activations from forward pass
                # (not stored compressed ones like NREM)
                if hasattr(mach, 'compute_context_gates'):
                    mach.compute_context_gates()
                mach.hebbian_step(
                    reward=reward, step_idx=0, n_steps=1, device=device
                )

                dreams.append((dream_op, reward, ep_td))

        return dreams

    def set_neuromod(self, gamma, avg_decay):
        """Interface compatibility."""
        pass

    def reconsolidate(self, td_error):
        """Interface compatibility."""
        pass

    def decay_all(self):
        """No-op."""
        pass

    def save(self, path=None):
        """Persist memory state to disk."""
        path = path or self.save_path
        if path is None:
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'episodes': self.episodes.cpu(),
            'ep_keys': self.ep_keys.cpu(),
            'ep_td_errors': self.ep_td_errors.cpu(),
            'ep_timestamps': self.ep_timestamps.cpu(),
            'ep_activations': self.ep_activations.cpu(),
            'ep_valid': self.ep_valid.cpu(),
            'write_head': self._write_head,
            'global_step': self._global_step,
        }, path)

    def _load(self, path):
        if not os.path.exists(path):
            return
        data = torch.load(path, map_location='cpu', weights_only=True)
        if 'episodes' in data and data['episodes'].shape == self.episodes.shape:
            self.episodes.copy_(data['episodes'])
            self.ep_keys.copy_(data['ep_keys'])
            self.ep_td_errors.copy_(data['ep_td_errors'])
            self.ep_valid.copy_(data['ep_valid'])
            if 'ep_timestamps' in data:
                self.ep_timestamps.copy_(data['ep_timestamps'])
            if 'ep_activations' in data and data['ep_activations'].shape == self.ep_activations.shape:
                self.ep_activations.copy_(data['ep_activations'])
            if 'write_head' in data:
                self._write_head = data['write_head']
            if 'global_step' in data:
                self._global_step = data['global_step']

    def __len__(self):
        """Number of stored episodes."""
        return int(self.ep_valid.sum().item())

    def total_episodes(self):
        """Total episodes stored."""
        return len(self)

    def __repr__(self):
        return f"Hippocampus({len(self)}/{self.capacity} episodes)"
