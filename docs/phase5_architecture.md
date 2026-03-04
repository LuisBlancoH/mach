# Phase 5: Brain-Like Meta-Learner Architecture

## Motivation

Phases 2-4 achieved genuine meta-learning (demo reading) but fail to generalize
to unseen operations. The architecture memorizes a lookup table (task → patch weights)
instead of learning structured task representations.

Root cause: no information bottleneck between "understanding the task" and
"executing the task." All internal representations are d_meta=128 throughout,
allowing unconstrained memorization.

## Brain Mapping

| Brain Region | Role | Phase 5 Component |
|-------------|------|-------------------|
| Sensory cortex | Feature extraction | ObservationProjection (d_model → d_obs) |
| Hippocampus | Episodic sequential memory | GRU (d_obs → d_gru) |
| PFC (dlPFC) | Working memory, task representation | TaskState (d_task, sparse, gated) |
| Premotor cortex | Action planning | ActionCompiler (d_task → coefficients) |
| Motor cortex | Execution | BasisVectors → DifferentiablePatch |
| Basal ganglia | Value estimation, gradient shaping | Critic (separate, loss-only) |

## Architecture

```
Input (Qwen hidden states at patch layers)
    |
    v
ObservationProjection: pool + project hidden states (d_model → d_obs)
    |
    v
GRU: sequential episodic memory (d_obs → d_gru)
    |                                        [Hippocampus]
    v
Gated Task Update:                           [PFC]
    gate = sigmoid(W_gate @ [gru_memory, task_state])
    candidate = tanh(W_cand @ [gru_memory, task_state])
    task_state = gate * candidate + (1 - gate) * task_state
    |
    v
Sparsity penalty: L1 on task_state           [Sparse PFC coding]
    |
    v
ActionCompiler: task_state → coefficients + gates    [Premotor]
    |
    v
BasisVectors: coefficients → delta_W         [Motor execution]
    |
    v
DifferentiablePatch: accumulate writes        [Patch application]
```

## Key Dimensions

```python
d_obs = 64      # observation projection output
d_gru = 64      # GRU hidden state (hippocampal memory)
d_task = 32     # task state (PFC working memory) — THE bottleneck
n_basis = 8     # basis vectors per patch (unchanged)
```

d_task << d_gru << d_meta. The bottleneck forces structure.

## What's New vs Phase 2

| Component | Phase 2 | Phase 5 |
|-----------|---------|---------|
| Observation | obs_proj (d_model → d_meta=128) | obs_proj (d_model → d_obs=64) |
| Memory | GRU (d_meta=128) | GRU (d_obs → d_gru=64) |
| Task repr | None (flat d_meta throughout) | TaskState (d_task=32, sparse, gated) |
| Processing | MetaLearnerTransformer + tf_mem | Gated task update (no transformer) |
| Action | ActionHead (d_meta → coeffs) | ActionCompiler (d_task → coeffs) |
| Memory update | MemoryHead (d_meta → tf_mem) | Gated update (built into task state) |
| Reward input | reward_signals → fire() | None (critic shapes loss only) |
| Sparsity | None | L1 penalty on task_state |

## What's Removed

- MetaLearnerTransformer (replaced by gated task update — simpler, more constrained)
- MemoryHead (replaced by gated update mechanism)
- RewardProjection (no reward input to fire)
- tf_mem (replaced by task_state)

## What's Kept

- ObservationProjection (resized to d_obs)
- SimpleGRU (resized to d_gru)
- BasisVectors + DifferentiablePatch (unchanged)
- ActionHead logic (resized input to d_task, renamed ActionCompiler)
- Critic (Phase 3+, separate module, shapes CE loss only)

## Gated Task Update (Detail)

The task state update is a GRU-like gated recurrence on the task representation:

```python
class TaskState(nn.Module):
    def __init__(self, d_gru, d_task):
        self.gate_net = nn.Linear(d_gru + d_task, d_task)
        self.candidate_net = nn.Linear(d_gru + d_task, d_task)

    def forward(self, gru_memory, task_state):
        combined = torch.cat([gru_memory, task_state])
        gate = torch.sigmoid(self.gate_net(combined))
        candidate = torch.tanh(self.candidate_net(combined))
        new_task_state = gate * candidate + (1 - gate) * task_state
        return new_task_state
```

The gate decides how much each observation updates the task representation.
Early demos should update heavily (gate ≈ 1). Later redundant demos should
update less (gate ≈ 0). The model learns this from data.

## Sparsity Loss

```python
sparsity_loss = beta * task_state.abs().mean()
total_loss = ce_loss + sparsity_loss
```

beta is a hyperparameter (start with 0.01, tune). Higher beta = more sparsity =
more generalization pressure but less capacity. Lower beta = less sparsity =
more capacity but risk of memorization.

With d_task=32 and sparsity, expect ~4-8 active dimensions per task.
Novel tasks that share features with known tasks can generalize because
the active dimensions combine in new ways.

## Why This Should Generalize

1. **Information bottleneck**: d_task=32 with sparsity means ~4-8 effective
   dimensions. Can't memorize 9+ arbitrary operations in 4-8 dims.

2. **Compositional reuse**: If dim 3 encodes "scale factor for operand a" and
   dim 7 encodes "scale factor for operand b", then novel (c1, c2)
   combinations naturally fall in the right place.

3. **Gated updates**: The model learns WHICH observations are informative,
   not just what they contain. This transfers across tasks.

4. **No reward shortcut**: Critic shapes gradient only. The task state
   must be derived purely from observations.

## Training

Same outer loop as Phase 2:
- Episode: observe demos → update task state → write patches → evaluate on test problems
- Loss: CE on test problem predictions + sparsity penalty on task_state
- Optimizer: Adam on all meta-learner parameters
- Critic (optional): separate optimizer, TD loss, modulates CE gradient

## Estimated Parameters

- ObservationProjection: d_model(2560) * d_obs(64) ≈ 164K
- GRU: 3 * d_obs(64) * d_gru(64) * 2 ≈ 25K
- TaskState gate + candidate: 2 * (d_gru + d_task)(96) * d_task(32) ≈ 6K
- ActionCompiler: d_task(32) * (n_patches * n_basis * 3) ≈ 10K
- BasisVectors: n_patches(4) * n_basis(8) * (d_model + hidden)(2560+64) * 2 ≈ 336K

Total: ~540K parameters (down from ~5M in Phase 2)

The 10x parameter reduction is a feature, not a bug. Fewer parameters =
less memorization capacity = more pressure to generalize.

## Verification Plan

1. Train on 6 linear combinations, eval on 3 held-out
2. Train on 7 arithmetic ops, eval on held-out compound ops
3. Mismatch eval: wrong demos should hurt accuracy (proves demo reading)
4. Compare against Phase 2 on same tasks at same episode counts

Success = held-out accuracy > baseline (proves generalization beyond memorization)
