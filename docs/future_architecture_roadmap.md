# Future Architecture Roadmap: Toward Brain-Like General Meta-Learning

## Overview

Phase 5 introduces the information bottleneck (task state). These are the next steps to make the meta-learner truly general-purpose, each mapped to brain mechanisms.

## 1. Deliberation (PFC iterative refinement)

**Brain**: PFC has recurrent attractor dynamics — it "thinks" by iterating until settling on a stable representation.

**Currently**: One task state update per demo, no iteration.

**Implementation**: After all demos are observed, run N refinement steps where task state updates from its own representation — no new input, just internal processing.

```python
for _ in range(n_think_steps):
    task_state = self.deliberate(task_state)  # self-attention or recurrence
```

Cheap (d_task=32), and the model learns effective iteration count via gating (gate→0 = stop changing).

**Priority**: HIGH — cheapest change, most likely to help immediately.

## 2. Self-Evaluation (anterior cingulate cortex, error monitoring)

**Brain**: ACC monitors conflict and errors, signals PFC to adjust strategy.

**Currently**: No feedback about whether patches work.

**Implementation**: After writing patches, do a quick "mental simulation" — run Qwen on a demo problem with current patches, check if output matches known answer. Feed error back into task state for correction.

```python
# Write patches
self.apply_writes(writes)
# Mental simulation on a demo we already know the answer to
predicted = run_qwen(demo_input)
error = compare(predicted, demo_answer)
# Correct
correction_writes = self.error_correction(task_state, error)
self.apply_writes(correction_writes)
```

Cerebellum-like forward model: predict, compare, correct — but driven by task state.

**Priority**: HIGH — closes the loop, lets model verify its own patches.

## 3. Richer Observation (sensory cortex → association cortex)

**Brain**: Processes at multiple levels simultaneously, not a single snapshot.

**Currently**: Pool one layer's last-token hidden state.

**Implementation**: Observe all 4 patch layers, not just middle one. Each layer captures different abstraction levels. Also observe token-level distributions (softmax output) — the meta-learner "sees" what Qwen currently believes.

```
Layer 9 hiddens  → "low-level features"
Layer 18 hiddens → "mid-level"
Layer 27 hiddens → "high-level"
Logits           → "Qwen's current belief"
```

Cross-attend or concatenate across layers before projecting to d_obs.

**Priority**: MEDIUM — more signal for the same architecture.

## 4. Feedback (PFC → sensory cortex, top-down attention)

**Brain**: PFC tells visual cortex what to look for. Massive top-down connections.

**Currently**: Strictly feedforward information flow.

**Implementation**: Condition observation projection on current task state. After initial demos form a rough task state, re-process demos with task-state-modulated attention.

```python
# First pass: generic observation
task_state = initial_update(demos)
# Second pass: task-guided observation
attention_bias = self.feedback_proj(task_state)  # d_task → d_obs
refined_obs = self.obs_proj(hidden_states, bias=attention_bias)
task_state = refined_update(refined_obs, task_state)
```

Like re-reading examples after forming a hypothesis about the pattern.

**Priority**: MEDIUM — powerful but adds complexity.

## 5. Richer Writes (motor cortex → multiple effectors)

**Brain**: Neuromodulators don't rewrite synapses — they change the gain of entire circuits. Dopamine modulates excitability, not connectivity.

**Currently**: Additive residual stream patches at 4 layers, rank 8.

**Implementation options**:
- **Attention head gating**: Scale individual attention heads up/down. Some heads specialize in arithmetic, copying, reasoning — meta-learner learns which to amplify.
- **Key/value biases**: Small additive biases to attention keys or values, steering what tokens attend to.
- **Dense small patches**: Write at all 36 layers with very small patches instead of 4 layers with larger ones.

```python
# Attention head gating
head_gates = self.head_gate_compiler(task_state)  # → 32 scalars per layer
# Applied: attention_output[head_i] *= gate[head_i]
```

**Priority**: LOW — needs careful design to avoid instability.

## 6. Consolidation (hippocampus → cortex transfer, sleep)

**Brain**: Sleep replays episodes, gradually burns common patterns into cortical synapses. Skills become automatic.

**Currently**: Every episode starts from zero task state.

**Implementation**: After training on many episodes, cluster common patch patterns across tasks. Distill into "permanent" base patch weights. Meta-learner then writes delta from nearest known pattern.

```python
# After N episodes, cluster the patch writes
centroids = cluster(all_episode_patches, k=16)
# Permanent patches = centroids
# Meta-learner now writes: select_centroid + small_correction
```

Like skill automatization — a pianist doesn't re-derive hand positions each time.

**Priority**: LOW — only makes sense after many tasks work.

## Implementation Order

1. **Bottleneck** (Phase 5) ← WE ARE HERE
2. **Deliberation** — internal iteration on task state
3. **Self-evaluation** — verify patches, correct errors
4. **Richer observation** — multi-layer, multi-position
5. **Feedback** — top-down task-guided observation
6. **Richer writes** — attention gating, more sites
7. **Consolidation** — sleep, skill permanence

Each step makes the meta-learner more general without changing what Qwen provides. The frozen LLM already knows how to do most things — the meta-learner just needs to learn how to steer it.
