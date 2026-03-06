# MACH Roadmap: Known Limitations & Solutions

## 1. Patch Expressiveness (Medium Priority)

**Problem**: 4 patched layers, rank-16 updates, 256-dim bottleneck. Patches can only add small corrections. If the base LLM has zero capability on a task, patches may not be enough.

**Solutions**:
- Patch all 36 layers (MACHDenseHebbian already exists) instead of just 4
- Increase rank from 16 to 32-64 for richer updates
- Increase hidden_dim beyond 256
- Brain analogy: plasticity happens at every synapse, not just 4 locations

## 2. Residual Stream Only (Medium Priority)

**Problem**: Patches only add to the residual stream. Can't modify how attention routes information between positions. Some tasks may need new information flow paths.

**Solutions**:
- Add attention output patches: hook self_attn, add learned correction before residual connection
- Controls *what information gets routed*, not just what passes through
- Plan already exists (DualHebbianPatchedModel in plan file)
- Qwen uses fused SDPA — can't inject into attention scores, but CAN modify attention output
- Brain analogy: plasticity modifies both dendrites (residual) and synaptic connections (attention)

## 3. Learning Speed (Low Priority)

**Problem**: Hebbian outer products (rank-16) are less expressive than full backprop. May need many steps to learn what gradient descent gets in one.

**Solutions**:
- Higher-rank updates (rank-64 or full-rank small matrices)
- Multiple Hebbian steps per experience (within-episode replay)
- Hippocampal replay already implemented — could replay more aggressively
- Brain analogy: synapses update continuously, not once per experience

## 4. Meta-Training Distribution (Highest Priority)

**Problem**: Nuclei only trained on arithmetic. They learn arithmetic-shaped learning strategies, not general-purpose ones. Architecture supports generality but training is narrow.

**Solutions**:
- Diverse meta-training curriculum beyond arithmetic:
  - Code completion (reward: compiles / passes tests)
  - Text classification (reward: correct label)
  - Translation (reward: BLEU score)
  - Reasoning / QA (reward: correct answer)
  - Summarization (reward: ROUGE or human preference)
  - Pattern completion (reward: exact match)
  - Few-shot classification (reward: correct on query)
- Each task provides a reward signal; same Hebbian machinery adapts
- Nuclei learn domain-general learning strategies from diverse experience
- Brain analogy: evolution trained on everything life threw at organisms

**Priority**: This is the highest leverage change. The architecture is already general — the training is what's narrow. A nucleus trained on 100 task types will generalize to task 101 far better than one trained only on arithmetic.

## Architecture Summary (Current)

| Component | Brain Equivalent | Params |
|---|---|---|
| Frozen Qwen 3-4B | Cortex | 0 (frozen) |
| 4 Hebbian patches | Synaptic plasticity | ~2M buffers |
| Activation Hebbian rule | Three-factor learning | ~500K |
| Critic GRU | Brainstem integration | ~8K |
| Value head (VTA) | Dopamine RPE | ~65 |
| Eta nucleus | Dopamine modulation | ~600 |
| Decay nucleus | ACh retention | ~600 |
| Exploration nucleus | Locus coeruleus / NE | ~600 |
| PFC context gates | Selective disinhibition | ~178K |
| Hippocampus | Episodic memory | 0 (numpy) |
| **Total trainable** | | **~1.96M** |
