# Phase 2b Results — Mixed-Difficulty Episodes

**Date**: 2026-03-03
**Hardware**: NVIDIA A100 (80GB)
**Base model**: Qwen/Qwen3-4B (frozen, gradient checkpointing)
**Meta-learner params**: 920,072
**Config**: lr=3e-4, 1000 episodes, 20 problems/episode, mixed d5/d6/d7
**Checkpoint**: Loaded from Phase 2 (`checkpoints/phase2_mach.pt`)

## Key Change from Phase 2

Each episode contains problems sampled randomly from d5, d6, and d7 (instead of a single difficulty per curriculum stage). A static write strategy can't simultaneously optimize for 2x2 and 3x3 multiplication.

## Evaluation Results (10 episodes x 20 problems per difficulty, every 200 episodes)

| Episode | Difficulty | Base | Early (0-4) | Late (15-19) | Delta | vs Base |
|---------|-----------|------|-------------|--------------|-------|---------|
| 200 | d5 | 81% | 96% | 94% | -2% | +13pp |
| 200 | d6 | 70% | 92% | 88% | -4% | +18pp |
| 200 | d7 | 9% | 36% | 40% | +4% | +31pp |
| 400 | d5 | 86% | 96% | 98% | +2% | +12pp |
| 400 | d6 | 62% | 82% | 84% | +2% | +22pp |
| 400 | d7 | 7% | 44% | 48% | +4% | +41pp |
| **600** | **d5** | **89%** | **98%** | **98%** | **+0%** | **+9pp** |
| **600** | **d6** | **69%** | **92%** | **88%** | **-4%** | **+19pp** |
| **600** | **d7** | **7%** | **38%** | **52%** | **+14%** | **+45pp** |
| 800 | d5 | 83% | 96% | 98% | +2% | +15pp |
| 800 | d6 | 61% | 88% | 86% | -2% | +25pp |
| 800 | d7 | 6% | 38% | 46% | +8% | +40pp |

## Within-Episode Learning Delta on d7

| Episode | Delta | Note |
|---------|-------|------|
| 200 | +4% | Early signal |
| 400 | +4% | Holding |
| **600** | **+14%** | **Peak — crosses 10pp threshold** |
| 800 | +8% | Partial regression toward static strategy |

## Comparison with Phase 2

| Metric | Phase 2 | Phase 2b |
|--------|---------|----------|
| Best delta (any difficulty) | +24% (transient, ep1600 d9) | +14% (ep600 d7) |
| Sustained delta at convergence | 0% (all difficulties) | +8% on d7 |
| Delta collapses to 0? | Yes, by ep1800 | No — held +4-14% across 600 episodes |
| d7 vs base improvement | +25pp | +40pp |

**Phase 2b verdict**: Mixed difficulties sustain within-episode learning on d7. The +14% peak at ep600 crosses the strict 10pp threshold. Unlike Phase 2, the delta does not fully collapse — it settled at +8% at ep800. Write mechanism remains strong (+40pp on d7 over base).

## Phase 2b+undetach Results (Observation Path Gradient)

Same setup but with `--undetach-obs`: gradient flows through obs_proj and GRU.

| Episode | Difficulty | Base | Early | Late | Delta | 2b Delta (detached) |
|---------|-----------|------|-------|------|-------|---------------------|
| 200 | d7 | 6% | 40% | 44% | +4% | +4% |
| 400 | d7 | 3% | 52% | 44% | -8% | +4% |
| 600 | d7 | 7% | 44% | 36% | -8% | **+14%** |

**Verdict: undetaching hurts.** d7 delta went negative (-8%) while detached 2b peaked at +14%.

### Why It Failed (Diagnostics at ep600)

Gradient norms with undetach:
- obs_proj: 0.000134 (barely any signal despite undetaching)
- gru: 0.000034 (weaker than detached 2b's 0.001)
- action_head: 0.045, basis: 0.131 (1000x stronger)

The gradient path from loss → action_head → transformer → token[0] → GRU → obs_proj is too deep — the signal vanishes. Worse, the small noisy gradient interferes with the indirect signal the GRU was getting before.

**Conclusion**: The observation path needs a *direct* loss signal, not diluted backprop through the write path. This is exactly what the critic (Phase 3) provides.

## Next Steps

1. **Phase 3**: Critic provides loss directly on transformer hidden states, creating a short gradient path that forces observation-dependent behavior
2. Phase 3 uses mixed difficulties by default (lesson from 2b)
3. `--undetach-obs` flag available on Phase 3 if critic provides enough gradient for obs_proj to learn
