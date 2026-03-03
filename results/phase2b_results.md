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

## Next Experiment

Phase 2b with `--undetach-obs`: allow gradient through obs_proj and GRU so the meta-learner learns WHAT to observe, not just what to write. Currently obs_proj is randomly initialized and never trained — the meta-learner's observation is noise projected through an untrained linear layer.

```bash
python scripts/run_phase2b.py --checkpoint checkpoints/phase2_mach.pt --undetach-obs
```
