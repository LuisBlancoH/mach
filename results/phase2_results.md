# Phase 2 Results — Meta-Learner Training

**Date**: 2026-03-03
**Hardware**: NVIDIA A100 (80GB)
**Base model**: Qwen/Qwen3-4B (frozen, gradient checkpointing)
**Meta-learner params**: 920,072
**Config**: lr=3e-4, 2000 episodes, progressive length (5→10→15→20 problems)
**Curriculum**: d5 (ep0-500) → d6 (ep500-1000) → d7 (ep1000-1500) → d9 (ep1500-2000)

## Final Evaluation (20 episodes × 20 problems per difficulty)

| Difficulty | Description | Base Qwen | Early (0-4) | Late (15-19) | Delta | vs Base |
|-----------|-------------|-----------|-------------|--------------|-------|---------|
| 5 | 2x2 multiplication | 86.8% | 95.0% | 99.0% | +4.0pp | +12.2pp |
| 6 | 3x2 multiplication | 65.8% | 87.0% | 91.0% | +4.0pp | +25.3pp |
| 7 | 3x3 multiplication | 6.5% | 35.0% | 32.0% | -3.0pp | +25.5pp |
| 9 | mixed hard | 60.2% | 79.0% | 77.0% | -2.0pp | +16.8pp |

**Ablation**: Random writes accuracy = 68% on d6 (base = 66%, learned = 87-91%).

**Strict verdict: FAIL** — no difficulty shows late - early ≥ 10pp at convergence.

**Qualified verdict: WRITE MECHANISM VALIDATED** — meta-learner writes improve Qwen by 12-25pp across all difficulties. Random writes give +2pp; learned writes give +25pp.

## Within-Episode Learning (Transient)

During training, within-episode learning appeared on first exposure to new difficulties:

| Eval | Difficulty | Base | Early | Late | Delta | Note |
|------|-----------|------|-------|------|-------|------|
| ep200 | d5 | 88% | 96% | 92% | -4% | Already saturated |
| ep400 | d5 | 86% | 96% | 96% | 0% | Static strategy converged |
| ep600 | d6 | 57% | 78% | 86% | +8% | First exposure to d6 |
| ep800 | d6 | 56% | 82% | 88% | +6% | Early catching up |
| ep1000 | d7 | 10% | 40% | 48% | +8% | First exposure to d7 |
| ep1200 | d7 | 12% | 30% | 36% | +6% | Gradient dip |
| ep1400 | d7 | 7% | 52% | 54% | +2% | Static strategy converged |
| **ep1600** | **d9** | **53%** | **58%** | **82%** | **+24%** | **First exposure to mixed** |
| ep1800 | d9 | 61% | 78% | 78% | 0% | Static strategy converged |

**Key finding**: +24pp within-episode learning at ep1600 on first exposure to d9 (mixed). The meta-learner CAN adapt within an episode when the task is novel. With continued training, it converges to a static strategy that front-loads the solution.

## Diagnostics

### Basis vector norms (grew from 0.01 init)
- patch0 down_V: 3.75, up_U: 2.79 (largest — early layer patches most active)
- patch2: smallest norms (~1.0-2.1)

### Gradient hierarchy
- action_head, basis: 0.04-0.15 (strongest — direct write path)
- transformer: 0.006-0.020
- gru, memory_head, reward_proj: 0.001-0.005 (weakest — observation path mostly detached)

### Patch delta norms (per episode, 20 problems)
- Range 0.9-2.2 across patches
- Patch 0 and 1 (layers 9, 18) have largest deltas

## Interpretation

The meta-learner learned to produce a single excellent set of write coefficients that it applies from the first firing. It does not meaningfully adapt its writes based on within-episode observations or rewards. The weak gradient signal through the observation path (gru: 0.001) provides insufficient incentive to condition writes on state.

Within-episode learning appeared transiently on first exposure to new task distributions (d9 mixed: +24pp). This proves the architecture CAN support adaptation, but the meta-learner quickly finds a static strategy that eliminates the need.

## Next Steps

1. **Phase 2b**: Mixed d5/d6/d7 within every episode to prevent static convergence. Script ready: `python scripts/run_phase2b.py --checkpoint checkpoints/phase2_mach.pt`
2. **Phase 3**: Add critic (basal ganglia) — provides a learned value signal conditioned on meta-learner state, giving stronger incentive to adapt writes to observations.
3. **Undetach observation**: Allow gradient through GRU/obs_proj so the meta-learner learns WHAT to observe, not just what to write.
