# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MACH (Meta-learned Adaptive Cognitive Hierarchy) is a ~5M parameter neural network wrapper around a frozen Qwen2.5-4B language model. It gives the frozen LLM the ability to learn new computation from user feedback at inference time by writing into small "cortical patch" MLPs injected into the LLM's residual stream.

The full design spec is in `mach_v1_implementation_spec_v2.md`.

## Setup and Commands

```bash
# Install dependencies
pip install torch transformers accelerate safetensors

# Download the base model
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-4B')"

# Run Phase 1 (baseline evaluation only)
python scripts/run_phase1.py --baseline-only

# Run Phase 1 (direct patch training)
python scripts/run_phase1.py

# Run Phase 2 (meta-learning)
python scripts/run_phase2.py
```

Hardware target: NVIDIA A100 (80GB VRAM). Qwen runs in float16; patch parameters use float32.

## Architecture

### Phase 1: Direct Patch Training

Validates that small MLPs (CorticalPatch) injected at ~25%, 50%, 75%, and near-final layers of frozen Qwen can learn arithmetic via direct backpropagation. Patches add to the residual stream via forward hooks. Success = >10pp accuracy improvement over baseline on at least one difficulty level.

### Phase 2: Meta-Learner Training

Validates the core write mechanism — a meta-learner that produces patch weight modifications through basis vector outer products:

```
Observation: Qwen hidden states → ObservationProjection → SimpleGRU (detached, no gradient)
Firing:      GRU memory + reward signals → MetaLearnerTransformer → ActionHead → coefficients + gates
Writing:     BasisVectors.compute_delta_W(coefficients, gate) → DifferentiablePatch.accumulate_write()
Evaluation:  Qwen forward WITH modified patches → cross-entropy loss → backprop through write path
```

The gradient path flows: loss → Qwen output → DifferentiablePatch (using base + delta_W) → BasisVectors.compute_delta_W (einsum) → ActionHead → MetaLearnerTransformer. The observation path is detached — the meta-learner gets gradient only about what to write, not what to observe.

Success = late-episode accuracy exceeds early-episode accuracy by >=10pp (within-episode learning).

### Key Design Constraints

- **Qwen is always frozen** — all base model parameters have `requires_grad = False`
- **DifferentiablePatch** base weights are fixed at zero; all computation comes from accumulated delta tensors that remain in the computational graph
- **GATE_SCALE = 0.1** — sigmoid output is multiplied by this to keep initial writes small
- **Phase 2 is simplified** — no cerebellum, no critic, no planning loop, no surprise gating; fixed firing cadence (once per problem)

## File Structure

```
config.py                    # All hyperparameters (D_META=128, N_BASIS=8, etc.)
models/
    patches.py               # CorticalPatch (Phase 1), DifferentiablePatch (Phase 2)
    basis_vectors.py         # BasisVectors — rank-1 outer products for patch writes
    gru.py                   # SimpleGRU (Phase 2), SurpriseGatedGRU (Phase 4+)
    meta_learner.py          # MetaLearnerTransformer, TransformerBlock
    action_head.py           # ActionHead — think_0 → patch write coefficients
    memory_head.py           # MemoryHead — think_1 → transformer memory update
    observation.py           # ObservationProjection (d_model → d_meta)
    reward_projection.py     # RewardProjection (Phase 2 placeholder for critic)
    universal_module.py      # MACHPhase2, full UniversalModule (later phases)
data/arithmetic.py           # generate_arithmetic_problems, extract_number
training/
    phase1_direct.py         # Direct backprop patch training
    phase2_meta_train.py     # Meta-training outer loop
    episode.py               # run_episode (single meta-training episode)
evaluation/
    baseline.py              # Evaluate base Qwen
    evaluate.py              # evaluate_model, compare results
    ablations.py             # Random writes baseline, etc.
scripts/
    run_phase1.py            # Phase 1 entry point
    run_phase2.py            # Phase 2 entry point
```

## Phase Roadmap (Post Phase 2)

- **Phase 3**: Add Critic (basal ganglia — TD learning, evaluates meta-learner hidden states)
- **Phase 4**: Add Cerebellum (prediction error, surprise gating on GRU, correction vector)
- **Phase 5**: Surprise-triggered firing (variable cadence replaces fixed)
- **Phase 6**: Planning loop (multi-iteration critic-gated proposals)
- **Phase 7**: Sleep (trajectory replay, hindsight critic, patch consolidation)
- **Phase 8**: Slow module (second universal module writing into both Qwen patches and fast module weights)
