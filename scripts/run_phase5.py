#!/usr/bin/env python3
"""Phase 5: Brain-like meta-learner with task bottleneck."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

try:
    import wandb
except ImportError:
    wandb = None

from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from models.universal_module import MACHPhase5, MACHPatchedModel
from training.phase5_train import (
    meta_train_phase5, DEFAULT_CURRICULUM, LINEAR_CURRICULUM,
    CONTINUOUS_LINEAR_CURRICULUM, TOKEN_MAP_CURRICULUM, MIXED_CURRICULUM,
)


def load_base_model():
    print(f"Loading {config.BASE_MODEL} on {config.DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, dtype=config.DTYPE
    ).to(config.DEVICE)

    for param in model.parameters():
        param.requires_grad = False
    model.gradient_checkpointing_enable()

    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  d_model={d_model}, n_layers={n_layers}")

    return model, tokenizer, d_model, n_layers


def create_mach_phase5(d_model, n_layers, n_deliberation_steps=0,
                       d_task=None, task_noise=0.0, multi_layer_obs=False,
                       n_patch_layers=None, n_basis=None, d_obs=None,
                       consolidation=False, ema_decay=None):
    if d_task is None:
        d_task = config.PHASE5_D_TASK
    if n_patch_layers is None:
        n_patch_layers = config.PHASE5_N_PATCH_LAYERS
    if n_basis is None:
        n_basis = config.N_BASIS
    if d_obs is None:
        d_obs = config.PHASE5_D_OBS
    if ema_decay is None:
        ema_decay = config.PHASE5_EMA_DECAY

    # Generate evenly-spaced patch layers
    if n_patch_layers == 4:
        patch_layers = [
            n_layers // 4,
            n_layers // 2,
            3 * n_layers // 4,
            n_layers - 2,
        ]
    else:
        step = n_layers // n_patch_layers
        patch_layers = [i * step for i in range(n_patch_layers)]
        # Ensure last layer isn't the very last (output layer)
        patch_layers = [min(l, n_layers - 2) for l in patch_layers]

    # Auto-enable multi-layer obs when using more than 4 patch layers
    if n_patch_layers > 4:
        multi_layer_obs = True

    # Auto-adjust d_obs to be divisible by n_patch_layers when multi-layer
    if multi_layer_obs and d_obs % len(patch_layers) != 0:
        d_obs = (d_obs // len(patch_layers) + 1) * len(patch_layers)
        print(f"  Adjusted d_obs to {d_obs} (divisible by {len(patch_layers)} layers)")

    d_gru = d_obs  # GRU hidden matches obs dim

    print(f"Patch layers ({len(patch_layers)}): {patch_layers}")
    print(f"d_obs={d_obs}, d_gru={d_gru}, d_task={d_task}, n_basis={n_basis}, "
          f"deliberation_steps={n_deliberation_steps}, "
          f"task_noise={task_noise}, multi_layer_obs={multi_layer_obs}, "
          f"consolidation={consolidation}")

    mach = MACHPhase5(
        d_model=d_model,
        n_layers=n_layers,
        patch_layers=patch_layers,
        hidden_dim=config.PATCH_HIDDEN_DIM,
        d_obs=d_obs,
        d_gru=d_gru,
        d_task=d_task,
        n_basis=n_basis,
        n_deliberation_steps=n_deliberation_steps,
        task_noise=task_noise,
        multi_layer_obs=multi_layer_obs,
        consolidation=consolidation,
        ema_decay=ema_decay,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in mach.parameters())
    print(f"MACH Phase 5 total parameters: {n_params:,}")

    return mach, patch_layers


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Brain-like meta-learner"
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--task", type=str, default="few_shot",
        choices=["few_shot", "linear", "continuous_linear",
                 "token_map", "mixed"],
        help="Task type: few_shot (7 ops), linear (6 train combos), "
             "continuous_linear (random coefficients 0-5), "
             "token_map (random symbol substitution), "
             "mixed (random mix of continuous_linear + token_map)"
    )
    parser.add_argument(
        "--sparsity-beta", type=float, default=None,
        help="L1 sparsity penalty weight on task state"
    )
    parser.add_argument(
        "--decorr-beta", type=float, default=None,
        help="Decorrelation (lateral inhibition) loss weight"
    )
    parser.add_argument(
        "--deliberation-steps", type=int, default=None,
        help="Number of deliberation steps (0=none, 3+=iterative refinement)"
    )
    parser.add_argument(
        "--d-task", type=int, default=None,
        help="Task state dimensionality (default: 32, try 4-8 to force composition)"
    )
    parser.add_argument(
        "--task-noise", type=float, default=None,
        help="Noise std on task state for forgetting regularization"
    )
    parser.add_argument(
        "--energy-beta", type=float, default=None,
        help="Unified metabolic cost weight (replaces sparsity + decorr)"
    )
    parser.add_argument(
        "--self-eval-steps", type=int, default=None,
        help="Self-evaluation rounds: observe own patched output on demos"
    )
    parser.add_argument(
        "--multi-layer-obs", action="store_true",
        help="Observe all 4 patch layers instead of just the middle one"
    )
    parser.add_argument(
        "--n-patch-layers", type=int, default=None,
        help="Number of evenly-spaced patch layers (4=default, 12=searchable)"
    )
    parser.add_argument(
        "--n-basis", type=int, default=None,
        help="Number of basis vectors per patch (8=default, 16=searchable)"
    )
    parser.add_argument(
        "--d-obs", type=int, default=None,
        help="Observation projection dimension (64=default, 96=searchable)"
    )
    parser.add_argument(
        "--td-modulation", type=float, default=None,
        help="TD-weighted CE loss: critic modulates gradient (0=off, 0.5=moderate)"
    )
    parser.add_argument(
        "--critic-beta", type=float, default=None,
        help="Critic loss weight (default: 0.1)"
    )
    parser.add_argument(
        "--satisfaction-threshold", type=float, default=None,
        help="Critic value above which self-eval stops early (default: 0.5)"
    )
    parser.add_argument(
        "--consolidation", action="store_true",
        help="Enable cross-episode slow memory (consolidation/sleep)"
    )
    parser.add_argument(
        "--ema-decay", type=float, default=None,
        help="Slow memory EMA decay rate (default: 0.95)"
    )
    args = parser.parse_args()

    if args.task == "continuous_linear":
        curriculum = CONTINUOUS_LINEAR_CURRICULUM
    elif args.task == "linear":
        curriculum = LINEAR_CURRICULUM
    elif args.task == "token_map":
        curriculum = TOKEN_MAP_CURRICULUM
    elif args.task == "mixed":
        curriculum = MIXED_CURRICULUM
    else:
        curriculum = DEFAULT_CURRICULUM

    d_task_actual = args.d_task or config.PHASE5_D_TASK
    n_basis_actual = args.n_basis or config.N_BASIS
    n_patch_layers_actual = args.n_patch_layers or config.PHASE5_N_PATCH_LAYERS
    d_obs_actual = args.d_obs or config.PHASE5_D_OBS
    run_name = f"phase5-{args.task}-d{d_task_actual}-L{n_patch_layers_actual}-B{n_basis_actual}"

    if wandb is not None:
        wandb.init(
            project="mach",
            name=run_name,
            config={
                "base_model": config.BASE_MODEL,
                "d_obs": d_obs_actual,
                "d_gru": d_obs_actual,
                "d_task": d_task_actual,
                "n_basis": n_basis_actual,
                "n_patch_layers": n_patch_layers_actual,
                "sparsity_beta": args.sparsity_beta or config.PHASE5_SPARSITY_BETA,
                "decorr_beta": args.decorr_beta or config.PHASE5_DECORR_BETA,
                "lr": args.lr or config.PHASE5_LR,
                "episodes": args.episodes or config.PHASE5_EPISODES,
                "architecture": "phase5",
                "deliberation_steps": args.deliberation_steps or config.PHASE5_N_DELIBERATION_STEPS,
                "task_noise": args.task_noise or config.PHASE5_TASK_NOISE,
                "energy_beta": args.energy_beta or config.PHASE5_ENERGY_BETA,
                "multi_layer_obs": args.multi_layer_obs,
                "td_modulation": args.td_modulation or config.PHASE5_TD_MODULATION,
                "critic_beta": args.critic_beta or config.PHASE5_CRITIC_BETA,
                "consolidation": args.consolidation,
                "ema_decay": args.ema_decay or config.PHASE5_EMA_DECAY,
                "task": args.task,
                "device": str(config.DEVICE),
            },
        )

    n_delib = args.deliberation_steps if args.deliberation_steps is not None \
        else config.PHASE5_N_DELIBERATION_STEPS
    task_noise = args.task_noise if args.task_noise is not None \
        else config.PHASE5_TASK_NOISE

    multi_layer_obs = args.multi_layer_obs or config.PHASE5_MULTI_LAYER_OBS

    base_model, tokenizer, d_model, n_layers = load_base_model()
    mach, patch_layers = create_mach_phase5(
        d_model, n_layers, n_delib, d_task=args.d_task, task_noise=task_noise,
        multi_layer_obs=multi_layer_obs,
        n_patch_layers=args.n_patch_layers, n_basis=args.n_basis,
        d_obs=args.d_obs,
        consolidation=args.consolidation, ema_decay=args.ema_decay,
    )
    patched_model = MACHPatchedModel(base_model, mach)

    save_path = f"checkpoints/phase5_{args.task}_d{d_task_actual}_L{n_patch_layers_actual}_B{n_basis_actual}.pt"
    os.makedirs("checkpoints", exist_ok=True)

    meta_train_phase5(
        base_model, mach, patched_model, tokenizer, config.DEVICE,
        n_episodes=args.episodes, lr=args.lr,
        checkpoint_path=args.checkpoint,
        save_path=save_path,
        curriculum=curriculum,
        sparsity_beta=args.sparsity_beta,
        decorr_beta=args.decorr_beta,
        energy_beta=args.energy_beta,
        n_self_eval_steps=args.self_eval_steps,
        td_modulation=args.td_modulation,
        critic_beta=args.critic_beta,
        satisfaction_threshold=args.satisfaction_threshold,
    )

    torch.save(mach.state_dict(), save_path)
    print(f"\nSaved final checkpoint to {save_path}")

    if wandb is not None:
        wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
