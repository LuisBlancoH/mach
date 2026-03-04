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
    CONTINUOUS_LINEAR_CURRICULUM,
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
                       d_task=None, task_noise=0.0, multi_layer_obs=False):
    if d_task is None:
        d_task = config.PHASE5_D_TASK
    patch_layers = [
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 2,
    ]
    print(f"Patch layers: {patch_layers}")
    print(f"d_obs={config.PHASE5_D_OBS}, d_gru={config.PHASE5_D_GRU}, "
          f"d_task={d_task}, "
          f"deliberation_steps={n_deliberation_steps}, "
          f"task_noise={task_noise}, "
          f"multi_layer_obs={multi_layer_obs}")

    mach = MACHPhase5(
        d_model=d_model,
        n_layers=n_layers,
        patch_layers=patch_layers,
        hidden_dim=config.PATCH_HIDDEN_DIM,
        d_obs=config.PHASE5_D_OBS,
        d_gru=config.PHASE5_D_GRU,
        d_task=d_task,
        n_basis=config.N_BASIS,
        n_deliberation_steps=n_deliberation_steps,
        task_noise=task_noise,
        multi_layer_obs=multi_layer_obs,
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
        choices=["few_shot", "linear", "continuous_linear"],
        help="Task type: few_shot (7 ops), linear (6 train combos), "
             "continuous_linear (random coefficients 0-5)"
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
    args = parser.parse_args()

    if args.task == "continuous_linear":
        curriculum = CONTINUOUS_LINEAR_CURRICULUM
    elif args.task == "linear":
        curriculum = LINEAR_CURRICULUM
    else:
        curriculum = DEFAULT_CURRICULUM

    d_task_actual = args.d_task or config.PHASE5_D_TASK
    run_name = f"phase5-{args.task}-d{d_task_actual}"

    if wandb is not None:
        wandb.init(
            project="mach",
            name=run_name,
            config={
                "base_model": config.BASE_MODEL,
                "d_obs": config.PHASE5_D_OBS,
                "d_gru": config.PHASE5_D_GRU,
                "d_task": d_task_actual,
                "n_basis": config.N_BASIS,
                "sparsity_beta": args.sparsity_beta or config.PHASE5_SPARSITY_BETA,
                "decorr_beta": args.decorr_beta or config.PHASE5_DECORR_BETA,
                "lr": args.lr or config.PHASE5_LR,
                "episodes": args.episodes or config.PHASE5_EPISODES,
                "architecture": "phase5",
                "deliberation_steps": args.deliberation_steps or config.PHASE5_N_DELIBERATION_STEPS,
                "task_noise": args.task_noise or config.PHASE5_TASK_NOISE,
                "energy_beta": args.energy_beta or config.PHASE5_ENERGY_BETA,
                "multi_layer_obs": args.multi_layer_obs,
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
    )
    patched_model = MACHPatchedModel(base_model, mach)

    save_path = f"checkpoints/phase5_{args.task}_d{d_task_actual}.pt"
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
    )

    torch.save(mach.state_dict(), save_path)
    print(f"\nSaved final checkpoint to {save_path}")

    if wandb is not None:
        wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
