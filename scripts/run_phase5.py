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


def create_mach_phase5(d_model, n_layers):
    patch_layers = [
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 2,
    ]
    print(f"Patch layers: {patch_layers}")
    print(f"d_obs={config.PHASE5_D_OBS}, d_gru={config.PHASE5_D_GRU}, "
          f"d_task={config.PHASE5_D_TASK}")

    mach = MACHPhase5(
        d_model=d_model,
        n_layers=n_layers,
        patch_layers=patch_layers,
        hidden_dim=config.PATCH_HIDDEN_DIM,
        d_obs=config.PHASE5_D_OBS,
        d_gru=config.PHASE5_D_GRU,
        d_task=config.PHASE5_D_TASK,
        n_basis=config.N_BASIS,
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
        choices=["few_shot", "linear"],
        help="Task type: few_shot (7 ops) or linear (6 train combos)"
    )
    parser.add_argument(
        "--sparsity-beta", type=float, default=None,
        help="L1 sparsity penalty weight on task state"
    )
    args = parser.parse_args()

    curriculum = LINEAR_CURRICULUM if args.task == "linear" else DEFAULT_CURRICULUM
    run_name = f"phase5-{args.task}"

    if wandb is not None:
        wandb.init(
            project="mach",
            name=run_name,
            config={
                "base_model": config.BASE_MODEL,
                "d_obs": config.PHASE5_D_OBS,
                "d_gru": config.PHASE5_D_GRU,
                "d_task": config.PHASE5_D_TASK,
                "n_basis": config.N_BASIS,
                "sparsity_beta": args.sparsity_beta or config.PHASE5_SPARSITY_BETA,
                "lr": args.lr or config.PHASE5_LR,
                "episodes": args.episodes or config.PHASE5_EPISODES,
                "architecture": "phase5",
                "task": args.task,
                "device": str(config.DEVICE),
            },
        )

    base_model, tokenizer, d_model, n_layers = load_base_model()
    mach, patch_layers = create_mach_phase5(d_model, n_layers)
    patched_model = MACHPatchedModel(base_model, mach)

    save_path = f"checkpoints/phase5_{args.task}.pt"
    os.makedirs("checkpoints", exist_ok=True)

    meta_train_phase5(
        base_model, mach, patched_model, tokenizer, config.DEVICE,
        n_episodes=args.episodes, lr=args.lr,
        checkpoint_path=args.checkpoint,
        save_path=save_path,
        curriculum=curriculum,
        sparsity_beta=args.sparsity_beta,
    )

    torch.save(mach.state_dict(), save_path)
    print(f"\nSaved final checkpoint to {save_path}")

    if wandb is not None:
        wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
