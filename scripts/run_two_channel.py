#!/usr/bin/env python3
"""Two-Channel: Modulation (Channel 1) + Writing (Channel 2) architecture."""

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
from models.universal_module import MACHTwoChannel, TwoChannelPatchedModel
from training.two_channel_train import (
    meta_train_two_channel,
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


def create_mach_two_channel(d_model, n_layers, n_patch_layers=None,
                            d_obs=None, d_task=None, n_basis=None,
                            n_prims=None, write_cost_scale=None):
    if n_patch_layers is None:
        n_patch_layers = config.PHASE5_N_PATCH_LAYERS
    if d_obs is None:
        d_obs = 96
    if d_task is None:
        d_task = config.PHASE5_D_TASK
    if n_basis is None:
        n_basis = config.N_BASIS
    if n_prims is None:
        n_prims = config.TWO_CHANNEL_N_PRIMS
    if write_cost_scale is None:
        write_cost_scale = config.TWO_CHANNEL_WRITE_COST_SCALE

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
        patch_layers = [min(l, n_layers - 2) for l in patch_layers]

    # Ensure d_obs is divisible by n_patch_layers
    if d_obs % len(patch_layers) != 0:
        d_obs = (d_obs // len(patch_layers) + 1) * len(patch_layers)
        print(f"  Adjusted d_obs to {d_obs} (divisible by {len(patch_layers)} layers)")

    print(f"Patch layers ({len(patch_layers)}): {patch_layers}")
    print(f"d_obs={d_obs}, d_task={d_task}, n_basis={n_basis}, n_prims={n_prims}")
    print(f"write_cost_scale={write_cost_scale}")

    mach = MACHTwoChannel(
        d_model=d_model,
        n_layers=n_layers,
        patch_layers=patch_layers,
        hidden_dim=config.PATCH_HIDDEN_DIM,
        d_obs=d_obs,
        d_task=d_task,
        n_basis=n_basis,
        n_prims=n_prims,
        write_cost_scale=write_cost_scale,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in mach.parameters())
    print(f"MACH Two-Channel total parameters: {n_params:,}")

    return mach, patch_layers


def main():
    parser = argparse.ArgumentParser(
        description="Two-Channel: Modulation + Writing architecture"
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--task", type=str, default="continuous_linear",
        choices=["continuous_linear", "token_map", "mixed"],
    )
    parser.add_argument("--d-task", type=int, default=None)
    parser.add_argument("--d-obs", type=int, default=96)
    parser.add_argument("--n-basis", type=int, default=None)
    parser.add_argument("--n-patch-layers", type=int, default=None)
    parser.add_argument("--n-prims", type=int, default=None)
    parser.add_argument("--write-cost-scale", type=float, default=None)
    parser.add_argument("--energy-beta", type=float, default=None)
    args = parser.parse_args()

    if args.task == "token_map":
        curriculum = TOKEN_MAP_CURRICULUM
    elif args.task == "mixed":
        curriculum = MIXED_CURRICULUM
    else:
        curriculum = CONTINUOUS_LINEAR_CURRICULUM

    d_task_actual = args.d_task or config.PHASE5_D_TASK
    n_basis_actual = args.n_basis or config.N_BASIS
    n_patch_layers_actual = args.n_patch_layers or config.PHASE5_N_PATCH_LAYERS
    n_prims_actual = args.n_prims or config.TWO_CHANNEL_N_PRIMS
    run_name = (
        f"twochan-{args.task}-d{d_task_actual}"
        f"-L{n_patch_layers_actual}-B{n_basis_actual}-P{n_prims_actual}"
    )

    if wandb is not None:
        wandb.init(
            project="mach",
            name=run_name,
            config={
                "base_model": config.BASE_MODEL,
                "d_obs": args.d_obs,
                "d_task": d_task_actual,
                "n_basis": n_basis_actual,
                "n_patch_layers": n_patch_layers_actual,
                "n_prims": n_prims_actual,
                "write_cost_scale": args.write_cost_scale or config.TWO_CHANNEL_WRITE_COST_SCALE,
                "lr": args.lr or config.PHASE5_LR,
                "episodes": args.episodes or config.PHASE5_EPISODES,
                "energy_beta": args.energy_beta or config.TWO_CHANNEL_ENERGY_BETA,
                "architecture": "two_channel",
                "task": args.task,
                "device": str(config.DEVICE),
            },
        )

    base_model, tokenizer, d_model, n_layers = load_base_model()
    mach, patch_layers = create_mach_two_channel(
        d_model, n_layers,
        n_patch_layers=args.n_patch_layers,
        d_obs=args.d_obs,
        d_task=args.d_task,
        n_basis=args.n_basis,
        n_prims=args.n_prims,
        write_cost_scale=args.write_cost_scale,
    )
    patched_model = TwoChannelPatchedModel(base_model, mach)

    save_path = (
        f"checkpoints/twochan_{args.task}_d{d_task_actual}"
        f"_L{n_patch_layers_actual}_B{n_basis_actual}_P{n_prims_actual}.pt"
    )
    os.makedirs("checkpoints", exist_ok=True)

    meta_train_two_channel(
        base_model, mach, patched_model, tokenizer, config.DEVICE,
        n_episodes=args.episodes, lr=args.lr,
        checkpoint_path=args.checkpoint,
        save_path=save_path,
        curriculum=curriculum,
        energy_beta=args.energy_beta,
    )

    torch.save(mach.state_dict(), save_path)
    print(f"\nSaved final checkpoint to {save_path}")

    if wandb is not None:
        wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
