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
from models.universal_module import (
    MACHTwoChannel, TwoChannelPatchedModel,
    MACHDemoRead, MACHOracleMinimal, IterativePatchedModel,
    MACHHebbian, HebbianPatchedModel,
    MACHActivationHebbian, ActivationHebbianPatchedModel,
    MACHDualHebbian, DualHebbianPatchedModel,
    MACHCoprocessor, CoprocessorPatchedModel,
    MACHDenseHebbian, DenseHebbianPatchedModel,
)
from training.two_channel_train import (
    meta_train_two_channel, meta_train_demoread, meta_train_hebbian,
    ablate_hebbian,
    CONTINUOUS_LINEAR_CURRICULUM, TOKEN_MAP_CURRICULUM, MIXED_CURRICULUM,
    FEW_SHOT_BASIC_CURRICULUM, DIVERSE_OPS_CURRICULUM,
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
        choices=["continuous_linear", "token_map", "mixed", "few_shot_basic", "diverse_ops"],
    )
    parser.add_argument("--d-task", type=int, default=None)
    parser.add_argument("--d-obs", type=int, default=96)
    parser.add_argument("--n-basis", type=int, default=None)
    parser.add_argument("--n-rank", type=int, default=None,
                        help="Hebbian update rank (default: config.HEBBIAN_N_RANK=2)")
    parser.add_argument("--n-patch-layers", type=int, default=None)
    parser.add_argument("--n-prims", type=int, default=None)
    parser.add_argument("--write-cost-scale", type=float, default=None)
    parser.add_argument("--energy-beta", type=float, default=None)
    parser.add_argument("--demoread", action="store_true",
                        help="Use MACHDemoRead (DemoEncoder + skip connection)")
    parser.add_argument("--oracle", action="store_true",
                        help="Oracle mode: feed true [c1,c2] instead of demos")
    parser.add_argument("--oracle-minimal", action="store_true",
                        help="Minimal oracle: Linear(2) → patches, no GRU/transformer")
    parser.add_argument("--hebbian", action="store_true",
                        help="Three-factor Hebbian learning with critic")
    parser.add_argument("--act-hebbian", action="store_true",
                        help="Activation-derived Hebbian (no basis vectors)")
    parser.add_argument("--frozen-proj", action="store_true",
                        help="Freeze hebb_rule projections (reservoir-style)")
    parser.add_argument("--dual-hebbian", action="store_true",
                        help="Dual Hebbian: residual patches + attention output patches")
    parser.add_argument("--coprocessor", action="store_true",
                        help="Coprocessor: direct injection + residual patches")
    parser.add_argument("--dense-hebbian", action="store_true",
                        help="Dense Hebbian: 12 layers + top-down gain modulation")
    parser.add_argument("--consolidation", action="store_true",
                        help="Enable memory consolidation across episodes")
    parser.add_argument("--ema-decay", type=float, default=0.95,
                        help="EMA decay for consolidation (default: 0.95)")
    parser.add_argument("--delta-decay", type=float, default=1.0,
                        help="Decay on accumulated deltas (1.0=no decay, 0.9=EMA). Prevents Hebbian drift.")
    parser.add_argument("--consolidation-interval", type=int, default=0,
                        help="Consolidate every N Hebbian steps (0=episode-based only). Enables continuous learning.")
    parser.add_argument("--cot", action="store_true",
                        help="Chain of thought: model generates thinking tokens before answering")
    parser.add_argument("--max-thinking", type=int, default=32,
                        help="Max thinking tokens for CoT (default: 32)")
    parser.add_argument("--ablate", action="store_true",
                        help="Run Hebbian ablation (requires --hebbian/--act-hebbian --checkpoint)")
    parser.add_argument("--continuous", action="store_true",
                        help="Continuous training: no episodes, no resets, truncated backprop")
    parser.add_argument("--truncation-window", type=int, default=20,
                        help="Truncated backprop window for continuous training (default: 20)")
    parser.add_argument("--n-steps", type=int, default=40000,
                        help="Total steps for continuous training (default: 40000)")
    parser.add_argument("--context-size", type=int, default=0,
                        help="Number of past solved problems as context (0=off, 5-10=typical)")
    parser.add_argument("--thinking-tokens", type=int, default=0,
                        help="Max thinking tokens before answering (0=off, 16-32=typical)")
    args = parser.parse_args()

    if args.task == "token_map":
        curriculum = TOKEN_MAP_CURRICULUM
    elif args.task == "mixed":
        curriculum = MIXED_CURRICULUM
    elif args.task == "few_shot_basic":
        curriculum = FEW_SHOT_BASIC_CURRICULUM
    elif args.task == "diverse_ops":
        curriculum = DIVERSE_OPS_CURRICULUM
    else:
        curriculum = CONTINUOUS_LINEAR_CURRICULUM

    n_basis_actual = args.n_basis or config.N_BASIS
    n_patch_layers_actual = args.n_patch_layers or config.PHASE5_N_PATCH_LAYERS

    base_model, tokenizer, d_model, n_layers = load_base_model()

    if args.coprocessor:
        # MACHCoprocessor: direct injection (read early layer → process → inject at later layer)
        arch_name = "coprocessor"
        n_rank = args.n_rank if args.n_rank is not None else config.HEBBIAN_N_RANK
        d_proj = config.HEBBIAN_D_PROJ
        d_copro = config.COPRO_D_MODEL
        n_copro_layers = config.COPRO_N_LAYERS
        run_name = (
            f"copro-{args.task}"
            f"-L{n_patch_layers_actual}-C{d_copro}x{n_copro_layers}"
        )

        # Generate patch layers
        if n_patch_layers_actual == 4:
            patch_layers = [
                n_layers // 4, n_layers // 2,
                3 * n_layers // 4, n_layers - 2,
            ]
        else:
            step = n_layers // n_patch_layers_actual
            patch_layers = [i * step for i in range(n_patch_layers_actual)]
            patch_layers = [min(l, n_layers - 2) for l in patch_layers]

        print(f"Patch layers ({len(patch_layers)}): {patch_layers}")
        print(f"Coprocessor: d={d_copro}, layers={n_copro_layers}")
        print(f"  read_layer={n_layers // 4}, write_layer={3 * n_layers // 4}")

        mach = MACHCoprocessor(
            d_model=d_model,
            n_layers=n_layers,
            patch_layers=patch_layers,
            hidden_dim=config.PATCH_HIDDEN_DIM,
            copro_hidden_dim=d_copro,
            n_copro_layers=n_copro_layers,
            n_rank=n_rank,
            d_proj=d_proj,
            delta_decay=args.delta_decay,
        ).to(config.DEVICE)

        n_params = sum(p.numel() for p in mach.parameters())
        print(f"MACHCoprocessor total parameters: {n_params:,}")

        patched_model = CoprocessorPatchedModel(base_model, mach)

        save_path = (
            f"checkpoints/copro_{args.task}"
            f"_L{n_patch_layers_actual}_C{d_copro}x{n_copro_layers}.pt"
        )
        os.makedirs("checkpoints", exist_ok=True)

        if wandb is not None:
            wandb.init(
                project="mach",
                name=run_name,
                config={
                    "base_model": config.BASE_MODEL,
                    "d_copro": d_copro,
                    "n_copro_layers": n_copro_layers,
                    "n_rank": n_rank,
                    "d_proj": d_proj,
                    "n_patch_layers": n_patch_layers_actual,
                    "lr": args.lr or config.PHASE5_LR,
                    "episodes": args.episodes or config.PHASE5_EPISODES,
                    "architecture": "coprocessor",
                    "task": args.task,
                    "device": str(config.DEVICE),
                },
            )

        if args.ablate:
            if args.checkpoint:
                mach.load_state_dict(torch.load(args.checkpoint, map_location=config.DEVICE))
                print(f"Loaded checkpoint: {args.checkpoint}")
            mach.eval()
            ablate_hebbian(
                base_model, mach, patched_model, tokenizer, config.DEVICE,
            )
        else:
            meta_train_hebbian(
                base_model, mach, patched_model, tokenizer, config.DEVICE,
                n_episodes=args.episodes, lr=args.lr,
                checkpoint_path=args.checkpoint,
                save_path=save_path,
                curriculum=curriculum,
                chain_of_thought=args.cot,
                max_thinking_tokens=args.max_thinking,
            )

    elif args.dense_hebbian:
        # MACHDenseHebbian: 12+ layers + top-down gain modulation
        arch_name = "dense_hebbian"
        n_rank = args.n_rank if args.n_rank is not None else config.HEBBIAN_N_RANK
        d_proj = config.HEBBIAN_D_PROJ
        dense_hidden = 64  # smaller patches, more layers
        n_dense_layers = n_patch_layers_actual if n_patch_layers_actual >= 8 else 12
        run_name = (
            f"dense-hebbian-{args.task}"
            f"-L{n_dense_layers}-R{n_rank}-P{d_proj}-H{dense_hidden}"
        )

        # Generate dense patch layers (every 3rd layer for 12 layers)
        step = n_layers // n_dense_layers
        patch_layers = [i * step for i in range(n_dense_layers)]
        patch_layers = [min(l, n_layers - 2) for l in patch_layers]

        print(f"Patch layers ({len(patch_layers)}): {patch_layers}")
        print(f"n_rank={n_rank}, d_proj={d_proj}, hidden_dim={dense_hidden}")

        mach = MACHDenseHebbian(
            d_model=d_model,
            n_layers=n_layers,
            patch_layers=patch_layers,
            hidden_dim=dense_hidden,
            n_rank=n_rank,
            d_proj=d_proj,
            consolidation=args.consolidation,
            ema_decay=args.ema_decay,
            delta_decay=args.delta_decay,
            consolidation_interval=args.consolidation_interval,
        ).to(config.DEVICE)

        n_params = sum(p.numel() for p in mach.parameters())
        print(f"MACHDenseHebbian total parameters: {n_params:,}")

        patched_model = DenseHebbianPatchedModel(base_model, mach)

        save_path = (
            f"checkpoints/dense_hebbian_{args.task}"
            f"_L{n_dense_layers}_R{n_rank}_P{d_proj}_H{dense_hidden}.pt"
        )
        os.makedirs("checkpoints", exist_ok=True)

        if wandb is not None:
            wandb.init(
                project="mach",
                name=run_name,
                config={
                    "base_model": config.BASE_MODEL,
                    "n_rank": n_rank,
                    "d_proj": d_proj,
                    "hidden_dim": dense_hidden,
                    "n_patch_layers": n_dense_layers,
                    "lr": args.lr or config.PHASE5_LR,
                    "episodes": args.episodes or config.PHASE5_EPISODES,
                    "architecture": "dense_hebbian",
                    "task": args.task,
                    "device": str(config.DEVICE),
                },
            )

        if args.ablate:
            if args.checkpoint:
                mach.load_state_dict(torch.load(args.checkpoint, map_location=config.DEVICE))
                print(f"Loaded checkpoint: {args.checkpoint}")
            mach.eval()
            ablate_hebbian(
                base_model, mach, patched_model, tokenizer, config.DEVICE,
            )
        else:
            meta_train_hebbian(
                base_model, mach, patched_model, tokenizer, config.DEVICE,
                n_episodes=args.episodes, lr=args.lr,
                checkpoint_path=args.checkpoint,
                save_path=save_path,
                curriculum=curriculum,
                chain_of_thought=args.cot,
                max_thinking_tokens=args.max_thinking,
            )

    elif args.dual_hebbian:
        # MACHDualHebbian: residual patches + attention output patches
        arch_name = "dual_hebbian"
        n_rank = args.n_rank if args.n_rank is not None else config.HEBBIAN_N_RANK
        d_proj = config.HEBBIAN_D_PROJ
        attn_hidden = config.ATTN_PATCH_HIDDEN_DIM
        run_name = (
            f"dual-hebbian-{args.task}"
            f"-L{n_patch_layers_actual}-R{n_rank}-P{d_proj}-A{attn_hidden}"
        )

        # Generate patch layers
        if n_patch_layers_actual == 4:
            patch_layers = [
                n_layers // 4, n_layers // 2,
                3 * n_layers // 4, n_layers - 2,
            ]
        else:
            step = n_layers // n_patch_layers_actual
            patch_layers = [i * step for i in range(n_patch_layers_actual)]
            patch_layers = [min(l, n_layers - 2) for l in patch_layers]

        print(f"Patch layers ({len(patch_layers)}): {patch_layers}")
        print(f"n_rank={n_rank}, d_proj={d_proj}, attn_hidden={attn_hidden}")

        mach = MACHDualHebbian(
            d_model=d_model,
            n_layers=n_layers,
            patch_layers=patch_layers,
            hidden_dim=config.PATCH_HIDDEN_DIM,
            attn_hidden_dim=attn_hidden,
            n_rank=n_rank,
            d_proj=d_proj,
            delta_decay=args.delta_decay,
        ).to(config.DEVICE)

        n_params = sum(p.numel() for p in mach.parameters())
        print(f"MACHDualHebbian total parameters: {n_params:,}")

        patched_model = DualHebbianPatchedModel(base_model, mach)

        save_path = (
            f"checkpoints/dual_hebbian_{args.task}"
            f"_L{n_patch_layers_actual}_R{n_rank}_P{d_proj}_A{attn_hidden}.pt"
        )
        os.makedirs("checkpoints", exist_ok=True)

        if wandb is not None:
            wandb.init(
                project="mach",
                name=run_name,
                config={
                    "base_model": config.BASE_MODEL,
                    "n_rank": n_rank,
                    "d_proj": d_proj,
                    "attn_hidden_dim": attn_hidden,
                    "n_patch_layers": n_patch_layers_actual,
                    "lr": args.lr or config.PHASE5_LR,
                    "episodes": args.episodes or config.PHASE5_EPISODES,
                    "architecture": "dual_hebbian",
                    "task": args.task,
                    "device": str(config.DEVICE),
                },
            )

        if args.ablate:
            if args.checkpoint:
                mach.load_state_dict(torch.load(args.checkpoint, map_location=config.DEVICE))
                print(f"Loaded checkpoint: {args.checkpoint}")
            mach.eval()
            ablate_hebbian(
                base_model, mach, patched_model, tokenizer, config.DEVICE,
            )
        else:
            meta_train_hebbian(
                base_model, mach, patched_model, tokenizer, config.DEVICE,
                n_episodes=args.episodes, lr=args.lr,
                checkpoint_path=args.checkpoint,
                save_path=save_path,
                curriculum=curriculum,
                chain_of_thought=args.cot,
                max_thinking_tokens=args.max_thinking,
            )

    elif args.act_hebbian:
        # MACHActivationHebbian: activation-derived Hebbian (no basis vectors)
        arch_name = "act_hebbian"
        n_rank = args.n_rank if args.n_rank is not None else config.HEBBIAN_N_RANK
        d_proj = 128 if args.frozen_proj else config.HEBBIAN_D_PROJ
        frozen = args.frozen_proj
        suffix = "-frozen" if frozen else ""
        run_name = (
            f"act-hebbian-{args.task}"
            f"-L{n_patch_layers_actual}-R{n_rank}-P{d_proj}{suffix}"
        )

        # Generate patch layers
        if n_patch_layers_actual == 4:
            patch_layers = [
                n_layers // 4, n_layers // 2,
                3 * n_layers // 4, n_layers - 2,
            ]
        else:
            step = n_layers // n_patch_layers_actual
            patch_layers = [i * step for i in range(n_patch_layers_actual)]
            patch_layers = [min(l, n_layers - 2) for l in patch_layers]

        print(f"Patch layers ({len(patch_layers)}): {patch_layers}")
        print(f"n_rank={n_rank}, d_proj={d_proj}, frozen_proj={frozen}")

        mach = MACHActivationHebbian(
            d_model=d_model,
            n_layers=n_layers,
            patch_layers=patch_layers,
            hidden_dim=config.PATCH_HIDDEN_DIM,
            n_rank=n_rank,
            d_proj=d_proj,
            frozen_projections=frozen,
            consolidation=args.consolidation,
            ema_decay=args.ema_decay,
            delta_decay=args.delta_decay,
            consolidation_interval=args.consolidation_interval,
        ).to(config.DEVICE)

        n_params = sum(p.numel() for p in mach.parameters())
        print(f"MACHActivationHebbian total parameters: {n_params:,}")

        patched_model = ActivationHebbianPatchedModel(base_model, mach)

        save_path = (
            f"checkpoints/act_hebbian_{args.task}"
            f"_L{n_patch_layers_actual}_R{n_rank}_P{d_proj}{suffix}.pt"
        )
        os.makedirs("checkpoints", exist_ok=True)

        if wandb is not None:
            wandb.init(
                project="mach",
                name=run_name,
                config={
                    "base_model": config.BASE_MODEL,
                    "n_rank": n_rank,
                    "d_proj": d_proj,
                    "n_patch_layers": n_patch_layers_actual,
                    "lr": args.lr or config.PHASE5_LR,
                    "episodes": args.episodes or config.PHASE5_EPISODES,
                    "architecture": "act_hebbian",
                    "task": args.task,
                    "device": str(config.DEVICE),
                },
            )

        if args.ablate:
            if args.checkpoint:
                mach.load_state_dict(torch.load(args.checkpoint, map_location=config.DEVICE))
                print(f"Loaded checkpoint: {args.checkpoint}")
            mach.eval()
            ablate_hebbian(
                base_model, mach, patched_model, tokenizer, config.DEVICE,
            )
        elif args.continuous:
            from training.two_channel_train import meta_train_continuous
            meta_train_continuous(
                base_model, mach, patched_model, tokenizer, config.DEVICE,
                n_steps=args.n_steps, lr=args.lr,
                truncation_window=args.truncation_window,
                checkpoint_path=args.checkpoint,
                save_path=save_path,
                curriculum=curriculum,
                context_size=args.context_size,
                thinking_tokens=args.thinking_tokens,
            )
        else:
            meta_train_hebbian(
                base_model, mach, patched_model, tokenizer, config.DEVICE,
                n_episodes=args.episodes, lr=args.lr,
                checkpoint_path=args.checkpoint,
                save_path=save_path,
                curriculum=curriculum,
                chain_of_thought=args.cot,
                max_thinking_tokens=args.max_thinking,
            )

    elif args.hebbian:
        # MACHHebbian: three-factor Hebbian learning with critic
        arch_name = "hebbian"
        run_name = (
            f"hebbian-{args.task}"
            f"-L{n_patch_layers_actual}-B{n_basis_actual}"
        )

        # Generate patch layers
        if n_patch_layers_actual == 4:
            patch_layers = [
                n_layers // 4, n_layers // 2,
                3 * n_layers // 4, n_layers - 2,
            ]
        else:
            step = n_layers // n_patch_layers_actual
            patch_layers = [i * step for i in range(n_patch_layers_actual)]
            patch_layers = [min(l, n_layers - 2) for l in patch_layers]

        print(f"Patch layers ({len(patch_layers)}): {patch_layers}")
        print(f"n_basis={n_basis_actual}")

        mach = MACHHebbian(
            d_model=d_model,
            n_layers=n_layers,
            patch_layers=patch_layers,
            hidden_dim=config.PATCH_HIDDEN_DIM,
            n_basis=n_basis_actual,
            delta_decay=args.delta_decay,
        ).to(config.DEVICE)

        n_params = sum(p.numel() for p in mach.parameters())
        print(f"MACHHebbian total parameters: {n_params:,}")

        patched_model = HebbianPatchedModel(base_model, mach)

        save_path = (
            f"checkpoints/hebbian_{args.task}"
            f"_L{n_patch_layers_actual}_B{n_basis_actual}.pt"
        )
        os.makedirs("checkpoints", exist_ok=True)

        if wandb is not None:
            wandb.init(
                project="mach",
                name=run_name,
                config={
                    "base_model": config.BASE_MODEL,
                    "n_basis": n_basis_actual,
                    "n_patch_layers": n_patch_layers_actual,
                    "lr": args.lr or config.PHASE5_LR,
                    "episodes": args.episodes or config.PHASE5_EPISODES,
                    "architecture": "hebbian",
                    "task": args.task,
                    "device": str(config.DEVICE),
                },
            )

        if args.ablate:
            if args.checkpoint:
                mach.load_state_dict(torch.load(args.checkpoint, map_location=config.DEVICE))
                print(f"Loaded checkpoint: {args.checkpoint}")
            mach.eval()
            ablate_hebbian(
                base_model, mach, patched_model, tokenizer, config.DEVICE,
            )
        else:
            meta_train_hebbian(
                base_model, mach, patched_model, tokenizer, config.DEVICE,
                n_episodes=args.episodes, lr=args.lr,
                checkpoint_path=args.checkpoint,
                save_path=save_path,
                curriculum=curriculum,
                chain_of_thought=args.cot,
                max_thinking_tokens=args.max_thinking,
            )

    elif args.demoread or args.oracle or args.oracle_minimal:
        # MACHDemoRead / oracle / oracle-minimal
        if args.oracle_minimal:
            arch_name = "oracle_minimal"
        elif args.oracle:
            arch_name = "oracle"
        else:
            arch_name = "demoread"
        run_name = (
            f"{arch_name}-{args.task}"
            f"-L{n_patch_layers_actual}-B{n_basis_actual}"
        )

        # Generate patch layers
        if n_patch_layers_actual == 4:
            patch_layers = [
                n_layers // 4, n_layers // 2,
                3 * n_layers // 4, n_layers - 2,
            ]
        else:
            step = n_layers // n_patch_layers_actual
            patch_layers = [i * step for i in range(n_patch_layers_actual)]
            patch_layers = [min(l, n_layers - 2) for l in patch_layers]

        print(f"Patch layers ({len(patch_layers)}): {patch_layers}")
        print(f"n_basis={n_basis_actual}, d_meta={config.D_META}")

        if args.oracle_minimal:
            mach = MACHOracleMinimal(
                d_model=d_model,
                n_layers=n_layers,
                patch_layers=patch_layers,
                hidden_dim=config.PATCH_HIDDEN_DIM,
                n_basis=n_basis_actual,
            ).to(config.DEVICE)
        else:
            mach = MACHDemoRead(
                d_model=d_model,
                n_layers=n_layers,
                patch_layers=patch_layers,
                hidden_dim=config.PATCH_HIDDEN_DIM,
                d_meta=config.D_META,
                n_basis=n_basis_actual,
                oracle=args.oracle,
            ).to(config.DEVICE)

        n_params = sum(p.numel() for p in mach.parameters())
        print(f"{arch_name} total parameters: {n_params:,}")

        patched_model = IterativePatchedModel(base_model, mach)

        save_path = (
            f"checkpoints/{arch_name}_{args.task}"
            f"_L{n_patch_layers_actual}_B{n_basis_actual}.pt"
        )
        os.makedirs("checkpoints", exist_ok=True)

        if wandb is not None:
            wandb.init(
                project="mach",
                name=run_name,
                config={
                    "base_model": config.BASE_MODEL,
                    "n_basis": n_basis_actual,
                    "n_patch_layers": n_patch_layers_actual,
                    "d_meta": config.D_META,
                    "lr": args.lr or config.PHASE5_LR,
                    "episodes": args.episodes or config.PHASE5_EPISODES,
                    "architecture": arch_name,
                    "task": args.task,
                    "device": str(config.DEVICE),
                },
            )

        meta_train_demoread(
            base_model, mach, patched_model, tokenizer, config.DEVICE,
            n_episodes=args.episodes, lr=args.lr,
            checkpoint_path=args.checkpoint,
            save_path=save_path,
            curriculum=curriculum,
        )

    else:
        # Original two-channel mode
        arch_name = "two_channel"
        d_task_actual = args.d_task or config.PHASE5_D_TASK
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
