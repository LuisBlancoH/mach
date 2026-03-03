#!/usr/bin/env python3
"""
Phase 2b: Mixed-difficulty episodes to force within-episode adaptation.

Phase 2 showed the meta-learner learns excellent static patches (+47pp on d7)
but doesn't adapt within an episode (delta ≈ 0%). This experiment tests whether
mixing difficulties within each episode forces the meta-learner to condition
its writes on observations rather than using a fixed strategy.

Key change: each problem in an episode is sampled from a random difficulty.
A static write strategy can't simultaneously help 2x2 and 3x3 multiplication.
"""

import argparse
import random
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
from data.arithmetic import generate_arithmetic_problems
from models.universal_module import MACHPhase2, MACHPatchedModel
from training.episode import run_episode
from training.phase2_meta_train import (
    _run_validation, _log_diagnostics, get_difficulty
)


def generate_mixed_episode(n_problems, difficulties=(5, 6, 7)):
    """Generate an episode with problems from mixed difficulties."""
    problems = []
    for _ in range(n_problems):
        diff = random.choice(difficulties)
        p = generate_arithmetic_problems(1, diff)[0]
        p["difficulty"] = diff
        problems.append(p)
    return problems


def meta_train_mixed(base_model, mach, patched_model, tokenizer, device,
                     n_episodes=1000, lr=None, checkpoint_path=None):
    """
    Phase 2b: meta-training with mixed-difficulty episodes.
    Optionally loads from a Phase 2 checkpoint to continue training.
    """
    if lr is None:
        lr = config.PHASE2_LR

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        mach.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("  Loaded.")

    meta_params = []
    meta_params += list(mach.obs_proj.parameters())
    meta_params += list(mach.gru.parameters())
    meta_params += list(mach.basis.parameters())
    meta_params += list(mach.transformer.parameters())
    meta_params += list(mach.action_head.parameters())
    meta_params += list(mach.memory_head.parameters())
    meta_params += list(mach.reward_proj.parameters())

    optimizer = torch.optim.Adam(meta_params, lr=lr)

    n_meta_params = sum(p.numel() for p in meta_params)
    print(f"Meta-learner trainable parameters: {n_meta_params:,}")

    use_chunked = False

    for episode_idx in range(n_episodes):
        n_problems = 20
        problems = generate_mixed_episode(n_problems, difficulties=(5, 6, 7))

        optimizer.zero_grad()

        try:
            if use_chunked:
                from training.episode import run_episode_chunked
                loss_val, rewards, problem_losses = run_episode_chunked(
                    base_model, mach, patched_model, tokenizer,
                    problems, device, optimizer
                )
                loss_scalar = loss_val
            else:
                loss, rewards, problem_losses = run_episode(
                    base_model, mach, patched_model, tokenizer,
                    problems, device
                )
                loss.backward()
                loss_scalar = loss.item()

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at episode {episode_idx}. Switching to chunked.")
            torch.cuda.empty_cache()
            use_chunked = True
            optimizer.zero_grad()
            from training.episode import run_episode_chunked
            loss_val, rewards, problem_losses = run_episode_chunked(
                base_model, mach, patched_model, tokenizer,
                problems, device, optimizer
            )
            loss_scalar = loss_val

        torch.nn.utils.clip_grad_norm_(meta_params, max_norm=config.PHASE2_GRAD_CLIP)
        optimizer.step()

        if episode_idx % 10 == 0:
            avg_reward = sum(rewards) / len(rewards)
            early_acc = sum(1 for r in rewards[:5] if r > 0) / 5
            late_acc = sum(1 for r in rewards[-5:] if r > 0) / 5

            # Per-difficulty breakdown
            diff_correct = {}
            diff_total = {}
            for j, p in enumerate(problems):
                d = p.get("difficulty", "?")
                diff_total[d] = diff_total.get(d, 0) + 1
                if rewards[j] > 0:
                    diff_correct[d] = diff_correct.get(d, 0) + 1

            diff_str = " ".join(
                f"d{d}={diff_correct.get(d,0)}/{diff_total[d]}"
                for d in sorted(diff_total)
            )

            print(f"Episode {episode_idx:4d} | mixed n={n_problems} | "
                  f"loss={loss_scalar:.4f} avg_r={avg_reward:.2f} | "
                  f"early={early_acc:.0%} late={late_acc:.0%} | {diff_str}")

            if wandb is not None:
                log_dict = {
                    "episode": episode_idx,
                    "loss": loss_scalar,
                    "avg_reward": avg_reward,
                    "early_accuracy": early_acc,
                    "late_accuracy": late_acc,
                    "learning_delta": late_acc - early_acc,
                }
                for j, pl in enumerate(problem_losses):
                    log_dict[f"problem_loss/{j}"] = pl
                wandb.log(log_dict)

        # Validation every 200 episodes
        if episode_idx % 200 == 0 and episode_idx > 0:
            _log_diagnostics(mach, meta_params, episode_idx)
            # Eval on each difficulty separately
            for eval_diff in [5, 6, 7]:
                _run_validation(base_model, mach, patched_model, tokenizer,
                                device, eval_diff, episode_idx)


def main():
    parser = argparse.ArgumentParser(description="MACH Phase 2b: Mixed-Difficulty Episodes")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load Phase 2 checkpoint to continue training")
    parser.add_argument("--undetach-obs", action="store_true",
                        help="Allow gradient through obs_proj and GRU")
    args = parser.parse_args()

    if wandb is not None:
        wandb.init(
            project="mach",
            name="phase2b-mixed-difficulty",
            config={
                "base_model": config.BASE_MODEL,
                "d_meta": config.D_META,
                "n_basis": config.N_BASIS,
                "lr": args.lr or config.PHASE2_LR,
                "episodes": args.episodes,
                "problems_per_episode": 20,
                "difficulties": [5, 6, 7],
                "detach_obs": not args.undetach_obs,
                "device": str(config.DEVICE),
            },
        )

    print(f"Loading {config.BASE_MODEL} on {config.DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, dtype=config.DTYPE
    ).to(config.DEVICE)
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.gradient_checkpointing_enable()

    d_model = base_model.config.hidden_size
    n_layers = base_model.config.num_hidden_layers
    print(f"  d_model={d_model}, n_layers={n_layers}")

    patch_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2]
    print(f"Patch layers: {patch_layers}")

    detach_obs = not args.undetach_obs
    if not detach_obs:
        print("*** Observation path UNDETACHED — gradient flows through obs_proj and GRU ***")
    mach = MACHPhase2(
        d_model=d_model, n_layers=n_layers, patch_layers=patch_layers,
        hidden_dim=config.PATCH_HIDDEN_DIM, d_meta=config.D_META, n_basis=config.N_BASIS,
        detach_obs=detach_obs,
    ).to(config.DEVICE)
    patched_model = MACHPatchedModel(base_model, mach)

    n_params = sum(p.numel() for p in mach.parameters())
    print(f"MACH Phase 2b total parameters: {n_params:,}")

    meta_train_mixed(
        base_model, mach, patched_model, tokenizer, config.DEVICE,
        n_episodes=args.episodes, lr=args.lr, checkpoint_path=args.checkpoint,
    )

    # Save checkpoint
    save_path = "checkpoints/phase2b_mach.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(mach.state_dict(), save_path)
    print(f"\nSaved to {save_path}")

    if wandb is not None:
        wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
