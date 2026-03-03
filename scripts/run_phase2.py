#!/usr/bin/env python3
"""Phase 2: Meta-learner training — validate the core write mechanism."""

import argparse
import sys
import os

# Allow imports from project root
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
from training.phase2_meta_train import meta_train
from training.episode import run_episode
from evaluation.ablations import random_writes_baseline


def load_base_model():
    print(f"Loading {config.BASE_MODEL} on {config.DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, dtype=config.DTYPE
    ).to(config.DEVICE)

    # Freeze all base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()

    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  d_model={d_model}, n_layers={n_layers}")

    return model, tokenizer, d_model, n_layers


def create_mach(d_model, n_layers):
    patch_layers = [
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 2,
    ]
    print(f"Patch layers: {patch_layers}")

    mach = MACHPhase2(
        d_model=d_model,
        n_layers=n_layers,
        patch_layers=patch_layers,
        hidden_dim=config.PATCH_HIDDEN_DIM,
        d_meta=config.D_META,
        n_basis=config.N_BASIS,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in mach.parameters())
    print(f"MACH Phase 2 total parameters: {n_params:,}")

    return mach, patch_layers


def final_evaluation(base_model, mach, patched_model, tokenizer, device,
                     n_episodes=10, n_problems=20):
    """Run evaluation episodes and report early vs late accuracy."""
    print(f"\n=== Final Evaluation ({n_episodes} episodes, {n_problems} problems each) ===")

    all_early = []
    all_late = []
    all_rewards = []

    for ep in range(n_episodes):
        difficulty = 6  # 3x2 multiplication (Phase 1's best target)
        problems = generate_arithmetic_problems(n_problems, difficulty)

        with torch.no_grad():
            # Still need grad for write path through patches
            pass

        # Run episode (no backward needed)
        loss, rewards, _ = run_episode(
            base_model, mach, patched_model, tokenizer, problems, device
        )

        early_acc = sum(1 for r in rewards[:5] if r > 0) / 5
        late_acc = sum(1 for r in rewards[-5:] if r > 0) / 5

        all_early.append(early_acc)
        all_late.append(late_acc)
        all_rewards.extend(rewards)

        print(f"  Episode {ep}: early={early_acc:.0%} late={late_acc:.0%} "
              f"delta={late_acc - early_acc:+.0%}")

    avg_early = sum(all_early) / len(all_early)
    avg_late = sum(all_late) / len(all_late)
    avg_reward = sum(all_rewards) / len(all_rewards)
    delta = avg_late - avg_early

    print(f"\n  Average early accuracy: {avg_early:.1%}")
    print(f"  Average late accuracy:  {avg_late:.1%}")
    print(f"  Learning delta:         {delta:+.1%}")
    print(f"  Average reward:         {avg_reward:.2f}")

    passed = delta >= 0.10
    print(f"\n  Phase 2 {'PASS' if passed else 'FAIL'}: "
          f"late - early = {delta:+.1%} ({'≥' if passed else '<'} 10pp)")

    if wandb is not None:
        wandb.log({
            "final/avg_early_accuracy": avg_early,
            "final/avg_late_accuracy": avg_late,
            "final/learning_delta": delta,
            "final/avg_reward": avg_reward,
            "final/passed": passed,
        })

    return passed


def run_ablation(base_model, mach, patched_model, tokenizer, device):
    """Compare meta-learner writes to random writes."""
    print("\n=== Ablation: Random Writes vs Learned Writes ===")

    difficulty = 6
    n_episodes = 5
    n_problems = 20

    # Random writes baseline
    random_rewards = []
    for ep in range(n_episodes):
        mach.reset_episode()
        problems = generate_arithmetic_problems(n_problems, difficulty)

        for i, problem in enumerate(problems):
            # Apply random writes instead of meta-learner writes
            writes = random_writes_baseline(
                mach.n_patches, config.N_BASIS, device
            )
            mach.apply_writes(writes)

        # Evaluate (just check last few problems)
        for problem in problems[-5:]:
            input_ids = tokenizer(
                problem["prompt"], return_tensors="pt"
            ).input_ids.to(device)
            full_text = problem["prompt"] + problem["answer"]
            encoding = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_len = len(tokenizer(problem["prompt"]).input_ids)
            labels = encoding.input_ids.clone()
            labels[0, :prompt_len] = -100

            with torch.no_grad():
                output = patched_model(input_ids=encoding.input_ids, labels=labels)
                logits = output.logits
                pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                from data.arithmetic import extract_number
                predicted = extract_number(pred_text)
                correct = (predicted == problem["answer"])
                random_rewards.append(1.0 if correct else -1.0)

    random_acc = sum(1 for r in random_rewards if r > 0) / len(random_rewards)
    print(f"  Random writes accuracy: {random_acc:.1%}")
    if wandb is not None:
        wandb.log({"ablation/random_writes_accuracy": random_acc})


def main():
    parser = argparse.ArgumentParser(description="MACH Phase 2: Meta-Learner Training")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip random writes ablation")
    args = parser.parse_args()

    if wandb is not None:
        wandb.init(
            project="mach",
            name="phase2-meta-training",
            config={
                "base_model": config.BASE_MODEL,
                "d_meta": config.D_META,
                "n_basis": config.N_BASIS,
                "n_meta_layers": config.N_META_LAYERS,
                "patch_hidden_dim": config.PATCH_HIDDEN_DIM,
                "gate_scale": config.GATE_SCALE,
                "lr": args.lr or config.PHASE2_LR,
                "episodes": args.episodes or config.PHASE2_EPISODES,
                "problems_per_episode": config.PHASE2_PROBLEMS_PER_EPISODE,
                "grad_clip": config.PHASE2_GRAD_CLIP,
                "device": str(config.DEVICE),
            },
        )

    # Load base model
    base_model, tokenizer, d_model, n_layers = load_base_model()

    # Create MACH Phase 2
    mach, patch_layers = create_mach(d_model, n_layers)
    patched_model = MACHPatchedModel(base_model, mach)

    # Meta-train
    meta_train(
        base_model, mach, patched_model, tokenizer, config.DEVICE,
        n_episodes=args.episodes, lr=args.lr,
    )

    # Final evaluation
    passed = final_evaluation(
        base_model, mach, patched_model, tokenizer, config.DEVICE
    )

    # Ablation
    if not args.skip_ablation:
        run_ablation(base_model, mach, patched_model, tokenizer, config.DEVICE)

    # Save meta-learner checkpoint
    save_path = "checkpoints/phase2_mach.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(mach.state_dict(), save_path)
    print(f"\nSaved MACH checkpoint to {save_path}")

    if wandb is not None:
        wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
