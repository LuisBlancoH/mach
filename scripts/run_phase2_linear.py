#!/usr/bin/env python3
"""Phase 2 + linear combinations: f(a,b) = c1*a + c2*b with c1,c2 in {0,1,2}."""

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
from data.arithmetic import (
    extract_number, generate_linear_episode,
    LINEAR_COEFFS, LINEAR_TRAIN, LINEAR_HELDOUT,
)
from models.universal_module import MACHPhase2, MACHPatchedModel
from training.phase2_fewshot_train import meta_train_phase2_fewshot


LINEAR_CURRICULUM = [
    (0, 300, "single"),    # Warm up on d6 (multiplication)
    (300, 2000, "linear"),  # Linear combinations
]


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


def create_mach_phase2(d_model, n_layers, detach_obs=True):
    patch_layers = [
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 2,
    ]
    print(f"Patch layers: {patch_layers}")
    print(f"Detach obs: {detach_obs}")

    mach = MACHPhase2(
        d_model=d_model,
        n_layers=n_layers,
        patch_layers=patch_layers,
        hidden_dim=config.PATCH_HIDDEN_DIM,
        d_meta=config.D_META,
        n_basis=config.N_BASIS,
        detach_obs=detach_obs,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in mach.parameters())
    print(f"MACH Phase 2 total parameters: {n_params:,}")

    return mach, patch_layers


def _eval_linear_combo(base_model, mach, patched_model, tokenizer, device,
                       coeffs, n_episodes, n_problems, n_demos):
    """Evaluate one linear combination, return accuracy."""
    c1, c2 = coeffs
    test_correct = 0
    test_total = 0

    for ep in range(n_episodes):
        problems = generate_linear_episode(
            n_problems, n_demos=n_demos, coeffs=(c1, c2)
        )
        mach.reset_episode()

        for i, problem in enumerate(problems):
            input_ids = tokenizer(
                problem["prompt"], return_tensors="pt"
            ).input_ids.to(device)

            gru_memory = mach.observe(base_model, input_ids)
            reward_signals = torch.zeros(
                3, device=device, dtype=torch.float32
            )
            writes = mach.fire(gru_memory, reward_signals)
            mach.apply_writes(writes)

            if problem["is_demo"]:
                continue

            full_text = problem["prompt"] + problem["answer"]
            encoding = tokenizer(
                full_text, return_tensors="pt"
            ).to(device)
            prompt_len = len(tokenizer(problem["prompt"]).input_ids)

            with torch.no_grad():
                output = patched_model(input_ids=encoding.input_ids)
                logits = (
                    output.logits
                    if hasattr(output, 'logits') else output[0]
                )
                pred_tokens = logits[
                    0, prompt_len - 1:-1
                ].argmax(dim=-1)
                pred_text = tokenizer.decode(
                    pred_tokens, skip_special_tokens=True
                ).strip()
                predicted = extract_number(pred_text)
                correct = (predicted == problem["answer"])

            test_correct += int(correct)
            test_total += 1

    return test_correct / test_total if test_total > 0 else 0


def final_evaluation(base_model, mach, patched_model, tokenizer, device,
                     n_episodes=20, n_problems=20, n_demos=5):
    """Evaluate all 9 linear combinations, split by train/held-out."""
    print(f"\n{'='*60}")
    print(f"FINAL LINEAR COMBINATION EVALUATION")
    print(f"{n_episodes} episodes x {n_problems} problems "
          f"({n_demos} demos + {n_problems - n_demos} test)")
    print(f"{'='*60}")

    print(f"\n--- TRAINED combinations ---")
    train_accs = []
    for c1, c2 in LINEAR_TRAIN:
        acc = _eval_linear_combo(
            base_model, mach, patched_model, tokenizer, device,
            (c1, c2), n_episodes, n_problems, n_demos,
        )
        train_accs.append(acc)
        print(f"  {c1}a+{c2}b: {acc:.0%}")

    print(f"\n--- HELD-OUT combinations (never seen during training) ---")
    heldout_accs = []
    for c1, c2 in LINEAR_HELDOUT:
        acc = _eval_linear_combo(
            base_model, mach, patched_model, tokenizer, device,
            (c1, c2), n_episodes, n_problems, n_demos,
        )
        heldout_accs.append(acc)
        print(f"  {c1}a+{c2}b: {acc:.0%}")

    train_avg = sum(train_accs) / len(train_accs)
    heldout_avg = sum(heldout_accs) / len(heldout_accs)

    print(f"\n{'='*60}")
    print(f"Train avg: {train_avg:.0%}  |  Held-out avg: {heldout_avg:.0%}")
    print(f"If held-out > 0: genuine function induction")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 + linear combinations: c1*a + c2*b"
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint to load (default: train from scratch)"
    )
    parser.add_argument(
        "--from-scratch", action="store_true",
        help="Train from scratch without loading checkpoint"
    )
    parser.add_argument(
        "--undetach-obs", action="store_true",
        help="Allow gradient through obs_proj and GRU"
    )
    args = parser.parse_args()

    detach_obs = not args.undetach_obs
    run_name = "phase2-linear"
    if args.undetach_obs:
        run_name += "-undetach"

    if wandb is not None:
        wandb.init(
            project="mach",
            name=run_name,
            config={
                "base_model": config.BASE_MODEL,
                "d_meta": config.D_META,
                "n_basis": config.N_BASIS,
                "lr": args.lr or config.PHASE2_LR,
                "episodes": args.episodes or config.PHASE2_EPISODES,
                "architecture": "phase2",
                "task": "linear",
                "coeffs": list(LINEAR_COEFFS),
                "detach_obs": detach_obs,
                "no_rewards": True,
                "device": str(config.DEVICE),
            },
        )

    base_model, tokenizer, d_model, n_layers = load_base_model()
    mach, patch_layers = create_mach_phase2(
        d_model, n_layers, detach_obs=detach_obs
    )
    patched_model = MACHPatchedModel(base_model, mach)

    save_path = "checkpoints/phase2_linear.pt"
    os.makedirs("checkpoints", exist_ok=True)

    checkpoint = None if args.from_scratch else args.checkpoint
    meta_train_phase2_fewshot(
        base_model, mach, patched_model, tokenizer, config.DEVICE,
        n_episodes=args.episodes, lr=args.lr,
        checkpoint_path=checkpoint,
        save_path=save_path,
        no_rewards=True,
        curriculum=LINEAR_CURRICULUM,
    )

    final_evaluation(
        base_model, mach, patched_model, tokenizer, config.DEVICE,
    )

    torch.save(mach.state_dict(), save_path)
    print(f"\nSaved final checkpoint to {save_path}")

    if wandb is not None:
        wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
