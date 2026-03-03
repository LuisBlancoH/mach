#!/usr/bin/env python3
"""Phase 1: Direct patch training on arithmetic problems."""

import argparse
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from data.arithmetic import generate_arithmetic_problems
from evaluation.baseline import evaluate_model
from models.patches import PatchedModel
from training.phase1_direct import train_patches_direct


DIFFICULTY_LABELS = {
    1: "4-digit addition",
    2: "5-digit addition",
    3: "4-digit subtraction",
    4: "5-digit subtraction",
    5: "2x2 multiplication",
    6: "3x2 multiplication",
    7: "3x3 multiplication",
    8: "4-digit division",
    9: "mixed hard",
    10: "6-digit addition",
}

N_DIFFICULTIES = 10


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


def run_baseline(model, tokenizer):
    print("\n=== Baseline Evaluation ===")
    results = {}
    for diff in range(1, N_DIFFICULTIES + 1):
        print(f"  Difficulty {diff}/{N_DIFFICULTIES} ({DIFFICULTY_LABELS[diff]}):")
        problems = generate_arithmetic_problems(config.PHASE1_TEST_PROBLEMS, diff)
        acc = evaluate_model(model, tokenizer, problems, label=f"baseline d{diff}")
        results[diff] = acc
        print(f"  -> {acc:.2%}")
        wandb.log({f"baseline/diff{diff}_accuracy": acc})
    return results


def run_training(model, tokenizer, d_model, n_layers, baseline_results):
    # Determine patch layers (quarter points)
    patch_layers = [
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 2,
    ]
    print(f"\nPatch layers: {patch_layers}")

    patched_model = PatchedModel(
        model, d_model, patch_layers, hidden_dim=config.PATCH_HIDDEN_DIM
    )
    # Move patch parameters to device in float32
    patched_model.patches.float().to(config.DEVICE)

    # Count trainable params
    n_patch_params = sum(p.numel() for p in patched_model.patches.parameters())
    print(f"Trainable patch parameters: {n_patch_params:,}")

    # Only train on difficulty levels where baseline < 80%
    train_diffs = [d for d, acc in baseline_results.items() if acc < 0.80]
    if not train_diffs:
        print("\nNo difficulty levels below 80% baseline — nothing to train on!")
        train_diffs = [d for d, acc in sorted(baseline_results.items(), key=lambda x: x[1])][:3]
        print(f"Falling back to 3 weakest levels: {train_diffs}")

    skip_diffs = [d for d in range(1, N_DIFFICULTIES + 1) if d not in train_diffs]
    if skip_diffs:
        print(f"\nSkipping levels with baseline >= 80%: {skip_diffs}")

    for diff in train_diffs:
        print(f"\n=== Training on difficulty {diff} ({DIFFICULTY_LABELS[diff]}) "
              f"[baseline: {baseline_results[diff]:.2%}] ===")

        train_problems = generate_arithmetic_problems(config.PHASE1_TRAIN_PROBLEMS, diff)
        test_problems = generate_arithmetic_problems(config.PHASE1_TEST_PROBLEMS, diff)

        train_patches_direct(
            patched_model, tokenizer, train_problems, test_problems,
            device=config.DEVICE, difficulty=diff,
            epochs=config.PHASE1_EPOCHS, lr=config.PHASE1_LR,
        )

    # Final evaluation across all difficulties
    print("\n=== Final Patched Evaluation ===")
    patched_results = {}
    for diff in range(1, N_DIFFICULTIES + 1):
        problems = generate_arithmetic_problems(config.PHASE1_TEST_PROBLEMS, diff)
        acc = evaluate_model(patched_model, tokenizer, problems, label=f"final d{diff}")
        patched_results[diff] = acc
        print(f"  -> {acc:.2%}")
        wandb.log({f"final/diff{diff}_accuracy": acc})

    return patched_results


def main():
    parser = argparse.ArgumentParser(description="MACH Phase 1: Direct Patch Training")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run baseline evaluation, no training")
    args = parser.parse_args()

    wandb.init(
        project="mach",
        name="phase1-baseline" if args.baseline_only else "phase1-patch-training",
        config={
            "base_model": config.BASE_MODEL,
            "patch_hidden_dim": config.PATCH_HIDDEN_DIM,
            "lr": config.PHASE1_LR,
            "epochs": config.PHASE1_EPOCHS,
            "train_problems": config.PHASE1_TRAIN_PROBLEMS,
            "test_problems": config.PHASE1_TEST_PROBLEMS,
            "device": str(config.DEVICE),
        },
    )

    model, tokenizer, d_model, n_layers = load_base_model()
    baseline_results = run_baseline(model, tokenizer)

    if args.baseline_only:
        wandb.finish()
        return

    patched_results = run_training(model, tokenizer, d_model, n_layers, baseline_results)

    # Print and log comparison
    print("\n=== Baseline vs Patched ===")
    summary = {}
    for diff in range(1, N_DIFFICULTIES + 1):
        base = baseline_results[diff]
        patched = patched_results[diff]
        delta = patched - base
        marker = " <<<" if delta > 0.10 else ""
        print(f"  Difficulty {diff} ({DIFFICULTY_LABELS[diff]}): "
              f"{base:.2%} -> {patched:.2%} (delta: {delta:+.2%}){marker}")
        summary[f"delta/diff{diff}"] = delta
    wandb.log(summary)
    wandb.finish()


if __name__ == "__main__":
    main()
