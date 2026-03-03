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
    1: "1-digit addition",
    2: "2-digit addition",
    3: "3-digit addition",
    4: "2-digit subtraction",
    5: "3-digit subtraction",
    6: "1x1 multiplication",
    7: "2x1 multiplication",
}


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
    for diff in range(1, 8):
        print(f"  Difficulty {diff}/7 ({DIFFICULTY_LABELS[diff]}):")
        problems = generate_arithmetic_problems(config.PHASE1_TEST_PROBLEMS, diff)
        acc = evaluate_model(model, tokenizer, problems, label=f"baseline d{diff}")
        results[diff] = acc
        print(f"  -> {acc:.2%}")
        wandb.log({f"baseline/diff{diff}_accuracy": acc})
    return results


def run_training(model, tokenizer, d_model, n_layers):
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

    # Train on each difficulty level where baseline struggles
    for diff in range(1, 8):
        print(f"\n=== Training on difficulty {diff} ({DIFFICULTY_LABELS[diff]}) ===")

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
    for diff in range(1, 8):
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

    patched_results = run_training(model, tokenizer, d_model, n_layers)

    # Print and log comparison
    print("\n=== Baseline vs Patched ===")
    summary = {}
    for diff in range(1, 8):
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
