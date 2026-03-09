#!/usr/bin/env python3
"""Predictive Coding experiment: cortical patches that predict and correct.

Tests whether predictive coding patches (predict → error → precision-weight → correct)
outperform simple MLP patches for teaching frozen Qwen new computation.

Usage:
    # Baseline only
    python scripts/run_predictive_coding.py --baseline-only

    # Direct backprop training on diverse operations (continuous)
    python scripts/run_predictive_coding.py --continuous --n-steps 10000

    # Phase 1 style (per-difficulty training)
    python scripts/run_predictive_coding.py
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from models.predictive_coding import (
    PredictiveCodingNetwork,
    PredictiveCodingPatchedModel,
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


def main():
    parser = argparse.ArgumentParser(
        description="Predictive Coding Patches Experiment"
    )
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--continuous", action="store_true",
                        help="Continuous training on diverse ops")
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--d-repr", type=int, default=128,
                        help="Representation dimension")
    parser.add_argument("--prediction-weight", type=float, default=0.1,
                        help="Weight of prediction loss")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-patch-layers", type=int, default=4)
    args = parser.parse_args()

    model, tokenizer, d_model, n_layers = load_base_model()

    # Patch layers at quarter points
    if args.n_patch_layers == 4:
        patch_layers = [
            n_layers // 4, n_layers // 2,
            3 * n_layers // 4, n_layers - 2,
        ]
    else:
        step = n_layers // args.n_patch_layers
        patch_layers = [min(i * step, n_layers - 2) for i in range(args.n_patch_layers)]

    print(f"Patch layers: {patch_layers}")

    # Create predictive coding network
    pc_network = PredictiveCodingNetwork(
        d_model=d_model,
        d_repr=args.d_repr,
        patch_layers=patch_layers,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in pc_network.parameters())
    print(f"Predictive coding params: {n_params:,}")

    patched_model = PredictiveCodingPatchedModel(model, pc_network)

    if args.baseline_only:
        from evaluation.baseline import evaluate_model
        from data.arithmetic import generate_arithmetic_problems

        print("\n=== Baseline Evaluation ===")
        for diff in [6, 7, 8, 9]:
            problems = generate_arithmetic_problems(500, diff)
            acc = evaluate_model(model, tokenizer, problems,
                                 label=f"baseline d{diff}")
            print(f"  Difficulty {diff}: {acc:.2%}")

        print("\n=== Predictive Coding (untrained) ===")
        for diff in [6, 7, 8, 9]:
            problems = generate_arithmetic_problems(500, diff)
            acc = evaluate_model(patched_model, tokenizer, problems,
                                 label=f"pc-untrained d{diff}")
            print(f"  Difficulty {diff}: {acc:.2%}")
        return

    if args.continuous:
        from training.predictive_coding_train import (
            train_predictive_coding_continuous,
        )

        save_path = (
            f"checkpoints/pc_diverse_ops"
            f"_L{len(patch_layers)}_R{args.d_repr}.pt"
        )
        os.makedirs("checkpoints", exist_ok=True)

        print(f"\n=== Continuous Predictive Coding Training ===")
        print(f"  n_steps={args.n_steps}, lr={args.lr}")
        print(f"  prediction_weight={args.prediction_weight}")
        print(f"  save_path={save_path}")

        train_predictive_coding_continuous(
            patched_model, pc_network, tokenizer,
            device=config.DEVICE,
            n_steps=args.n_steps,
            lr=args.lr,
            prediction_weight=args.prediction_weight,
            save_path=save_path,
        )
    else:
        # Phase 1 style: per-difficulty training
        from data.arithmetic import generate_arithmetic_problems
        from evaluation.baseline import evaluate_model
        from training.predictive_coding_train import train_predictive_coding

        difficulties = [6, 7, 8, 9]

        # Baseline
        print("\n=== Baseline Evaluation ===")
        baseline_results = {}
        for diff in difficulties:
            problems = generate_arithmetic_problems(500, diff)
            acc = evaluate_model(model, tokenizer, problems,
                                 label=f"baseline d{diff}")
            baseline_results[diff] = acc
            print(f"  Difficulty {diff}: {acc:.2%}")

        # Train
        for diff in difficulties:
            print(f"\n=== Training difficulty {diff} "
                  f"[baseline: {baseline_results[diff]:.2%}] ===")

            train_problems = generate_arithmetic_problems(
                config.PHASE1_TRAIN_PROBLEMS, diff
            )
            test_problems = generate_arithmetic_problems(
                config.PHASE1_TEST_PROBLEMS, diff
            )

            train_predictive_coding(
                patched_model, pc_network, tokenizer,
                train_problems, test_problems,
                device=config.DEVICE,
                epochs=args.epochs,
                lr=args.lr,
                prediction_weight=args.prediction_weight,
                eval_fn=evaluate_model,
            )

        # Final comparison
        print("\n=== Baseline vs Predictive Coding ===")
        for diff in difficulties:
            problems = generate_arithmetic_problems(500, diff)
            pc_acc = evaluate_model(patched_model, tokenizer, problems,
                                    label=f"final d{diff}")
            base = baseline_results[diff]
            delta = pc_acc - base
            marker = " <<<" if delta > 0.10 else ""
            print(f"  Diff {diff}: {base:.2%} -> {pc_acc:.2%} "
                  f"(delta: {delta:+.2%}){marker}")


if __name__ == "__main__":
    main()
