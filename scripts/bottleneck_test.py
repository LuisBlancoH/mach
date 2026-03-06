#!/usr/bin/env python3
"""
Bottleneck test: what's limiting accuracy?

Tests the same operation with varying numbers of Hebbian update problems.
If more problems = higher accuracy, the bottleneck is steering precision
(Hebbian rule needs more updates to converge).
If accuracy plateaus early, the bottleneck is something else.

Usage:
    python scripts/bottleneck_test.py --checkpoint checkpoints/act_hebbian_diverse_ops_L4_R2_P32.pt
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from models.universal_module import MACHActivationHebbian, ActivationHebbianPatchedModel
from data.arithmetic import generate_few_shot_episode, extract_number


def run_bottleneck_test(checkpoint_path, ops=None, problem_counts=None, n_trials=5, delta_decay=1.0, n_rank=None, context_size=0):
    if ops is None:
        ops = ["add", "sub", "mul", "div", "gcd", "abs_diff", "mod", "max"]
    if problem_counts is None:
        problem_counts = [5, 10, 20, 40, 60]

    print(f"Loading {config.BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, dtype=config.DTYPE
    ).to(config.DEVICE)
    for param in model.parameters():
        param.requires_grad = False

    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    patch_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2]

    mach = MACHActivationHebbian(
        d_model=d_model, n_layers=n_layers, patch_layers=patch_layers,
        hidden_dim=config.PATCH_HIDDEN_DIM, n_rank=n_rank if n_rank is not None else config.HEBBIAN_N_RANK,
        d_proj=config.HEBBIAN_D_PROJ, delta_decay=delta_decay,
    ).to(config.DEVICE)

    if checkpoint_path:
        mach.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE), strict=False)
        print(f"Loaded: {checkpoint_path}")

    patched_model = ActivationHebbianPatchedModel(model, mach)
    mach.eval()

    print(f"\n{'op':<12}", end="")
    for n in problem_counts:
        print(f"| n={n:<3} ", end="")
    print()
    print("-" * (12 + 9 * len(problem_counts)))

    for op in ops:
        print(f"{op:<12}", end="")
        for n_problems in problem_counts:
            correct_total = 0
            total = 0

            for trial in range(n_trials):
                problems = generate_few_shot_episode(
                    n_problems, n_demos=0, op_type=op
                )
                mach.reset_episode()
                context_buffer = []

                for step, problem in enumerate(problems):
                    # Build context from past solved problems
                    if context_size > 0 and context_buffer:
                        context_str = "".join(context_buffer[-context_size:])
                        full_prompt = context_str + problem["prompt"]
                    else:
                        full_prompt = problem["prompt"]

                    full_text = full_prompt + problem["answer"]
                    encoding = tokenizer(full_text, return_tensors="pt").to(config.DEVICE)
                    prompt_len = len(tokenizer(full_prompt).input_ids)

                    with torch.no_grad():
                        output = patched_model(input_ids=encoding.input_ids)
                        logits = output.logits
                        pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
                        pred_text = tokenizer.decode(
                            pred_tokens, skip_special_tokens=True
                        ).strip()
                        predicted = extract_number(pred_text)
                        correct = (predicted == problem["answer"])
                        reward = 1.0 if correct else -1.0

                    # Add solved problem to context
                    if context_size > 0:
                        context_buffer.append(f"{problem['prompt']}{problem['answer']}\n")

                    mach.hebbian_step(reward, step, n_problems, config.DEVICE)

                    # Only count last 25% as test accuracy
                    if step >= n_problems * 0.75:
                        correct_total += int(correct)
                        total += 1

            acc = correct_total / max(total, 1)
            print(f"| {acc:>4.0%}  ", end="")
        print()

    print("\nIf accuracy rises with n: bottleneck is steering precision (needs more updates)")
    print("If accuracy plateaus: bottleneck is elsewhere (Hebbian rule capacity, model capability)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--delta-decay", type=float, default=1.0,
                        help="Decay on accumulated deltas (1.0=no decay, 0.9=EMA)")
    parser.add_argument("--n-rank", type=int, default=None,
                        help="Hebbian rank (default: from config)")
    parser.add_argument("--context-size", type=int, default=0,
                        help="Number of past solved problems as context (0=off)")
    args = parser.parse_args()
    print(f"Delta decay: {args.delta_decay}")
    run_bottleneck_test(args.checkpoint, delta_decay=args.delta_decay,
                        n_rank=args.n_rank, context_size=args.context_size)
