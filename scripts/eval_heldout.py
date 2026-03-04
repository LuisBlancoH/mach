#!/usr/bin/env python3
"""
Held-out evaluation: test on operations never seen during training.

Training used: add, sub, mul, div
Held-out operations: mod, max, min, add_mul (a*b + a), square_diff (a^2 - b^2)

If the model genuinely learns from demos (not just classifying among 4 known ops),
it should show non-trivial accuracy on held-out operations.

Also tests: train on {add, mul} only, evaluate on {sub, div} — can it
generalize to unseen but structurally similar operations?
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from data.arithmetic import extract_number
from models.universal_module import MACHPhase2, MACHPhase3, MACHPatchedModel


# ---- Held-out operation generators ----

def _make_operands_heldout(op_type):
    """Generate operands and answer for held-out operations.

    All ops use a,b in [10, 99] with a >= b to match training ranges.
    """
    a = random.randint(10, 99)
    b = random.randint(10, 99)
    if b > a:
        a, b = b, a

    if op_type == "add":
        return a, b, a + b
    elif op_type == "sub":
        return a, b, a - b
    elif op_type == "mul":
        return a, b, a * b
    elif op_type == "div":
        return a, b, a // b
    elif op_type == "mod":
        return a, b, a % b
    elif op_type == "max":
        return a, b, max(a, b)
    elif op_type == "min":
        return a, b, min(a, b)
    elif op_type == "add_mul":
        return a, b, a * b + a
    elif op_type == "square_diff":
        return a, b, a * a - b * b
    else:
        raise ValueError(f"Unknown op_type: {op_type}")


def generate_heldout_episode(n_problems, n_demos, op_type):
    """Generate a few-shot episode for any operation type."""
    problems = []
    for i in range(n_problems):
        a, b, answer = _make_operands_heldout(op_type)
        is_demo = (i < n_demos)

        if is_demo:
            prompt = f"{a} ? {b} = {answer}"
        else:
            prompt = f"{a} ? {b} = "

        problems.append({
            "prompt": prompt,
            "answer": str(answer),
            "a": a, "b": b,
            "op": op_type,
            "is_demo": is_demo,
        })
    return problems


def run_eval(base_model, mach, patched_model, tokenizer, device, problems):
    """Run one episode, return (correct, total)."""
    mach.reset_episode()
    is_phase2 = isinstance(mach, MACHPhase2) and not isinstance(mach, MACHPhase3)

    last_reward = 0.0
    cumulative_reward = 0.0
    last_value = torch.tensor(0.0, device=device)
    last_td_error = torch.tensor(0.0, device=device)

    correct_count = 0
    test_count = 0

    for i, problem in enumerate(problems):
        input_ids = tokenizer(
            problem["prompt"], return_tensors="pt"
        ).input_ids.to(device)

        gru_memory = mach.observe(base_model, input_ids)

        if is_phase2:
            reward_signals = torch.tensor(
                [last_reward, cumulative_reward, float(i)],
                device=device, dtype=torch.float32
            )
            writes = mach.fire(gru_memory, reward_signals)
        else:
            writes = mach.fire(gru_memory, last_value, last_td_error)
            with torch.no_grad():
                current_value = mach.get_value()

        mach.apply_writes(writes)

        if problem["is_demo"]:
            if not is_phase2:
                last_value = current_value.detach()
            continue

        full_text = problem["prompt"] + problem["answer"]
        encoding = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_len = len(tokenizer(problem["prompt"]).input_ids)

        with torch.no_grad():
            output = patched_model(input_ids=encoding.input_ids)
            logits = output.logits if hasattr(output, 'logits') else output[0]
            pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
            pred_text = tokenizer.decode(
                pred_tokens, skip_special_tokens=True
            ).strip()
            predicted = extract_number(pred_text)
            correct = (predicted == problem["answer"])

        correct_count += int(correct)
        test_count += 1

        reward = 1.0 if correct else -1.0
        last_reward = reward
        cumulative_reward += reward

        if not is_phase2:
            td_target = torch.tensor(reward, device=device, dtype=torch.float32)
            last_td_error = (td_target - last_value).detach()
            last_value = current_value.detach()

    return correct_count, test_count


def main():
    parser = argparse.ArgumentParser(
        description="Held-out evaluation: test on unseen operations"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Checkpoint to evaluate"
    )
    parser.add_argument(
        "--phase", type=int, default=2, choices=[2, 3],
        help="Architecture phase (2 or 3)"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=20,
        help="Episodes per operation"
    )
    args = parser.parse_args()

    n_problems = 20
    n_demos = 5

    print(f"Loading {config.BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, dtype=config.DTYPE
    ).to(config.DEVICE)
    for param in base_model.parameters():
        param.requires_grad = False

    d_model = base_model.config.hidden_size
    n_layers = base_model.config.num_hidden_layers
    patch_layers = [
        n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2
    ]

    if args.phase == 2:
        mach = MACHPhase2(
            d_model=d_model, n_layers=n_layers, patch_layers=patch_layers,
            hidden_dim=config.PATCH_HIDDEN_DIM, d_meta=config.D_META,
            n_basis=config.N_BASIS, detach_obs=True,
        ).to(config.DEVICE)
    else:
        mach = MACHPhase3(
            d_model=d_model, n_layers=n_layers, patch_layers=patch_layers,
            hidden_dim=config.PATCH_HIDDEN_DIM, d_meta=config.D_META,
            n_basis=config.N_BASIS, detach_obs=True,
        ).to(config.DEVICE)

    state_dict = torch.load(args.checkpoint, map_location=config.DEVICE)
    mach.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")

    patched_model = MACHPatchedModel(base_model, mach)

    trained_ops = ["add", "sub", "mul", "div"]
    heldout_ops = ["mod", "max", "min", "add_mul", "square_diff"]

    print(f"\n{'='*70}")
    print(f"HELD-OUT EVALUATION — Phase {args.phase}")
    print(f"{args.n_episodes} episodes x {n_problems} problems "
          f"({n_demos} demos + {n_problems - n_demos} test)")
    print(f"{'='*70}")

    # Control: trained operations
    print(f"\n--- TRAINED OPERATIONS (control) ---")
    for op in trained_ops:
        correct = 0
        total = 0
        for _ in range(args.n_episodes):
            problems = generate_heldout_episode(n_problems, n_demos, op)
            c, t = run_eval(
                base_model, mach, patched_model, tokenizer,
                config.DEVICE, problems
            )
            correct += c
            total += t
        acc = correct / total if total > 0 else 0
        print(f"  {op:15s}: {acc:.0%}  ({correct}/{total})")

    # Held-out: novel operations
    print(f"\n--- HELD-OUT OPERATIONS (never seen during training) ---")
    for op in heldout_ops:
        correct = 0
        total = 0
        for _ in range(args.n_episodes):
            problems = generate_heldout_episode(n_problems, n_demos, op)
            c, t = run_eval(
                base_model, mach, patched_model, tokenizer,
                config.DEVICE, problems
            )
            correct += c
            total += t
        acc = correct / total if total > 0 else 0
        print(f"  {op:15s}: {acc:.0%}  ({correct}/{total})")

    # Baseline: no patches (raw Qwen on "A ? B = " format)
    print(f"\n--- BASELINE (no patches, raw Qwen) ---")
    for op in trained_ops + heldout_ops:
        correct = 0
        total = 0
        for _ in range(args.n_episodes):
            problems = generate_heldout_episode(n_problems, n_demos, op)
            mach.reset_episode()  # zeros all patches
            for problem in problems:
                if problem["is_demo"]:
                    continue
                full_text = problem["prompt"] + problem["answer"]
                encoding = tokenizer(full_text, return_tensors="pt").to(config.DEVICE)
                prompt_len = len(tokenizer(problem["prompt"]).input_ids)
                with torch.no_grad():
                    output = patched_model(input_ids=encoding.input_ids)
                    logits = output.logits if hasattr(output, 'logits') else output[0]
                    pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
                    pred_text = tokenizer.decode(
                        pred_tokens, skip_special_tokens=True
                    ).strip()
                    predicted = extract_number(pred_text)
                    c = (predicted == problem["answer"])
                correct += int(c)
                total += 1
        acc = correct / total if total > 0 else 0
        print(f"  {op:15s}: {acc:.0%}  ({correct}/{total})")

    print(f"\n{'='*70}")
    print("If genuine meta-learning: held-out > baseline (patches help novel ops)")
    print("If just 4-op classifier: held-out ~ baseline (patches only help known ops)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
