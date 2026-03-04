#!/usr/bin/env python3
"""
Mismatch evaluation: show demos for one operation, test on another.
If the model uses demos, mismatched demos should HURT accuracy.
If the model uses number ranges, mismatched demos won't matter.

Tests:
1. Matched: demos=add, test=add (control)
2. Mismatched: demos=mul, test=add (should fail if using demos)
3. No demos: skip demos entirely (baseline)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from data.arithmetic import extract_number, _make_operands
from models.universal_module import MACHPhase2, MACHPhase3, MACHPatchedModel


def generate_mismatch_episode(n_problems, n_demos, demo_op, test_op):
    """Generate episode with demos from one op and tests from another."""
    problems = []
    for i in range(n_problems):
        is_demo = (i < n_demos)
        if is_demo:
            a, b, answer = _make_operands(demo_op)
            prompt = f"{a} ? {b} = {answer}"
        else:
            a, b, answer = _make_operands(test_op)
            prompt = f"{a} ? {b} = "

        problems.append({
            "prompt": prompt,
            "answer": str(answer),
            "a": a, "b": b,
            "op": test_op if not is_demo else demo_op,
            "is_demo": is_demo,
        })
    return problems


def generate_no_demo_episode(n_problems, test_op):
    """Generate episode with no demos at all."""
    problems = []
    for i in range(n_problems):
        a, b, answer = _make_operands(test_op)
        prompt = f"{a} ? {b} = "
        problems.append({
            "prompt": prompt,
            "answer": str(answer),
            "a": a, "b": b,
            "op": test_op,
            "is_demo": False,
        })
    return problems


def run_eval(base_model, mach, patched_model, tokenizer, device,
             problems, n_demos):
    """Run one episode, return test accuracy."""
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
        description="Mismatch evaluation: do demos actually matter?"
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
        help="Episodes per condition"
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
    patch_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2]

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

    ops = ["add", "sub", "mul", "div", "mod", "max", "min"]

    print(f"\n{'='*70}")
    print(f"MISMATCH EVALUATION — Phase {args.phase}")
    print(f"{args.n_episodes} episodes x {n_problems} problems "
          f"({n_demos} demos + {n_problems - n_demos} test)")
    print(f"{'='*70}")

    # Test 1: Matched (control)
    print(f"\n--- MATCHED (control) ---")
    for test_op in ops:
        correct = 0
        total = 0
        for _ in range(args.n_episodes):
            problems = generate_mismatch_episode(
                n_problems, n_demos, demo_op=test_op, test_op=test_op
            )
            c, t = run_eval(
                base_model, mach, patched_model, tokenizer,
                config.DEVICE, problems, n_demos
            )
            correct += c
            total += t
        acc = correct / total if total > 0 else 0
        print(f"  demo={test_op} test={test_op}: {acc:.0%}")

    # Test 2: Mismatched
    print(f"\n--- MISMATCHED (demos from wrong operation) ---")
    for test_op in ops:
        for demo_op in ops:
            if demo_op == test_op:
                continue
            correct = 0
            total = 0
            for _ in range(args.n_episodes):
                problems = generate_mismatch_episode(
                    n_problems, n_demos, demo_op=demo_op, test_op=test_op
                )
                c, t = run_eval(
                    base_model, mach, patched_model, tokenizer,
                    config.DEVICE, problems, n_demos
                )
                correct += c
                total += t
            acc = correct / total if total > 0 else 0
            print(f"  demo={demo_op} test={test_op}: {acc:.0%}")

    # Test 3: No demos
    print(f"\n--- NO DEMOS (baseline) ---")
    for test_op in ops:
        correct = 0
        total = 0
        for _ in range(args.n_episodes):
            problems = generate_no_demo_episode(n_problems, test_op)
            c, t = run_eval(
                base_model, mach, patched_model, tokenizer,
                config.DEVICE, problems, n_demos=0
            )
            correct += c
            total += t
        acc = correct / total if total > 0 else 0
        print(f"  demo=none test={test_op}: {acc:.0%}")

    print(f"\n{'='*70}")
    print("If demos matter: matched >> mismatched, and mismatched ~ no-demo")
    print("If number ranges leak: matched ~ mismatched >> no-demo")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
