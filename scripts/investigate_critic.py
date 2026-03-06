#!/usr/bin/env python3
"""
Investigate within-episode adaptation and neuromodulation dynamics.

For each operation, runs N problems and tracks:
- Critic predictions vs actual rewards
- TD error magnitude over time
- Per-patch eta/decay/exploration values
- Patch delta norms
- Per-step and rolling accuracy (first10 vs last10 = adaptation signal)

Usage:
    python scripts/investigate_critic.py --checkpoint checkpoints/act_hebbian_diverse_ops_L4_R16_P32.pt --n-rank 16
    python scripts/investigate_critic.py --checkpoint checkpoints/act_hebbian_diverse_ops_L4_R16_P32.pt --n-rank 16 --ops mod max min
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


def investigate(checkpoint_path, n_rank=None, ops=None, n_problems=60):
    if ops is None:
        ops = ["add", "sub", "div", "gcd"]
    if n_rank is None:
        n_rank = config.HEBBIAN_N_RANK

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
        hidden_dim=config.PATCH_HIDDEN_DIM, n_rank=n_rank,
        d_proj=config.HEBBIAN_D_PROJ,
    ).to(config.DEVICE)

    if checkpoint_path:
        mach.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE), strict=False)
        print(f"Loaded: {checkpoint_path}")

    patched_model = ActivationHebbianPatchedModel(model, mach)
    mach.eval()

    # Baseline: frozen Qwen without patches
    print(f"\n{'='*90}")
    print("BASELINE (frozen Qwen, no patches)")
    print(f"{'='*90}")
    patched_model.remove_hooks()
    for op in ops:
        problems = generate_few_shot_episode(n_problems, n_demos=0, op_type=op)
        correct_count = 0
        for problem in problems:
            full_text = problem["prompt"] + problem["answer"]
            encoding = tokenizer(full_text, return_tensors="pt").to(config.DEVICE)
            prompt_len = len(tokenizer(problem["prompt"]).input_ids)
            with torch.no_grad():
                output = model(input_ids=encoding.input_ids)
                logits = output.logits
                pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                predicted = extract_number(pred_text)
                if predicted == problem["answer"]:
                    correct_count += 1
        print(f"  {op:<12} {correct_count}/{n_problems} ({correct_count/n_problems:.0%})")
    # Re-register hooks
    patched_model._register_hooks()

    for op in ops:
        print(f"\n{'='*90}")
        print(f"Operation: {op} ({n_problems} problems)")
        print(f"{'='*90}")
        print(f"{'step':>4} {'ok':>3} {'reward':>6} {'td_err':>7} "
              f"{'eta':>20} {'decay':>20} {'Δnorm':>6} {'acc10':>5}")
        print("-" * 90)

        mach.reset_episode()
        problems = generate_few_shot_episode(n_problems, n_demos=0, op_type=op)

        rewards_history = []
        delta_norms_history = []

        for step, problem in enumerate(problems):
            full_text = problem["prompt"] + problem["answer"]
            encoding = tokenizer(full_text, return_tensors="pt").to(config.DEVICE)
            prompt_len = len(tokenizer(problem["prompt"]).input_ids)

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

            rewards_history.append(reward)

            # Hebbian step
            value, _ = mach.hebbian_step(reward, step, n_problems, config.DEVICE)

            td_error = mach._last_td_error
            etas = mach._last_etas
            decays = mach._last_decays

            # Patch delta norms
            total_norm = sum(
                (p.delta_down.norm().item() if p.delta_down is not None else 0) +
                (p.delta_up.norm().item() if p.delta_up is not None else 0)
                for p in mach.patches
            )
            delta_norms_history.append(total_norm)

            # Rolling accuracy (last 10)
            recent = rewards_history[-10:]
            acc_10 = sum(1 for r in recent if r > 0) / len(recent)

            eta_str = "/".join(f"{e:.2f}" for e in etas)
            decay_str = "/".join(f"{d:.2f}" for d in decays)

            print(f"{step:4d} {'Y' if correct else 'N':>3} {reward:>6.1f} {td_error:>+7.3f} "
                  f"  [{eta_str}] [{decay_str}] {total_norm:>6.2f} {acc_10:>4.0%}")

        # Summary
        total_correct = sum(1 for r in rewards_history if r > 0)
        first_10 = sum(1 for r in rewards_history[:10] if r > 0)
        last_10 = sum(1 for r in rewards_history[-10:] if r > 0)
        first_half = sum(1 for r in rewards_history[:n_problems//2] if r > 0)
        last_half = sum(1 for r in rewards_history[n_problems//2:] if r > 0)
        half = n_problems // 2
        print(f"\nSummary: {total_correct}/{n_problems} correct ({total_correct/n_problems:.0%})")
        print(f"  first10={first_10}/10  last10={last_10}/10  (adaptation: {last_10-first_10:+d})")
        print(f"  first_half={first_half}/{half}  last_half={last_half}/{half}  (adaptation: {last_half-first_half:+d})")
        print(f"  Delta norm: start={delta_norms_history[0]:.3f} → end={delta_norms_history[-1]:.3f} "
              f"(Δ={delta_norms_history[-1]-delta_norms_history[0]:+.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-rank", type=int, default=None)
    parser.add_argument("--n-problems", type=int, default=60)
    parser.add_argument("--ops", nargs="+", default=None)
    args = parser.parse_args()
    investigate(args.checkpoint, n_rank=args.n_rank, n_problems=args.n_problems,
                ops=args.ops)
