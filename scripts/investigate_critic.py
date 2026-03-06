#!/usr/bin/env python3
"""
Investigate why patches drift: is it the critic, eta, or decay?

For each operation, runs N problems and tracks:
- Critic predictions vs actual rewards (is critic accurate?)
- TD error magnitude over time (does it shrink when doing well?)
- Eta values (does learning rate decrease when consistent?)
- Patch delta norms (are patches stabilizing or wandering?)
- Per-step accuracy (when does drift start?)

Usage:
    python scripts/investigate_critic.py --checkpoint checkpoints/act_hebbian_diverse_ops_L4_R16_P32.pt --n-rank 16
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

    print(f"\neta_scale={mach.eta_scale.item():.4f}, decay_base={mach.decay_base.item():.4f}")
    print(f"decay = sigmoid(decay_base) = {torch.sigmoid(mach.decay_base).item():.4f}")
    print(f"eta at td_error=0: clamp(0) = 0.000")
    print(f"eta at td_error=1: clamp({mach.eta_scale.item():.3f}) = {min(mach.eta_scale.item() * 1.0, 1.0):.4f}")
    print(f"eta at td_error=2: clamp({2*mach.eta_scale.item():.3f}) = {min(mach.eta_scale.item() * 2.0, 1.0):.4f}")

    patched_model = ActivationHebbianPatchedModel(model, mach)
    mach.eval()

    for op in ops:
        print(f"\n{'='*80}")
        print(f"Operation: {op} ({n_problems} problems)")
        print(f"{'='*80}")
        print(f"{'step':>4} {'correct':>7} {'reward':>6} {'critic':>6} {'td_err':>6} "
              f"{'eta':>6} {'gate':>8} {'Δnorm':>8} {'acc_10':>6}")
        print("-" * 75)

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

            # Get critic prediction BEFORE hebbian step
            act_summary = mach.get_activation_summary()
            act_summary = act_summary / (act_summary.norm() + 1e-8)
            critic_val = mach.critic(mach.critic_proj(act_summary)).squeeze(-1).item()

            # Hebbian step
            value, _ = mach.hebbian_step(reward, step, n_problems, config.DEVICE)

            td_error = mach._last_td_error
            eta = mach._last_etas[0].item()
            gate = eta * td_error * mach.gate_scale

            # Patch delta norms
            total_norm = sum(
                p.delta_down.norm().item() + p.delta_up.norm().item()
                for p in mach.patches
            )
            delta_norms_history.append(total_norm)

            # Rolling accuracy (last 10)
            recent = rewards_history[-10:]
            acc_10 = sum(1 for r in recent if r > 0) / len(recent)

            print(f"{step:4d} {'✓' if correct else '✗':>7} {reward:>6.1f} {critic_val:>6.3f} "
                  f"{td_error:>6.3f} {eta:>6.4f} {gate:>8.5f} {total_norm:>8.3f} {acc_10:>5.0%}")

        # Summary
        total_correct = sum(1 for r in rewards_history if r > 0)
        first_10 = sum(1 for r in rewards_history[:10] if r > 0)
        last_10 = sum(1 for r in rewards_history[-10:] if r > 0)
        print(f"\nSummary: {total_correct}/{n_problems} correct "
              f"(first10={first_10}/10, last10={last_10}/10)")
        print(f"Delta norm: start={delta_norms_history[0]:.3f} → end={delta_norms_history[-1]:.3f} "
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
