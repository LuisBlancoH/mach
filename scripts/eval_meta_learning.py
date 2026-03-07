"""
Meta-learning speed test: how fast does the system adapt to NEW tasks?

The real test of meta-learning: train on some ops, then introduce held-out ops
and measure how quickly accuracy rises — with the full continuous system
(gradient + Hebbian), not frozen params.

Compares:
1. FULL SYSTEM: gradient descent on meta-params + Hebbian patch updates
2. HEBBIAN ONLY: frozen meta-params, only Hebbian updates (old eval)
3. BASELINE: frozen Qwen, no patches at all
4. SPARSE REWARD: full system but feedback only every N steps

Usage:
    python scripts/eval_meta_learning.py \
        --checkpoint checkpoints/act_hebbian_diverse_ops_L4_R16_P32.pt \
        --n-rank 16 \
        --held-out-ops mod max min \
        --n-steps 200 \
        --sparse-interval 5
"""

import argparse
import os
import sys
import random
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.arithmetic import generate_few_shot_episode, extract_number
from training.two_channel_train import graded_reward


def run_adaptation_test(base_model, mach, patched_model, tokenizer, device,
                        op_type, n_steps=200, mode="full", sparse_interval=1,
                        lr=1e-4):
    """
    Run n_steps of a single operation and track accuracy over time.

    mode:
        "full" — gradient descent + Hebbian (continuous deployment)
        "hebbian" — Hebbian only, no gradient (frozen meta-params)
        "baseline" — no patches, no updates
        "sparse" — full system but reward only every sparse_interval steps
        "reward_only" — NO CE loss, only critic loss + Hebbian (simulates deployment)
    """
    mach.reset_episode()

    if mode in ("full", "sparse", "reward_only"):
        mach.train()
        meta_params = [p for p in mach.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(meta_params, lr=lr)
    else:
        mach.eval()
        optimizer = None

    problems = generate_few_shot_episode(n_steps, n_demos=0, op_type=op_type)

    results = []  # (step, correct, reward, td_error)
    window_ce = torch.tensor(0.0, device=device, requires_grad=True)
    window_critic_losses = []
    truncation_window = 20

    for step, problem in enumerate(problems):
        full_text = problem["prompt"] + problem["answer"]
        encoding = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_len = len(tokenizer(problem["prompt"]).input_ids)
        labels = encoding.input_ids.clone()
        labels[0, :prompt_len] = -100

        if mode == "baseline":
            with torch.no_grad():
                output = base_model(input_ids=encoding.input_ids)
                logits = output.logits
                pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                predicted = extract_number(pred_text)
                correct = (predicted == problem["answer"])
                results.append((step, int(correct), 0, 0))
            continue

        if mode in ("full", "sparse"):
            output = patched_model(input_ids=encoding.input_ids, labels=labels)
            window_ce = window_ce + output.loss
        elif mode == "reward_only":
            # Forward pass WITH patches (so they affect output) but NO CE loss
            # Gradient flows only through critic loss → machinery
            output = patched_model(input_ids=encoding.input_ids, labels=labels)
            # Don't add output.loss to window_ce — CE is not available at deployment
        else:
            with torch.no_grad():
                output = patched_model(input_ids=encoding.input_ids, labels=labels)

        with torch.no_grad():
            logits = output.logits
            pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            predicted = extract_number(pred_text)
            correct = (predicted == problem["answer"])
            reward = graded_reward(predicted, problem["answer"])

        # Sparse reward: only give feedback every N steps
        if mode == "sparse" and step % sparse_interval != 0:
            reward = 0.0  # no feedback this step

        # Hebbian step
        if mode in ("full", "sparse", "reward_only"):
            value, _ = mach.hebbian_step(reward, step, n_steps, device)
            critic_target = torch.tensor(reward, device=device, dtype=torch.float32)
            critic_loss = (value - critic_target) ** 2
            window_critic_losses.append(critic_loss)
        elif mode == "hebbian":
            with torch.no_grad():
                value, _ = mach.hebbian_step(reward, step, n_steps, device)

        td_error = mach._last_td_error if hasattr(mach, '_last_td_error') else 0
        results.append((step, int(correct), reward, td_error))

        # Truncated backprop
        if mode in ("full", "sparse", "reward_only") and (step + 1) % truncation_window == 0:
            if window_critic_losses:
                avg_critic = torch.stack(window_critic_losses).mean()
                # reward_only: critic loss ONLY (no CE — simulates deployment)
                if mode == "reward_only":
                    total_loss = 0.5 * avg_critic
                else:
                    total_loss = window_ce + 0.5 * avg_critic
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(meta_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Detach everything
            for patch in mach.patches:
                if patch.delta_down is not None:
                    patch.delta_down = patch.delta_down.detach()
                if patch.delta_up is not None:
                    patch.delta_up = patch.delta_up.detach()
                if patch.delta_gain is not None:
                    patch.delta_gain = patch.delta_gain.detach()
            if hasattr(mach, 'attn_patches'):
                for patch in mach.attn_patches:
                    if patch.delta_down is not None:
                        patch.delta_down = patch.delta_down.detach()
                    if patch.delta_up is not None:
                        patch.delta_up = patch.delta_up.detach()
                    if patch.delta_gain is not None:
                        patch.delta_gain = patch.delta_gain.detach()
            if hasattr(mach, '_critic_state'):
                mach._critic_state = mach._critic_state.detach()
            for attr in ('_eta_state', '_decay_state', '_expl_state', '_pfc_state'):
                if hasattr(mach, attr):
                    setattr(mach, attr, getattr(mach, attr).detach())
            for rule_attr in ('hebb_rule', 'attn_hebb_rule'):
                rule = getattr(mach, rule_attr, None)
                if rule is not None and hasattr(rule, '_traces') and rule._traces is not None:
                    for p_traces in rule._traces:
                        for r in range(len(p_traces)):
                            p_traces[r] = p_traces[r].detach()
            for key in list(mach._pre_activations.keys()):
                mach._pre_activations[key] = mach._pre_activations[key].detach()
            for key in list(mach._post_activations.keys()):
                mach._post_activations[key] = mach._post_activations[key].detach()
            if hasattr(mach, '_attn_pre_activations'):
                for key in list(mach._attn_pre_activations.keys()):
                    mach._attn_pre_activations[key] = mach._attn_pre_activations[key].detach()
            if hasattr(mach, '_attn_post_activations'):
                for key in list(mach._attn_post_activations.keys()):
                    mach._attn_post_activations[key] = mach._attn_post_activations[key].detach()

            window_ce = torch.tensor(0.0, device=device, requires_grad=True)
            window_critic_losses = []

    return results


def compute_learning_curve(results, window=20):
    """Compute rolling accuracy from results."""
    steps = [r[0] for r in results]
    corrects = [r[1] for r in results]
    curve = []
    for i in range(len(corrects)):
        start = max(0, i - window + 1)
        acc = sum(corrects[start:i+1]) / (i - start + 1)
        curve.append((steps[i], acc))
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-rank", type=int, default=16)
    parser.add_argument("--held-out-ops", nargs="+", default=["mod", "max", "min"])
    parser.add_argument("--train-ops", nargs="+", default=["add", "div", "gcd"])
    parser.add_argument("--n-steps", type=int, default=200,
                        help="Steps per operation per mode")
    parser.add_argument("--sparse-interval", type=int, default=5,
                        help="Feedback every N steps in sparse mode")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading Qwen/Qwen3-4B...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(config.DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    d_model = base_model.config.hidden_size
    n_layers = base_model.config.num_hidden_layers

    from models.universal_module import MACHActivationHebbian, ActivationHebbianPatchedModel

    n_patch_layers = 4
    patch_layers = [
        n_layers // 4, n_layers // 2,
        3 * n_layers // 4, n_layers - 2,
    ]

    # --- BASELINE (no patches) ---
    print("\n" + "=" * 80)
    print("BASELINE (frozen Qwen, no patches)")
    print("=" * 80)
    for op in args.held_out_ops + args.train_ops:
        # Just need a dummy mach for this
        mach = MACHActivationHebbian(
            d_model=d_model, n_layers=n_layers, patch_layers=patch_layers,
            hidden_dim=config.PATCH_HIDDEN_DIM, n_rank=args.n_rank, d_proj=32,
        ).to(config.DEVICE)
        patched_model = ActivationHebbianPatchedModel(base_model, mach)
        results = run_adaptation_test(
            base_model, mach, patched_model, tokenizer, config.DEVICE,
            op, n_steps=args.n_steps, mode="baseline",
        )
        acc = sum(r[1] for r in results) / len(results)
        print(f"  {op:12s} | {acc:.0%}")
        patched_model.remove_hooks()
        del mach, patched_model

    # --- Test each mode ---
    modes = [
        ("HEBBIAN ONLY (frozen meta-params)", "hebbian"),
        ("FULL SYSTEM (gradient + Hebbian)", "full"),
        ("REWARD ONLY (no CE — simulates deployment)", "reward_only"),
        (f"SPARSE REWARD (every {args.sparse_interval} steps)", "sparse"),
    ]

    for mode_name, mode_key in modes:
        print("\n" + "=" * 80)
        print(mode_name)
        print("=" * 80)

        all_ops = args.held_out_ops + args.train_ops
        for op in all_ops:
            is_held_out = op in args.held_out_ops

            # Fresh model from checkpoint each time
            mach = MACHActivationHebbian(
                d_model=d_model, n_layers=n_layers, patch_layers=patch_layers,
                hidden_dim=config.PATCH_HIDDEN_DIM, n_rank=args.n_rank, d_proj=32,
            ).to(config.DEVICE)

            state = torch.load(args.checkpoint, map_location=config.DEVICE)
            mach.load_state_dict(state, strict=False)

            patched_model = ActivationHebbianPatchedModel(base_model, mach)

            results = run_adaptation_test(
                base_model, mach, patched_model, tokenizer, config.DEVICE,
                op, n_steps=args.n_steps, mode=mode_key,
                sparse_interval=args.sparse_interval, lr=args.lr,
            )

            curve = compute_learning_curve(results, window=20)

            # Report accuracy at key points
            tag = "HELD-OUT" if is_held_out else "trained"
            first20 = sum(r[1] for r in results[:20]) / 20
            last20 = sum(r[1] for r in results[-20:]) / 20
            first50 = sum(r[1] for r in results[:50]) / 50
            last50 = sum(r[1] for r in results[-50:]) / 50
            total = sum(r[1] for r in results) / len(results)
            delta = last50 - first50

            print(f"  {op:12s} [{tag:8s}] | "
                  f"first20={first20:.0%} last20={last20:.0%} | "
                  f"first50={first50:.0%} last50={last50:.0%} | "
                  f"Δ={delta:+.0%} total={total:.0%}")

            patched_model.remove_hooks()
            del mach, patched_model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
