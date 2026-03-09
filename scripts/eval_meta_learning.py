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
                        lr=1e-4, hippocampus=None, thinking_tokens=0):
    """
    Run n_steps of a single operation and track accuracy over time.

    mode:
        "full" — gradient descent + Hebbian (continuous deployment)
        "hebbian" — Hebbian only, no gradient (frozen meta-params)
        "baseline" — no patches, no updates
        "sparse" — full system but reward only every sparse_interval steps
        "reward_only" — NO CE loss, only critic loss + Hebbian (simulates deployment)

    thinking_tokens: if > 0, generate N tokens before answering (inner speech/CoT)
    """
    mach.reset_episode()

    if mode in ("full", "sparse", "reward_only"):
        mach.train()
        mach_params = [p for p in mach.parameters() if p.requires_grad]
        hipp_params = []
        if hippocampus is not None:
            hipp_params = [p for p in hippocampus.parameters() if p.requires_grad]
        meta_params = mach_params + hipp_params
        optimizer = torch.optim.Adam(meta_params, lr=lr)
    else:
        mach.eval()
        mach_params = []
        hipp_params = []
        optimizer = None

    problems = generate_few_shot_episode(n_steps, n_demos=0, op_type=op_type)

    results = []  # (step, correct, reward, td_error)
    window_ce = torch.tensor(0.0, device=device, requires_grad=True)
    window_critic_losses = []
    window_nuclei_losses = []
    window_hipp_losses = []
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

        # Hippocampus retrieval: reinstate similar neural states
        if hippocampus is not None and len(hippocampus) > 0:
            act_summary = mach.get_activation_summary()
            act_summary = act_summary / (act_summary.norm() + 1e-8)
            td_err = mach._last_td_error if hasattr(mach, '_last_td_error') else 0
            hippocampus.retrieve_and_reinstate(
                mach, act_summary, td_err, top_k=3, device=device
            )

        if thinking_tokens > 0 and mode != "baseline":
            # CoT: generate thinking tokens, then score answer
            prompt_ids = encoding.input_ids[0, :prompt_len].unsqueeze(0)
            with torch.no_grad():
                generated = prompt_ids
                for _ in range(thinking_tokens):
                    out = patched_model(input_ids=generated)
                    next_token = out.logits[0, -1:].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                    tok_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
                    if "=" in tok_text or "\n" in tok_text:
                        break

            answer_ids = tokenizer(problem["answer"], return_tensors="pt",
                                   add_special_tokens=False).input_ids.to(device)
            full_ids = torch.cat([generated, answer_ids], dim=1)
            cot_labels = full_ids.clone()
            cot_labels[0, :generated.shape[1]] = -100

            if mode in ("full", "sparse"):
                output = patched_model(input_ids=full_ids, labels=cot_labels)
                window_ce = window_ce + output.loss
            elif mode == "reward_only":
                output = patched_model(input_ids=full_ids, labels=cot_labels)
            else:
                with torch.no_grad():
                    output = patched_model(input_ids=full_ids, labels=cot_labels)

            with torch.no_grad():
                pred_tokens = output.logits[0, generated.shape[1] - 1:-1].argmax(dim=-1)
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                predicted = extract_number(pred_text)
                correct = (predicted == problem["answer"])
                reward = graded_reward(predicted, problem["answer"])
        else:
            # Direct: no thinking
            if mode in ("full", "sparse"):
                output = patched_model(input_ids=encoding.input_ids, labels=labels)
                window_ce = window_ce + output.loss
            elif mode == "reward_only":
                output = patched_model(input_ids=encoding.input_ids, labels=labels)
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
            # TD bootstrapped critic loss
            if hasattr(mach, '_prev_critic_value') and mach._prev_critic_value is not None:
                prev_gamma = mach._prev_gamma if hasattr(mach, '_prev_gamma') and mach._prev_gamma is not None else mach.gamma
                td_target = mach._prev_reward + prev_gamma * value.detach()
                critic_loss = (mach._prev_critic_value - td_target) ** 2
                window_critic_losses.append(critic_loss)
            mach._prev_critic_value = value
            mach._prev_reward = torch.tensor(reward, device=device, dtype=torch.float32)
            mach._prev_gamma = mach._current_gamma.detach() if hasattr(mach, '_current_gamma') else mach.gamma
            if hasattr(mach, '_nuclei_loss'):
                window_nuclei_losses.append(mach._nuclei_loss)
        elif mode == "hebbian":
            with torch.no_grad():
                value, _ = mach.hebbian_step(reward, step, n_steps, device)

        td_error = mach._last_td_error if hasattr(mach, '_last_td_error') else 0

        # Hippocampus: reconsolidate + store + update dynamics
        if hippocampus is not None and mode != "baseline":
            gamma = mach._last_gamma if hasattr(mach, '_last_gamma') else 0.95
            avg_decay = mach._last_decays.mean().item() if hasattr(mach, '_last_decays') and mach._last_decays is not None else 0.9
            hippocampus.set_neuromod(gamma, avg_decay)
            hippocampus.reconsolidate(td_error)
            act_summary = mach.get_activation_summary()
            act_summary = act_summary / (act_summary.norm() + 1e-8)
            hippocampus.store(mach, act_summary, reward, td_error, global_step=step)
            # Local REINFORCE loss for key_proj (learns at inference too)
            if mode in ("full", "sparse", "reward_only"):
                hipp_local = hippocampus.compute_local_loss(td_error)
                if hipp_local.abs().item() > 0:
                    window_hipp_losses.append(hipp_local)

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
                # Nuclei auxiliary loss
                if window_nuclei_losses:
                    avg_nuclei = torch.stack(window_nuclei_losses).mean()
                    total_loss = total_loss + 0.1 * avg_nuclei
                # Hippocampus local loss (key_proj learns at inference)
                if window_hipp_losses:
                    avg_hipp = torch.stack(window_hipp_losses).mean()
                    total_loss = total_loss + 0.05 * avg_hipp
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(mach_params, max_norm=1.0)
                if hipp_params:
                    torch.nn.utils.clip_grad_norm_(hipp_params, max_norm=1.0)
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
            for attr in ('_eta_state', '_decay_state', '_expl_state', '_gamma_state', '_pfc_state'):
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

            # Detach TD bootstrapping state
            if hasattr(mach, '_prev_critic_value') and mach._prev_critic_value is not None:
                mach._prev_critic_value = mach._prev_critic_value.detach()

            window_ce = torch.tensor(0.0, device=device, requires_grad=True)
            window_critic_losses = []
            window_nuclei_losses = []
            window_hipp_losses = []

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
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test: 25 steps, 1 op per category")
    parser.add_argument("--memory-path", type=str, default=None,
                        help="Path to hippocampal memory file (adds +hippocampus eval modes)")
    parser.add_argument("--sleep-cycles", type=int, default=0,
                        help="Run N sleep cycles (NREM+REM) before eval to measure consolidation effect")
    parser.add_argument("--thinking-tokens", type=int, default=0,
                        help="CoT: generate N thinking tokens before answering (0=off)")
    args = parser.parse_args()

    if args.quick:
        args.n_steps = 25
        args.held_out_ops = [args.held_out_ops[0]]  # just first held-out op
        args.train_ops = [args.train_ops[0]]          # just first trained op

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
    # (name, mode_key, use_hipp, thinking_tokens, sleep_type)
    # sleep_type: None, "both", "nrem", "rem"
    modes = [
        ("HEBBIAN ONLY (frozen meta-params)", "hebbian", False, 0, None),
        ("FULL SYSTEM (gradient + Hebbian)", "full", False, 0, None),
        ("REWARD ONLY (no CE — simulates deployment)", "reward_only", False, 0, None),
        (f"SPARSE REWARD (every {args.sparse_interval} steps)", "sparse", False, 0, None),
    ]
    if args.memory_path:
        modes.extend([
            ("FULL + HIPPOCAMPUS", "full", True, 0, None),
            ("REWARD ONLY + HIPPOCAMPUS", "reward_only", True, 0, None),
        ])
    if args.sleep_cycles > 0 and args.memory_path:
        modes.extend([
            (f"FULL + HIPP + NREM ONLY ({args.sleep_cycles} cycles)", "full", True, 0, "nrem"),
            (f"FULL + HIPP + REM ONLY ({args.sleep_cycles} cycles)", "full", True, 0, "rem"),
            (f"FULL + HIPP + SLEEP ({args.sleep_cycles} cycles)", "full", True, 0, "both"),
            (f"REWARD ONLY + HIPP + SLEEP ({args.sleep_cycles} cycles)", "reward_only", True, 0, "both"),
        ])
    if args.thinking_tokens > 0:
        modes.extend([
            (f"FULL + CoT ({args.thinking_tokens} tokens)", "full", False, args.thinking_tokens, None),
            (f"REWARD ONLY + CoT ({args.thinking_tokens} tokens)", "reward_only", False, args.thinking_tokens, None),
        ])
        if args.memory_path:
            modes.extend([
                (f"FULL + HIPP + CoT ({args.thinking_tokens} tokens)", "full", True, args.thinking_tokens, None),
            ])

    for mode_name, mode_key, use_hipp, cot_tokens, sleep_type in modes:
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

            # Create hippocampus if needed (fresh per op, or loaded from file)
            hipp = None
            if use_hipp and args.memory_path:
                from models.hippocampus import Hippocampus
                key_dim = len(mach.patch_layers) * mach.hebb_rule.d_proj
                hipp = Hippocampus(
                    key_dim=key_dim, pfc_dim=32, n_patches=mach.n_patches,
                    save_path=args.memory_path,
                ).to(config.DEVICE)

            # Pre-eval sleep: run NREM and/or REM cycles to consolidate before testing
            if sleep_type is not None and hipp is not None and len(hipp) > 0:
                total_nrem = 0
                total_rem = 0
                rem_td_errors = []
                rem_critic_vals = []
                pre_norm = sum(
                    (p.delta_down.norm().item() if p.delta_down is not None else 0) +
                    (p.delta_up.norm().item() if p.delta_up is not None else 0)
                    for p in mach.patches
                )
                for cycle in range(args.sleep_cycles):
                    if sleep_type in ("nrem", "both"):
                        n_nrem = hipp.replay_nrem(mach, n_replays=4, device=config.DEVICE)
                        total_nrem += n_nrem
                    if sleep_type in ("rem", "both"):
                        dreams = hipp.replay_rem(
                            mach, patched_model, tokenizer, n_dreams=2, device=config.DEVICE
                        )
                        total_rem += len(dreams)
                        for d in dreams:
                            rem_td_errors.append(d['td_error'])
                            rem_critic_vals.append(d['critic_value'])
                post_norm = sum(
                    (p.delta_down.norm().item() if p.delta_down is not None else 0) +
                    (p.delta_up.norm().item() if p.delta_up is not None else 0)
                    for p in mach.patches
                )
                rem_avg_td = sum(abs(t) for t in rem_td_errors) / len(rem_td_errors) if rem_td_errors else 0
                rem_avg_val = sum(rem_critic_vals) / len(rem_critic_vals) if rem_critic_vals else 0
                parts = []
                if total_nrem > 0:
                    parts.append(f"NREM={total_nrem}")
                if total_rem > 0:
                    parts.append(f"REM={total_rem} avg_|td|={rem_avg_td:.4f} critic_val={rem_avg_val:.4f}")
                print(f"    Sleep: {', '.join(parts)} | patch_Δ={post_norm - pre_norm:+.4f}")

            results = run_adaptation_test(
                base_model, mach, patched_model, tokenizer, config.DEVICE,
                op, n_steps=args.n_steps, mode=mode_key,
                sparse_interval=args.sparse_interval, lr=args.lr,
                hippocampus=hipp, thinking_tokens=cot_tokens,
            )

            curve = compute_learning_curve(results, window=20)

            # Report accuracy at key points
            tag = "HELD-OUT" if is_held_out else "trained"
            n = len(results)
            w1 = min(20, n // 2)
            w2 = min(50, n // 2)
            first_w1 = sum(r[1] for r in results[:w1]) / max(w1, 1)
            last_w1 = sum(r[1] for r in results[-w1:]) / max(w1, 1)
            first_w2 = sum(r[1] for r in results[:w2]) / max(w2, 1)
            last_w2 = sum(r[1] for r in results[-w2:]) / max(w2, 1)
            total = sum(r[1] for r in results) / n
            delta = last_w2 - first_w2

            hipp_str = ""
            if hipp is not None:
                hipp_str = f" mem={len(hipp)}"

            print(f"  {op:12s} [{tag:8s}] | "
                  f"first{w1}={first_w1:.0%} last{w1}={last_w1:.0%} | "
                  f"first{w2}={first_w2:.0%} last{w2}={last_w2:.0%} | "
                  f"Δ={delta:+.0%} total={total:.0%}{hipp_str}")

            patched_model.remove_hooks()
            del mach, patched_model, hipp
            torch.cuda.empty_cache()

    # --- Sequential multi-op eval (simulates real deployment) ---
    if args.memory_path:
        all_ops = args.held_out_ops + args.train_ops
        steps_per_op = min(50, args.n_steps // 5)  # shorter blocks, more cycles
        n_cycles = 5

        for mode_name, mode_key, use_hipp in [
            ("SEQUENTIAL: full (no hipp)", "full", False),
            ("SEQUENTIAL: full + hippocampus", "full", True),
            ("SEQUENTIAL: reward_only (no hipp)", "reward_only", False),
            ("SEQUENTIAL: reward_only + hippocampus", "reward_only", True),
        ]:
            print("\n" + "=" * 80)
            print(f"{mode_name} — {steps_per_op} steps/op × {n_cycles} cycles")
            print("=" * 80)

            op_results, hipp_count = run_sequential_eval(
                base_model, d_model, n_layers, patch_layers, tokenizer,
                args.checkpoint, args.n_rank, all_ops, steps_per_op, n_cycles,
                args.lr, mode_key, use_hipp, args.memory_path,
            )

            for op in all_ops:
                is_held_out = op in args.held_out_ops
                tag = "HELD-OUT" if is_held_out else "trained"
                accs = [a for _, a in op_results[op]]
                first = accs[0]
                last = accs[-1]
                avg = sum(accs) / len(accs)
                cycle_str = " → ".join(f"{a:.0%}" for a in accs)
                print(f"  {op:12s} [{tag:8s}] | {cycle_str} | avg={avg:.0%}")

            if hipp_count > 0:
                print(f"  Hippocampus: {hipp_count} memories")


def run_sequential_eval(base_model, d_model, n_layers, patch_layers, tokenizer,
                        checkpoint, n_rank, ops, steps_per_op, n_cycles, lr,
                        mode, use_hipp, memory_path):
    """Run ops sequentially with shared mach + hippocampus across all ops.

    Simulates real deployment: ops interleave, patches persist, hippocampus
    builds memories across ops and reinstates when ops repeat.

    Returns dict of {op: [(cycle, accuracy)]}
    """
    from models.hippocampus import Hippocampus

    mach = MACHActivationHebbian(
        d_model=d_model, n_layers=n_layers, patch_layers=patch_layers,
        hidden_dim=config.PATCH_HIDDEN_DIM, n_rank=n_rank, d_proj=32,
    ).to(config.DEVICE)
    state = torch.load(checkpoint, map_location=config.DEVICE)
    mach.load_state_dict(state, strict=False)
    patched_model = ActivationHebbianPatchedModel(base_model, mach)

    hipp = None
    if use_hipp:
        key_dim = len(mach.patch_layers) * mach.hebb_rule.d_proj
        hipp = Hippocampus(
            key_dim=key_dim, pfc_dim=32, n_patches=mach.n_patches,
        ).to(config.DEVICE)

    # No reset_episode — patches and state persist across everything
    mach.reset_episode()
    mach.train()
    mach_params = [p for p in mach.parameters() if p.requires_grad]
    hipp_params = []
    if hipp is not None:
        hipp_params = [p for p in hipp.parameters() if p.requires_grad]
    meta_params = mach_params + hipp_params
    optimizer = torch.optim.Adam(meta_params, lr=lr)

    op_results = {op: [] for op in ops}
    truncation_window = 20
    window_ce = torch.tensor(0.0, device=config.DEVICE, requires_grad=True)
    window_critic_losses = []
    window_nuclei_losses = []
    window_hipp_losses = []
    global_step = 0

    for cycle in range(n_cycles):
        for op in ops:
            problems = generate_few_shot_episode(steps_per_op, n_demos=0, op_type=op)
            corrects = []

            for problem in problems:
                full_text = problem["prompt"] + problem["answer"]
                encoding = tokenizer(full_text, return_tensors="pt").to(config.DEVICE)
                prompt_len = len(tokenizer(problem["prompt"]).input_ids)
                labels = encoding.input_ids.clone()
                labels[0, :prompt_len] = -100

                # Hippocampus retrieval
                if hipp is not None and len(hipp) > 0:
                    act_summary = mach.get_activation_summary()
                    act_summary = act_summary / (act_summary.norm() + 1e-8)
                    td_err = mach._last_td_error if hasattr(mach, '_last_td_error') else 0
                    hipp.retrieve_and_reinstate(mach, act_summary, td_err, top_k=3, device=config.DEVICE)

                if mode == "full":
                    output = patched_model(input_ids=encoding.input_ids, labels=labels)
                    window_ce = window_ce + output.loss
                else:  # reward_only
                    output = patched_model(input_ids=encoding.input_ids, labels=labels)

                with torch.no_grad():
                    logits = output.logits
                    pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
                    pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                    predicted = extract_number(pred_text)
                    correct = (predicted == problem["answer"])
                    reward = graded_reward(predicted, problem["answer"])

                corrects.append(int(correct))

                value, _ = mach.hebbian_step(reward, 0, 1, config.DEVICE)

                # Hippocampus store + reconsolidate
                if hipp is not None:
                    td_error = mach._last_td_error if hasattr(mach, '_last_td_error') else 0
                    gamma = mach._last_gamma if hasattr(mach, '_last_gamma') else 0.95
                    avg_decay = mach._last_decays.mean().item() if hasattr(mach, '_last_decays') and mach._last_decays is not None else 0.9
                    hipp.set_neuromod(gamma, avg_decay)
                    hipp.reconsolidate(td_error)
                    act_summary = mach.get_activation_summary()
                    act_summary = act_summary / (act_summary.norm() + 1e-8)
                    hipp.store(mach, act_summary, reward, td_error)
                    # Local REINFORCE loss for key_proj
                    hipp_local = hipp.compute_local_loss(td_error)
                    if hipp_local.abs().item() > 0:
                        window_hipp_losses.append(hipp_local)

                # Critic loss
                if hasattr(mach, '_prev_critic_value') and mach._prev_critic_value is not None:
                    prev_gamma = mach._prev_gamma if hasattr(mach, '_prev_gamma') and mach._prev_gamma is not None else mach.gamma
                    td_target = mach._prev_reward + prev_gamma * value.detach()
                    critic_loss = (mach._prev_critic_value - td_target) ** 2
                    window_critic_losses.append(critic_loss)
                mach._prev_critic_value = value
                mach._prev_reward = torch.tensor(reward, device=config.DEVICE, dtype=torch.float32)
                mach._prev_gamma = mach._current_gamma.detach() if hasattr(mach, '_current_gamma') else mach.gamma
                if hasattr(mach, '_nuclei_loss'):
                    window_nuclei_losses.append(mach._nuclei_loss)

                global_step += 1

                # Truncated backprop
                if global_step % truncation_window == 0:
                    if window_critic_losses:
                        avg_critic = torch.stack(window_critic_losses).mean()
                        if mode == "reward_only":
                            total_loss = 0.5 * avg_critic
                        else:
                            total_loss = window_ce + 0.5 * avg_critic
                        if window_nuclei_losses:
                            avg_nuclei = torch.stack(window_nuclei_losses).mean()
                            total_loss = total_loss + 0.1 * avg_nuclei
                        if window_hipp_losses:
                            avg_hipp = torch.stack(window_hipp_losses).mean()
                            total_loss = total_loss + 0.05 * avg_hipp
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(mach_params, max_norm=1.0)
                        if hipp_params:
                            torch.nn.utils.clip_grad_norm_(hipp_params, max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                    # Detach all state
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
                    for attr in ('_eta_state', '_decay_state', '_expl_state', '_gamma_state', '_pfc_state'):
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
                    if hasattr(mach, '_attn_pre'):
                        for key in list(mach._attn_pre.keys()):
                            mach._attn_pre[key] = mach._attn_pre[key].detach()
                    if hasattr(mach, '_attn_post'):
                        for key in list(mach._attn_post.keys()):
                            mach._attn_post[key] = mach._attn_post[key].detach()
                    if hasattr(mach, '_prev_critic_value') and mach._prev_critic_value is not None:
                        mach._prev_critic_value = mach._prev_critic_value.detach()

                    window_ce = torch.tensor(0.0, device=config.DEVICE, requires_grad=True)
                    window_critic_losses = []
                    window_nuclei_losses = []
                    window_hipp_losses = []

            acc = sum(corrects) / len(corrects)
            op_results[op].append((cycle, acc))

    patched_model.remove_hooks()
    hipp_count = len(hipp) if hipp is not None else 0
    return op_results, hipp_count


if __name__ == "__main__":
    main()
