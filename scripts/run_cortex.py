#!/usr/bin/env python3
"""
Cortex experiment: PC network that reads from Qwen and proposes output.

Qwen runs fully intact (frozen). The cortex observes hidden states at
multiple layers, processes through a predictive coding hierarchy, and
produces a modified final hidden state for generation.

No patching. No corruption. Qwen computes normally.

Usage:
    # Continuous training on diverse operations
    python scripts/run_cortex.py --n-steps 10000

    # Baseline comparison
    python scripts/run_cortex.py --baseline-only
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from models.cortex import Cortex, CortexModel


def load_base_model():
    print(f"Loading {config.BASE_MODEL} on {config.DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, dtype=config.DTYPE
    ).to(config.DEVICE)

    for param in model.parameters():
        param.requires_grad = False

    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  d_model={d_model}, n_layers={n_layers}")
    return model, tokenizer, d_model, n_layers


def main():
    parser = argparse.ArgumentParser(description="Cortex Experiment")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--d-cortical", type=int, default=512)
    parser.add_argument("--n-cortex-layers", type=int, default=4)
    parser.add_argument("--n-settle", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-observe-layers", type=int, default=4)
    args = parser.parse_args()

    model, tokenizer, d_model, n_layers = load_base_model()

    # Observation layers at quarter points
    if args.n_observe_layers == 4:
        observe_layers = [
            n_layers // 4, n_layers // 2,
            3 * n_layers // 4, n_layers - 2,
        ]
    else:
        step = n_layers // args.n_observe_layers
        observe_layers = [min(i * step, n_layers - 2)
                          for i in range(args.n_observe_layers)]

    print(f"Observe layers: {observe_layers}")

    # Create cortex
    cortex = Cortex(
        d_model=d_model,
        d_cortical=args.d_cortical,
        n_layers=args.n_cortex_layers,
        n_settle=args.n_settle,
        observe_layers=observe_layers,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in cortex.parameters())
    n_buffers = sum(b.numel() for b in cortex.buffers())
    print(f"Cortex params (genome): {n_params:,}")
    print(f"Cortex buffers (plastic): {n_buffers:,}")

    cortex_model = CortexModel(model, cortex)

    if args.baseline_only:
        from evaluation.baseline import evaluate_model
        from data.arithmetic import generate_arithmetic_problems

        print("\n=== Baseline (pure Qwen) ===")
        for diff in [6, 7, 8, 9]:
            problems = generate_arithmetic_problems(200, diff)
            acc = evaluate_model(model, tokenizer, problems,
                                 label=f"baseline d{diff}")
            print(f"  Difficulty {diff}: {acc:.2%}")

        print("\n=== Cortex (untrained) ===")
        for diff in [6, 7, 8, 9]:
            problems = generate_arithmetic_problems(200, diff)
            acc = evaluate_model(cortex_model, tokenizer, problems,
                                 label=f"cortex d{diff}")
            print(f"  Difficulty {diff}: {acc:.2%}")
        return

    # === Training ===
    from training.two_channel_train import (
        DIVERSE_TRAIN_OPS, generate_few_shot_episode
    )
    from data.arithmetic import extract_number

    # Optimizer for genome (input/output projections, critic, norms, precision)
    genome_params = []
    genome_params.extend(cortex.input_projs.parameters())
    genome_params.extend(cortex.fuse.parameters())
    genome_params.extend(cortex.output_proj.parameters())
    genome_params.extend(cortex.critic.parameters())
    for layer in cortex.layers:
        genome_params.append(layer.precision_logit)
        genome_params.extend(layer.norm.parameters())

    optimizer = torch.optim.Adam(genome_params, lr=args.lr)

    # Load checkpoint
    save_path = (f"checkpoints/cortex_diverse_ops"
                 f"_L{args.n_cortex_layers}_C{args.d_cortical}.pt")
    os.makedirs("checkpoints", exist_ok=True)
    start_step = 0
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=config.DEVICE, weights_only=False)
        cortex.load_state_dict(ckpt['cortex'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt.get('step', 0)
        print(f"  Resumed from step {start_step}")

    print(f"\n=== Cortex Training ===")
    print(f"  n_steps={args.n_steps}, lr={args.lr}")
    print(f"  d_cortical={args.d_cortical}, n_layers={args.n_cortex_layers}")
    print(f"  n_settle={args.n_settle}")
    print(f"  save_path={save_path}")

    cortex.train()
    current_op = random.choice(DIVERSE_TRAIN_OPS)
    op_step_count = 0
    op_switch_interval = 20

    all_correct = []
    step_timer = time.time()

    for step in range(start_step, args.n_steps):
        # Switch operation periodically
        op_step_count += 1
        if op_step_count >= op_switch_interval:
            current_op = random.choice(DIVERSE_TRAIN_OPS)
            op_step_count = 0
            cortex.reset_episode()

        # Generate one problem
        problems = generate_few_shot_episode(1, n_demos=0, op_type=current_op)
        p = problems[0]

        full_text = p["prompt"] + p["answer"]
        encoding = tokenizer(full_text, return_tensors="pt").to(config.DEVICE)
        input_ids = encoding.input_ids

        prompt_len = len(tokenizer(p["prompt"]).input_ids)
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100

        # Forward: Qwen intact → cortex observes → cortex proposes
        outputs = cortex_model(input_ids=input_ids, labels=labels)
        ce_loss = outputs.loss

        # Gradient step for genome
        optimizer.zero_grad()
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(genome_params, 1.0)
        optimizer.step()

        # Check accuracy via generation
        with torch.no_grad():
            prompt_ids = tokenizer(
                p["prompt"], return_tensors="pt"
            ).input_ids.to(config.DEVICE)
            gen = cortex_model.generate(prompt_ids, max_new_tokens=20)
            gen_text = tokenizer.decode(
                gen[0][prompt_ids.shape[1]:], skip_special_tokens=True
            )
            pred_num = extract_number(gen_text)
            correct = (pred_num == p.get("numerical_answer"))
            all_correct.append(1.0 if correct else 0.0)

        # Hebbian update
        reward = 1.0 if correct else -0.5
        cortex.hebbian_step(reward, device=config.DEVICE)

        # Logging
        if (step + 1) % 100 == 0:
            recent = all_correct[-100:]
            acc = sum(recent) / len(recent)
            elapsed = time.time() - step_timer
            rate = 100 / elapsed if elapsed > 0 else 0
            step_timer = time.time()

            print(
                f"Step {step+1:5d} | op={current_op:<10} | "
                f"acc(100)={acc:.0%} ce={ce_loss.item():.3f} "
                f"[{rate:.1f} st/s]"
            )

        # Diagnostics + checkpoint
        if (step + 1) % 2000 == 0:
            diag = cortex.get_diagnostics()
            print("  Diagnostics:")
            for key in sorted(diag.keys()):
                val = diag[key]
                print(f"    {key}: {val:.4f}")

            # Gradient norms
            print("  Gradient norms:")
            for name, param in cortex.named_parameters():
                if param.grad is not None:
                    print(f"    {name}: {param.grad.norm().item():.6f}")

            # Save checkpoint
            torch.save({
                'cortex': cortex.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step + 1,
            }, save_path)
            print(f"  Checkpoint saved to {save_path}")

            # Quick eval
            print(f"  --- Validation (step {step+1}) ---")
            eval_ops = DIVERSE_TRAIN_OPS[:6]
            for eval_op in eval_ops:
                eval_problems = generate_few_shot_episode(
                    20, n_demos=0, op_type=eval_op
                )
                n_correct = 0
                for ep in eval_problems:
                    with torch.no_grad():
                        prompt_ids = tokenizer(
                            ep["prompt"], return_tensors="pt"
                        ).input_ids.to(config.DEVICE)
                        gen = cortex_model.generate(
                            prompt_ids, max_new_tokens=20
                        )
                        gen_text = tokenizer.decode(
                            gen[0][prompt_ids.shape[1]:],
                            skip_special_tokens=True
                        )
                        pred_num = extract_number(gen_text)
                        if pred_num == ep.get("numerical_answer"):
                            n_correct += 1
                print(f"  EVAL {eval_op:<10} | "
                      f"{n_correct}/{len(eval_problems)} = "
                      f"{n_correct/len(eval_problems):.0%}")


if __name__ == "__main__":
    main()
