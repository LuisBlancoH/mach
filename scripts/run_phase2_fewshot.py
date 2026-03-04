#!/usr/bin/env python3
"""Ablation: Phase 2 architecture + few-shot task (no critic, no Phase 4)."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

try:
    import wandb
except ImportError:
    wandb = None

from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from data.arithmetic import extract_number, generate_few_shot_episode
from models.universal_module import MACHPhase2, MACHPatchedModel
from training.phase2_fewshot_train import meta_train_phase2_fewshot


EVAL_OPS = {
    "add": "addition",
    "sub": "subtraction",
    "mul": "multiplication",
    "div": "division",
}


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


def create_mach_phase2(d_model, n_layers, detach_obs=True):
    patch_layers = [
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 2,
    ]
    print(f"Patch layers: {patch_layers}")
    print(f"Detach obs: {detach_obs}")

    mach = MACHPhase2(
        d_model=d_model,
        n_layers=n_layers,
        patch_layers=patch_layers,
        hidden_dim=config.PATCH_HIDDEN_DIM,
        d_meta=config.D_META,
        n_basis=config.N_BASIS,
        detach_obs=detach_obs,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in mach.parameters())
    print(f"MACH Phase 2 (ablation) total parameters: {n_params:,}")

    return mach, patch_layers


def final_evaluation(base_model, mach, patched_model, tokenizer, device,
                     n_episodes=20, n_problems=20, n_demos=5):
    """Few-shot evaluation per operation."""
    print(f"\n{'='*60}")
    print(f"FINAL FEW-SHOT EVALUATION (Phase 2 ablation)")
    print(f"{n_episodes} episodes x {n_problems} problems "
          f"({n_demos} demos + {n_problems - n_demos} test)")
    print(f"{'='*60}")

    any_passed = False
    n_test = n_problems - n_demos

    for op_type, label in EVAL_OPS.items():
        test_correct = 0
        test_total = 0
        test_correct_early = 0
        test_correct_late = 0
        test_total_early = 0
        test_total_late = 0

        for ep in range(n_episodes):
            problems = generate_few_shot_episode(
                n_problems, n_demos=n_demos, op_type=op_type
            )
            mach.reset_episode()
            last_reward = 0.0
            cumulative_reward = 0.0
            test_idx = 0

            for i, problem in enumerate(problems):
                input_ids = tokenizer(
                    problem["prompt"], return_tensors="pt"
                ).input_ids.to(device)

                gru_memory = mach.observe(base_model, input_ids)
                reward_signals = torch.tensor(
                    [last_reward, cumulative_reward, float(i)],
                    device=device, dtype=torch.float32
                )
                writes = mach.fire(gru_memory, reward_signals)
                mach.apply_writes(writes)

                if problem["is_demo"]:
                    continue

                full_text = problem["prompt"] + problem["answer"]
                encoding = tokenizer(
                    full_text, return_tensors="pt"
                ).to(device)
                prompt_len = len(tokenizer(problem["prompt"]).input_ids)

                with torch.no_grad():
                    output = patched_model(input_ids=encoding.input_ids)
                    logits = (
                        output.logits
                        if hasattr(output, 'logits') else output[0]
                    )
                    pred_tokens = logits[
                        0, prompt_len - 1:-1
                    ].argmax(dim=-1)
                    pred_text = tokenizer.decode(
                        pred_tokens, skip_special_tokens=True
                    ).strip()
                    predicted = extract_number(pred_text)
                    correct = (predicted == problem["answer"])

                test_correct += int(correct)
                test_total += 1

                if test_idx < 5:
                    test_correct_early += int(correct)
                    test_total_early += 1
                if test_idx >= n_test - 5:
                    test_correct_late += int(correct)
                    test_total_late += 1
                test_idx += 1

                reward = 1.0 if correct else -1.0
                last_reward = reward
                cumulative_reward += reward

        test_acc = test_correct / test_total if test_total > 0 else 0
        early_acc = (
            test_correct_early / test_total_early
            if test_total_early > 0 else 0
        )
        late_acc = (
            test_correct_late / test_total_late
            if test_total_late > 0 else 0
        )
        delta = late_acc - early_acc
        passed = test_acc >= 0.50

        if passed:
            any_passed = True

        marker = " <<< PASS" if passed else ""
        print(f"\n  {op_type} ({label}):")
        print(f"    Test accuracy:  {test_acc:.1%}{marker}")
        print(f"    Early test:     {early_acc:.1%}")
        print(f"    Late test:      {late_acc:.1%}")
        print(f"    Delta:          {delta:+.1%}")

        if wandb is not None:
            wandb.log({
                f"final/{op_type}_test": test_acc,
                f"final/{op_type}_early": early_acc,
                f"final/{op_type}_late": late_acc,
                f"final/{op_type}_delta": delta,
            })

    print(f"\n{'='*60}")
    print(f"Ablation {'PASS' if any_passed else 'FAIL'}: "
          f"{'at least one' if any_passed else 'no'} operation >= 50%")
    print(f"{'='*60}")

    return any_passed


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: Phase 2 + Few-Shot (no critic, no Phase 4)"
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/phase2_mach.pt",
        help="Phase 2 checkpoint to load"
    )
    parser.add_argument(
        "--from-scratch", action="store_true",
        help="Train from scratch without loading checkpoint"
    )
    parser.add_argument(
        "--undetach-obs", action="store_true",
        help="Allow gradient through obs_proj and GRU"
    )
    args = parser.parse_args()

    detach_obs = not args.undetach_obs
    run_name = "ablation-phase2-fewshot"
    if args.undetach_obs:
        run_name += "-undetach"

    if wandb is not None:
        wandb.init(
            project="mach",
            name=run_name,
            config={
                "base_model": config.BASE_MODEL,
                "d_meta": config.D_META,
                "n_basis": config.N_BASIS,
                "lr": args.lr or config.PHASE2_LR,
                "episodes": args.episodes or config.PHASE2_EPISODES,
                "architecture": "phase2",
                "task": "few_shot",
                "ablation": True,
                "from_scratch": args.from_scratch,
                "detach_obs": detach_obs,
                "device": str(config.DEVICE),
            },
        )

    base_model, tokenizer, d_model, n_layers = load_base_model()
    mach, patch_layers = create_mach_phase2(d_model, n_layers, detach_obs=detach_obs)
    patched_model = MACHPatchedModel(base_model, mach)

    save_path = "checkpoints/phase2_fewshot_ablation.pt"
    os.makedirs("checkpoints", exist_ok=True)

    checkpoint = None if args.from_scratch else args.checkpoint
    meta_train_phase2_fewshot(
        base_model, mach, patched_model, tokenizer, config.DEVICE,
        n_episodes=args.episodes, lr=args.lr,
        checkpoint_path=checkpoint,
        save_path=save_path,
    )

    passed = final_evaluation(
        base_model, mach, patched_model, tokenizer, config.DEVICE
    )

    torch.save(mach.state_dict(), save_path)
    print(f"\nSaved final checkpoint to {save_path}")

    if wandb is not None:
        wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
