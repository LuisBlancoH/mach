#!/usr/bin/env python3
"""Ablation: Phase 3 architecture + few-shot task (no Phase 4 additions)."""

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
from models.universal_module import MACHPhase3, MACHPatchedModel
from training.phase3_fewshot_train import meta_train_phase3_fewshot


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


def create_mach_phase3(d_model, n_layers):
    patch_layers = [
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 2,
    ]
    print(f"Patch layers: {patch_layers}")

    mach = MACHPhase3(
        d_model=d_model,
        n_layers=n_layers,
        patch_layers=patch_layers,
        hidden_dim=config.PATCH_HIDDEN_DIM,
        d_meta=config.D_META,
        n_basis=config.N_BASIS,
        detach_obs=True,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in mach.parameters())
    print(f"MACH Phase 3 (ablation) total parameters: {n_params:,}")

    return mach, patch_layers


def final_evaluation(base_model, mach, patched_model, tokenizer, device,
                     n_episodes=20, n_problems=20, n_demos=5):
    """Few-shot evaluation per operation."""
    print(f"\n{'='*60}")
    print(f"FINAL FEW-SHOT EVALUATION (Phase 3 ablation)")
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
            last_value = torch.tensor(0.0, device=device)
            last_td_error = torch.tensor(0.0, device=device)
            test_idx = 0

            for i, problem in enumerate(problems):
                input_ids = tokenizer(
                    problem["prompt"], return_tensors="pt"
                ).input_ids.to(device)

                gru_memory = mach.observe(base_model, input_ids)
                writes = mach.fire(gru_memory, last_value, last_td_error)

                with torch.no_grad():
                    current_value = mach.get_value()

                mach.apply_writes(writes)

                if problem["is_demo"]:
                    last_value = current_value.detach()
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
                if test_idx > 1:
                    td_target = torch.tensor(
                        reward, device=device, dtype=torch.float32
                    )
                    last_td_error = (td_target - last_value).detach()
                last_value = current_value.detach()

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
        description="Ablation: Phase 3 + Few-Shot (no Phase 4 additions)"
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/phase3_mach.pt",
        help="Phase 3 checkpoint to load"
    )
    parser.add_argument(
        "--from-scratch", action="store_true",
        help="Train from scratch without loading checkpoint"
    )
    args = parser.parse_args()

    if wandb is not None:
        wandb.init(
            project="mach",
            name="ablation-phase3-fewshot",
            config={
                "base_model": config.BASE_MODEL,
                "d_meta": config.D_META,
                "n_basis": config.N_BASIS,
                "lr": args.lr or config.PHASE3_LR,
                "episodes": args.episodes or config.PHASE3_EPISODES,
                "critic_loss_weight": config.PHASE3_CRITIC_LOSS_WEIGHT,
                "gamma": config.PHASE3_GAMMA,
                "architecture": "phase3",
                "task": "few_shot",
                "ablation": True,
                "device": str(config.DEVICE),
            },
        )

    base_model, tokenizer, d_model, n_layers = load_base_model()
    mach, patch_layers = create_mach_phase3(d_model, n_layers)
    patched_model = MACHPatchedModel(base_model, mach)

    checkpoint = None if args.from_scratch else args.checkpoint
    meta_train_phase3_fewshot(
        base_model, mach, patched_model, tokenizer, config.DEVICE,
        n_episodes=args.episodes, lr=args.lr,
        checkpoint_path=checkpoint,
    )

    passed = final_evaluation(
        base_model, mach, patched_model, tokenizer, config.DEVICE
    )

    save_path = "checkpoints/phase3_fewshot_ablation.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(mach.state_dict(), save_path)
    print(f"\nSaved ablation checkpoint to {save_path}")

    if wandb is not None:
        wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
