#!/usr/bin/env python3
"""Phase 3: Add Critic (basal ganglia) to meta-learner."""

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
from data.arithmetic import generate_arithmetic_problems, extract_number
from models.universal_module import MACHPhase3, MACHPatchedModel
from training.phase3_meta_train import meta_train_phase3


EVAL_DIFFICULTIES = {
    5: "2x2 multiplication",
    6: "3x2 multiplication",
    7: "3x3 multiplication",
    9: "mixed hard",
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
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in mach.parameters())
    print(f"MACH Phase 3 total parameters: {n_params:,}")

    return mach, patch_layers


def final_evaluation(base_model, mach, patched_model, tokenizer, device,
                     n_episodes=20, n_problems=20):
    """
    Final evaluation across all curriculum difficulties.
    Reports base accuracy, early/late meta-learner accuracy, delta,
    and critic value statistics per difficulty.
    """
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION — {n_episodes} episodes x {n_problems} problems per difficulty")
    print(f"{'='*60}")

    any_passed = False

    for difficulty, label in EVAL_DIFFICULTIES.items():
        base_correct = 0
        base_total = 0
        meta_correct_early = 0
        meta_correct_late = 0
        meta_total_early = 0
        meta_total_late = 0
        all_rewards = []
        all_values = []

        for ep in range(n_episodes):
            problems = generate_arithmetic_problems(n_problems, difficulty)

            # --- Meta-learner evaluation ---
            mach.reset_episode()
            rewards = []
            last_value = torch.tensor(0.0, device=device)
            last_td_error = torch.tensor(0.0, device=device)

            for i, problem in enumerate(problems):
                input_ids = tokenizer(
                    problem["prompt"], return_tensors="pt"
                ).input_ids.to(device)

                gru_memory = mach.observe(base_model, input_ids)
                writes = mach.fire(gru_memory, last_value, last_td_error)

                with torch.no_grad():
                    current_value = mach.get_value()
                all_values.append(current_value.item())

                mach.apply_writes(writes)

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

                reward = 1.0 if correct else -1.0
                rewards.append(reward)

                # TD for next step
                if i > 0:
                    td_target = rewards[-2] + config.PHASE3_GAMMA * current_value
                    last_td_error = (td_target - last_value).detach()
                last_value = current_value.detach()

                if i < 5:
                    meta_correct_early += int(correct)
                    meta_total_early += 1
                if i >= n_problems - 5:
                    meta_correct_late += int(correct)
                    meta_total_late += 1

            all_rewards.extend(rewards)

            # --- Base Qwen evaluation (zero patches) ---
            mach.reset_episode()
            for problem in problems:
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
                    base_correct += int(predicted == problem["answer"])
                    base_total += 1

        base_acc = base_correct / base_total
        early_acc = meta_correct_early / meta_total_early
        late_acc = meta_correct_late / meta_total_late
        delta = late_acc - early_acc
        improvement = late_acc - base_acc
        passed = delta >= 0.10
        mean_val = sum(all_values) / len(all_values) if all_values else 0

        if passed:
            any_passed = True

        marker = " <<< PASS" if passed else ""
        print(f"\n  d{difficulty} ({label}):")
        print(f"    Base Qwen:    {base_acc:.1%}")
        print(f"    Early (0-4):  {early_acc:.1%}")
        print(f"    Late (15-19): {late_acc:.1%}")
        print(f"    Delta:        {delta:+.1%}{marker}")
        print(f"    vs Base:      {improvement:+.1%}")
        print(f"    Avg critic V: {mean_val:.3f}")

        if wandb is not None:
            wandb.log({
                f"final/d{difficulty}_base": base_acc,
                f"final/d{difficulty}_early": early_acc,
                f"final/d{difficulty}_late": late_acc,
                f"final/d{difficulty}_delta": delta,
                f"final/d{difficulty}_vs_base": improvement,
                f"final/d{difficulty}_passed": passed,
                f"final/d{difficulty}_avg_value": mean_val,
            })

    print(f"\n{'='*60}")
    print(f"Phase 3 {'PASS' if any_passed else 'FAIL'}: "
          f"{'at least one' if any_passed else 'no'} difficulty shows "
          f"late - early >= 10pp")
    print(f"{'='*60}")

    return any_passed


def main():
    parser = argparse.ArgumentParser(description="MACH Phase 3: Critic (Basal Ganglia)")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase2_mach.pt",
                        help="Phase 2 checkpoint to load")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Train from scratch without loading Phase 2 checkpoint")
    args = parser.parse_args()

    if wandb is not None:
        wandb.init(
            project="mach",
            name="phase3-critic",
            config={
                "base_model": config.BASE_MODEL,
                "d_meta": config.D_META,
                "n_basis": config.N_BASIS,
                "lr": args.lr or config.PHASE3_LR,
                "episodes": args.episodes or config.PHASE3_EPISODES,
                "critic_loss_weight": config.PHASE3_CRITIC_LOSS_WEIGHT,
                "gamma": config.PHASE3_GAMMA,
                "device": str(config.DEVICE),
            },
        )

    base_model, tokenizer, d_model, n_layers = load_base_model()
    mach, patch_layers = create_mach_phase3(d_model, n_layers)
    patched_model = MACHPatchedModel(base_model, mach)

    checkpoint = None if args.from_scratch else args.checkpoint
    meta_train_phase3(
        base_model, mach, patched_model, tokenizer, config.DEVICE,
        n_episodes=args.episodes, lr=args.lr, checkpoint_path=checkpoint,
    )

    passed = final_evaluation(
        base_model, mach, patched_model, tokenizer, config.DEVICE
    )

    save_path = "checkpoints/phase3_mach.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(mach.state_dict(), save_path)
    print(f"\nSaved Phase 3 checkpoint to {save_path}")

    if wandb is not None:
        wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
