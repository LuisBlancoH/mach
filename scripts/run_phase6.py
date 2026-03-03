#!/usr/bin/env python3
"""Phase 6: Planning Loop (Critic-Gated Iterative Proposals)."""

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
from models.universal_module import MACHPhase6, MACHPatchedModel
from training.phase6_meta_train import meta_train_phase6


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


def create_mach_phase6(d_model, n_layers, max_planning_iters=None, detach_obs=True):
    if max_planning_iters is None:
        max_planning_iters = config.MAX_PLANNING_ITERS

    patch_layers = [
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 2,
    ]
    print(f"Patch layers: {patch_layers}")

    mach = MACHPhase6(
        d_model=d_model,
        n_layers=n_layers,
        patch_layers=patch_layers,
        hidden_dim=config.PATCH_HIDDEN_DIM,
        d_meta=config.D_META,
        n_basis=config.N_BASIS,
        detach_obs=detach_obs,
        max_planning_iters=max_planning_iters,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in mach.parameters())
    print(f"MACH Phase 6 total parameters: {n_params:,}")
    print(f"Planning iterations: {max_planning_iters}")

    return mach, patch_layers


def final_evaluation(base_model, mach, patched_model, tokenizer, device,
                     n_episodes=20, n_problems=20):
    """
    Final evaluation across all curriculum difficulties.
    Reports base accuracy, early/late meta-learner accuracy, delta,
    critic value statistics, and commit distribution.
    """
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION — {n_episodes} episodes x {n_problems} problems per difficulty")
    print(f"Planning iterations: {mach.max_planning_iters}")
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
        commit_counts = [0] * mach.max_planning_iters
        committed_vs_others = []
        total_problems = 0

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
                    iter_values = mach.get_all_iteration_values()
                    committed_iter = mach.get_committed_iteration()

                all_values.append(current_value.item())
                commit_counts[committed_iter] += 1
                total_problems += 1

                # Committed vs non-committed
                committed_val = iter_values[committed_iter].item()
                other_vals = [
                    v.item() for k, v in enumerate(iter_values) if k != committed_iter
                ]
                if other_vals:
                    committed_vs_others.append(
                        (committed_val, sum(other_vals) / len(other_vals))
                    )

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

                if i > 0:
                    td_target = rewards[-2] + config.PHASE6_GAMMA * current_value
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

        # Commit distribution
        commit_dist = [c / total_problems for c in commit_counts] if total_problems > 0 else commit_counts
        commit_str = " ".join(f"i{k}={commit_dist[k]:.0%}" for k in range(mach.max_planning_iters))
        not_all_i0 = commit_dist[0] < 1.0

        # Committed vs others
        if committed_vs_others:
            avg_committed = sum(c for c, _ in committed_vs_others) / len(committed_vs_others)
            avg_other = sum(o for _, o in committed_vs_others) / len(committed_vs_others)
            critic_correct = avg_committed > avg_other
        else:
            avg_committed = avg_other = 0
            critic_correct = False

        marker = " <<< PASS" if passed else ""
        print(f"\n  d{difficulty} ({label}):")
        print(f"    Base Qwen:       {base_acc:.1%}")
        print(f"    Early (0-4):     {early_acc:.1%}")
        print(f"    Late (15-19):    {late_acc:.1%}")
        print(f"    Delta:           {delta:+.1%}{marker}")
        print(f"    vs Base:         {improvement:+.1%}")
        print(f"    Avg critic V:    {mean_val:.3f}")
        print(f"    Commit dist:     {commit_str} {'(diverse)' if not_all_i0 else '(all i0)'}")
        print(f"    Critic gating:   committed={avg_committed:.3f} other={avg_other:.3f} "
              f"{'OK' if critic_correct else 'BAD'}")

        if wandb is not None:
            wandb.log({
                f"final/d{difficulty}_base": base_acc,
                f"final/d{difficulty}_early": early_acc,
                f"final/d{difficulty}_late": late_acc,
                f"final/d{difficulty}_delta": delta,
                f"final/d{difficulty}_vs_base": improvement,
                f"final/d{difficulty}_passed": passed,
                f"final/d{difficulty}_avg_value": mean_val,
                f"final/d{difficulty}_critic_correct": critic_correct,
                f"final/d{difficulty}_commit_diverse": not_all_i0,
                **{f"final/d{difficulty}_commit_iter_{k}": commit_dist[k]
                   for k in range(mach.max_planning_iters)},
            })

    print(f"\n{'='*60}")
    print(f"Phase 6 {'PASS' if any_passed else 'FAIL'}: "
          f"{'at least one' if any_passed else 'no'} difficulty shows "
          f"late - early >= 10pp")
    print(f"{'='*60}")

    return any_passed


def main():
    parser = argparse.ArgumentParser(
        description="MACH Phase 6: Planning Loop (Critic-Gated Iterative Proposals)"
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase3_mach.pt",
                        help="Phase 3 checkpoint to load")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Train from scratch without loading Phase 3 checkpoint")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="Override max planning iterations (default: MAX_PLANNING_ITERS)")
    args = parser.parse_args()

    if wandb is not None:
        wandb.init(
            project="mach",
            name="phase6-planning-loop",
            config={
                "base_model": config.BASE_MODEL,
                "d_meta": config.D_META,
                "n_basis": config.N_BASIS,
                "lr": args.lr or config.PHASE6_LR,
                "episodes": args.episodes or config.PHASE6_EPISODES,
                "critic_loss_weight": config.PHASE6_CRITIC_LOSS_WEIGHT,
                "gamma": config.PHASE6_GAMMA,
                "max_planning_iters": args.max_iters or config.MAX_PLANNING_ITERS,
                "device": str(config.DEVICE),
            },
        )

    base_model, tokenizer, d_model, n_layers = load_base_model()
    mach, patch_layers = create_mach_phase6(
        d_model, n_layers, max_planning_iters=args.max_iters
    )
    patched_model = MACHPatchedModel(base_model, mach)

    checkpoint = None if args.from_scratch else args.checkpoint
    meta_train_phase6(
        base_model, mach, patched_model, tokenizer, config.DEVICE,
        n_episodes=args.episodes, lr=args.lr, checkpoint_path=checkpoint,
    )

    passed = final_evaluation(
        base_model, mach, patched_model, tokenizer, config.DEVICE
    )

    save_path = "checkpoints/phase6_mach.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(mach.state_dict(), save_path)
    print(f"\nSaved Phase 6 checkpoint to {save_path}")

    if wandb is not None:
        wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
