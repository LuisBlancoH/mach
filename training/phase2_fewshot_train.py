"""
Ablation: Phase 2 architecture + few-shot task.
No critic, no cerebellum, no surprise, no TD modulation, no obs pathway.
Tests whether the minimal observe-write-evaluate loop is sufficient.
"""

import random

import torch

try:
    import wandb
except ImportError:
    wandb = None

from data.arithmetic import (
    generate_arithmetic_problems, extract_number, generate_few_shot_episode,
)
import config


DEFAULT_CURRICULUM = [
    (0, 300, "single"),       # Warm up on d6 (multiplication)
    (300, 2000, "few_shot"),  # Few-shot: hidden operation, must infer from demos
]


def get_episode_mode(episode_idx, curriculum):
    for start, end, mode in curriculum:
        if start <= episode_idx < end:
            return mode
    return curriculum[-1][2]


def generate_episode_problems(n_problems, mode):
    if mode == "single":
        return generate_arithmetic_problems(n_problems, 6)
    else:  # few_shot
        return generate_few_shot_episode(n_problems)


def run_episode_phase2_fewshot(base_model, mach, patched_model, tokenizer,
                                problems, device, no_rewards=False):
    """
    Phase 2 episode with few-shot demo handling.
    Plain CE loss. No critic, no value estimates.
    """
    mach.reset_episode()

    rewards = []
    problem_losses = []
    qwen_loss = torch.tensor(0.0, device=device, requires_grad=True)

    last_reward = 0.0
    cumulative_reward = 0.0

    for i, problem in enumerate(problems):
        input_ids = tokenizer(
            problem["prompt"], return_tensors="pt"
        ).input_ids.to(device)

        # Step 1: Observe
        gru_memory = mach.observe(base_model, input_ids)

        # Step 2: Fire meta-learner (Phase 2 reward signals)
        if no_rewards:
            reward_signals = torch.zeros(3, device=device, dtype=torch.float32)
        else:
            reward_signals = torch.tensor(
                [last_reward, cumulative_reward, float(i)],
                device=device, dtype=torch.float32
            )
        writes = mach.fire(gru_memory, reward_signals)

        # Step 3: Apply writes
        mach.apply_writes(writes)

        # Demo problems: observe and write, but skip evaluation
        if problem.get("is_demo", False):
            rewards.append(0.0)
            problem_losses.append(0.0)
            last_reward = 0.0
            continue

        # Step 4: Forward Qwen with patches
        full_text = problem["prompt"] + problem["answer"]
        encoding = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_len = len(tokenizer(problem["prompt"]).input_ids)
        labels = encoding.input_ids.clone()
        labels[0, :prompt_len] = -100

        output = patched_model(input_ids=encoding.input_ids, labels=labels)

        # Step 5: Compute reward
        with torch.no_grad():
            logits = output.logits
            pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
            pred_text = tokenizer.decode(
                pred_tokens, skip_special_tokens=True
            ).strip()
            predicted = extract_number(pred_text)
            correct = (predicted == problem["answer"])
            reward = 1.0 if correct else -1.0

        rewards.append(reward)
        problem_losses.append(output.loss.item())

        # Step 6: Plain CE loss
        qwen_loss = qwen_loss + output.loss

        if not no_rewards:
            last_reward = reward
            cumulative_reward += reward

    return qwen_loss, rewards, problem_losses


def meta_train_phase2_fewshot(base_model, mach, patched_model, tokenizer,
                               device, n_episodes=None, lr=None,
                               curriculum=None, checkpoint_path=None,
                               save_path=None, no_rewards=False):
    """Phase 2 ablation: few-shot task with Phase 2 architecture (no critic)."""
    if n_episodes is None:
        n_episodes = config.PHASE2_EPISODES
    if lr is None:
        lr = config.PHASE2_LR
    if curriculum is None:
        curriculum = DEFAULT_CURRICULUM

    if checkpoint_path is not None:
        print(f"Loading Phase 2 checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        mach.load_state_dict(state_dict, strict=False)

    meta_params = []
    meta_params += list(mach.obs_proj.parameters())
    meta_params += list(mach.gru.parameters())
    meta_params += list(mach.basis.parameters())
    meta_params += list(mach.transformer.parameters())
    meta_params += list(mach.action_head.parameters())
    meta_params += list(mach.memory_head.parameters())
    meta_params += list(mach.reward_proj.parameters())

    optimizer = torch.optim.Adam(meta_params, lr=lr)

    n_meta = sum(p.numel() for p in meta_params)
    print(f"Phase 2 (fewshot ablation) trainable parameters: {n_meta:,}")
    print(f"No rewards: {no_rewards}")

    def get_n_problems(episode_idx):
        if episode_idx < 100:
            return 5
        elif episode_idx < 300:
            return 10
        elif episode_idx < 600:
            return 15
        else:
            return config.PHASE2_PROBLEMS_PER_EPISODE

    for episode_idx in range(n_episodes):
        mode = get_episode_mode(episode_idx, curriculum)
        n_problems = get_n_problems(episode_idx)
        problems = generate_episode_problems(n_problems, mode)

        optimizer.zero_grad()

        try:
            loss, rewards, problem_losses = run_episode_phase2_fewshot(
                base_model, mach, patched_model, tokenizer,
                problems, device, no_rewards=no_rewards
            )
            loss.backward()
            loss_scalar = loss.item()

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at episode {episode_idx} with {n_problems} "
                  f"problems. Reducing to {max(5, n_problems // 2)}.")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            problems = problems[:max(5, n_problems // 2)]
            loss, rewards, problem_losses = run_episode_phase2_fewshot(
                base_model, mach, patched_model, tokenizer,
                problems, device, no_rewards=no_rewards
            )
            loss.backward()
            loss_scalar = loss.item()

        torch.nn.utils.clip_grad_norm_(
            meta_params, max_norm=config.PHASE2_GRAD_CLIP
        )
        optimizer.step()

        if episode_idx % 10 == 0:
            # Filter test-only rewards (exclude demos)
            test_rewards = [r for j, r in enumerate(rewards)
                           if not problems[j].get("is_demo", False)]
            if not test_rewards:
                test_rewards = rewards
            avg_reward = sum(test_rewards) / len(test_rewards)
            n_early = min(5, len(test_rewards))
            n_late = min(5, len(test_rewards))
            early_acc = sum(
                1 for r in test_rewards[:n_early] if r > 0
            ) / n_early
            late_acc = sum(
                1 for r in test_rewards[-n_late:] if r > 0
            ) / n_late

            diff_str = ""
            if mode == "few_shot":
                diff_correct = {}
                diff_total = {}
                for j, p in enumerate(problems):
                    if p.get("is_demo", False):
                        continue
                    d = p.get("difficulty", "?")
                    diff_total[d] = diff_total.get(d, 0) + 1
                    if rewards[j] > 0:
                        diff_correct[d] = diff_correct.get(d, 0) + 1
                diff_str = " | " + " ".join(
                    f"{d}={diff_correct.get(d,0)}/{diff_total[d]}"
                    for d in sorted(diff_total)
                )

            print(
                f"Episode {episode_idx:4d} | {mode} n={len(problems):2d} | "
                f"loss={loss_scalar:.4f} | "
                f"avg_r={avg_reward:.2f} "
                f"early={early_acc:.0%} late={late_acc:.0%}"
                f"{diff_str}"
            )

            log_dict = {
                "episode": episode_idx,
                "n_problems": len(problems),
                "total_loss": loss_scalar,
                "avg_reward": avg_reward,
                "early_accuracy": early_acc,
                "late_accuracy": late_acc,
                "learning_delta": late_acc - early_acc,
            }

            for j, pl in enumerate(problem_losses):
                log_dict[f"problem_loss/{j}"] = pl

            if wandb is not None:
                wandb.log(log_dict)

        # Diagnostics + validation + checkpoint every 200 episodes + final
        if (episode_idx % 200 == 0 and episode_idx > 0) or \
                episode_idx == n_episodes - 1:
            _log_diagnostics(mach, meta_params, episode_idx)
            if mode == "few_shot":
                for op in ["add", "sub", "mul", "div"]:
                    _run_few_shot_validation(
                        base_model, mach, patched_model, tokenizer,
                        device, op, episode_idx,
                        no_rewards=no_rewards,
                    )
            else:
                for eval_diff in [6, 7]:
                    _run_standard_validation(
                        base_model, mach, patched_model, tokenizer,
                        device, eval_diff, episode_idx,
                        no_rewards=no_rewards,
                    )

            if save_path is not None:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(mach.state_dict(), save_path)
                print(f"  Checkpoint saved to {save_path}")


def _run_few_shot_validation(base_model, mach, patched_model, tokenizer,
                              device, op_type, episode_idx, n_episodes=10,
                              n_problems=20, n_demos=5, no_rewards=False):
    """Evaluate few-shot performance for a specific operation."""
    test_correct = 0
    test_total = 0
    test_correct_early = 0
    test_correct_late = 0
    test_total_early = 0
    test_total_late = 0
    n_test = n_problems - n_demos

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
            if no_rewards:
                reward_signals = torch.zeros(
                    3, device=device, dtype=torch.float32
                )
            else:
                reward_signals = torch.tensor(
                    [last_reward, cumulative_reward, float(i)],
                    device=device, dtype=torch.float32
                )
            writes = mach.fire(gru_memory, reward_signals)
            mach.apply_writes(writes)

            if problem["is_demo"]:
                continue

            full_text = problem["prompt"] + problem["answer"]
            encoding = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_len = len(tokenizer(problem["prompt"]).input_ids)

            with torch.no_grad():
                output = patched_model(input_ids=encoding.input_ids)
                logits = (
                    output.logits if hasattr(output, 'logits') else output[0]
                )
                pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
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
            if not no_rewards:
                last_reward = reward
                cumulative_reward += reward

    test_acc = test_correct / test_total if test_total > 0 else 0
    early_acc = (
        test_correct_early / test_total_early if test_total_early > 0 else 0
    )
    late_acc = (
        test_correct_late / test_total_late if test_total_late > 0 else 0
    )
    delta = late_acc - early_acc

    print(
        f"  EVAL ep{episode_idx} {op_type:3s} | "
        f"test={test_acc:.0%} early={early_acc:.0%} late={late_acc:.0%} "
        f"delta={delta:+.0%}"
    )

    if wandb is not None:
        wandb.log({
            f"eval/{op_type}_test": test_acc,
            f"eval/{op_type}_early": early_acc,
            f"eval/{op_type}_late": late_acc,
            f"eval/{op_type}_delta": delta,
        })


def _run_standard_validation(base_model, mach, patched_model, tokenizer,
                              device, difficulty, episode_idx, n_episodes=10,
                              n_problems=20, no_rewards=False):
    """Standard per-difficulty eval (for warmup phase)."""
    meta_correct_early = 0
    meta_correct_late = 0
    meta_total_early = 0
    meta_total_late = 0

    for ep in range(n_episodes):
        problems = generate_arithmetic_problems(n_problems, difficulty)
        mach.reset_episode()
        last_reward = 0.0
        cumulative_reward = 0.0

        for i, problem in enumerate(problems):
            input_ids = tokenizer(
                problem["prompt"], return_tensors="pt"
            ).input_ids.to(device)

            gru_memory = mach.observe(base_model, input_ids)
            if no_rewards:
                reward_signals = torch.zeros(
                    3, device=device, dtype=torch.float32
                )
            else:
                reward_signals = torch.tensor(
                    [last_reward, cumulative_reward, float(i)],
                    device=device, dtype=torch.float32
                )
            writes = mach.fire(gru_memory, reward_signals)
            mach.apply_writes(writes)

            full_text = problem["prompt"] + problem["answer"]
            encoding = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_len = len(tokenizer(problem["prompt"]).input_ids)

            with torch.no_grad():
                output = patched_model(input_ids=encoding.input_ids)
                logits = (
                    output.logits if hasattr(output, 'logits') else output[0]
                )
                pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
                pred_text = tokenizer.decode(
                    pred_tokens, skip_special_tokens=True
                ).strip()
                predicted = extract_number(pred_text)
                correct = (predicted == problem["answer"])

            reward = 1.0 if correct else -1.0
            if not no_rewards:
                last_reward = reward
                cumulative_reward += reward

            if i < 5:
                meta_correct_early += int(correct)
                meta_total_early += 1
            if i >= n_problems - 5:
                meta_correct_late += int(correct)
                meta_total_late += 1

    early_acc = (
        meta_correct_early / meta_total_early if meta_total_early > 0 else 0
    )
    late_acc = (
        meta_correct_late / meta_total_late if meta_total_late > 0 else 0
    )
    delta = late_acc - early_acc

    print(
        f"  EVAL ep{episode_idx} d={difficulty} | "
        f"early={early_acc:.0%} late={late_acc:.0%} "
        f"delta={delta:+.0%}"
    )


def _log_diagnostics(mach, meta_params, episode_idx):
    """Log gradient norms and weight stats."""
    diag = {}

    component_grad_norms = {}
    for name, param in mach.named_parameters():
        if param.grad is not None:
            component = name.split('.')[0]
            norm = param.grad.norm().item()
            if component not in component_grad_norms:
                component_grad_norms[component] = []
            component_grad_norms[component].append(norm)

    for component, norms in component_grad_norms.items():
        avg_norm = sum(norms) / len(norms)
        diag[f"grad_norm/{component}"] = avg_norm

    for i in range(mach.basis.n_patches):
        diag[f"basis_norm/patch{i}_down_U"] = (
            mach.basis.down_U[i].norm().item()
        )
        diag[f"basis_norm/patch{i}_down_V"] = (
            mach.basis.down_V[i].norm().item()
        )

    for i, patch in enumerate(mach.patches):
        if patch.delta_down is not None:
            diag[f"patch_delta/patch{i}_down"] = (
                patch.delta_down.norm().item()
            )
            diag[f"patch_delta/patch{i}_up"] = (
                patch.delta_up.norm().item()
            )

    print(f"  Diagnostics at episode {episode_idx}:")
    for k, v in sorted(diag.items()):
        print(f"    {k}: {v:.6f}")

    if wandb is not None:
        wandb.log({f"diag/{k}": v for k, v in diag.items()})
