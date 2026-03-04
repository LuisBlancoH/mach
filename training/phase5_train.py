"""
Phase 5: Brain-like meta-learner training.
Gated task state, sparse PFC, no reward input, smaller architecture.
"""

import random

import torch

try:
    import wandb
except ImportError:
    wandb = None

from data.arithmetic import (
    generate_arithmetic_problems, extract_number, generate_few_shot_episode,
    generate_linear_episode, LINEAR_TRAIN, LINEAR_HELDOUT,
)
import config


DEFAULT_CURRICULUM = [
    (0, 300, "single"),
    (300, 2000, "few_shot"),
]

LINEAR_CURRICULUM = [
    (0, 300, "single"),
    (300, 2000, "linear"),
]

CONTINUOUS_LINEAR_CURRICULUM = [
    (0, 300, "single"),
    (300, 2000, "continuous_linear"),
]


def get_episode_mode(episode_idx, curriculum):
    for start, end, mode in curriculum:
        if start <= episode_idx < end:
            return mode
    return curriculum[-1][2]


def generate_episode_problems(n_problems, mode):
    if mode == "single":
        return generate_arithmetic_problems(n_problems, 6)
    elif mode == "linear":
        return generate_linear_episode(n_problems)
    elif mode == "continuous_linear":
        return generate_linear_episode(n_problems, continuous=True, max_coeff=5)
    else:  # few_shot
        return generate_few_shot_episode(n_problems)


def run_episode_phase5(base_model, mach, patched_model, tokenizer,
                       problems, device):
    """
    Phase 5 episode. No reward signals. Sparsity loss on task state.
    """
    mach.reset_episode()

    rewards = []
    problem_losses = []
    sparsity_losses = []
    qwen_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for i, problem in enumerate(problems):
        input_ids = tokenizer(
            problem["prompt"], return_tensors="pt"
        ).input_ids.to(device)

        # Observe
        gru_memory = mach.observe(base_model, input_ids)

        # Fire (no reward signals)
        writes = mach.fire(gru_memory)

        # Sparsity loss on task state
        task_state = mach.get_task_state()
        sparsity_losses.append(task_state.abs().mean())

        # Apply writes
        mach.apply_writes(writes)

        # Skip demo evaluation
        if problem.get("is_demo", False):
            rewards.append(0.0)
            problem_losses.append(0.0)
            continue

        # Evaluate with patches
        full_text = problem["prompt"] + problem["answer"]
        encoding = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_len = len(tokenizer(problem["prompt"]).input_ids)
        labels = encoding.input_ids.clone()
        labels[0, :prompt_len] = -100

        output = patched_model(input_ids=encoding.input_ids, labels=labels)

        # Reward
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
        qwen_loss = qwen_loss + output.loss

    # Aggregate sparsity loss
    sparsity_loss = sum(sparsity_losses) / len(sparsity_losses)

    return qwen_loss, sparsity_loss, rewards, problem_losses


def compute_decorrelation_loss(task_state_buffer, current_task_state):
    """
    Lateral inhibition: penalize correlation between task state dimensions.
    Forces each dimension to encode independent features.

    Uses detached buffer for statistics but the current (live) task state
    for the gradient signal — so the loss backprops through the current episode.
    """
    if len(task_state_buffer) < 10:
        return torch.tensor(0.0)
    # Stack buffer (detached) + current state (live gradient)
    detached_states = torch.stack(task_state_buffer)
    all_states = torch.cat([detached_states, current_task_state.unsqueeze(0)])
    # Center
    mean = all_states.mean(dim=0, keepdim=True)
    centered = all_states - mean
    # Correlation matrix
    norms = centered.norm(dim=0, keepdim=True).clamp(min=1e-8)
    normed = centered / norms
    corr = normed.T @ normed / all_states.shape[0]
    # Penalize off-diagonal correlations
    eye = torch.eye(corr.shape[0], device=corr.device)
    decorr_loss = (corr - eye).pow(2).mean()
    return decorr_loss


def meta_train_phase5(base_model, mach, patched_model, tokenizer,
                      device, n_episodes=None, lr=None, curriculum=None,
                      checkpoint_path=None, save_path=None,
                      sparsity_beta=None, decorr_beta=None,
                      energy_beta=None):
    """Phase 5 meta-training loop."""
    if n_episodes is None:
        n_episodes = config.PHASE5_EPISODES
    if lr is None:
        lr = config.PHASE5_LR
    if curriculum is None:
        curriculum = DEFAULT_CURRICULUM
    if sparsity_beta is None:
        sparsity_beta = config.PHASE5_SPARSITY_BETA
    if decorr_beta is None:
        decorr_beta = config.PHASE5_DECORR_BETA
    if energy_beta is None:
        energy_beta = config.PHASE5_ENERGY_BETA

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        mach.load_state_dict(state_dict, strict=False)

    meta_params = list(mach.parameters())
    optimizer = torch.optim.Adam(meta_params, lr=lr)

    use_energy = energy_beta > 0
    n_meta = sum(p.numel() for p in meta_params)
    print(f"Phase 5 trainable parameters: {n_meta:,}")
    if use_energy:
        print(f"Using unified energy loss (beta={energy_beta})")
    else:
        print(f"Sparsity beta: {sparsity_beta}, decorr beta: {decorr_beta}")

    # Rolling buffer of task states for decorrelation loss
    task_state_buffer = []
    BUFFER_SIZE = 50

    def get_n_problems(episode_idx):
        if episode_idx < 100:
            return 5
        elif episode_idx < 300:
            return 10
        elif episode_idx < 600:
            return 15
        else:
            return config.PHASE5_PROBLEMS_PER_EPISODE

    for episode_idx in range(n_episodes):
        mode = get_episode_mode(episode_idx, curriculum)
        n_problems = get_n_problems(episode_idx)
        problems = generate_episode_problems(n_problems, mode)

        optimizer.zero_grad()

        try:
            ce_loss, sparsity_loss, rewards, problem_losses = \
                run_episode_phase5(
                    base_model, mach, patched_model, tokenizer,
                    problems, device,
                )

            if use_energy:
                # Unified free energy: prediction error + metabolic cost
                metabolic = mach.metabolic_cost()
                total_loss = ce_loss + energy_beta * metabolic
                energy_scalar = metabolic.item()
                sparsity_scalar = sparsity_loss.item()
                decorr_scalar = 0.0
            else:
                # Separate losses: sparsity + decorrelation
                task_state = mach.get_task_state()
                if task_state is not None:
                    decorr_loss = compute_decorrelation_loss(
                        task_state_buffer, task_state
                    )
                    decorr_loss = decorr_loss.to(device)
                    task_state_buffer.append(task_state.detach())
                    if len(task_state_buffer) > BUFFER_SIZE:
                        task_state_buffer.pop(0)
                else:
                    decorr_loss = torch.tensor(0.0, device=device)

                total_loss = ce_loss + sparsity_beta * sparsity_loss \
                    + decorr_beta * decorr_loss
                energy_scalar = 0.0
                decorr_scalar = decorr_loss.item()
                sparsity_scalar = sparsity_loss.item()

            total_loss.backward()
            loss_scalar = ce_loss.item()

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at episode {episode_idx}. Reducing problems.")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            problems = problems[:max(5, len(problems) // 2)]
            ce_loss, sparsity_loss, rewards, problem_losses = \
                run_episode_phase5(
                    base_model, mach, patched_model, tokenizer,
                    problems, device,
                )
            total_loss = ce_loss + sparsity_beta * sparsity_loss
            total_loss.backward()
            loss_scalar = ce_loss.item()
            sparsity_scalar = sparsity_loss.item()
            decorr_scalar = 0.0
            energy_scalar = 0.0

        torch.nn.utils.clip_grad_norm_(
            meta_params, max_norm=config.PHASE5_GRAD_CLIP
        )
        optimizer.step()

        if episode_idx % 10 == 0:
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

            # Task state stats
            task_state = mach.get_task_state()
            if task_state is not None:
                ts = task_state.detach()
                n_active = (ts.abs() > 0.1).sum().item()
                ts_max = ts.abs().max().item()
            else:
                n_active = 0
                ts_max = 0.0

            diff_str = ""
            if mode in ("few_shot", "linear", "continuous_linear"):
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

            if use_energy:
                cost_str = f"energy={energy_scalar:.3f}"
            else:
                cost_str = f"sp={sparsity_scalar:.3f} dc={decorr_scalar:.3f}"

            print(
                f"Episode {episode_idx:4d} | {mode} n={len(problems):2d} | "
                f"ce={loss_scalar:.4f} {cost_str} | "
                f"avg_r={avg_reward:.2f} "
                f"early={early_acc:.0%} late={late_acc:.0%} | "
                f"active={n_active:.0f}/{mach.d_task} max={ts_max:.2f}"
                f"{diff_str}"
            )

            if wandb is not None:
                wandb.log({
                    "episode": episode_idx,
                    "ce_loss": loss_scalar,
                    "sparsity_loss": sparsity_scalar,
                    "decorr_loss": decorr_scalar,
                    "energy_cost": energy_scalar,
                    "avg_reward": avg_reward,
                    "early_accuracy": early_acc,
                    "late_accuracy": late_acc,
                    "task_state_active": n_active,
                    "task_state_max": ts_max,
                })

        # Diagnostics + validation + checkpoint
        if (episode_idx % 200 == 0 and episode_idx > 0) or \
                episode_idx == n_episodes - 1:
            _log_diagnostics(mach, meta_params, episode_idx)
            if mode == "few_shot":
                for op in ["add", "sub", "mul", "div", "mod", "max", "min"]:
                    _run_few_shot_validation(
                        base_model, mach, patched_model, tokenizer,
                        device, op, episode_idx,
                    )
            elif mode in ("linear", "continuous_linear"):
                # For continuous: eval on sample of combos to check generalization
                print(f"  --- TRAIN combos ---")
                for coeffs in LINEAR_TRAIN:
                    _run_linear_validation(
                        base_model, mach, patched_model, tokenizer,
                        device, coeffs, episode_idx,
                    )
                print(f"  --- HELD-OUT combos ---")
                for coeffs in LINEAR_HELDOUT:
                    _run_linear_validation(
                        base_model, mach, patched_model, tokenizer,
                        device, coeffs, episode_idx,
                    )
                if mode == "continuous_linear":
                    print(f"  --- NOVEL combos (never in fixed pool) ---")
                    for coeffs in [(3, 1), (1, 4), (4, 3), (5, 0), (0, 5)]:
                        _run_linear_validation(
                            base_model, mach, patched_model, tokenizer,
                            device, coeffs, episode_idx,
                        )
            else:
                for eval_diff in [6, 7]:
                    _run_standard_validation(
                        base_model, mach, patched_model, tokenizer,
                        device, eval_diff, episode_idx,
                    )

            if save_path is not None:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(mach.state_dict(), save_path)
                print(f"  Checkpoint saved to {save_path}")


def _run_few_shot_validation(base_model, mach, patched_model, tokenizer,
                              device, op_type, episode_idx, n_episodes=10,
                              n_problems=20, n_demos=5):
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
        test_idx = 0

        for i, problem in enumerate(problems):
            input_ids = tokenizer(
                problem["prompt"], return_tensors="pt"
            ).input_ids.to(device)

            gru_memory = mach.observe(base_model, input_ids)
            writes = mach.fire(gru_memory)
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


def _run_linear_validation(base_model, mach, patched_model, tokenizer,
                            device, coeffs, episode_idx, n_episodes=10,
                            n_problems=20, n_demos=5):
    """Evaluate linear combination performance for specific (c1, c2)."""
    test_correct = 0
    test_total = 0

    for ep in range(n_episodes):
        problems = generate_linear_episode(
            n_problems, n_demos=n_demos, coeffs=coeffs
        )
        mach.reset_episode()

        for i, problem in enumerate(problems):
            input_ids = tokenizer(
                problem["prompt"], return_tensors="pt"
            ).input_ids.to(device)

            gru_memory = mach.observe(base_model, input_ids)
            writes = mach.fire(gru_memory)
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

    test_acc = test_correct / test_total if test_total > 0 else 0
    c1, c2 = coeffs
    label = f"{c1}a+{c2}b"

    print(f"  EVAL ep{episode_idx} {label:8s} | test={test_acc:.0%}")

    if wandb is not None:
        wandb.log({f"eval/linear_{c1}_{c2}": test_acc})


def _run_standard_validation(base_model, mach, patched_model, tokenizer,
                              device, difficulty, episode_idx, n_episodes=10,
                              n_problems=20):
    """Standard per-difficulty eval (for warmup phase)."""
    meta_correct_early = 0
    meta_correct_late = 0
    meta_total_early = 0
    meta_total_late = 0

    for ep in range(n_episodes):
        problems = generate_arithmetic_problems(n_problems, difficulty)
        mach.reset_episode()

        for i, problem in enumerate(problems):
            input_ids = tokenizer(
                problem["prompt"], return_tensors="pt"
            ).input_ids.to(device)

            gru_memory = mach.observe(base_model, input_ids)
            writes = mach.fire(gru_memory)
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

    # Task state info
    task_state = mach.get_task_state()
    if task_state is not None:
        ts = task_state.detach()
        diag["task_state/l1_norm"] = ts.abs().mean().item()
        diag["task_state/n_active_01"] = (ts.abs() > 0.1).sum().item()
        diag["task_state/n_active_05"] = (ts.abs() > 0.5).sum().item()
        diag["task_state/max_abs"] = ts.abs().max().item()

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
