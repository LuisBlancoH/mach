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
    generate_linear_episode, generate_token_mapping_episode,
    LINEAR_TRAIN, LINEAR_HELDOUT,
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

TOKEN_MAP_CURRICULUM = [
    (0, 300, "single"),
    (300, 2000, "token_map"),
]

MIXED_CURRICULUM = [
    (0, 300, "single"),
    (300, 2000, "mixed"),
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
    elif mode == "token_map":
        return generate_token_mapping_episode(n_problems)
    elif mode == "mixed":
        # Randomly pick a task type each episode
        sub_mode = random.choice(["continuous_linear", "token_map"])
        return generate_episode_problems(n_problems, sub_mode)
    else:  # few_shot
        return generate_few_shot_episode(n_problems)


def run_episode_phase5(base_model, mach, patched_model, tokenizer,
                       problems, device, n_self_eval_steps=0,
                       td_modulation=0.0, gamma=0.0,
                       satisfaction_threshold=0.5):
    """
    Phase 5 episode. No reward signals to fire(). Sparsity loss on task state.
    Optional TD-weighted CE loss: critic modulates gradient magnitude.
    Optional self-evaluation: satisfaction-gated when critic is active,
    fixed-step when critic is inactive.
    """
    mach.reset_episode()

    rewards = []
    problem_losses = []
    sparsity_losses = []
    qwen_loss = torch.tensor(0.0, device=device, requires_grad=True)
    critic_loss = torch.tensor(0.0, device=device, requires_grad=True)
    td_errors = []
    demo_problems = []
    use_critic = td_modulation > 0
    self_eval_steps_used = 0

    # Track value from previous test problem for TD targets
    last_value = None

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

        # Get critic value estimate (before applying writes)
        if use_critic:
            value = mach.get_value()

        # Apply writes
        mach.apply_writes(writes)

        # Skip demo evaluation
        if problem.get("is_demo", False):
            demo_problems.append(problem)
            rewards.append(0.0)
            problem_losses.append(0.0)

            # Self-evaluation after last demo
            # Satisfaction-gated when critic is active, fixed-step otherwise
            if n_self_eval_steps > 0 and i == len(problems) - 1 or \
                    (i + 1 < len(problems) and
                     not problems[i + 1].get("is_demo", False)):
                for step in range(n_self_eval_steps):
                    demo = demo_problems[step % len(demo_problems)]
                    demo_ids = tokenizer(
                        demo["prompt"], return_tensors="pt"
                    ).input_ids.to(device)

                    # Observe own patched output on demo
                    gru_memory = mach.observe_patched(
                        patched_model, demo_ids
                    )
                    writes = mach.fire(gru_memory)
                    sparsity_losses.append(
                        mach.get_task_state().abs().mean()
                    )
                    mach.apply_writes(writes)
                    self_eval_steps_used += 1

                    # Satisfaction gate: stop early if critic is confident
                    if use_critic:
                        satisfaction = mach.get_value()
                        if satisfaction.item() > satisfaction_threshold:
                            break

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
            if problem.get("difficulty") == "token_map":
                correct = (pred_text == problem["answer"])
            else:
                predicted = extract_number(pred_text)
                correct = (predicted == problem["answer"])
            reward = 1.0 if correct else -1.0

        rewards.append(reward)
        problem_losses.append(output.loss.item())

        if use_critic:
            # TD-weighted CE: surprising outcomes get amplified gradient
            with torch.no_grad():
                td_error = abs(reward - value.item())
                td_errors.append(td_error)
            ce_weight = 1.0 + td_modulation * td_error
            qwen_loss = qwen_loss + output.loss * ce_weight

            # Critic TD loss: train value estimator
            with torch.no_grad():
                # Next value for TD target (0 if last problem)
                next_value = 0.0  # Will be updated if there's a next test
                td_target = reward + gamma * next_value
            critic_loss = critic_loss + (value - td_target) ** 2

            # Update TD target for previous problem's next_value
            last_value = value
        else:
            qwen_loss = qwen_loss + output.loss

    # Aggregate sparsity loss
    sparsity_loss = sum(sparsity_losses) / len(sparsity_losses)

    avg_td_error = sum(td_errors) / len(td_errors) if td_errors else 0.0

    return qwen_loss, sparsity_loss, rewards, problem_losses, \
        critic_loss, avg_td_error, self_eval_steps_used


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
                      energy_beta=None, n_self_eval_steps=None,
                      td_modulation=None, critic_beta=None,
                      satisfaction_threshold=None):
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
    if n_self_eval_steps is None:
        n_self_eval_steps = config.PHASE5_N_SELF_EVAL_STEPS
    if td_modulation is None:
        td_modulation = config.PHASE5_TD_MODULATION
    if critic_beta is None:
        critic_beta = config.PHASE5_CRITIC_BETA
    if satisfaction_threshold is None:
        satisfaction_threshold = config.PHASE5_SATISFACTION_THRESHOLD

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        mach.load_state_dict(state_dict, strict=False)

    meta_params = list(mach.parameters())
    optimizer = torch.optim.Adam(meta_params, lr=lr)

    use_energy = energy_beta > 0
    use_critic = td_modulation > 0
    gamma = config.PHASE5_GAMMA
    n_meta = sum(p.numel() for p in meta_params)
    print(f"Phase 5 trainable parameters: {n_meta:,}")
    if use_energy:
        print(f"Using unified energy loss (beta={energy_beta})")
    else:
        print(f"Sparsity beta: {sparsity_beta}, decorr beta: {decorr_beta}")
    if use_critic:
        print(f"Critic: td_modulation={td_modulation}, critic_beta={critic_beta}, gamma={gamma}")

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
            ce_loss, sparsity_loss, rewards, problem_losses, \
                ep_critic_loss, avg_td_error, ep_self_eval_steps = \
                run_episode_phase5(
                    base_model, mach, patched_model, tokenizer,
                    problems, device,
                    n_self_eval_steps=n_self_eval_steps,
                    td_modulation=td_modulation,
                    gamma=gamma,
                    satisfaction_threshold=satisfaction_threshold,
                )

            if use_energy:
                # Unified free energy: prediction error + metabolic cost
                metabolic = mach.metabolic_cost()
                total_loss = ce_loss + energy_beta * metabolic
                if use_critic:
                    total_loss = total_loss + critic_beta * ep_critic_loss
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

                # Obs gate sparsity: push unused observation layers toward zero
                obs_gates = mach.get_obs_gates()
                obs_gate_loss = obs_gates.mean() if obs_gates is not None \
                    else torch.tensor(0.0, device=device)

                total_loss = ce_loss + sparsity_beta * sparsity_loss \
                    + decorr_beta * decorr_loss \
                    + sparsity_beta * obs_gate_loss
                if use_critic:
                    total_loss = total_loss + critic_beta * ep_critic_loss
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
            ce_loss, sparsity_loss, rewards, problem_losses, \
                ep_critic_loss, avg_td_error, ep_self_eval_steps = \
                run_episode_phase5(
                    base_model, mach, patched_model, tokenizer,
                    problems, device,
                    n_self_eval_steps=n_self_eval_steps,
                    td_modulation=td_modulation,
                    gamma=gamma,
                    satisfaction_threshold=satisfaction_threshold,
                )
            total_loss = ce_loss + sparsity_beta * sparsity_loss
            if use_critic:
                total_loss = total_loss + critic_beta * ep_critic_loss
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

            critic_str = ""
            if use_critic:
                eval_str = f" eval={ep_self_eval_steps}" \
                    if n_self_eval_steps > 0 else ""
                critic_str = f" | td={avg_td_error:.3f}{eval_str}"

            print(
                f"Episode {episode_idx:4d} | {mode} n={len(problems):2d} | "
                f"ce={loss_scalar:.4f} {cost_str}{critic_str} | "
                f"avg_r={avg_reward:.2f} "
                f"early={early_acc:.0%} late={late_acc:.0%} | "
                f"active={n_active:.0f}/{mach.d_task} max={ts_max:.2f}"
                f"{diff_str}"
            )

            if wandb is not None:
                log_dict = {
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
                }
                if use_critic:
                    log_dict["avg_td_error"] = avg_td_error
                    log_dict["critic_loss"] = ep_critic_loss.item()
                    if n_self_eval_steps > 0:
                        log_dict["self_eval_steps"] = ep_self_eval_steps
                wandb.log(log_dict)

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
                        n_self_eval_steps=n_self_eval_steps,
                    )
                print(f"  --- HELD-OUT combos ---")
                for coeffs in LINEAR_HELDOUT:
                    _run_linear_validation(
                        base_model, mach, patched_model, tokenizer,
                        device, coeffs, episode_idx,
                        n_self_eval_steps=n_self_eval_steps,
                    )
                if mode == "continuous_linear":
                    print(f"  --- NOVEL combos (never in fixed pool) ---")
                    for coeffs in [(3, 1), (1, 4), (4, 3), (5, 0), (0, 5)]:
                        _run_linear_validation(
                            base_model, mach, patched_model, tokenizer,
                            device, coeffs, episode_idx,
                            n_self_eval_steps=n_self_eval_steps,
                        )
            elif mode in ("token_map", "mixed"):
                _run_token_map_validation(
                    base_model, mach, patched_model, tokenizer,
                    device, episode_idx,
                    n_self_eval_steps=n_self_eval_steps,
                )
                if mode == "mixed":
                    # Also eval linear combos
                    print(f"  --- LINEAR sample ---")
                    for coeffs in [(1, 0), (0, 1), (1, 1), (2, 1)]:
                        _run_linear_validation(
                            base_model, mach, patched_model, tokenizer,
                            device, coeffs, episode_idx,
                            n_self_eval_steps=n_self_eval_steps,
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
                            n_problems=20, n_demos=5, n_self_eval_steps=0):
    """Evaluate linear combination performance for specific (c1, c2)."""
    test_correct = 0
    test_total = 0

    for ep in range(n_episodes):
        problems = generate_linear_episode(
            n_problems, n_demos=n_demos, coeffs=coeffs
        )
        mach.reset_episode()
        demo_problems = []

        for i, problem in enumerate(problems):
            input_ids = tokenizer(
                problem["prompt"], return_tensors="pt"
            ).input_ids.to(device)

            gru_memory = mach.observe(base_model, input_ids)
            writes = mach.fire(gru_memory)
            mach.apply_writes(writes)

            if problem["is_demo"]:
                demo_problems.append(problem)
                # Self-eval after last demo
                if n_self_eval_steps > 0 and (
                    i + 1 >= len(problems) or
                    not problems[i + 1].get("is_demo", False)
                ):
                    for step in range(n_self_eval_steps):
                        demo = demo_problems[step % len(demo_problems)]
                        demo_ids = tokenizer(
                            demo["prompt"], return_tensors="pt"
                        ).input_ids.to(device)
                        gru_mem = mach.observe_patched(
                            patched_model, demo_ids
                        )
                        w = mach.fire(gru_mem)
                        mach.apply_writes(w)
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


def _run_token_map_validation(base_model, mach, patched_model, tokenizer,
                               device, episode_idx, n_episodes=10,
                               n_problems=20, n_demos=5,
                               n_self_eval_steps=0):
    """Evaluate token mapping performance across random permutations."""
    test_correct = 0
    test_total = 0
    # Track per-symbol accuracy
    symbol_correct = 0
    symbol_total = 0

    for ep in range(n_episodes):
        problems = generate_token_mapping_episode(
            n_problems, n_demos=n_demos
        )
        mach.reset_episode()
        demo_problems = []

        for i, problem in enumerate(problems):
            input_ids = tokenizer(
                problem["prompt"], return_tensors="pt"
            ).input_ids.to(device)

            gru_memory = mach.observe(base_model, input_ids)
            writes = mach.fire(gru_memory)
            mach.apply_writes(writes)

            if problem["is_demo"]:
                demo_problems.append(problem)
                # Self-eval after last demo
                if n_self_eval_steps > 0 and (
                    i + 1 >= len(problems) or
                    not problems[i + 1].get("is_demo", False)
                ):
                    for step in range(n_self_eval_steps):
                        demo = demo_problems[step % len(demo_problems)]
                        demo_ids = tokenizer(
                            demo["prompt"], return_tensors="pt"
                        ).input_ids.to(device)
                        gru_mem = mach.observe_patched(
                            patched_model, demo_ids
                        )
                        w = mach.fire(gru_mem)
                        mach.apply_writes(w)
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
                correct = (pred_text == problem["answer"])

                # Per-symbol accuracy
                pred_syms = pred_text.split()
                answer_syms = problem["answer"].split()
                for j, ans_sym in enumerate(answer_syms):
                    symbol_total += 1
                    if j < len(pred_syms) and pred_syms[j] == ans_sym:
                        symbol_correct += 1

            test_correct += int(correct)
            test_total += 1

    test_acc = test_correct / test_total if test_total > 0 else 0
    sym_acc = symbol_correct / symbol_total if symbol_total > 0 else 0

    print(
        f"  EVAL ep{episode_idx} token_map | "
        f"exact={test_acc:.0%} per_symbol={sym_acc:.0%}"
    )

    if wandb is not None:
        wandb.log({
            "eval/token_map_exact": test_acc,
            "eval/token_map_symbol": sym_acc,
        })


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

    # Observation gates (self-discovering architecture)
    obs_gates = mach.get_obs_gates()
    if obs_gates is not None:
        for i, (layer_idx, gate_val) in enumerate(
            zip(mach.patch_layers, obs_gates.tolist())
        ):
            diag[f"obs_gate/layer{layer_idx}"] = gate_val
        diag["obs_gate/n_active"] = (obs_gates > 0.3).sum().item()
        diag["obs_gate/mean"] = obs_gates.mean().item()

    print(f"  Diagnostics at episode {episode_idx}:")
    for k, v in sorted(diag.items()):
        print(f"    {k}: {v:.6f}")

    if wandb is not None:
        wandb.log({f"diag/{k}": v for k, v in diag.items()})
