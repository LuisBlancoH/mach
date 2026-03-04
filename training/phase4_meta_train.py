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
    elif mode == "mixed":
        problems = []
        for _ in range(n_problems):
            diff = random.choice([5, 6, 7])
            p = generate_arithmetic_problems(1, diff)[0]
            p["difficulty"] = diff
            problems.append(p)
        return problems
    elif mode == "diverse":
        problems = []
        op_diffs = [1, 3, 6, 8]
        for _ in range(n_problems):
            diff = random.choice(op_diffs)
            p = generate_arithmetic_problems(1, diff)[0]
            p["difficulty"] = diff
            problems.append(p)
        return problems
    else:  # few_shot
        return generate_few_shot_episode(n_problems)


def run_episode_phase4(base_model, mach, patched_model, tokenizer,
                       problems, device, gamma=None):
    """
    Phase 4 episode: cerebellum + surprise gating + gamma=0 critic.

    Returns:
        total_loss: qwen_loss + critic_loss_weight * critic_loss
        cerebellum_loss: predictor MSE (for separate optimizer)
        rewards: list of reward floats
        problem_losses: list of per-problem Qwen CE losses
        diagnostics: dict with critic + cerebellum metrics
    """
    if gamma is None:
        gamma = config.PHASE4_GAMMA
    critic_loss_weight = config.PHASE4_CRITIC_LOSS_WEIGHT
    td_modulation = config.PHASE4_TD_MODULATION
    recency_alpha = config.PHASE4_RECENCY_ALPHA

    mach.reset_episode()

    rewards = []
    problem_losses = []
    values = []
    qwen_loss = torch.tensor(0.0, device=device, requires_grad=True)
    critic_loss = torch.tensor(0.0, device=device, requires_grad=True)

    last_value = torch.tensor(0.0, device=device)
    last_td_error = torch.tensor(0.0, device=device)
    last_reward = 0.0

    for i, problem in enumerate(problems):
        input_ids = tokenizer(
            problem["prompt"], return_tensors="pt"
        ).input_ids.to(device)

        # Step 1: Observe (with cerebellum surprise + correction)
        gru_memory = mach.observe(base_model, input_ids)

        # Step 2: Fire meta-learner
        writes = mach.fire(gru_memory, last_value, last_td_error)

        # Step 3: Get value estimate
        current_value = mach.get_value()
        values.append(current_value)

        # Step 4: Apply writes
        mach.apply_writes(writes)

        # Demo problems: observe and write, but skip evaluation
        if problem.get("is_demo", False):
            rewards.append(0.0)
            problem_losses.append(0.0)
            last_value = current_value.detach()
            last_reward = 0.0
            continue

        # Step 5: Forward Qwen with patches
        full_text = problem["prompt"] + problem["answer"]
        encoding = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_len = len(tokenizer(problem["prompt"]).input_ids)
        labels = encoding.input_ids.clone()
        labels[0, :prompt_len] = -100

        output = patched_model(input_ids=encoding.input_ids, labels=labels)

        # Step 6: Compute reward
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

        # Step 7: Accumulate Qwen CE loss with two modulations:
        # 1. TD modulation (dopamine): unexpected outcomes get more gradient
        # 2. Recency weighting: later problems matter more (eligibility trace)
        advantage = reward - current_value.detach().item()
        td_weight = 1.0 + td_modulation * abs(advantage)
        recency_weight = 1.0 + recency_alpha * (i / max(len(problems) - 1, 1))
        qwen_loss = qwen_loss + td_weight * recency_weight * output.loss

        # Step 8: TD error and critic loss for PREVIOUS step
        # gamma=0: td_target = last_reward (no bootstrap)
        if i > 0:
            td_target = torch.tensor(
                last_reward, device=device, dtype=torch.float32
            )
            if gamma > 0:
                td_target = td_target + gamma * current_value.detach()
            td_error = td_target - values[i - 1]
            critic_loss = critic_loss + (
                values[i - 1] - td_target.detach()
            ) ** 2
            last_td_error = td_error.detach()

        last_value = current_value.detach()
        last_reward = reward

    # Terminal TD for last step
    if len(values) > 0:
        td_target_final = torch.tensor(
            last_reward, device=device, dtype=torch.float32
        )
        critic_loss = critic_loss + (
            values[-1] - td_target_final.detach()
        ) ** 2

    total_loss = qwen_loss + critic_loss_weight * critic_loss

    # Cerebellum predictor loss (separate optimizer)
    cerebellum_loss = mach.get_cerebellum_loss()

    # Diagnostics
    cerebellum_diag = mach.get_cerebellum_diagnostics()
    diagnostics = {
        "critic_loss": critic_loss.item(),
        "qwen_loss": qwen_loss.item(),
        "mean_value": (
            sum(v.item() for v in values) / len(values) if values else 0
        ),
        "value_range": (
            max(v.item() for v in values) - min(v.item() for v in values)
        ) if len(values) > 1 else 0,
        **cerebellum_diag,
    }

    return total_loss, cerebellum_loss, rewards, problem_losses, diagnostics


def meta_train_phase4(base_model, mach, patched_model, tokenizer, device,
                      n_episodes=None, lr=None, cerebellum_lr=None,
                      curriculum=None, checkpoint_path=None):
    """
    Phase 4 meta-training loop.

    Two optimizers:
    1. meta_optimizer: all meta params + correction_proj
    2. cerebellum_optimizer: predictor params only (online supervised)
    """
    if n_episodes is None:
        n_episodes = config.PHASE4_EPISODES
    if lr is None:
        lr = config.PHASE4_LR
    if cerebellum_lr is None:
        cerebellum_lr = config.PHASE4_CEREBELLUM_LR
    if curriculum is None:
        curriculum = DEFAULT_CURRICULUM

    if checkpoint_path is not None:
        print(f"Loading Phase 3 checkpoint from {checkpoint_path}...")
        mach.load_phase3_checkpoint(checkpoint_path, device=device)

    # Meta params: everything except predictor
    meta_params = []
    meta_params += list(mach.obs_proj.parameters())
    meta_params += list(mach.gru.parameters())
    meta_params += list(mach.basis.parameters())
    meta_params += list(mach.transformer.parameters())
    meta_params += list(mach.action_head.parameters())
    meta_params += list(mach.memory_head.parameters())
    meta_params += list(mach.critic.parameters())
    meta_params += list(mach.critic_signal_proj.parameters())
    meta_params += list(mach.cerebellum.correction_proj.parameters())

    # Cerebellum predictor: separate optimizer
    cerebellum_params = list(mach.cerebellum.predictor.parameters())

    meta_optimizer = torch.optim.Adam(meta_params, lr=lr)
    cerebellum_optimizer = torch.optim.Adam(cerebellum_params, lr=cerebellum_lr)

    n_meta = sum(p.numel() for p in meta_params)
    n_cerebellum = sum(p.numel() for p in cerebellum_params)
    print(f"Phase 4 meta-learner trainable parameters: {n_meta:,}")
    print(f"Phase 4 cerebellum predictor parameters: {n_cerebellum:,}")
    print(f"Phase 4 gamma: {config.PHASE4_GAMMA}")
    print(f"Phase 4 surprise_scale: {mach.gru.surprise_scale}")

    def get_n_problems(episode_idx):
        if episode_idx < 100:
            return 5
        elif episode_idx < 300:
            return 10
        elif episode_idx < 600:
            return 15
        else:
            return config.PHASE4_PROBLEMS_PER_EPISODE

    for episode_idx in range(n_episodes):
        mode = get_episode_mode(episode_idx, curriculum)
        n_problems = get_n_problems(episode_idx)
        problems = generate_episode_problems(n_problems, mode)

        meta_optimizer.zero_grad()
        cerebellum_optimizer.zero_grad()

        try:
            loss, cerebellum_loss, rewards, problem_losses, diagnostics = \
                run_episode_phase4(
                    base_model, mach, patched_model, tokenizer,
                    problems, device
                )

            loss.backward()
            loss_scalar = loss.item()

            if cerebellum_loss.requires_grad:
                cerebellum_loss.backward()

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at episode {episode_idx} with {n_problems} "
                  f"problems. Reducing to {max(5, n_problems // 2)}.")
            torch.cuda.empty_cache()
            meta_optimizer.zero_grad()
            cerebellum_optimizer.zero_grad()
            problems = problems[:max(5, n_problems // 2)]
            loss, cerebellum_loss, rewards, problem_losses, diagnostics = \
                run_episode_phase4(
                    base_model, mach, patched_model, tokenizer,
                    problems, device
                )
            loss.backward()
            loss_scalar = loss.item()
            if cerebellum_loss.requires_grad:
                cerebellum_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            meta_params, max_norm=config.PHASE4_GRAD_CLIP
        )
        meta_optimizer.step()
        cerebellum_optimizer.step()

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

            surprise_str = ""
            if "avg_surprise" in diagnostics:
                surprise_str = (
                    f" surp={diagnostics['avg_surprise']:.3f}"
                    f"[{diagnostics.get('min_surprise', 0):.2f}-"
                    f"{diagnostics.get('max_surprise', 0):.2f}]"
                )

            cerebellum_str = ""
            if "cerebellum_loss" in diagnostics:
                cerebellum_str = (
                    f" cb={diagnostics['cerebellum_loss']:.4f}"
                )

            diff_str = ""
            if mode in ("mixed", "diverse", "few_shot"):
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
                f"loss={loss_scalar:.4f} "
                f"critic={diagnostics['critic_loss']:.4f}"
                f"{cerebellum_str} "
                f"avg_v={diagnostics['mean_value']:.3f} | "
                f"avg_r={avg_reward:.2f} "
                f"early={early_acc:.0%} late={late_acc:.0%}"
                f"{surprise_str}"
                f"{diff_str}"
            )

            log_dict = {
                "episode": episode_idx,
                "n_problems": len(problems),
                "total_loss": loss_scalar,
                "qwen_loss": diagnostics["qwen_loss"],
                "critic_loss": diagnostics["critic_loss"],
                "mean_value": diagnostics["mean_value"],
                "value_range": diagnostics["value_range"],
                "avg_reward": avg_reward,
                "early_accuracy": early_acc,
                "late_accuracy": late_acc,
                "learning_delta": late_acc - early_acc,
            }

            for k in ["avg_surprise", "max_surprise", "min_surprise",
                       "cerebellum_loss"]:
                if k in diagnostics:
                    log_dict[k] = diagnostics[k]

            for j, pl in enumerate(problem_losses):
                log_dict[f"problem_loss/{j}"] = pl

            if wandb is not None:
                wandb.log(log_dict)

        # Diagnostics + validation every 200 episodes + final
        if (episode_idx % 200 == 0 and episode_idx > 0) or \
                episode_idx == n_episodes - 1:
            _log_diagnostics_phase4(mach, meta_params, episode_idx)
            if mode == "few_shot":
                for op in ["add", "sub", "mul", "div"]:
                    _run_few_shot_validation(
                        base_model, mach, patched_model, tokenizer,
                        device, op, episode_idx
                    )
            else:
                for eval_diff in [6, 7]:
                    _run_validation_phase4(
                        base_model, mach, patched_model, tokenizer,
                        device, eval_diff, episode_idx
                    )


def _run_validation_phase4(base_model, mach, patched_model, tokenizer, device,
                           difficulty, episode_idx, n_episodes=10,
                           n_problems=20):
    """Held-out evaluation with cerebellum diagnostics."""
    base_correct = 0
    meta_correct_early = 0
    meta_correct_late = 0
    meta_total_early = 0
    meta_total_late = 0
    base_total = 0
    all_rewards = []
    all_values = []
    all_surprises = []

    for ep in range(n_episodes):
        problems = generate_arithmetic_problems(n_problems, difficulty)
        mach.reset_episode()

        last_value = torch.tensor(0.0, device=device)
        last_td_error = torch.tensor(0.0, device=device)
        rewards = []

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
            rewards.append(reward)

            # TD error with gamma=0
            if i > 0:
                td_target = torch.tensor(
                    rewards[-2], device=device, dtype=torch.float32
                )
                last_td_error = (td_target - last_value).detach()
            last_value = current_value.detach()

            if i < 5:
                meta_correct_early += int(correct)
                meta_total_early += 1
            if i >= n_problems - 5:
                meta_correct_late += int(correct)
                meta_total_late += 1

        cb_diag = mach.get_cerebellum_diagnostics()
        if "avg_surprise" in cb_diag:
            all_surprises.append(cb_diag["avg_surprise"])

        all_rewards.extend(rewards)

        # Baseline: base Qwen without patches
        mach.reset_episode()
        for problem in problems[:10]:
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
                base_correct += int(predicted == problem["answer"])
                base_total += 1

    base_acc = base_correct / base_total if base_total > 0 else 0
    early_acc = (
        meta_correct_early / meta_total_early if meta_total_early > 0 else 0
    )
    late_acc = (
        meta_correct_late / meta_total_late if meta_total_late > 0 else 0
    )
    delta = late_acc - early_acc
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
    mean_val = sum(all_values) / len(all_values) if all_values else 0
    avg_surprise = (
        sum(all_surprises) / len(all_surprises) if all_surprises else 0
    )

    print(
        f"  EVAL ep{episode_idx} d={difficulty} | "
        f"base={base_acc:.0%} early={early_acc:.0%} late={late_acc:.0%} "
        f"delta={delta:+.0%} avg_r={avg_reward:.2f} "
        f"avg_v={mean_val:.3f} avg_surp={avg_surprise:.3f}"
    )

    if wandb is not None:
        wandb.log({
            f"eval/d{difficulty}_base": base_acc,
            f"eval/d{difficulty}_early": early_acc,
            f"eval/d{difficulty}_late": late_acc,
            f"eval/d{difficulty}_delta": delta,
            f"eval/d{difficulty}_avg_reward": avg_reward,
            f"eval/d{difficulty}_avg_value": mean_val,
            f"eval/d{difficulty}_avg_surprise": avg_surprise,
        })


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
    all_surprises = []
    n_test = n_problems - n_demos

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

            # Evaluate test problem
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
            if test_idx > 1:
                td_target = torch.tensor(
                    reward, device=device, dtype=torch.float32
                )
                last_td_error = (td_target - last_value).detach()
            last_value = current_value.detach()

        cb_diag = mach.get_cerebellum_diagnostics()
        if "avg_surprise" in cb_diag:
            all_surprises.append(cb_diag["avg_surprise"])

    test_acc = test_correct / test_total if test_total > 0 else 0
    early_acc = (
        test_correct_early / test_total_early if test_total_early > 0 else 0
    )
    late_acc = (
        test_correct_late / test_total_late if test_total_late > 0 else 0
    )
    delta = late_acc - early_acc
    avg_surprise = (
        sum(all_surprises) / len(all_surprises) if all_surprises else 0
    )

    print(
        f"  EVAL ep{episode_idx} {op_type:3s} | "
        f"test={test_acc:.0%} early={early_acc:.0%} late={late_acc:.0%} "
        f"delta={delta:+.0%} avg_surp={avg_surprise:.3f}"
    )

    if wandb is not None:
        wandb.log({
            f"eval/{op_type}_test": test_acc,
            f"eval/{op_type}_early": early_acc,
            f"eval/{op_type}_late": late_acc,
            f"eval/{op_type}_delta": delta,
        })


def _log_diagnostics_phase4(mach, meta_params, episode_idx):
    """Log gradient norms including cerebellum components."""
    diag = {}

    component_grad_norms = {}
    for name, param in mach.named_parameters():
        if param.grad is not None:
            if name.startswith("cerebellum."):
                component = "cerebellum." + name.split('.')[1]
            else:
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
        diag[f"basis_norm/patch{i}_up_U"] = (
            mach.basis.up_U[i].norm().item()
        )
        diag[f"basis_norm/patch{i}_up_V"] = (
            mach.basis.up_V[i].norm().item()
        )

    for i, patch in enumerate(mach.patches):
        if patch.delta_down is not None:
            diag[f"patch_delta/patch{i}_down"] = (
                patch.delta_down.norm().item()
            )
            diag[f"patch_delta/patch{i}_up"] = (
                patch.delta_up.norm().item()
            )

    for name, param in mach.critic.named_parameters():
        diag[f"critic_weight/{name}"] = param.norm().item()

    for name, param in mach.cerebellum.named_parameters():
        diag[f"cerebellum_weight/{name}"] = param.norm().item()

    print(f"  Diagnostics at episode {episode_idx}:")
    for k, v in sorted(diag.items()):
        print(f"    {k}: {v:.6f}")

    if wandb is not None:
        wandb.log({f"diag/{k}": v for k, v in diag.items()})
