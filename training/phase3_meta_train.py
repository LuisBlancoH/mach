import random

import torch

try:
    import wandb
except ImportError:
    wandb = None

from data.arithmetic import generate_arithmetic_problems, extract_number
import config


DEFAULT_CURRICULUM = [
    (0, 300, "single"),     # Warm up on d6 while critic learns basic values
    (300, 2000, "mixed"),   # Mixed d5/d6/d7 to prevent static convergence
]


def get_episode_mode(episode_idx, curriculum):
    for start, end, mode in curriculum:
        if start <= episode_idx < end:
            return mode
    return curriculum[-1][2]


def generate_episode_problems(n_problems, mode):
    if mode == "single":
        return generate_arithmetic_problems(n_problems, 6)
    else:
        problems = []
        for _ in range(n_problems):
            diff = random.choice([5, 6, 7])
            p = generate_arithmetic_problems(1, diff)[0]
            p["difficulty"] = diff
            problems.append(p)
        return problems


def run_episode_phase3(base_model, mach, patched_model, tokenizer,
                       problems, device, gamma=None):
    """
    Phase 3 episode with critic value estimation and TD learning.

    Returns:
        total_loss: qwen_loss + critic_loss_weight * critic_loss (differentiable)
        rewards: list of reward floats
        problem_losses: list of per-problem Qwen CE loss values
        diagnostics: dict with critic-specific metrics
    """
    if gamma is None:
        gamma = config.PHASE3_GAMMA
    critic_loss_weight = config.PHASE3_CRITIC_LOSS_WEIGHT

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

        # Step 1: Observe (detached)
        gru_memory = mach.observe(base_model, input_ids)

        # Step 2: Fire meta-learner (with critic signals)
        writes = mach.fire(gru_memory, last_value, last_td_error)

        # Step 3: Get value estimate from current hidden states
        current_value = mach.get_value()
        values.append(current_value)

        # Step 4: Apply writes
        mach.apply_writes(writes)

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

        # Step 7: Accumulate Qwen CE loss
        qwen_loss = qwen_loss + output.loss

        # Step 8: TD error and critic loss for PREVIOUS step
        if i > 0:
            td_target = last_reward + gamma * current_value.detach()
            td_error = td_target - values[i - 1]
            critic_loss = critic_loss + (values[i - 1] - td_target.detach()) ** 2
            last_td_error = td_error.detach()

        last_value = current_value.detach()
        last_reward = reward

    # Terminal TD for last step (no future value)
    if len(values) > 0:
        td_target_final = torch.tensor(
            last_reward, device=device, dtype=torch.float32
        )
        critic_loss = critic_loss + (values[-1] - td_target_final.detach()) ** 2

    total_loss = qwen_loss + critic_loss_weight * critic_loss

    diagnostics = {
        "critic_loss": critic_loss.item(),
        "qwen_loss": qwen_loss.item(),
        "mean_value": sum(v.item() for v in values) / len(values) if values else 0,
        "value_range": (
            max(v.item() for v in values) - min(v.item() for v in values)
        ) if len(values) > 1 else 0,
    }

    return total_loss, rewards, problem_losses, diagnostics


def meta_train_phase3(base_model, mach, patched_model, tokenizer, device,
                      n_episodes=None, lr=None, curriculum=None,
                      checkpoint_path=None):
    """
    Phase 3 meta-training loop.

    Trainable: all Phase 2 params + critic + critic_signal_proj
    Frozen: base_model (Qwen), patches (written by meta-learner)
    """
    if n_episodes is None:
        n_episodes = config.PHASE3_EPISODES
    if lr is None:
        lr = config.PHASE3_LR
    if curriculum is None:
        curriculum = DEFAULT_CURRICULUM

    if checkpoint_path is not None:
        print(f"Loading Phase 2 checkpoint from {checkpoint_path}...")
        mach.load_phase2_checkpoint(checkpoint_path, device=device)

    meta_params = []
    meta_params += list(mach.obs_proj.parameters())
    meta_params += list(mach.gru.parameters())
    meta_params += list(mach.basis.parameters())
    meta_params += list(mach.transformer.parameters())
    meta_params += list(mach.action_head.parameters())
    meta_params += list(mach.memory_head.parameters())
    meta_params += list(mach.critic.parameters())
    meta_params += list(mach.critic_signal_proj.parameters())

    optimizer = torch.optim.Adam(meta_params, lr=lr)

    n_meta_params = sum(p.numel() for p in meta_params)
    print(f"Phase 3 meta-learner trainable parameters: {n_meta_params:,}")

    def get_n_problems(episode_idx):
        if episode_idx < 100:
            return 5
        elif episode_idx < 300:
            return 10
        elif episode_idx < 600:
            return 15
        else:
            return config.PHASE3_PROBLEMS_PER_EPISODE

    use_chunked = False

    for episode_idx in range(n_episodes):
        mode = get_episode_mode(episode_idx, curriculum)
        n_problems = get_n_problems(episode_idx)
        problems = generate_episode_problems(n_problems, mode)

        optimizer.zero_grad()

        try:
            loss, rewards, problem_losses, diagnostics = run_episode_phase3(
                base_model, mach, patched_model, tokenizer,
                problems, device
            )
            loss.backward()
            loss_scalar = loss.item()

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at episode {episode_idx} with {n_problems} problems. "
                  f"Reducing to {max(5, n_problems // 2)}.")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            problems = problems[:max(5, n_problems // 2)]
            loss, rewards, problem_losses, diagnostics = run_episode_phase3(
                base_model, mach, patched_model, tokenizer,
                problems, device
            )
            loss.backward()
            loss_scalar = loss.item()

        torch.nn.utils.clip_grad_norm_(
            meta_params, max_norm=config.PHASE3_GRAD_CLIP
        )
        optimizer.step()

        if episode_idx % 10 == 0:
            avg_reward = sum(rewards) / len(rewards)
            n_early = min(5, len(rewards))
            n_late = min(5, len(rewards))
            early_acc = sum(1 for r in rewards[:n_early] if r > 0) / n_early
            late_acc = sum(1 for r in rewards[-n_late:] if r > 0) / n_late

            # Per-difficulty breakdown for mixed mode
            diff_str = ""
            if mode == "mixed":
                diff_correct = {}
                diff_total = {}
                for j, p in enumerate(problems):
                    d = p.get("difficulty", "?")
                    diff_total[d] = diff_total.get(d, 0) + 1
                    if rewards[j] > 0:
                        diff_correct[d] = diff_correct.get(d, 0) + 1
                diff_str = " | " + " ".join(
                    f"d{d}={diff_correct.get(d,0)}/{diff_total[d]}"
                    for d in sorted(diff_total)
                )

            print(
                f"Episode {episode_idx:4d} | {mode} n={len(problems):2d} | "
                f"loss={loss_scalar:.4f} critic={diagnostics['critic_loss']:.4f} "
                f"avg_v={diagnostics['mean_value']:.3f} | "
                f"avg_r={avg_reward:.2f} early={early_acc:.0%} late={late_acc:.0%}"
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

            for j, pl in enumerate(problem_losses):
                log_dict[f"problem_loss/{j}"] = pl

            if wandb is not None:
                wandb.log(log_dict)

        # Diagnostics + validation every 200 episodes + final
        if (episode_idx % 200 == 0 and episode_idx > 0) or episode_idx == n_episodes - 1:
            _log_diagnostics_phase3(mach, meta_params, episode_idx)
            for eval_diff in [5, 6, 7]:
                _run_validation_phase3(
                    base_model, mach, patched_model, tokenizer,
                    device, eval_diff, episode_idx
                )


def _run_validation_phase3(base_model, mach, patched_model, tokenizer, device,
                           difficulty, episode_idx, n_episodes=10, n_problems=20):
    """
    Held-out evaluation: fresh problems, no gradient.
    Compares base Qwen (no patches) vs meta-learner writes.
    Also reports critic value statistics.
    """
    base_correct = 0
    meta_correct_early = 0
    meta_correct_late = 0
    meta_total_early = 0
    meta_total_late = 0
    base_total = 0
    all_rewards = []
    all_values = []

    for ep in range(n_episodes):
        problems = generate_arithmetic_problems(n_problems, difficulty)
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

            # TD error for next step
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

        # Baseline: base Qwen without patches
        mach.reset_episode()
        for problem in problems[:10]:
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

    base_acc = base_correct / base_total if base_total > 0 else 0
    early_acc = meta_correct_early / meta_total_early if meta_total_early > 0 else 0
    late_acc = meta_correct_late / meta_total_late if meta_total_late > 0 else 0
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
    delta = late_acc - early_acc
    mean_val = sum(all_values) / len(all_values) if all_values else 0

    print(f"  EVAL ep{episode_idx} d={difficulty} | "
          f"base={base_acc:.0%} early={early_acc:.0%} late={late_acc:.0%} "
          f"delta={delta:+.0%} avg_r={avg_reward:.2f} avg_v={mean_val:.3f}")

    if wandb is not None:
        wandb.log({
            f"eval/d{difficulty}_base": base_acc,
            f"eval/d{difficulty}_early": early_acc,
            f"eval/d{difficulty}_late": late_acc,
            f"eval/d{difficulty}_delta": delta,
            f"eval/d{difficulty}_avg_reward": avg_reward,
            f"eval/d{difficulty}_avg_value": mean_val,
        })


def _log_diagnostics_phase3(mach, meta_params, episode_idx):
    """Log gradient norms, basis norms, critic stats, patch deltas."""
    diag = {}

    # Gradient norms per component
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

    # Basis vector norms
    for i in range(mach.basis.n_patches):
        diag[f"basis_norm/patch{i}_down_U"] = mach.basis.down_U[i].norm().item()
        diag[f"basis_norm/patch{i}_down_V"] = mach.basis.down_V[i].norm().item()
        diag[f"basis_norm/patch{i}_up_U"] = mach.basis.up_U[i].norm().item()
        diag[f"basis_norm/patch{i}_up_V"] = mach.basis.up_V[i].norm().item()

    # Patch delta norms
    for i, patch in enumerate(mach.patches):
        if patch.delta_down is not None:
            diag[f"patch_delta/patch{i}_down"] = patch.delta_down.norm().item()
            diag[f"patch_delta/patch{i}_up"] = patch.delta_up.norm().item()

    # Critic weight norms
    for name, param in mach.critic.named_parameters():
        diag[f"critic_weight/{name}"] = param.norm().item()

    print(f"  Diagnostics at episode {episode_idx}:")
    for k, v in sorted(diag.items()):
        print(f"    {k}: {v:.6f}")

    if wandb is not None:
        wandb.log({f"diag/{k}": v for k, v in diag.items()})
