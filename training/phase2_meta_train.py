import torch

try:
    import wandb
except ImportError:
    wandb = None

from data.arithmetic import generate_arithmetic_problems
from training.episode import run_episode, run_episode_chunked
import config


# Curriculum: episode ranges mapped to difficulty levels
# Phase 1 results: d6 +18pp, d7 +32.8pp, d8 REGRESSED, d9 +19.2pp
# Multiplication is the sweet spot. Avoid division (d8).
DEFAULT_CURRICULUM = [
    (0, 500, 5),        # 2x2 multiplication (warm up — baseline ~50-60%)
    (500, 1000, 6),     # 3x2 multiplication (Phase 1 best: 65% -> 83%)
    (1000, 1500, 7),    # 3x3 multiplication (Phase 1 strong pass: 6% -> 39%)
    (1500, 2000, 9),    # mixed hard (Phase 1: 62% -> 82%)
]


def get_difficulty(episode_idx, curriculum):
    for start, end, diff in curriculum:
        if start <= episode_idx < end:
            return diff
    return curriculum[-1][2]


def meta_train(base_model, mach, patched_model, tokenizer, device,
               n_episodes=None, lr=None, curriculum=None):
    """
    Train the meta-learner to produce useful patch writes.

    Trainable: obs_proj, gru, basis vectors, transformer, action_head,
               memory_head, reward_proj
    Frozen: base_model (Qwen), patches (written by meta-learner, not by gradient)
    """
    if n_episodes is None:
        n_episodes = config.PHASE2_EPISODES
    if lr is None:
        lr = config.PHASE2_LR
    if curriculum is None:
        curriculum = DEFAULT_CURRICULUM

    # Collect all meta-learner parameters
    meta_params = []
    meta_params += list(mach.obs_proj.parameters())
    meta_params += list(mach.gru.parameters())
    meta_params += list(mach.basis.parameters())
    meta_params += list(mach.transformer.parameters())
    meta_params += list(mach.action_head.parameters())
    meta_params += list(mach.memory_head.parameters())
    meta_params += list(mach.reward_proj.parameters())

    optimizer = torch.optim.Adam(meta_params, lr=lr)

    n_meta_params = sum(p.numel() for p in meta_params)
    print(f"Meta-learner trainable parameters: {n_meta_params:,}")

    # Progressive episode length: start short, grow to full
    def get_n_problems(episode_idx):
        if episode_idx < 100:
            return 5
        elif episode_idx < 300:
            return 10
        elif episode_idx < 600:
            return 15
        else:
            return config.PHASE2_PROBLEMS_PER_EPISODE

    use_chunked = False  # Will flip to True on OOM

    for episode_idx in range(n_episodes):
        difficulty = get_difficulty(episode_idx, curriculum)
        n_problems = get_n_problems(episode_idx)
        problems = generate_arithmetic_problems(n_problems, difficulty)

        optimizer.zero_grad()

        try:
            if use_chunked:
                loss_val, rewards, problem_losses = run_episode_chunked(
                    base_model, mach, patched_model, tokenizer,
                    problems, device, optimizer
                )
                # Gradients already accumulated in chunked mode
                loss_scalar = loss_val
            else:
                loss, rewards, problem_losses = run_episode(
                    base_model, mach, patched_model, tokenizer,
                    problems, device
                )
                loss.backward()
                loss_scalar = loss.item()

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at episode {episode_idx} with {n_problems} problems. "
                  f"Switching to per-problem backward.")
            torch.cuda.empty_cache()
            use_chunked = True
            optimizer.zero_grad()

            loss_val, rewards, problem_losses = run_episode_chunked(
                base_model, mach, patched_model, tokenizer,
                problems, device, optimizer
            )
            loss_scalar = loss_val

        torch.nn.utils.clip_grad_norm_(meta_params, max_norm=config.PHASE2_GRAD_CLIP)
        optimizer.step()

        # Logging
        if episode_idx % 10 == 0:
            avg_reward = sum(rewards) / len(rewards)
            n_early = min(5, len(rewards))
            n_late = min(5, len(rewards))
            early_acc = sum(1 for r in rewards[:n_early] if r > 0) / n_early
            late_acc = sum(1 for r in rewards[-n_late:] if r > 0) / n_late

            print(f"Episode {episode_idx:4d} | d={difficulty} n={n_problems:2d} | "
                  f"loss={loss_scalar:.4f} avg_r={avg_reward:.2f} | "
                  f"early={early_acc:.0%} late={late_acc:.0%}")

            log_dict = {
                "episode": episode_idx,
                "difficulty": difficulty,
                "n_problems": n_problems,
                "loss": loss_scalar,
                "avg_reward": avg_reward,
                "early_accuracy": early_acc,
                "late_accuracy": late_acc,
                "learning_delta": late_acc - early_acc,
            }

            # Per-problem loss curve (shows within-episode learning)
            for j, pl in enumerate(problem_losses):
                log_dict[f"problem_loss/{j}"] = pl

            if wandb is not None:
                wandb.log(log_dict)

        # Validation + diagnostics every 200 episodes
        if episode_idx % 200 == 0 and episode_idx > 0:
            _run_validation(base_model, mach, patched_model, tokenizer,
                            device, difficulty, episode_idx)
            _log_diagnostics(mach, meta_params, episode_idx)


def _run_validation(base_model, mach, patched_model, tokenizer, device,
                    difficulty, episode_idx, n_episodes=10, n_problems=20):
    """
    Held-out evaluation: fresh problems, no gradient.
    Compares base Qwen (no patches) vs meta-learner writes.
    """
    from data.arithmetic import generate_arithmetic_problems, extract_number

    base_correct = 0
    meta_correct_early = 0
    meta_correct_late = 0
    meta_total_early = 0
    meta_total_late = 0
    base_total = 0
    all_rewards = []

    for ep in range(n_episodes):
        problems = generate_arithmetic_problems(n_problems, difficulty)
        mach.reset_episode()

        rewards = []
        for i, problem in enumerate(problems):
            input_ids = tokenizer(
                problem["prompt"], return_tensors="pt"
            ).input_ids.to(device)

            # Observe + fire + write (still need forward graph for patches)
            gru_memory = mach.observe(base_model, input_ids)
            reward_signals = torch.tensor(
                [rewards[-1] if rewards else 0.0,
                 sum(rewards), float(i)],
                device=device, dtype=torch.float32
            )
            writes = mach.fire(gru_memory, reward_signals)
            mach.apply_writes(writes)

            # Evaluate patched model
            full_text = problem["prompt"] + problem["answer"]
            encoding = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_len = len(tokenizer(problem["prompt"]).input_ids)

            with torch.no_grad():
                output = patched_model(input_ids=encoding.input_ids)
                logits = output.logits if hasattr(output, 'logits') else output[0]
                pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                predicted = extract_number(pred_text)
                correct = (predicted == problem["answer"])

            reward = 1.0 if correct else -1.0
            rewards.append(reward)

            if i < 5:
                meta_correct_early += int(correct)
                meta_total_early += 1
            if i >= n_problems - 5:
                meta_correct_late += int(correct)
                meta_total_late += 1

        all_rewards.extend(rewards)

        # Baseline: run same problems through base Qwen without patches
        # (reset patches so they're zero, then restore)
        mach.reset_episode()  # zeros out all deltas
        for problem in problems[:10]:  # sample 10 for speed
            full_text = problem["prompt"] + problem["answer"]
            encoding = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_len = len(tokenizer(problem["prompt"]).input_ids)
            with torch.no_grad():
                output = patched_model(input_ids=encoding.input_ids)
                logits = output.logits if hasattr(output, 'logits') else output[0]
                pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                predicted = extract_number(pred_text)
                base_correct += int(predicted == problem["answer"])
                base_total += 1

    base_acc = base_correct / base_total if base_total > 0 else 0
    early_acc = meta_correct_early / meta_total_early if meta_total_early > 0 else 0
    late_acc = meta_correct_late / meta_total_late if meta_total_late > 0 else 0
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
    delta = late_acc - early_acc

    print(f"  EVAL ep{episode_idx} d={difficulty} | "
          f"base={base_acc:.0%} early={early_acc:.0%} late={late_acc:.0%} "
          f"delta={delta:+.0%} avg_r={avg_reward:.2f}")

    if wandb is not None:
        wandb.log({
            "eval/episode": episode_idx,
            "eval/base_accuracy": base_acc,
            "eval/early_accuracy": early_acc,
            "eval/late_accuracy": late_acc,
            "eval/learning_delta": delta,
            "eval/avg_reward": avg_reward,
            "eval/improvement_over_base": late_acc - base_acc,
        })


def _log_diagnostics(mach, meta_params, episode_idx):
    """Log gradient norms, basis vector norms, gate values, etc."""
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

    # Patch delta norms (from last episode)
    for i, patch in enumerate(mach.patches):
        if patch.delta_down is not None:
            diag[f"patch_delta/patch{i}_down"] = patch.delta_down.norm().item()
            diag[f"patch_delta/patch{i}_up"] = patch.delta_up.norm().item()

    print(f"  Diagnostics at episode {episode_idx}:")
    for k, v in sorted(diag.items()):
        print(f"    {k}: {v:.6f}")

    if wandb is not None:
        wandb.log({f"diag/{k}": v for k, v in diag.items()})
