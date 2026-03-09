"""
Two-Channel training: modulation (Channel 1) + writing (Channel 2).
Reuses episode structure from phase5_concat but with channel ablation validation.
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


def graded_reward(predicted, actual):
    """Graded reward: +1 for correct, graded negative for incorrect.
    Wrong answers are always negative — close miss = gentle correction,
    way off = strong correction. Never reinforces wrong answers."""
    if predicted == actual:
        return 1.0
    try:
        p = int(predicted) if predicted is not None else 0
        a = int(actual)
    except (ValueError, TypeError):
        return -1.0
    # Relative error: how far off as fraction of answer magnitude
    scale = max(abs(a), 1)
    rel_error = abs(p - a) / scale
    # Wrong answers: -0.2 (very close) to -1.0 (way off)
    # close miss doesn't cancel learning as hard, but still corrective
    return -0.2 - 0.8 * min(rel_error, 1.0)


CONTINUOUS_LINEAR_CURRICULUM = [
    (0, 5000, "continuous_linear"),
]

TOKEN_MAP_CURRICULUM = [
    (0, 5000, "token_map"),
]

MIXED_CURRICULUM = [
    (0, 5000, "mixed"),
]

FEW_SHOT_CURRICULUM = [
    (0, 5000, "few_shot"),
]

FEW_SHOT_BASIC_CURRICULUM = [
    (0, 5000, "few_shot_basic"),
]

# Diverse ops for generalization training
# Train on 9 ops, hold out mod/max/min (Qwen CAN do these — fair generalization test)
# Also test "impossible" ops (Qwen fundamentally can't compute) as control group
DIVERSE_TRAIN_OPS = [
    "add", "sub", "mul", "div",
    "gcd", "abs_diff", "avg", "first", "second",
    "digit_sum_add", "bitwise_and",
]
DIVERSE_HELDOUT_OPS = ["mod", "max", "min"]
DIVERSE_IMPOSSIBLE_OPS = ["bitwise_xor"]

DIVERSE_OPS_CURRICULUM = [
    (0, 5000, "few_shot_diverse"),
]


def get_episode_mode(episode_idx, curriculum):
    for start, end, mode in curriculum:
        if start <= episode_idx < end:
            return mode
    return curriculum[-1][2]


def generate_episode_problems(n_problems, mode):
    if mode == "continuous_linear":
        return generate_linear_episode(n_problems, continuous=True, max_coeff=5)
    elif mode == "token_map":
        return generate_token_mapping_episode(n_problems)
    elif mode == "mixed":
        sub_mode = random.choice(["continuous_linear", "token_map"])
        return generate_episode_problems(n_problems, sub_mode)
    elif mode == "few_shot":
        return generate_few_shot_episode(n_problems)
    elif mode == "few_shot_basic":
        # Only basic ops (add/sub/mul/div), all as test problems (no demos)
        op = random.choice(["add", "sub", "mul", "div"])
        return generate_few_shot_episode(n_problems, n_demos=0, op_type=op)
    elif mode == "few_shot_diverse":
        # 12 diverse ops for generalization training
        op = random.choice(DIVERSE_TRAIN_OPS)
        return generate_few_shot_episode(n_problems, n_demos=0, op_type=op)
    else:
        return generate_few_shot_episode(n_problems)


def _contrastive_loss(mach, base_model, tokenizer, problems, device):
    """
    Contrastive auxiliary loss: gives demo_encoder direct gradient.

    Generate demos for a different coefficient pair, encode them,
    and push the two task_states apart (cosine distance).
    Same-task states should be similar; different-task states should differ.
    """
    # Get current task's coefficients
    c1 = problems[0].get("c1")
    c2 = problems[0].get("c2")
    if c1 is None:
        return torch.tensor(0.0, device=device)

    # Generate negative: different coefficients
    for _ in range(10):
        neg_c1 = random.randint(0, 5)
        neg_c2 = random.randint(0, 5)
        if (neg_c1, neg_c2) != (c1, c2):
            break

    neg_problems = generate_linear_episode(5, coeffs=(neg_c1, neg_c2))
    neg_demos = [p for p in neg_problems if p.get("is_demo", False)]
    neg_text = "\n".join(p["prompt"] for p in neg_demos)
    neg_ids = tokenizer(neg_text, return_tensors="pt").input_ids.to(device)

    # Encode negative demos (encoder only, no channel writes)
    with torch.no_grad():
        neg_embeds = base_model.model.embed_tokens(neg_ids.squeeze(0))
    neg_task_state = mach.demo_encoder(neg_embeds.float())

    # Positive: current task_state (already computed)
    pos_task_state = mach.get_task_state()

    # Cosine similarity: push apart
    cos_sim = torch.nn.functional.cosine_similarity(
        pos_task_state.unsqueeze(0), neg_task_state.unsqueeze(0)
    )
    # Loss: want cos_sim to be low (different tasks → different states)
    # margin=0: just push apart, don't require a specific margin
    return torch.clamp(cos_sim + 0.5, min=0.0)  # hinge: sim < -0.5


def run_episode_two_channel(base_model, mach, patched_model, tokenizer,
                            problems, device):
    """
    Two-channel episode: concatenate demos → one Qwen pass → both channels → evaluate.
    """
    mach.reset_episode()

    demos = [p for p in problems if p.get("is_demo", False)]
    tests = [p for p in problems if not p.get("is_demo", False)]

    # Phase A: Demo processing
    if demos:
        demo_text = "\n".join(p["prompt"] for p in demos)
        demo_ids = tokenizer(demo_text, return_tensors="pt").input_ids.to(device)
        mach.process_demos(base_model, demo_ids)

    # Phase B: Test evaluation (both channels fixed from demo writes)
    rewards = []
    problem_losses = []
    qwen_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for problem in tests:
        full_text = problem["prompt"] + problem["answer"]
        encoding = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_len = len(tokenizer(problem["prompt"]).input_ids)
        labels = encoding.input_ids.clone()
        labels[0, :prompt_len] = -100

        output = patched_model(input_ids=encoding.input_ids, labels=labels)

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
        qwen_loss = qwen_loss + output.loss

    return qwen_loss, rewards, problem_losses


def meta_train_two_channel(base_model, mach, patched_model, tokenizer,
                           device, n_episodes=None, lr=None,
                           curriculum=None, checkpoint_path=None,
                           save_path=None, energy_beta=None):
    """Two-channel meta-training loop with channel ablation diagnostics."""
    if n_episodes is None:
        n_episodes = config.PHASE5_EPISODES
    if lr is None:
        lr = config.PHASE5_LR
    if curriculum is None:
        curriculum = CONTINUOUS_LINEAR_CURRICULUM
    if energy_beta is None:
        energy_beta = config.TWO_CHANNEL_ENERGY_BETA

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        mach.load_state_dict(state_dict, strict=False)

    meta_params = list(mach.parameters())
    optimizer = torch.optim.Adam(meta_params, lr=lr)

    use_energy = energy_beta > 0
    n_meta = sum(p.numel() for p in meta_params)
    print(f"Two-Channel trainable parameters: {n_meta:,}")
    if use_energy:
        print(f"Using unified energy loss (beta={energy_beta})")
    print(f"Write cost scale: {mach.write_cost_scale}x")

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
            ce_loss, rewards, problem_losses = run_episode_two_channel(
                base_model, mach, patched_model, tokenizer,
                problems, device,
            )

            if use_energy:
                metabolic = mach.metabolic_cost()
                total_loss = ce_loss + energy_beta * metabolic
                energy_scalar = metabolic.item()
            else:
                total_loss = ce_loss
                energy_scalar = 0.0

            # Contrastive loss: direct gradient to demo_encoder
            contrastive = _contrastive_loss(
                mach, base_model, tokenizer, problems, device
            )
            contrastive_beta = 1.0
            total_loss = total_loss + contrastive_beta * contrastive

            total_loss.backward()
            loss_scalar = ce_loss.item()

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at episode {episode_idx}. Reducing problems.")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            problems = problems[:max(5, len(problems) // 2)]
            ce_loss, rewards, problem_losses = run_episode_two_channel(
                base_model, mach, patched_model, tokenizer,
                problems, device,
            )
            total_loss = ce_loss
            if use_energy:
                total_loss = total_loss + energy_beta * mach.metabolic_cost()
            contrastive = _contrastive_loss(
                mach, base_model, tokenizer, problems, device
            )
            total_loss = total_loss + contrastive
            total_loss.backward()
            loss_scalar = ce_loss.item()
            energy_scalar = 0.0

        torch.nn.utils.clip_grad_norm_(
            meta_params, max_norm=config.PHASE5_GRAD_CLIP
        )
        optimizer.step()

        if episode_idx % 10 == 0:
            test_rewards = list(rewards)
            avg_reward = sum(test_rewards) / max(len(test_rewards), 1)
            n_early = min(5, len(test_rewards))
            n_late = min(5, len(test_rewards))
            early_acc = sum(
                1 for r in test_rewards[:n_early] if r > 0
            ) / max(n_early, 1)
            late_acc = sum(
                1 for r in test_rewards[-n_late:] if r > 0
            ) / max(n_late, 1)

            task_state = mach.get_task_state()
            if task_state is not None:
                ts = task_state.detach()
                n_active = (ts.abs() > 0.1).sum().item()
                ts_max = ts.abs().max().item()
            else:
                n_active = 0
                ts_max = 0.0

            # Channel diagnostics
            prim_w_str = ""
            if mach.primitives._weights is not None:
                pw = mach.primitives._weights.detach()
                prim_w_str = f"prim_w={pw.abs().mean():.3f}"

            cost_str = f"energy={energy_scalar:.3f}" if use_energy else ""

            print(
                f"Episode {episode_idx:4d} | {mode} n={len(problems):2d} | "
                f"ce={loss_scalar:.4f} {cost_str} | "
                f"avg_r={avg_reward:.2f} "
                f"early={early_acc:.0%} late={late_acc:.0%} | "
                f"active={n_active:.0f}/{mach.d_task} max={ts_max:.2f} "
                f"{prim_w_str}"
            )

            if wandb is not None:
                log_dict = {
                    "episode": episode_idx,
                    "ce_loss": loss_scalar,
                    "energy_cost": energy_scalar,
                    "avg_reward": avg_reward,
                    "early_accuracy": early_acc,
                    "late_accuracy": late_acc,
                    "task_state_active": n_active,
                    "task_state_max": ts_max,
                }
                if mach.primitives._weights is not None:
                    log_dict["prim_weight_mean"] = pw.abs().mean().item()
                wandb.log(log_dict)

        # Diagnostics + validation + checkpoint
        if (episode_idx % 200 == 0 and episode_idx > 0) or \
                episode_idx == n_episodes - 1:
            _log_diagnostics(mach, meta_params, episode_idx)

            # Channel ablation validation
            if mode in ("continuous_linear",):
                _run_channel_ablation(
                    base_model, mach, patched_model, tokenizer,
                    device, episode_idx,
                )
                print(f"  --- TRAIN combos ---")
                for coeffs in LINEAR_TRAIN:
                    _run_linear_validation_two_channel(
                        base_model, mach, patched_model, tokenizer,
                        device, coeffs, episode_idx,
                    )
                print(f"  --- HELD-OUT combos ---")
                for coeffs in LINEAR_HELDOUT:
                    _run_linear_validation_two_channel(
                        base_model, mach, patched_model, tokenizer,
                        device, coeffs, episode_idx,
                    )
                print(f"  --- NOVEL combos (never in fixed pool) ---")
                for coeffs in [(3, 1), (1, 4), (4, 3), (5, 0), (0, 5)]:
                    _run_linear_validation_two_channel(
                        base_model, mach, patched_model, tokenizer,
                        device, coeffs, episode_idx,
                    )
            elif mode == "token_map":
                _run_token_map_validation_two_channel(
                    base_model, mach, patched_model, tokenizer,
                    device, episode_idx,
                )
            elif mode == "mixed":
                _run_token_map_validation_two_channel(
                    base_model, mach, patched_model, tokenizer,
                    device, episode_idx,
                )
                print(f"  --- LINEAR sample ---")
                for coeffs in [(1, 0), (0, 1), (1, 1), (2, 1)]:
                    _run_linear_validation_two_channel(
                        base_model, mach, patched_model, tokenizer,
                        device, coeffs, episode_idx,
                    )

            if save_path is not None:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(mach.state_dict(), save_path)
                print(f"  Checkpoint saved to {save_path}")


def _run_channel_ablation(base_model, mach, patched_model, tokenizer,
                          device, episode_idx, n_episodes=5, n_problems=20,
                          n_demos=5):
    """
    Run validation three ways to reveal channel contributions:
    1. Both channels (normal)
    2. Modulation only (zero patch deltas)
    3. Write only (zero primitive weights)
    """
    sample_coeffs = [(1, 0), (0, 1), (2, 0), (1, 1)]

    for mode_label, zero_patches, zero_prims in [
        ("BOTH", False, False),
        ("MOD_ONLY", True, False),
        ("WRITE_ONLY", False, True),
    ]:
        total_correct = 0
        total_count = 0

        for coeffs in sample_coeffs:
            for ep in range(n_episodes):
                problems = generate_linear_episode(
                    n_problems, n_demos=n_demos, coeffs=coeffs
                )
                mach.reset_episode()

                demos = [p for p in problems if p["is_demo"]]
                tests = [p for p in problems if not p["is_demo"]]

                if demos:
                    demo_text = "\n".join(p["prompt"] for p in demos)
                    demo_ids = tokenizer(
                        demo_text, return_tensors="pt"
                    ).input_ids.to(device)
                    with torch.no_grad():
                        mach.process_demos(base_model, demo_ids)

                # Ablation: temporarily modify state
                saved_weights = None
                saved_deltas = []
                if zero_patches:
                    # Save and zero patch deltas
                    for patch in mach.patches:
                        saved_deltas.append((
                            patch.delta_down, patch.delta_up, patch.delta_gain
                        ))
                        patch.delta_down = torch.zeros_like(patch.delta_down)
                        patch.delta_up = torch.zeros_like(patch.delta_up)
                        patch.delta_gain = torch.zeros_like(patch.delta_gain)
                if zero_prims:
                    # Save and zero primitive weights
                    saved_weights = mach.primitives._weights
                    if saved_weights is not None:
                        mach.primitives._weights = torch.zeros_like(saved_weights)

                for problem in tests:
                    full_text = problem["prompt"] + problem["answer"]
                    encoding = tokenizer(full_text, return_tensors="pt").to(device)
                    prompt_len = len(tokenizer(problem["prompt"]).input_ids)

                    with torch.no_grad():
                        output = patched_model(input_ids=encoding.input_ids)
                        logits = (
                            output.logits if hasattr(output, 'logits')
                            else output[0]
                        )
                        pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
                        pred_text = tokenizer.decode(
                            pred_tokens, skip_special_tokens=True
                        ).strip()
                        predicted = extract_number(pred_text)
                        correct = (predicted == problem["answer"])

                    total_correct += int(correct)
                    total_count += 1

                # Restore
                if zero_patches:
                    for j, patch in enumerate(mach.patches):
                        patch.delta_down = saved_deltas[j][0]
                        patch.delta_up = saved_deltas[j][1]
                        patch.delta_gain = saved_deltas[j][2]
                if zero_prims and saved_weights is not None:
                    mach.primitives._weights = saved_weights

        acc = total_correct / total_count if total_count > 0 else 0
        print(f"  ABLATION ep{episode_idx} {mode_label:12s} | acc={acc:.0%}")

        if wandb is not None:
            wandb.log({f"ablation/{mode_label.lower()}": acc})


def _run_linear_validation_two_channel(base_model, mach, patched_model,
                                       tokenizer, device, coeffs, episode_idx,
                                       n_episodes=10, n_problems=20, n_demos=5):
    """Evaluate linear combination performance with two-channel architecture."""
    test_correct = 0
    test_total = 0

    for ep in range(n_episodes):
        problems = generate_linear_episode(
            n_problems, n_demos=n_demos, coeffs=coeffs
        )
        mach.reset_episode()

        demos = [p for p in problems if p["is_demo"]]
        tests = [p for p in problems if not p["is_demo"]]

        if demos:
            demo_text = "\n".join(p["prompt"] for p in demos)
            demo_ids = tokenizer(
                demo_text, return_tensors="pt"
            ).input_ids.to(device)
            with torch.no_grad():
                mach.process_demos(base_model, demo_ids)

        for problem in tests:
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


def _run_token_map_validation_two_channel(base_model, mach, patched_model,
                                          tokenizer, device, episode_idx,
                                          n_episodes=10, n_problems=20,
                                          n_demos=5):
    """Evaluate token mapping performance with two-channel architecture."""
    test_correct = 0
    test_total = 0
    symbol_correct = 0
    symbol_total = 0

    for ep in range(n_episodes):
        problems = generate_token_mapping_episode(
            n_problems, n_demos=n_demos
        )
        mach.reset_episode()

        demos = [p for p in problems if p["is_demo"]]
        tests = [p for p in problems if not p["is_demo"]]

        if demos:
            demo_text = "\n".join(p["prompt"] for p in demos)
            demo_ids = tokenizer(
                demo_text, return_tensors="pt"
            ).input_ids.to(device)
            with torch.no_grad():
                mach.process_demos(base_model, demo_ids)

        for problem in tests:
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


def _log_diagnostics(mach, meta_params, episode_idx):
    """Log gradient norms and weight stats with channel breakdown."""
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

    # Channel 1: primitive stats
    if mach.primitives._weights is not None:
        pw = mach.primitives._weights.detach()
        diag["ch1/weight_mean"] = pw.abs().mean().item()
        diag["ch1/weight_max"] = pw.abs().max().item()
        diag["ch1/bias_norm"] = mach.primitives.bias.norm().item()
        diag["ch1/gain_norm"] = mach.primitives.gain.norm().item()

    # Channel 2: basis + patch stats
    for i in range(mach.basis.n_patches):
        diag[f"ch2/basis_norm/patch{i}_down_U"] = (
            mach.basis.down_U[i].norm().item()
        )
        diag[f"ch2/basis_norm/patch{i}_down_V"] = (
            mach.basis.down_V[i].norm().item()
        )

    for i, patch in enumerate(mach.patches):
        if patch.delta_down is not None:
            diag[f"ch2/patch_delta/patch{i}_down"] = (
                patch.delta_down.norm().item()
            )
            diag[f"ch2/patch_delta/patch{i}_up"] = (
                patch.delta_up.norm().item()
            )
        if patch.delta_gain is not None:
            diag[f"ch2/patch_delta/patch{i}_gain"] = (
                patch.delta_gain.norm().item()
            )

    print(f"  Diagnostics at episode {episode_idx}:")
    for k, v in sorted(diag.items()):
        print(f"    {k}: {v:.6f}")

    if wandb is not None:
        wandb.log({f"diag/{k}": v for k, v in diag.items()})


# ---- DemoRead Training ----


def _encode_demos(base_model, mach, tokenizer, demos, device):
    """Tokenize and encode demos through DemoEncoder → write patches."""
    demo_text = "\n".join(p["prompt"] for p in demos)
    demo_ids = tokenizer(demo_text, return_tensors="pt").input_ids.to(device)
    mach.process_demos(base_model, demo_ids, device)


def run_episode_demoread(base_model, mach, patched_model, tokenizer,
                         problems, device):
    """
    DemoRead episode: encode demos → write patches → evaluate tests.
    Gradient flows: test_loss → patches → basis → action_head → task_state → DemoEncoder.
    """
    mach.reset_episode()

    demos = [p for p in problems if p.get("is_demo", False)]
    tests = [p for p in problems if not p.get("is_demo", False)]

    # Oracle mode: feed true coefficients directly
    if mach.oracle:
        c1 = problems[0].get("c1", 1)
        c2 = problems[0].get("c2", 0)
        mach.process_oracle(c1, c2, device)
    elif demos:
        _encode_demos(base_model, mach, tokenizer, demos, device)

    # Evaluate on test problems (gradient flows through patches → encoder)
    rewards = []
    problem_losses = []
    qwen_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for problem in tests:
        full_text = problem["prompt"] + problem["answer"]
        encoding = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_len = len(tokenizer(problem["prompt"]).input_ids)
        labels = encoding.input_ids.clone()
        labels[0, :prompt_len] = -100

        output = patched_model(input_ids=encoding.input_ids, labels=labels)

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
        qwen_loss = qwen_loss + output.loss

    return qwen_loss, rewards, problem_losses


def meta_train_demoread(base_model, mach, patched_model, tokenizer,
                        device, n_episodes=None, lr=None,
                        curriculum=None, checkpoint_path=None,
                        save_path=None):
    """DemoRead meta-training loop."""
    if n_episodes is None:
        n_episodes = config.PHASE5_EPISODES
    if lr is None:
        lr = config.PHASE5_LR
    if curriculum is None:
        curriculum = CONTINUOUS_LINEAR_CURRICULUM

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        mach.load_state_dict(state_dict, strict=False)

    meta_params = list(mach.parameters())
    optimizer = torch.optim.Adam(meta_params, lr=lr)

    n_meta = sum(p.numel() for p in meta_params)
    print(f"MACHDemoRead trainable parameters: {n_meta:,}")

    def get_n_problems(episode_idx):
        if episode_idx < 100:
            return 10
        elif episode_idx < 300:
            return 15
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
            ce_loss, rewards, problem_losses = run_episode_demoread(
                base_model, mach, patched_model, tokenizer,
                problems, device,
            )

            total_loss = ce_loss
            coeff_loss_scalar = 0.0

            # Supervised auxiliary: predict c1, c2 from task_state
            c1 = problems[0].get("c1")
            c2 = problems[0].get("c2")
            pred_coeffs = mach.predict_coeffs()
            if c1 is not None and pred_coeffs is not None:
                target = torch.tensor(
                    [float(c1), float(c2)], device=device
                )
                coeff_loss = torch.nn.functional.mse_loss(
                    pred_coeffs, target
                )
                coeff_loss_scalar = coeff_loss.item()
                total_loss = total_loss + 0.1 * coeff_loss

            total_loss.backward()
            loss_scalar = ce_loss.item()

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at episode {episode_idx}. Reducing problems.")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            problems = problems[:max(5, len(problems) // 2)]
            ce_loss, rewards, problem_losses = run_episode_demoread(
                base_model, mach, patched_model, tokenizer,
                problems, device,
            )
            ce_loss.backward()
            loss_scalar = ce_loss.item()
            coeff_loss_scalar = 0.0

        torch.nn.utils.clip_grad_norm_(
            meta_params, max_norm=config.PHASE5_GRAD_CLIP
        )
        optimizer.step()

        if episode_idx % 10 == 0:
            test_rewards = list(rewards)
            avg_reward = sum(test_rewards) / max(len(test_rewards), 1)
            test_acc = sum(1 for r in test_rewards if r > 0) / max(
                len(test_rewards), 1
            )

            task_state = mach.get_task_state()
            ts_info = ""
            if task_state is not None:
                ts = task_state.detach()
                ts_info = f"ts_max={ts.abs().max():.2f}"

            coeff_info = ""
            if coeff_loss_scalar > 0:
                pred = mach.predict_coeffs()
                if pred is not None:
                    p = pred.detach()
                    c1 = problems[0].get("c1", "?")
                    c2 = problems[0].get("c2", "?")
                    coeff_info = (
                        f"coeff={coeff_loss_scalar:.3f} "
                        f"pred=[{p[0]:.1f},{p[1]:.1f}] "
                        f"true=[{c1},{c2}]"
                    )

            print(
                f"Episode {episode_idx:4d} | {mode} n={len(problems):2d} | "
                f"ce={loss_scalar:.4f} | "
                f"test={test_acc:.0%} avg_r={avg_reward:.2f} | "
                f"{ts_info} {coeff_info}"
            )

            if wandb is not None:
                wandb.log({
                    "episode": episode_idx,
                    "ce_loss": loss_scalar,
                    "avg_reward": avg_reward,
                    "test_accuracy": test_acc,
                })

        # Diagnostics + validation + checkpoint
        if (episode_idx % 200 == 0 and episode_idx > 0) or \
                episode_idx == n_episodes - 1:
            _log_demoread_diagnostics(mach, meta_params, episode_idx)

            if mode in ("continuous_linear",):
                print(f"  --- TRAIN combos ---")
                for coeffs in LINEAR_TRAIN:
                    _run_linear_validation_demoread(
                        base_model, mach, patched_model, tokenizer,
                        device, coeffs, episode_idx,
                    )
                print(f"  --- HELD-OUT combos ---")
                for coeffs in LINEAR_HELDOUT:
                    _run_linear_validation_demoread(
                        base_model, mach, patched_model, tokenizer,
                        device, coeffs, episode_idx,
                    )
                print(f"  --- NOVEL combos ---")
                for coeffs in [(3, 1), (1, 4), (4, 3), (5, 0), (0, 5)]:
                    _run_linear_validation_demoread(
                        base_model, mach, patched_model, tokenizer,
                        device, coeffs, episode_idx,
                    )
            elif mode == "token_map":
                _run_token_map_validation_demoread(
                    base_model, mach, patched_model, tokenizer,
                    device, episode_idx,
                )

            if save_path is not None:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(mach.state_dict(), save_path)
                print(f"  Checkpoint saved to {save_path}")


def _log_demoread_diagnostics(mach, meta_params, episode_idx):
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

    # Task state
    task_state = mach.get_task_state()
    if task_state is not None:
        ts = task_state.detach()
        diag["task_state/l1_norm"] = ts.abs().mean().item()
        diag["task_state/max_abs"] = ts.abs().max().item()

    # Patch delta stats
    for i, patch in enumerate(mach.patches):
        if patch.delta_down is not None:
            diag[f"patch_delta/patch{i}_down"] = patch.delta_down.norm().item()
            diag[f"patch_delta/patch{i}_up"] = patch.delta_up.norm().item()
        if patch.delta_gain is not None:
            diag[f"patch_delta/patch{i}_gain"] = patch.delta_gain.norm().item()

    # Basis vector norms (only for MACHHebbian with basis vectors)
    if hasattr(mach, 'basis'):
        for i in range(mach.basis.n_patches):
            diag[f"basis_norm/patch{i}_down_U"] = mach.basis.down_U[i].norm().item()
            diag[f"basis_norm/patch{i}_down_V"] = mach.basis.down_V[i].norm().item()

    # Attention patch stats (only for MACHDualHebbian)
    if hasattr(mach, 'attn_patches'):
        for i, patch in enumerate(mach.attn_patches):
            if patch.delta_down is not None:
                diag[f"attn_patch_delta/patch{i}_down"] = patch.delta_down.norm().item()
                diag[f"attn_patch_delta/patch{i}_up"] = patch.delta_up.norm().item()

    # Coprocessor stats (only for MACHCoprocessor)
    if hasattr(mach, 'copro_patches'):
        for i, p in enumerate(mach.copro_patches):
            if p.delta_down is not None:
                diag[f"copro/patch{i}_down"] = p.delta_down.norm().item()
                diag[f"copro/patch{i}_up"] = p.delta_up.norm().item()

    # Top-down gain stats (only for MACHDenseHebbian)
    if hasattr(mach, 'gain_net') and mach._gains is not None:
        gains = mach._gains.detach()
        diag["gain/mean"] = gains.mean().item()
        diag["gain/std"] = gains.std().item()
        diag["gain/min"] = gains.min().item()
        diag["gain/max"] = gains.max().item()

    # Consolidation stats
    if hasattr(mach, 'consolidation') and mach.consolidation:
        slow_norms = []
        for p in mach.patches:
            slow_norms.append(p.slow_down.norm().item())
        diag["consolidation/slow_mean_norm"] = sum(slow_norms) / len(slow_norms)
        diag["consolidation/slow_max_norm"] = max(slow_norms)

    print(f"  Diagnostics at episode {episode_idx}:")
    for k, v in sorted(diag.items()):
        print(f"    {k}: {v:.6f}")

    if wandb is not None:
        wandb.log({f"diag/{k}": v for k, v in diag.items()})


def _run_linear_validation_demoread(base_model, mach, patched_model,
                                    tokenizer, device, coeffs, episode_idx,
                                    n_episodes=10, n_problems=20, n_demos=5):
    """Evaluate linear combination performance with DemoRead architecture."""
    test_correct = 0
    test_total = 0

    for ep in range(n_episodes):
        problems = generate_linear_episode(
            n_problems, n_demos=n_demos, coeffs=coeffs
        )
        mach.reset_episode()

        demos = [p for p in problems if p["is_demo"]]
        tests = [p for p in problems if not p["is_demo"]]

        if mach.oracle:
            with torch.no_grad():
                mach.process_oracle(coeffs[0], coeffs[1], device)
        elif demos:
            with torch.no_grad():
                _encode_demos(base_model, mach, tokenizer, demos, device)

        for problem in tests:
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


def _run_token_map_validation_demoread(base_model, mach, patched_model,
                                       tokenizer, device, episode_idx,
                                       n_episodes=10, n_problems=20,
                                       n_demos=5):
    """Evaluate token mapping performance with DemoRead architecture."""
    test_correct = 0
    test_total = 0

    for ep in range(n_episodes):
        problems = generate_token_mapping_episode(
            n_problems, n_demos=n_demos
        )
        mach.reset_episode()

        demos = [p for p in problems if p["is_demo"]]
        tests = [p for p in problems if not p["is_demo"]]

        if demos:
            with torch.no_grad():
                _encode_demos(base_model, mach, tokenizer, demos, device)

        for problem in tests:
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

            test_correct += int(correct)
            test_total += 1

    test_acc = test_correct / test_total if test_total > 0 else 0

    print(
        f"  EVAL ep{episode_idx} token_map | exact={test_acc:.0%}"
    )

    if wandb is not None:
        wandb.log({"eval/token_map_exact": test_acc})


# ---- Hebbian Training ----


def run_episode_hebbian(base_model, mach, patched_model, tokenizer,
                        problems, device):
    """
    Hebbian episode: for each problem, forward pass → reward → TD error → local update.
    Every problem is both a learning opportunity and a test.
    """
    mach.reset_episode()

    rewards = []
    problem_losses = []
    total_ce = torch.tensor(0.0, device=device, requires_grad=True)
    critic_losses = []
    nuclei_losses = []

    for step, problem in enumerate(problems):
        # 1. Forward pass (hooks capture pre/post activations)
        full_text = problem["prompt"] + problem["answer"]
        encoding = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_len = len(tokenizer(problem["prompt"]).input_ids)
        labels = encoding.input_ids.clone()
        labels[0, :prompt_len] = -100

        output = patched_model(input_ids=encoding.input_ids, labels=labels)
        total_ce = total_ce + output.loss
        problem_losses.append(output.loss.item())

        # 2. Compute reward
        with torch.no_grad():
            logits = output.logits
            pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
            pred_text = tokenizer.decode(
                pred_tokens, skip_special_tokens=True
            ).strip()
            if problem.get("difficulty") == "token_map":
                correct = (pred_text == problem["answer"])
                reward = 1.0 if correct else -1.0
            else:
                predicted = extract_number(pred_text)
                correct = (predicted == problem["answer"])
                reward = graded_reward(predicted, problem["answer"])
        rewards.append(reward)

        # 3. Hebbian step: critic → TD error → local weight updates
        value, td_error = mach.hebbian_step(
            reward, step, len(problems), device
        )

        # 4. Critic loss: TD bootstrapped target
        if hasattr(mach, '_prev_critic_value') and mach._prev_critic_value is not None:
            prev_gamma = mach._prev_gamma if hasattr(mach, '_prev_gamma') and mach._prev_gamma is not None else mach.gamma
            td_target = mach._prev_reward + prev_gamma * value.detach()
            critic_loss = (mach._prev_critic_value - td_target) ** 2
            critic_losses.append(critic_loss)
        mach._prev_critic_value = value
        mach._prev_gamma = mach._current_gamma.detach() if hasattr(mach, '_current_gamma') else getattr(mach, 'gamma', 0.95)
        mach._prev_reward = torch.tensor(reward, device=device, dtype=torch.float32)
        if hasattr(mach, '_nuclei_loss'):
            nuclei_losses.append(mach._nuclei_loss)

    avg_critic_loss = torch.stack(critic_losses).mean() if critic_losses else torch.tensor(0.0, device=device)
    avg_nuclei_loss = torch.stack(nuclei_losses).mean() if nuclei_losses else torch.tensor(0.0, device=device)

    return total_ce, rewards, problem_losses, avg_critic_loss + 0.1 * avg_nuclei_loss


def run_episode_hebbian_cot(base_model, mach, patched_model, tokenizer,
                            problems, device, max_thinking_tokens=32):
    """
    Chain-of-thought Hebbian episode: model generates thinking tokens
    before answering. Patches steer every generation step.

    For each problem:
    1. Encode prompt
    2. Generate up to max_thinking_tokens freely (steered by patches)
    3. Then force the correct answer tokens for CE loss (training signal)
    4. Extract generated answer, compute reward
    5. Hebbian update from reward

    The thinking tokens give the model extra computation time.
    The CE loss on the correct answer trains the Hebbian rule.
    """
    mach.reset_episode()

    rewards = []
    problem_losses = []
    total_ce = torch.tensor(0.0, device=device, requires_grad=True)
    critic_losses = []
    nuclei_losses = []

    for step, problem in enumerate(problems):
        prompt_text = problem["prompt"]
        answer_text = problem["answer"]
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        # Phase 1: Generate thinking tokens (steered, no forced output)
        with torch.no_grad():
            generated = prompt_ids
            for _ in range(max_thinking_tokens):
                output = patched_model(input_ids=generated)
                next_token = output.logits[0, -1:].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if model generates "=" or a newline (it's ready to answer)
                next_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
                if "=" in next_text or "\n" in next_text:
                    break

        # The generated thinking tokens (everything after prompt)
        thinking_ids = generated[0, prompt_len:]

        # Phase 2: Forward pass with thinking + correct answer for CE loss
        # Context: prompt + thinking + answer
        answer_ids = tokenizer(answer_text, return_tensors="pt",
                               add_special_tokens=False).input_ids.to(device)
        full_ids = torch.cat([generated, answer_ids], dim=1)

        labels = full_ids.clone()
        # Only compute loss on answer tokens (not prompt, not thinking)
        labels[0, :generated.shape[1]] = -100

        output = patched_model(input_ids=full_ids, labels=labels)
        total_ce = total_ce + output.loss
        problem_losses.append(output.loss.item())

        # Phase 3: Check if model's generated answer was correct
        with torch.no_grad():
            # Look at what the model predicted right after thinking
            gen_logits = output.logits[0, generated.shape[1] - 1:-1]
            pred_tokens = gen_logits.argmax(dim=-1)
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

            if problem.get("difficulty") == "token_map":
                correct = (pred_text == answer_text)
            else:
                predicted = extract_number(pred_text)
                correct = (predicted == answer_text)
            reward = 1.0 if correct else -1.0
        rewards.append(reward)

        # Phase 4: Hebbian step
        value, td_error = mach.hebbian_step(
            reward, step, len(problems), device
        )

        # Phase 5: Critic loss: TD bootstrapped target
        if hasattr(mach, '_prev_critic_value') and mach._prev_critic_value is not None:
            prev_gamma = mach._prev_gamma if hasattr(mach, '_prev_gamma') and mach._prev_gamma is not None else mach.gamma
            td_target = mach._prev_reward + prev_gamma * value.detach()
            critic_loss = (mach._prev_critic_value - td_target) ** 2
            critic_losses.append(critic_loss)
        mach._prev_critic_value = value
        mach._prev_gamma = mach._current_gamma.detach() if hasattr(mach, '_current_gamma') else getattr(mach, 'gamma', 0.95)
        mach._prev_reward = torch.tensor(reward, device=device, dtype=torch.float32)
        if hasattr(mach, '_nuclei_loss'):
            nuclei_losses.append(mach._nuclei_loss)

    avg_critic_loss = torch.stack(critic_losses).mean() if critic_losses else torch.tensor(0.0, device=device)
    avg_nuclei_loss = torch.stack(nuclei_losses).mean() if nuclei_losses else torch.tensor(0.0, device=device)

    return total_ce, rewards, problem_losses, avg_critic_loss + 0.1 * avg_nuclei_loss


def meta_train_hebbian(base_model, mach, patched_model, tokenizer,
                       device, n_episodes=None, lr=None,
                       curriculum=None, checkpoint_path=None,
                       save_path=None, chain_of_thought=False,
                       max_thinking_tokens=32):
    """Hebbian meta-training loop."""
    if n_episodes is None:
        n_episodes = config.PHASE5_EPISODES
    if lr is None:
        lr = config.PHASE5_LR
    if curriculum is None:
        curriculum = CONTINUOUS_LINEAR_CURRICULUM

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        mach.load_state_dict(state_dict, strict=False)

    meta_params = list(mach.parameters())
    optimizer = torch.optim.Adam(meta_params, lr=lr)

    n_meta = sum(p.numel() for p in meta_params)
    print(f"MACHHebbian trainable parameters: {n_meta:,}")

    def get_n_problems(episode_idx):
        import random
        if episode_idx < 100:
            return 10
        elif episode_idx < 300:
            return random.randint(10, 30)
        else:
            # Random length: model can't rely on reset timing
            # Occasional long episodes (10%) force self-regulation
            if random.random() < 0.1:
                return random.randint(100, 200)
            return random.randint(10, 60)

    for episode_idx in range(n_episodes):
        mode = get_episode_mode(episode_idx, curriculum)
        n_problems = get_n_problems(episode_idx)
        problems = generate_episode_problems(n_problems, mode)

        optimizer.zero_grad()

        episode_fn = run_episode_hebbian_cot if chain_of_thought else run_episode_hebbian
        episode_kwargs = {}
        if chain_of_thought:
            episode_kwargs["max_thinking_tokens"] = max_thinking_tokens

        try:
            ce_loss, rewards, problem_losses, critic_loss = episode_fn(
                base_model, mach, patched_model, tokenizer,
                problems, device, **episode_kwargs,
            )

            total_loss = ce_loss + 0.5 * critic_loss
            total_loss.backward()
            loss_scalar = ce_loss.item()
            critic_scalar = critic_loss.item()

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at episode {episode_idx}. Reducing problems.")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            problems = problems[:max(5, len(problems) // 2)]
            ce_loss, rewards, problem_losses, critic_loss = episode_fn(
                base_model, mach, patched_model, tokenizer,
                problems, device, **episode_kwargs,
            )
            total_loss = ce_loss + 0.5 * critic_loss
            total_loss.backward()
            loss_scalar = ce_loss.item()
            critic_scalar = critic_loss.item()

        torch.nn.utils.clip_grad_norm_(
            meta_params, max_norm=config.PHASE5_GRAD_CLIP
        )
        optimizer.step()

        # Consolidate successful episodes into slow memory
        if hasattr(mach, 'consolidate'):
            avg_r = sum(rewards) / max(len(rewards), 1)
            mach.consolidate(avg_reward=avg_r)

        if episode_idx % 10 == 0:
            n_total = len(rewards)
            n_correct = sum(1 for r in rewards if r > 0)
            test_acc = n_correct / max(n_total, 1)
            avg_reward = sum(rewards) / max(n_total, 1)

            # Within-episode improvement: compare first half vs second half
            mid = max(1, n_total // 2)
            first_half = sum(1 for r in rewards[:mid] if r > 0) / max(mid, 1)
            second_half = sum(1 for r in rewards[mid:] if r > 0) / max(n_total - mid, 1)

            neuromod_str = ""
            if hasattr(mach, '_last_etas') and mach._last_etas is not None:
                neuromod_str = (f" | η={mach._last_etas[0].item():.3f}"
                               f" decay={mach._last_decays[0].item():.3f}"
                               f" expl={mach._last_exploration:.3f}")
            print(
                f"Episode {episode_idx:4d} | {mode} n={n_total:2d} | "
                f"ce={loss_scalar:.4f} critic={critic_scalar:.4f} | "
                f"acc={test_acc:.0%} 1st={first_half:.0%} 2nd={second_half:.0%} | "
                f"avg_r={avg_reward:.2f}{neuromod_str}"
            )

            if wandb is not None:
                wandb.log({
                    "episode": episode_idx,
                    "ce_loss": loss_scalar,
                    "critic_loss": critic_scalar,
                    "accuracy": test_acc,
                    "first_half_acc": first_half,
                    "second_half_acc": second_half,
                    "avg_reward": avg_reward,
                })

        # Diagnostics + validation
        if (episode_idx % 200 == 0 and episode_idx > 0) or \
                episode_idx == n_episodes - 1:
            _log_hebbian_diagnostics(mach, meta_params, episode_idx)

            if mode in ("continuous_linear",):
                print(f"  --- TRAIN combos ---")
                for coeffs in LINEAR_TRAIN:
                    _run_linear_validation_hebbian(
                        base_model, mach, patched_model,
                        tokenizer, device, coeffs, episode_idx,
                    )

                print(f"  --- HELD-OUT combos ---")
                for coeffs in LINEAR_HELDOUT:
                    _run_linear_validation_hebbian(
                        base_model, mach, patched_model,
                        tokenizer, device, coeffs, episode_idx,
                    )

                print(f"  --- NOVEL combos ---")
                novel = [(3, 1), (1, 4), (4, 3), (5, 0), (0, 5)]
                for coeffs in novel:
                    _run_linear_validation_hebbian(
                        base_model, mach, patched_model,
                        tokenizer, device, coeffs, episode_idx,
                    )

            if mode in ("few_shot_basic", "few_shot", "few_shot_diverse"):
                if mode == "few_shot_diverse":
                    eval_ops = DIVERSE_TRAIN_OPS[:6]  # sample of trained ops
                else:
                    eval_ops = ["add", "sub", "mul", "div"]
                print(f"  --- Operation classification ---")
                for op in eval_ops:
                    _run_op_validation_hebbian(
                        base_model, mach, patched_model,
                        tokenizer, device, op, episode_idx,
                    )

                # Run ablation every 400 episodes
                if episode_idx % 400 == 0 and episode_idx > 0:
                    mach.eval()
                    ablate_hebbian(
                        base_model, mach, patched_model,
                        tokenizer, device,
                    )
                    mach.train()

            if save_path:
                torch.save(mach.state_dict(), save_path)
                print(f"  Checkpoint saved to {save_path}")


def _repair_checkpoint(mach):
    """Repair corrupted PFC/context_gate weights if needed.
    PFC explosion (discovered at 98k): context_gate weights grew to ~50,
    saturating sigmoid, killing gradient, causing PFC state to grow to 10^17.
    """
    # Note: context_gate weights were near-init values even at 98k.
    # The problem was PFC state (10^17), not the gate weights themselves.
    # PFC normalization (unit norm) is the real fix. No gate repair needed.
    if hasattr(mach, '_pfc_state') and mach._pfc_state.norm() > 100:
        print(f"  ⚠ Repairing corrupted PFC state (norm={mach._pfc_state.norm():.1f})")
        mach._pfc_state = torch.zeros_like(mach._pfc_state)


def meta_train_continuous(base_model, mach, patched_model, tokenizer,
                          device, n_steps=40000, lr=None,
                          truncation_window=20, checkpoint_path=None,
                          save_path=None, curriculum=None,
                          context_size=0, thinking_tokens=0,
                          memory_path=None, dense_only=False,
                          ce_anneal_start=0.7):
    """
    Continuous Hebbian training: no episodes, no resets.

    Problems arrive in an endless stream. Truncated backprop every
    `truncation_window` steps (computational necessity, not a semantic
    boundary). Patches and slow memory persist across everything.

    If context_size > 0, the last N solved problems are prepended as context
    (fast explicit memory, like hippocampus). The LLM uses in-context
    learning for fast adaptation while patches handle slow adaptation.

    If thinking_tokens > 0, the model generates N tokens before answering
    (inner speech). Patches steer thinking; gradient flows through the answer.

    This matches deployment: the model must manage its own plasticity.
    """
    if lr is None:
        lr = config.PHASE5_LR
    if curriculum is None:
        curriculum = CONTINUOUS_LINEAR_CURRICULUM

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        mach.load_state_dict(state_dict, strict=False)
        _repair_checkpoint(mach)

    meta_params = list(mach.parameters())
    optimizer = torch.optim.Adam(meta_params, lr=lr)

    n_meta = sum(p.numel() for p in meta_params)
    ce_anneal_step = int(n_steps * ce_anneal_start)
    ce_anneal_end = n_steps  # CE reaches 0 at final step
    print(f"Continuous training: {n_meta:,} params, {n_steps} steps, "
          f"truncation every {truncation_window}, "
          f"CE anneal: 1.0 until step {ce_anneal_step}, then → 0.0")

    # Initialize patches once — never reset
    mach.reset_episode()

    # Running stats
    window_ce = torch.tensor(0.0, device=device, requires_grad=True)
    window_critic_losses = []
    window_nuclei_losses = []
    window_hipp_losses = []  # local REINFORCE losses for hippocampus key_proj
    # Gradient tracking: accumulate norms between checkpoints
    _grad_accum = {}  # component → list of norms
    ce_weight = 1.0  # current CE weight (annealed over training)
    all_rewards = []
    current_op = random.choice(DIVERSE_TRAIN_OPS)
    op_step_count = 0
    op_switch_interval = random.randint(10, 60)  # random task duration
    # Reward sparsity: varies per task block (trains patience/gamma nucleus)
    # 1 = every step, 5 = every 5th step, etc.
    if dense_only:
        sparse_options = [1]
    else:
        sparse_options = [1, 1, 1, 1, 5, 10, 20]  # mostly dense, sometimes sparse
    reward_interval = random.choice(sparse_options)

    # Hippocampus: compressed episodic memory
    hippocampus = None
    if memory_path is not None:
        from models.hippocampus import Hippocampus
        key_dim = len(mach.patch_layers) * mach.hebb_rule.d_proj
        hippocampus = Hippocampus(
            key_dim=key_dim,
            pfc_dim=mach.pfc_dim,
            n_patches=mach.n_patches,
            save_path=memory_path,
        ).to(device)
        # Add hippocampus params to optimizer
        for p in hippocampus.parameters():
            if p.requires_grad:
                optimizer.add_param_group({'params': [p], 'lr': lr})
        n_hipp_params = sum(p.numel() for p in hippocampus.parameters())
        print(f"Hippocampus: {hippocampus}, params={n_hipp_params:,}")

    # Context buffer: rolling history of recent problems (short-term)
    context_buffer = []  # list of "a ? b = answer\n" strings

    import time
    step_timer = time.time()

    # Sleep cycle counters
    sleep_nrem_total = 0
    sleep_rem_total = 0
    sleep_rem_td_errors = []
    sleep_rem_critic_values = []
    sleep_patch_deltas = []
    # Per-operation context gate tracking
    _op_gate_accum = {}  # {op: {patch_idx: [gate_values]}}
    _op_ctx_accum = {}   # {op: {patch_idx: [context_vectors]}}

    for step in range(n_steps):
        # Switch operation randomly (like encountering different tasks)
        op_step_count += 1
        if op_step_count >= op_switch_interval:
            # Sleep between tasks — like sleeping between days
            if hippocampus is not None and len(hippocampus) > 0:
                # Snapshot patch norms before sleep
                pre_sleep_norm = sum(
                    (p.delta_down.norm().item() if p.delta_down is not None else 0) +
                    (p.delta_up.norm().item() if p.delta_up is not None else 0)
                    for p in mach.patches
                )
                # NREM: replay compressed activations → Hebbian updates (sharp-wave ripples)
                n_nrem = hippocampus.replay_nrem(mach, n_replays=4, device=device)
                sleep_nrem_total += n_nrem
                # REM: full Qwen forward with free generation → internal surprise drives plasticity
                rem_dreams = hippocampus.replay_rem(
                    mach, patched_model, tokenizer, n_dreams=2, device=device
                )
                sleep_rem_total += len(rem_dreams)
                for d in rem_dreams:
                    sleep_rem_td_errors.append(d['td_error'])
                    sleep_rem_critic_values.append(d['critic_value'])
                # Snapshot patch norms after sleep
                post_sleep_norm = sum(
                    (p.delta_down.norm().item() if p.delta_down is not None else 0) +
                    (p.delta_up.norm().item() if p.delta_up is not None else 0)
                    for p in mach.patches
                )
                sleep_patch_deltas.append(post_sleep_norm - pre_sleep_norm)

            current_op = random.choice(DIVERSE_TRAIN_OPS)
            op_step_count = 0
            op_switch_interval = random.randint(10, 60)
            reward_interval = random.choice(sparse_options)
            if context_size > 0:
                context_buffer.clear()  # new task = clear episodic memory

        # Generate one problem
        problems = generate_few_shot_episode(1, n_demos=0, op_type=current_op)
        problem = problems[0]

        # Hippocampus retrieval: reinstate similar neural states (partial blend)
        hipp_alpha = 0.0
        if hippocampus is not None and len(hippocampus) > 0:
            act_summary = mach.get_activation_summary()
            act_summary = act_summary / (act_summary.norm() + 1e-8)
            td_err = mach._last_td_error if hasattr(mach, '_last_td_error') else 0
            hipp_alpha = hippocampus.retrieve_and_reinstate(
                mach, act_summary, td_err, top_k=3, device=device
            )

        # Short-term context buffer
        context_parts = []
        if context_size > 0 and context_buffer:
            context_parts.extend(context_buffer[-context_size:])

        if context_parts:
            full_prompt = "".join(context_parts) + problem["prompt"]
        else:
            full_prompt = problem["prompt"]

        # Forward pass (with optional thinking tokens)
        prompt_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        if thinking_tokens > 0:
            # Phase 1: Generate thinking tokens (steered by patches, no grad)
            with torch.no_grad():
                generated = prompt_ids
                for _ in range(thinking_tokens):
                    out = patched_model(input_ids=generated)
                    next_token = out.logits[0, -1:].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                    tok_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
                    if "=" in tok_text or "\n" in tok_text:
                        break

            # Phase 2: Append correct answer, compute CE loss on answer only
            answer_ids = tokenizer(problem["answer"], return_tensors="pt",
                                   add_special_tokens=False).input_ids.to(device)
            full_ids = torch.cat([generated, answer_ids], dim=1)
            labels = full_ids.clone()
            labels[0, :generated.shape[1]] = -100  # loss only on answer

            output = patched_model(input_ids=full_ids, labels=labels)
            window_ce = window_ce + output.loss

            # Check prediction (what model said after thinking)
            with torch.no_grad():
                pred_tokens = output.logits[0, generated.shape[1] - 1:-1].argmax(dim=-1)
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                predicted = extract_number(pred_text)
                correct = (predicted == problem["answer"])
                reward = graded_reward(predicted, problem["answer"])
        else:
            # Direct: no thinking
            full_text = full_prompt + problem["answer"]
            encoding = tokenizer(full_text, return_tensors="pt").to(device)
            labels = encoding.input_ids.clone()
            labels[0, :prompt_len] = -100

            output = patched_model(input_ids=encoding.input_ids, labels=labels)
            window_ce = window_ce + output.loss

            with torch.no_grad():
                logits = output.logits
                pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
                pred_text = tokenizer.decode(
                    pred_tokens, skip_special_tokens=True
                ).strip()
                predicted = extract_number(pred_text)
                correct = (predicted == problem["answer"])
                reward = graded_reward(predicted, problem["answer"])
        # Sparse reward: withhold feedback on non-feedback steps
        # The system still sees the problem and generates an answer,
        # but doesn't get told if it was right. Trains patience (gamma nucleus).
        if reward_interval > 1 and op_step_count % reward_interval != 0:
            reward = 0.0  # no feedback this step

        all_rewards.append(reward)

        # Update context buffer: show the correct answer (feedback)
        if context_size > 0:
            context_buffer.append(f"{problem['prompt']}{problem['answer']}\n")

        # Hebbian step (no episode info — step_idx and n_steps are meaningless)
        value, _ = mach.hebbian_step(reward, 0, 1, device)

        # Track per-op context gate values and PFC context vectors
        if hasattr(mach, '_context_gate_values') and mach._context_gate_values:
            if current_op not in _op_gate_accum:
                _op_gate_accum[current_op] = {}
            for i, g in mach._context_gate_values.items():
                _op_gate_accum[current_op].setdefault(i, []).append(g.detach().item())
        if hasattr(mach, '_pfc_context') and mach._pfc_context:
            if current_op not in _op_ctx_accum:
                _op_ctx_accum[current_op] = {}
            for i, ctx in mach._pfc_context.items():
                _op_ctx_accum[current_op].setdefault(i, []).append(ctx.detach().clone())

        # Hippocampus: update dynamics from nuclei, reconsolidate, then store
        if hippocampus is not None:
            # Neuromod → hippocampus: nuclei control memory dynamics
            gamma = mach._last_gamma if hasattr(mach, '_last_gamma') else 0.95
            avg_decay = mach._last_decays.mean().item() if hasattr(mach, '_last_decays') and mach._last_decays is not None else 0.9
            hippocampus.set_neuromod(gamma, avg_decay)
            # Reconsolidate: update memories retrieved last step based on outcome
            hippocampus.reconsolidate(mach._last_td_error)
            # Store current state (strength = |td_error|, no threshold)
            act_summary = mach.get_activation_summary()
            act_summary = act_summary / (act_summary.norm() + 1e-8)
            hippocampus.store(mach, act_summary, reward, mach._last_td_error,
                              global_step=step)

        # Critic loss: TD target for previous step's value prediction
        # V(s_{t-1}) should equal r_{t-1} + γ·V(s_t)
        # At step t, we know V(s_t) and can compute the target for V(s_{t-1})
        if hasattr(mach, '_prev_critic_value') and mach._prev_critic_value is not None:
            prev_gamma = mach._prev_gamma if hasattr(mach, '_prev_gamma') and mach._prev_gamma is not None else mach.gamma
            td_target = mach._prev_reward + prev_gamma * value.detach()
            critic_loss = (mach._prev_critic_value - td_target) ** 2
            window_critic_losses.append(critic_loss)
        mach._prev_critic_value = value
        mach._prev_gamma = mach._current_gamma.detach() if hasattr(mach, '_current_gamma') else getattr(mach, 'gamma', 0.95)
        mach._prev_reward = torch.tensor(reward, device=device, dtype=torch.float32)

        # Nuclei loss: direct RPE signal to neuromodulatory nuclei
        if hasattr(mach, '_nuclei_loss'):
            window_nuclei_losses.append(mach._nuclei_loss)

        # Hippocampus local loss: REINFORCE signal for key_proj (pattern separation)
        if hippocampus is not None and hasattr(mach, '_last_td_error'):
            hipp_local = hippocampus.compute_local_loss(mach._last_td_error)
            if hipp_local.abs().item() > 0:
                window_hipp_losses.append(hipp_local)

        # Truncated backprop every N steps
        if (step + 1) % truncation_window == 0:
            # CE annealing: full CE until ce_anneal_start, then linear decay to 0
            if step < ce_anneal_step:
                ce_weight = 1.0
            else:
                ce_weight = max(0.0, 1.0 - (step - ce_anneal_step) / (ce_anneal_end - ce_anneal_step))
            avg_critic = torch.stack(window_critic_losses).mean() if window_critic_losses else torch.tensor(0.0, device=device)
            total_loss = ce_weight * window_ce + 0.5 * avg_critic
            # Nuclei auxiliary loss: direct RPE → nuclei (like VTA plasticity)
            if window_nuclei_losses:
                avg_nuclei = torch.stack(window_nuclei_losses).mean()
                total_loss = total_loss + 0.1 * avg_nuclei
            # Hippocampus local loss: REINFORCE for key_proj pattern separation
            if window_hipp_losses:
                avg_hipp = torch.stack(window_hipp_losses).mean()
                total_loss = total_loss + 0.05 * avg_hipp
            total_loss.backward()

            # Capture gradient norms BEFORE clipping/zeroing
            for name, p in mach.named_parameters():
                comp = name.split('.')[0]
                if p.grad is not None and p.grad.abs().sum() > 0:
                    _grad_accum.setdefault(f"mach/{comp}", []).append(p.grad.norm().item())
                elif comp in ('pfc_gru', 'pfc_proj', 'context_gates', 'pfc_to_patch', 'critic_gru', 'critic_proj'):
                    # Track PFC/critic even when zero so we can diagnose death
                    grad_status = "NONE" if p.grad is None else "ZERO"
                    _grad_accum.setdefault(f"mach/{comp}({grad_status})", []).append(0.0)
            if hippocampus is not None:
                for name, p in hippocampus.named_parameters():
                    comp = name.split('.')[0]
                    if p.grad is not None and p.grad.abs().sum() > 0:
                        _grad_accum.setdefault(f"hipp/{comp}", []).append(p.grad.norm().item())
                    else:
                        _grad_accum.setdefault(f"hipp/{comp}(ZERO)", []).append(0.0)

            # Clip nuclei output layers individually — decay_out/eta_out gradients
            # grow proportional to accumulated patch deltas (avg=9.6, max=135 at 98k),
            # dominating the global norm and starving small components like gamma_gru.
            for nuc_name in ('decay_out', 'eta_out', 'expl_out', 'gamma_out'):
                nuc = getattr(mach, nuc_name, None)
                if nuc is not None:
                    nuc_params = [p for p in nuc.parameters() if p.grad is not None]
                    if nuc_params:
                        torch.nn.utils.clip_grad_norm_(nuc_params, max_norm=1.0)
            # Then clip remaining MACH params globally
            torch.nn.utils.clip_grad_norm_(
                meta_params, max_norm=config.PHASE5_GRAD_CLIP
            )
            if hippocampus is not None:
                hipp_params = [p for p in hippocampus.parameters() if p.requires_grad and p.grad is not None]
                if hipp_params:
                    torch.nn.utils.clip_grad_norm_(hipp_params, max_norm=1.0)
            optimizer.step()
            # Synaptic scaling: clamp context gate weights to prevent saturation.
            # Brain analogy: homeostatic synaptic scaling keeps weights in functional range.
            if hasattr(mach, 'context_gates') and hasattr(mach.context_gates, 'weight'):
                with torch.no_grad():
                    mach.context_gates.weight.clamp_(-2.0, 2.0)
                    mach.context_gates.bias.clamp_(-3.0, 3.0)
            optimizer.zero_grad()

            # Detach patch deltas and critic state (break computation graph, keep values)
            for patch in mach.patches:
                if patch.delta_down is not None:
                    patch.delta_down = patch.delta_down.detach()
                if patch.delta_up is not None:
                    patch.delta_up = patch.delta_up.detach()
                if patch.delta_gain is not None:
                    patch.delta_gain = patch.delta_gain.detach()
            # Detach attention patch deltas
            if hasattr(mach, 'attn_patches'):
                for patch in mach.attn_patches:
                    if patch.delta_down is not None:
                        patch.delta_down = patch.delta_down.detach()
                    if patch.delta_up is not None:
                        patch.delta_up = patch.delta_up.detach()
                    if patch.delta_gain is not None:
                        patch.delta_gain = patch.delta_gain.detach()
            if hasattr(mach, '_critic_state'):
                mach._critic_state = mach._critic_state.detach()
            # Detach nuclei GRU states at truncation boundary
            for attr in ('_eta_state', '_decay_state', '_expl_state', '_gamma_state', '_pfc_state'):
                if hasattr(mach, attr):
                    setattr(mach, attr, getattr(mach, attr).detach())
            # Detach eligibility traces (residual + attention)
            for rule_attr in ('hebb_rule', 'attn_hebb_rule'):
                rule = getattr(mach, rule_attr, None)
                if rule is not None and hasattr(rule, '_traces') and rule._traces is not None:
                    for p_traces in rule._traces:
                        for r in range(len(p_traces)):
                            p_traces[r] = p_traces[r].detach()
            # Detach stored activations (undetached obs path)
            for key in list(mach._pre_activations.keys()):
                mach._pre_activations[key] = mach._pre_activations[key].detach()
            for key in list(mach._post_activations.keys()):
                mach._post_activations[key] = mach._post_activations[key].detach()
            # Detach attention activations
            if hasattr(mach, '_attn_pre_activations'):
                for key in list(mach._attn_pre_activations.keys()):
                    mach._attn_pre_activations[key] = mach._attn_pre_activations[key].detach()
            if hasattr(mach, '_attn_post_activations'):
                for key in list(mach._attn_post_activations.keys()):
                    mach._attn_post_activations[key] = mach._attn_post_activations[key].detach()

            # Detach TD bootstrapping state
            if hasattr(mach, '_prev_critic_value') and mach._prev_critic_value is not None:
                mach._prev_critic_value = mach._prev_critic_value.detach()

            # Reset window accumulators
            window_ce = torch.tensor(0.0, device=device, requires_grad=True)
            window_critic_losses = []
            window_nuclei_losses = []
            window_hipp_losses = []

        # Logging
        if step % 100 == 0 and step > 0:
            recent = all_rewards[-100:]
            acc = sum(1 for r in recent if r == 1.0) / len(recent)  # exact match only
            avg_r = sum(recent) / len(recent)
            neuromod_str = ""
            if hasattr(mach, '_last_etas') and mach._last_etas is not None:
                td_str = f"td={mach._last_td_error:+.3f}" if hasattr(mach, '_last_td_error') else ""
                # Patch delta norms
                dnorm = sum(
                    (p.delta_down.norm().item() if p.delta_down is not None else 0) +
                    (p.delta_up.norm().item() if p.delta_up is not None else 0)
                    for p in mach.patches
                )
                eta_str = "/".join(f"{e:.2f}" for e in mach._last_etas)
                decay_str = "/".join(f"{d:.2f}" for d in mach._last_decays)
                expl_str = "/".join(f"{x:.2f}" for x in mach._last_expls) if hasattr(mach, '_last_expls') else f"{mach._last_exploration:.3f}"
                gamma_str = f" γ={mach._last_gamma:.2f}" if hasattr(mach, '_last_gamma') else ""
                neuromod_str = (f" | η=[{eta_str}]"
                               f" dec=[{decay_str}]"
                               f" expl=[{expl_str}]"
                               f"{gamma_str}"
                               f" {td_str} Δ={dnorm:.2f}")
            elapsed = time.time() - step_timer
            steps_per_sec = 100 / elapsed if elapsed > 0 else 0
            step_timer = time.time()
            hipp_str = ""
            if hippocampus is not None:
                ep_idx = hippocampus._last_ep_idx if hasattr(hippocampus, '_last_ep_idx') else -1
                hipp_str = f" | hipp: e{ep_idx}/{len(hippocampus)} α={hipp_alpha:.3f}"
            ce_w = ce_weight if step >= ce_anneal_step else 1.0
            ce_str = f" ce={ce_w:.2f}" if ce_w < 1.0 else ""
            print(
                f"Step {step:5d} | op={current_op:<10} | "
                f"acc(100)={acc:.0%} avg_r={avg_r:.2f}{neuromod_str}"
                f"{hipp_str}{ce_str}"
                f" [{steps_per_sec:.1f} st/s]"
            )
            if wandb is not None:
                wandb.log({
                    "step": step,
                    "accuracy_100": acc,
                    "avg_reward_100": avg_r,
                    "current_op": current_op,
                })

        # Diagnostics + checkpoint
        if step % 2000 == 0 and step > 0:
            # Print accumulated gradient norms (captured after each backward)
            if _grad_accum:
                print(f"  Gradient norms (avg over {len(next(iter(_grad_accum.values())))} backward passes):")
                for comp in sorted(_grad_accum.keys()):
                    norms = _grad_accum[comp]
                    avg = sum(norms) / len(norms)
                    mx = max(norms)
                    print(f"    {comp}: avg={avg:.6f} max={mx:.6f} ({len(norms)} samples)")
                _grad_accum.clear()
            _log_hebbian_diagnostics(mach, meta_params, step, hippocampus=hippocampus)
            # Per-operation context gate summary
            if _op_gate_accum:
                print("  Context gates by operation:")
                for op in sorted(_op_gate_accum.keys()):
                    gates_str = " ".join(
                        f"p{i}={sum(vs)/len(vs):.2f}"
                        for i, vs in sorted(_op_gate_accum[op].items())
                    )
                    print(f"    {op:<10} {gates_str}")
                _op_gate_accum.clear()
            # Per-operation PFC top-down context: cosine similarity between ops
            if _op_ctx_accum and len(_op_ctx_accum) > 1:
                # Compute mean context vector per (op, patch)
                op_means = {}
                for op, patches in _op_ctx_accum.items():
                    op_means[op] = {}
                    for i, vecs in patches.items():
                        op_means[op][i] = torch.stack(vecs).mean(dim=0)
                ops = sorted(op_means.keys())
                print("  PFC top-down context similarity (cosine) between operations:")
                for i in range(len(mach.patches)):
                    # Pairwise cosine sim for this patch
                    sims = []
                    for a_idx in range(len(ops)):
                        for b_idx in range(a_idx + 1, len(ops)):
                            va = op_means[ops[a_idx]].get(i)
                            vb = op_means[ops[b_idx]].get(i)
                            if va is not None and vb is not None:
                                cos = torch.nn.functional.cosine_similarity(va, vb, dim=0).item()
                                sims.append(cos)
                    if sims:
                        avg_cos = sum(sims) / len(sims)
                        min_cos = min(sims)
                        print(f"    patch{i}: avg_cos={avg_cos:.3f} min_cos={min_cos:.3f}")
                _op_ctx_accum.clear()
            # Sleep cycle stats
            if sleep_nrem_total > 0 or sleep_rem_total > 0:
                rem_avg_td = sum(abs(t) for t in sleep_rem_td_errors) / len(sleep_rem_td_errors) if sleep_rem_td_errors else 0
                rem_avg_val = sum(sleep_rem_critic_values) / len(sleep_rem_critic_values) if sleep_rem_critic_values else 0
                avg_patch_delta = sum(sleep_patch_deltas) / len(sleep_patch_deltas) if sleep_patch_deltas else 0
                print(f"  Sleep: NREM={sleep_nrem_total} replays, REM={sleep_rem_total} dreams")
                print(f"    REM avg_|td|={rem_avg_td:.4f}  avg_critic_val={rem_avg_val:.4f}  avg_patch_Δ={avg_patch_delta:+.4f}")
                if rem_avg_td < 0.01:
                    print(f"    ⚠ REM td≈0 — critic not surprised by dreams, sleep may be inert")
                sleep_nrem_total = 0
                sleep_rem_total = 0
                sleep_rem_td_errors = []
                sleep_rem_critic_values = []
                sleep_patch_deltas = []

            # Quick validation — save/restore continuous state
            print(f"  --- Operation validation (step {step}) ---")
            eval_ops = DIVERSE_TRAIN_OPS[:6]
            saved_deltas = [
                (p.delta_down.detach().clone() if p.delta_down is not None else None,
                 p.delta_up.detach().clone() if p.delta_up is not None else None,
                 p.delta_gain.detach().clone() if p.delta_gain is not None else None)
                for p in mach.patches
            ]
            saved_attn_deltas = [
                (p.delta_down.detach().clone() if p.delta_down is not None else None,
                 p.delta_up.detach().clone() if p.delta_up is not None else None,
                 p.delta_gain.detach().clone() if p.delta_gain is not None else None)
                for p in mach.attn_patches
            ] if hasattr(mach, 'attn_patches') else None
            saved_reward_ema = mach._reward_ema
            saved_critic_state = mach._critic_state.detach().clone() if hasattr(mach, '_critic_state') else None
            saved_nuclei = {
                attr: getattr(mach, attr).detach().clone()
                for attr in ('_eta_state', '_decay_state', '_expl_state', '_pfc_state')
                if hasattr(mach, attr)
            }
            mach_training = mach.training
            mach.eval()
            for op in eval_ops:
                _run_op_validation_hebbian(
                    base_model, mach, patched_model,
                    tokenizer, device, op, step,
                )
            # Restore continuous state
            for p, (dd, du, dg) in zip(mach.patches, saved_deltas):
                p.delta_down = dd
                p.delta_up = du
                p.delta_gain = dg
            if saved_attn_deltas is not None:
                for p, (dd, du, dg) in zip(mach.attn_patches, saved_attn_deltas):
                    p.delta_down = dd
                    p.delta_up = du
                    p.delta_gain = dg
            mach._reward_ema = saved_reward_ema
            if saved_critic_state is not None:
                mach._critic_state = saved_critic_state
            for attr, val in saved_nuclei.items():
                setattr(mach, attr, val)
            if mach_training:
                mach.train()

            if save_path:
                torch.save(mach.state_dict(), save_path)
                print(f"  Checkpoint saved to {save_path}")

            # Hippocampus: decay + save
            if hippocampus is not None:
                hippocampus.decay_all()
                hippocampus.save()
                print(f"  Hippocampus: {hippocampus}")


def _consolidation_replay(mach, patched_model, tokenizer, device, hippocampus,
                          batch_size=10):
    """Sleep-like consolidation: replay hippocampal memories through Hebbian loop.

    Patches absorb patterns from stored experiences. The explicit memory
    can then fade (via decay) while the implicit knowledge persists in weights.
    """
    batch = hippocampus.get_replay_batch(batch_size=batch_size)
    if not batch:
        return 0

    was_training = mach.training
    mach.eval()

    for text, stored_td_error in batch:
        # Parse experience text back into prompt + answer
        # Format: "a ? b = answer\n"
        parts = text.strip().split("= ")
        if len(parts) != 2:
            continue
        prompt = parts[0] + "= "
        answer = parts[1]

        full_text = prompt + answer
        encoding = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_len = len(tokenizer(prompt).input_ids)

        with torch.no_grad():
            output = patched_model(input_ids=encoding.input_ids)
            logits = output.logits
            pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            predicted = extract_number(pred_text)
            correct = (predicted == answer)
            reward = 1.0 if correct else -1.0

        # Hebbian update from replay (same as live experience)
        mach.hebbian_step(reward, 0, 1, device)

    if was_training:
        mach.train()

    return len(batch)


def _run_op_validation_hebbian(base_model, mach, patched_model,
                               tokenizer, device, op_type, episode_idx,
                               n_episodes=10, n_problems=20):
    """Evaluate operation classification with Hebbian learning.
    First half: learning. Second half: evaluation."""
    second_half_correct = 0
    second_half_total = 0

    for ep in range(n_episodes):
        problems = generate_few_shot_episode(
            n_problems, n_demos=0, op_type=op_type
        )
        mach.reset_episode()

        for step, problem in enumerate(problems):
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

            # Hebbian update (inference mode)
            with torch.no_grad():
                mach.hebbian_step(reward, step, n_problems, device)

            # Count second half only
            if step >= n_problems // 2:
                second_half_correct += int(correct)
                second_half_total += 1

    acc = second_half_correct / second_half_total if second_half_total > 0 else 0
    print(f"  EVAL ep{episode_idx} {op_type:4s} | 2nd_half={acc:.0%}")

    if wandb is not None:
        wandb.log({f"eval/op_{op_type}": acc})


def ablate_hebbian(base_model, mach, patched_model, tokenizer, device,
                   n_episodes=10, n_problems=20):
    """
    Ablation: compare Hebbian updates ON vs OFF.
    - WITH Hebbian: normal eval (reset, forward, hebbian_step each problem)
    - WITHOUT Hebbian: reset once, forward all problems, NO hebbian_step
    - INIT ONLY: reset once (gets random init), forward all, NO updates
    This tells us if the Hebbian updates contribute beyond random init + static basis.
    """
    # Use diverse sets: train + held-out (fair) + impossible (control)
    all_train = DIVERSE_TRAIN_OPS
    all_heldout = DIVERSE_HELDOUT_OPS
    all_impossible = DIVERSE_IMPOSSIBLE_OPS
    ops = all_train + all_heldout + all_impossible
    results = {"with_hebbian": {}, "no_update": {}, "no_init": {}}

    for op in ops:
        # --- WITH Hebbian updates ---
        correct_hebb = 0
        total_hebb = 0
        for ep in range(n_episodes):
            problems = generate_few_shot_episode(
                n_problems, n_demos=0, op_type=op
            )
            mach.reset_episode()
            for step, problem in enumerate(problems):
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
                with torch.no_grad():
                    mach.hebbian_step(reward, step, n_problems, device)
                if step >= n_problems // 2:
                    correct_hebb += int(correct)
                    total_hebb += 1
        results["with_hebbian"][op] = correct_hebb / max(total_hebb, 1)

        # --- NO Hebbian updates (but WITH random init) ---
        correct_no = 0
        total_no = 0
        for ep in range(n_episodes):
            problems = generate_few_shot_episode(
                n_problems, n_demos=0, op_type=op
            )
            mach.reset_episode()  # gets random init
            for step, problem in enumerate(problems):
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
                # NO hebbian_step
                if step >= n_problems // 2:
                    correct_no += int(correct)
                    total_no += 1
        results["no_update"][op] = correct_no / max(total_no, 1)

        # --- NO init, NO updates (zero patches) ---
        correct_zero = 0
        total_zero = 0
        old_init_std = mach.init_std
        mach.init_std = 0  # disable random init
        for ep in range(n_episodes):
            problems = generate_few_shot_episode(
                n_problems, n_demos=0, op_type=op
            )
            mach.reset_episode()  # zero patches
            for step, problem in enumerate(problems):
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
                if step >= n_problems // 2:
                    correct_zero += int(correct)
                    total_zero += 1
        mach.init_std = old_init_std
        results["no_init"][op] = correct_zero / max(total_zero, 1)

    trained_ops = all_train
    heldout_ops = all_heldout
    impossible_ops = all_impossible

    print("\n  === HEBBIAN ABLATION ===")
    print(f"  {'op':4s} | {'with_hebb':>10s} | {'no_update':>10s} | {'no_init':>10s}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for op in trained_ops:
        print(f"  {op:4s} | {results['with_hebbian'][op]:>9.0%} | {results['no_update'][op]:>9.0%} | {results['no_init'][op]:>9.0%}")

    avg_hebb = sum(results['with_hebbian'][op] for op in trained_ops) / len(trained_ops)
    avg_no = sum(results['no_update'][op] for op in trained_ops) / len(trained_ops)
    avg_zero = sum(results['no_init'][op] for op in trained_ops) / len(trained_ops)
    print(f"  {'tavg':4s} | {avg_hebb:>9.0%} | {avg_no:>9.0%} | {avg_zero:>9.0%}")

    if heldout_ops:
        print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}  (held-out: Qwen CAN do)")
        for op in heldout_ops:
            print(f"  {op:4s} | {results['with_hebbian'][op]:>9.0%} | {results['no_update'][op]:>9.0%} | {results['no_init'][op]:>9.0%}")
        avg_hebb_h = sum(results['with_hebbian'][op] for op in heldout_ops) / len(heldout_ops)
        avg_no_h = sum(results['no_update'][op] for op in heldout_ops) / len(heldout_ops)
        avg_zero_h = sum(results['no_init'][op] for op in heldout_ops) / len(heldout_ops)
        print(f"  {'havg':4s} | {avg_hebb_h:>9.0%} | {avg_no_h:>9.0%} | {avg_zero_h:>9.0%}")

    if impossible_ops:
        print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}  (impossible: Qwen CAN'T do)")
        for op in impossible_ops:
            print(f"  {op:4s} | {results['with_hebbian'][op]:>9.0%} | {results['no_update'][op]:>9.0%} | {results['no_init'][op]:>9.0%}")
        avg_hebb_i = sum(results['with_hebbian'][op] for op in impossible_ops) / len(impossible_ops)
        avg_no_i = sum(results['no_update'][op] for op in impossible_ops) / len(impossible_ops)
        avg_zero_i = sum(results['no_init'][op] for op in impossible_ops) / len(impossible_ops)
        print(f"  {'iavg':4s} | {avg_hebb_i:>9.0%} | {avg_no_i:>9.0%} | {avg_zero_i:>9.0%}")
    print()

    if wandb is not None:
        for op in ops:
            wandb.log({
                f"ablation/{op}_with_hebbian": results["with_hebbian"][op],
                f"ablation/{op}_no_update": results["no_update"][op],
                f"ablation/{op}_no_init": results["no_init"][op],
            })

    return results


def _log_hebbian_diagnostics(mach, meta_params, episode_idx, hippocampus=None):
    """Log gradient norms and weight stats for Hebbian architecture."""
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
        max_norm = max(norms)
        diag[f"grad_norm/{component}"] = avg_norm
        diag[f"grad_max/{component}"] = max_norm

    # Hippocampus stats (gradient norms are captured in _grad_accum, not here)
    if hippocampus is not None:

        # Hippocampus weight norms (are parameters growing/changing?)
        for name, param in hippocampus.named_parameters():
            component = name.split('.')[0]
            diag[f"hipp_weight/{name}"] = param.data.norm().item()

        # Hippocampus buffer stats
        diag["hipp/n_episodes"] = len(hippocampus)
        if hasattr(hippocampus, '_last_alpha'):
            diag["hipp/last_alpha"] = hippocampus._last_alpha

    # Patch delta stats
    for i, patch in enumerate(mach.patches):
        if patch.delta_down is not None:
            diag[f"patch_delta/patch{i}_down"] = patch.delta_down.norm().item()
            diag[f"patch_delta/patch{i}_up"] = patch.delta_up.norm().item()

    # Basis vector norms (only for MACHHebbian with basis vectors)
    if hasattr(mach, 'basis'):
        for i in range(mach.basis.n_patches):
            diag[f"basis_norm/patch{i}_down_U"] = mach.basis.down_U[i].norm().item()
            diag[f"basis_norm/patch{i}_down_V"] = mach.basis.down_V[i].norm().item()

    # Attention patch stats (only for MACHDualHebbian)
    if hasattr(mach, 'attn_patches'):
        for i, patch in enumerate(mach.attn_patches):
            if patch.delta_down is not None:
                diag[f"attn_patch_delta/patch{i}_down"] = patch.delta_down.norm().item()
                diag[f"attn_patch_delta/patch{i}_up"] = patch.delta_up.norm().item()

    # Coprocessor stats (only for MACHCoprocessor)
    if hasattr(mach, 'copro_patches'):
        for i, p in enumerate(mach.copro_patches):
            if p.delta_down is not None:
                diag[f"copro/patch{i}_down"] = p.delta_down.norm().item()
                diag[f"copro/patch{i}_up"] = p.delta_up.norm().item()

    # Top-down gain stats (only for MACHDenseHebbian)
    if hasattr(mach, 'gain_net') and mach._gains is not None:
        gains = mach._gains.detach()
        diag["gain/mean"] = gains.mean().item()
        diag["gain/std"] = gains.std().item()
        diag["gain/min"] = gains.min().item()
        diag["gain/max"] = gains.max().item()

    # PFC / context gate diagnostics
    if hasattr(mach, '_context_gate_values') and mach._context_gate_values:
        for i, gate in mach._context_gate_values.items():
            g = gate.detach()
            if g.dim() == 0:
                diag[f"context_gate/patch{i}"] = g.item()  # scalar gate
            else:
                diag[f"context_gate/patch{i}_mean"] = g.mean().item()
                diag[f"context_gate/patch{i}_std"] = g.std().item()
    else:
        diag["context_gate/EMPTY"] = 1.0
    if hasattr(mach, '_pfc_state') and mach._pfc_state is not None:
        pfc = mach._pfc_state.detach()
        diag["pfc/state_norm"] = pfc.norm().item()
        diag["pfc/state_mean"] = pfc.mean().item()
        diag["pfc/requires_grad"] = float(mach._pfc_state.requires_grad)
    # PFC/critic/context_gates weight norms
    for name in ('pfc_gru', 'pfc_proj', 'critic_gru', 'critic_proj'):
        mod = getattr(mach, name, None)
        if mod is not None:
            for pname, p in mod.named_parameters():
                diag[f"weight/{name}.{pname}"] = p.data.norm().item()
    if hasattr(mach, 'context_gates'):
        mod = mach.context_gates
        if hasattr(mod, 'named_parameters'):
            for pname, p in mod.named_parameters():
                diag[f"weight/context_gates.{pname}"] = p.data.norm().item()
    if hasattr(mach, '_pfc_context') and mach._pfc_context:
        for i, ctx in mach._pfc_context.items():
            c = ctx.detach()
            diag[f"pfc_context/patch{i}_norm"] = c.norm().item()
    if hasattr(mach, 'pfc_to_patch'):
        for i, proj in enumerate(mach.pfc_to_patch):
            diag[f"weight/pfc_to_patch{i}.weight"] = proj.weight.data.norm().item()

    # Neuromodulation stats
    if hasattr(mach, '_last_etas') and mach._last_etas is not None:
        diag["neuromod/eta"] = mach._last_etas.mean().item()
        diag["neuromod/decay"] = mach._last_decays.mean().item()
        diag["neuromod/reward_ema"] = mach._reward_ema
    if hasattr(mach, '_last_td_error'):
        diag["neuromod/td_error"] = mach._last_td_error
    if hasattr(mach, '_last_exploration'):
        diag["neuromod/exploration"] = mach._last_exploration
    if hasattr(mach, '_last_gamma'):
        diag["neuromod/gamma"] = mach._last_gamma

    print(f"  Diagnostics at episode {episode_idx}:")
    for k, v in sorted(diag.items()):
        print(f"    {k}: {v:.6f}")

    if wandb is not None:
        wandb.log({f"diag/{k}": v for k, v in diag.items()})


def _run_linear_validation_hebbian(base_model, mach, patched_model,
                                   tokenizer, device, coeffs, episode_idx,
                                   n_episodes=10, n_problems=20):
    """Evaluate linear combination performance with Hebbian architecture.
    Uses first few problems as learning opportunities, evaluates later ones."""
    test_correct = 0
    test_total = 0

    for ep in range(n_episodes):
        problems = generate_linear_episode(
            n_problems, n_demos=0, coeffs=coeffs
        )
        mach.reset_episode()

        for step, problem in enumerate(problems):
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

            # Hebbian update (inference mode — no gradient)
            with torch.no_grad():
                mach.hebbian_step(reward, step, n_problems, device)

            # Only count second half as test accuracy
            if step >= n_problems // 2:
                test_correct += int(correct)
                test_total += 1

    test_acc = test_correct / test_total if test_total > 0 else 0
    c1, c2 = coeffs
    label = f"{c1}a+{c2}b"

    print(f"  EVAL ep{episode_idx} {label:8s} | test={test_acc:.0%}")

    if wandb is not None:
        wandb.log({f"eval/linear_{c1}_{c2}": test_acc})
