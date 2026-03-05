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


CONTINUOUS_LINEAR_CURRICULUM = [
    (0, 5000, "continuous_linear"),
]

TOKEN_MAP_CURRICULUM = [
    (0, 5000, "token_map"),
]

MIXED_CURRICULUM = [
    (0, 5000, "mixed"),
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
    else:  # few_shot
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

    # Basis vector norms
    for i in range(mach.basis.n_patches):
        diag[f"basis_norm/patch{i}_down_U"] = mach.basis.down_U[i].norm().item()
        diag[f"basis_norm/patch{i}_down_V"] = mach.basis.down_V[i].norm().item()

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
            else:
                predicted = extract_number(pred_text)
                correct = (predicted == problem["answer"])
            reward = 1.0 if correct else -1.0
        rewards.append(reward)

        # 3. Hebbian step: critic → TD error → local weight updates
        value, td_error = mach.hebbian_step(
            reward, step, len(problems), device
        )

        # 4. Critic loss (TD learning)
        critic_target = torch.tensor(reward, device=device, dtype=torch.float32)
        critic_loss = (value - critic_target) ** 2
        critic_losses.append(critic_loss)

    avg_critic_loss = torch.stack(critic_losses).mean() if critic_losses else torch.tensor(0.0, device=device)

    return total_ce, rewards, problem_losses, avg_critic_loss


def meta_train_hebbian(base_model, mach, patched_model, tokenizer,
                       device, n_episodes=None, lr=None,
                       curriculum=None, checkpoint_path=None,
                       save_path=None):
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
        if episode_idx < 100:
            return 10
        elif episode_idx < 300:
            return 15
        else:
            return config.PHASE5_PROBLEMS_PER_EPISODE

    for episode_idx in range(n_episodes):
        mode = get_episode_mode(episode_idx, curriculum)
        n_problems = get_n_problems(episode_idx)
        problems = generate_episode_problems(n_problems, mode)

        optimizer.zero_grad()

        try:
            ce_loss, rewards, problem_losses, critic_loss = run_episode_hebbian(
                base_model, mach, patched_model, tokenizer,
                problems, device,
            )

            total_loss = ce_loss + 0.1 * critic_loss
            total_loss.backward()
            loss_scalar = ce_loss.item()
            critic_scalar = critic_loss.item()

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at episode {episode_idx}. Reducing problems.")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            problems = problems[:max(5, len(problems) // 2)]
            ce_loss, rewards, problem_losses, critic_loss = run_episode_hebbian(
                base_model, mach, patched_model, tokenizer,
                problems, device,
            )
            total_loss = ce_loss + 0.1 * critic_loss
            total_loss.backward()
            loss_scalar = ce_loss.item()
            critic_scalar = critic_loss.item()

        torch.nn.utils.clip_grad_norm_(
            meta_params, max_norm=config.PHASE5_GRAD_CLIP
        )
        optimizer.step()

        if episode_idx % 10 == 0:
            n_total = len(rewards)
            n_correct = sum(1 for r in rewards if r > 0)
            test_acc = n_correct / max(n_total, 1)
            avg_reward = sum(rewards) / max(n_total, 1)

            # Within-episode improvement: compare first half vs second half
            mid = max(1, n_total // 2)
            first_half = sum(1 for r in rewards[:mid] if r > 0) / max(mid, 1)
            second_half = sum(1 for r in rewards[mid:] if r > 0) / max(n_total - mid, 1)

            print(
                f"Episode {episode_idx:4d} | {mode} n={n_total:2d} | "
                f"ce={loss_scalar:.4f} critic={critic_scalar:.4f} | "
                f"acc={test_acc:.0%} 1st={first_half:.0%} 2nd={second_half:.0%} | "
                f"avg_r={avg_reward:.2f}"
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

            if save_path:
                torch.save(mach.state_dict(), save_path)
                print(f"  Checkpoint saved to {save_path}")


def _log_hebbian_diagnostics(mach, meta_params, episode_idx):
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
        diag[f"grad_norm/{component}"] = avg_norm

    # Patch delta stats
    for i, patch in enumerate(mach.patches):
        if patch.delta_down is not None:
            diag[f"patch_delta/patch{i}_down"] = patch.delta_down.norm().item()
            diag[f"patch_delta/patch{i}_up"] = patch.delta_up.norm().item()

    # Basis vector norms
    for i in range(mach.basis.n_patches):
        diag[f"basis_norm/patch{i}_down_U"] = mach.basis.down_U[i].norm().item()
        diag[f"basis_norm/patch{i}_down_V"] = mach.basis.down_V[i].norm().item()

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
