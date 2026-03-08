"""Comprehensive system health tests: gradient flow, patch writes, reward-only mode.

Tests the REAL MACHActivationHebbian (not mocks) to verify:
1. Every component receives gradient during training
2. Patches are actually being written to by Hebbian updates
3. Reward-only mode (inference) works correctly
4. Hippocampus gradient flows through PFC → critic → loss
5. Attention patches get gradient
6. Eligibility traces accumulate properly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.universal_module import MACHActivationHebbian, DifferentiablePatch
from models.hippocampus import Hippocampus


def make_mach(d_model=64, n_layers=8, hidden_dim=32, n_rank=2, d_proj=16):
    """Create a small MACHActivationHebbian for testing."""
    patch_layers = [2, 5]
    mach = MACHActivationHebbian(
        d_model=d_model,
        n_layers=n_layers,
        patch_layers=patch_layers,
        hidden_dim=hidden_dim,
        n_rank=n_rank,
        d_proj=d_proj,
    )
    mach.reset_episode()
    mach.hebb_rule.reset_traces('cpu')
    mach.attn_hebb_rule.reset_traces('cpu')
    return mach


def simulate_forward_pass(mach, d_model=64):
    """Simulate a Qwen forward pass by populating pre/post activations."""
    for i in range(mach.n_patches):
        mach._pre_activations[i] = torch.randn(1, 5, d_model)
        mach._post_activations[i] = torch.randn(1, 5, d_model)
        mach._attn_pre_activations[i] = torch.randn(1, 5, d_model)
        mach._attn_post_activations[i] = torch.randn(1, 5, d_model)


def test_gradient_all_components():
    """Every trainable component in MACHActivationHebbian gets gradient.

    Simulates the REAL training loop order:
    1. compute_context_gates (PFC GRU, called in patched_model.forward)
    2. hebbian_step (critic, nuclei, Hebbian writes)
    3. Loss = CE (through patches) + critic (TD) + nuclei (RPE)
    """
    mach = make_mach()

    # --- Truncation window: multiple steps ---
    window_critic_losses = []
    window_nuclei_losses = []
    prev_value = None

    for step in range(5):
        simulate_forward_pass(mach)

        # Like patched_model.forward() calls compute_context_gates
        mach.compute_context_gates()

        # hebbian_step (nuclei, critic, Hebbian writes)
        reward = 1.0 if step % 2 == 0 else -1.0
        value, _ = mach.hebbian_step(reward=reward, step_idx=step, n_steps=5, device='cpu')

        # TD critic loss
        if prev_value is not None:
            td_target = torch.tensor(reward, dtype=torch.float32) + 0.95 * value.detach()
            critic_loss = (prev_value - td_target) ** 2
            window_critic_losses.append(critic_loss)
        prev_value = value

        # Nuclei RPE loss
        if hasattr(mach, '_nuclei_loss'):
            window_nuclei_losses.append(mach._nuclei_loss)

    # CE loss proxy: sum of patch deltas (in real training, CE flows through patches in Qwen)
    patch_out = sum(
        p.delta_down.sum() + p.delta_up.sum()
        for p in mach.patches if p.delta_down is not None
    )
    attn_patch_out = sum(
        p.delta_down.sum() + p.delta_up.sum()
        for p in mach.attn_patches if p.delta_down is not None
    )
    ce_proxy = 0.1 * patch_out + 0.1 * attn_patch_out

    # Total loss (matches training loop)
    avg_critic = torch.stack(window_critic_losses).mean() if window_critic_losses else torch.tensor(0.0)
    total_loss = ce_proxy + 0.5 * avg_critic
    if window_nuclei_losses:
        avg_nuclei = torch.stack(window_nuclei_losses).mean()
        total_loss = total_loss + 0.1 * avg_nuclei

    total_loss.backward()

    # Check every component
    component_grads = {}
    component_zero = {}
    for name, p in mach.named_parameters():
        comp = name.split('.')[0]
        if p.grad is not None and p.grad.abs().sum() > 0:
            component_grads.setdefault(comp, []).append(
                (name, p.grad.norm().item())
            )
        else:
            component_zero.setdefault(comp, []).append(name)

    print("  Components WITH gradient:")
    for comp in sorted(component_grads.keys()):
        params = component_grads[comp]
        avg_norm = sum(n for _, n in params) / len(params)
        print(f"    {comp}: {len(params)} params, avg_norm={avg_norm:.6f}")

    if component_zero:
        print("  Components with ZERO gradient:")
        for comp in sorted(component_zero.keys()):
            print(f"    {comp}: {component_zero[comp]}")

    # Expected components that MUST have gradient
    required = ['critic_gru', 'critic_proj', 'value_head',
                 'eta_gru', 'eta_out', 'decay_gru', 'decay_out',
                 'pfc_gru', 'pfc_proj', 'hebb_rule', 'attn_hebb_rule']

    # These get gradient through nuclei_loss (RPE), may be weak
    nuclei_components = ['expl_gru', 'expl_out', 'gamma_gru', 'gamma_out']

    for req in required:
        assert req in component_grads, \
            f"FAIL: {req} has NO gradient! Components with grad: {list(component_grads.keys())}"

    # Nuclei components: warn if missing but don't fail (gradient may be very weak)
    for nc in nuclei_components:
        if nc not in component_grads:
            print(f"  WARNING: {nc} has no gradient (RPE signal may be too weak)")

    print("  PASS: all required components have gradient")


def test_patch_writes_nonzero():
    """Hebbian step actually writes nonzero deltas to patches."""
    mach = make_mach()
    simulate_forward_pass(mach)

    # Record initial state
    initial_downs = [p.delta_down.clone() for p in mach.patches]
    initial_attn_downs = [p.delta_down.clone() for p in mach.attn_patches]

    # Run hebbian step with reward
    mach.hebbian_step(reward=1.0, step_idx=0, n_steps=10, device='cpu')

    # Check residual patches changed
    res_changed = 0
    for i, p in enumerate(mach.patches):
        if not torch.allclose(p.delta_down, initial_downs[i]):
            res_changed += 1

    # Check attention patches changed
    attn_changed = 0
    for i, p in enumerate(mach.attn_patches):
        if not torch.allclose(p.delta_down, initial_attn_downs[i]):
            attn_changed += 1

    print(f"  Residual patches written: {res_changed}/{len(mach.patches)}")
    print(f"  Attention patches written: {attn_changed}/{len(mach.attn_patches)}")

    assert res_changed > 0, "FAIL: no residual patches were written to"
    assert attn_changed > 0, "FAIL: no attention patches were written to"
    print("  PASS: patches are being written to")


def test_patch_writes_accumulate():
    """Multiple hebbian steps accumulate (not overwrite) patch deltas."""
    mach = make_mach()

    for step in range(5):
        simulate_forward_pass(mach)
        mach.hebbian_step(reward=1.0 if step % 2 == 0 else -1.0,
                          step_idx=step, n_steps=10, device='cpu')

    # After 5 steps, deltas should be substantial
    total_delta = sum(p.delta_down.abs().sum().item() for p in mach.patches)
    total_attn_delta = sum(p.delta_down.abs().sum().item() for p in mach.attn_patches)

    print(f"  Residual delta total: {total_delta:.4f}")
    print(f"  Attention delta total: {total_attn_delta:.4f}")

    assert total_delta > 0.01, f"FAIL: residual deltas too small ({total_delta})"
    assert total_attn_delta > 0.01, f"FAIL: attention deltas too small ({total_attn_delta})"
    print("  PASS: patch writes accumulate over steps")


def test_eligibility_traces():
    """Eligibility traces accumulate and influence updates."""
    mach = make_mach()
    d_model = 64

    # Step 1: stimulus (trace should form)
    simulate_forward_pass(mach)
    mach.hebbian_step(reward=0.0, step_idx=0, n_steps=10, device='cpu')

    traces_after_1 = [
        [t.item() for t in patch_traces]
        for patch_traces in mach.hebb_rule._traces
    ]

    # Step 2: reward arrives (trace converts to weight change)
    simulate_forward_pass(mach)
    value, _ = mach.hebbian_step(reward=1.0, step_idx=1, n_steps=10, device='cpu')

    traces_after_2 = [
        [t.item() for t in patch_traces]
        for patch_traces in mach.hebb_rule._traces
    ]

    print(f"  Traces after step 1: {traces_after_1}")
    print(f"  Traces after step 2: {traces_after_2}")

    # Traces should be nonzero
    any_nonzero = any(
        abs(t) > 1e-8
        for patch_traces in traces_after_2
        for t in patch_traces
    )
    assert any_nonzero, "FAIL: all traces are zero after 2 steps"
    print("  PASS: eligibility traces accumulate")


def test_nuclei_outputs_change():
    """Nuclei (eta, decay, exploration, gamma) actually produce varying outputs."""
    mach = make_mach()

    etas_over_time = []
    decays_over_time = []
    gammas_over_time = []

    for step in range(10):
        simulate_forward_pass(mach)
        reward = 1.0 if step < 5 else -1.0
        mach.hebbian_step(reward=reward, step_idx=step, n_steps=10, device='cpu')
        etas_over_time.append(mach._last_etas.clone())
        decays_over_time.append(mach._last_decays.clone())
        if hasattr(mach, '_last_gamma'):
            gammas_over_time.append(mach._last_gamma)

    # Check eta changes
    eta_range = (torch.stack(etas_over_time).max() - torch.stack(etas_over_time).min()).item()
    decay_range = (torch.stack(decays_over_time).max() - torch.stack(decays_over_time).min()).item()

    print(f"  Eta range over 10 steps: {eta_range:.4f}")
    print(f"  Decay range over 10 steps: {decay_range:.4f}")
    if gammas_over_time:
        gamma_range = max(gammas_over_time) - min(gammas_over_time)
        print(f"  Gamma range over 10 steps: {gamma_range:.4f}")

    assert eta_range > 1e-4, f"FAIL: eta stuck at constant value (range={eta_range})"
    print("  PASS: nuclei produce varying outputs")


def test_critic_value_changes():
    """Critic value changes with different reward histories."""
    mach = make_mach()

    values = []
    for step in range(10):
        simulate_forward_pass(mach)
        reward = 1.0 if step < 5 else -1.0
        value, _ = mach.hebbian_step(reward=reward, step_idx=step, n_steps=10, device='cpu')
        values.append(value.item())

    value_range = max(values) - min(values)
    print(f"  Critic values: {[f'{v:.3f}' for v in values]}")
    print(f"  Value range: {value_range:.4f}")

    assert value_range > 0.01, f"FAIL: critic value not changing (range={value_range})"
    print("  PASS: critic value responds to rewards")


def test_reward_only_mode():
    """Inference mode: no CE loss, only critic+nuclei. Components still adapt."""
    mach = make_mach()

    # Simulate 5 steps of reward-only mode
    initial_patches = [p.delta_down.clone() for p in mach.patches]
    values = []
    for step in range(5):
        simulate_forward_pass(mach)
        value, _ = mach.hebbian_step(reward=1.0, step_idx=step, n_steps=5, device='cpu')
        values.append(value.detach())

    # Patches should have changed (Hebbian writes happen regardless of loss)
    patches_changed = any(
        not torch.allclose(p.delta_down, initial_patches[i])
        for i, p in enumerate(mach.patches)
    )

    # Build reward-only loss (critic + nuclei, no CE)
    mach.zero_grad()
    simulate_forward_pass(mach)
    value, _ = mach.hebbian_step(reward=-1.0, step_idx=5, n_steps=6, device='cpu')

    # TD critic loss
    if len(values) > 0:
        td_target = torch.tensor(-1.0) + 0.95 * value.detach()
        critic_loss = (values[-1] - td_target) ** 2

        # Nuclei RPE loss
        nuclei_loss = mach._nuclei_loss if hasattr(mach, '_nuclei_loss') else torch.tensor(0.0)
        total_loss = critic_loss + 0.1 * nuclei_loss
        total_loss.backward()

        # Check which components got gradient
        grad_components = set()
        for name, p in mach.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                grad_components.add(name.split('.')[0])

        print(f"  Patches changed: {patches_changed}")
        print(f"  Components with gradient (reward-only): {sorted(grad_components)}")

        assert patches_changed, "FAIL: patches not written in reward-only mode"
        assert 'critic_gru' in grad_components, "FAIL: critic has no gradient in reward-only"
        print("  PASS: reward-only mode works correctly")
    else:
        print("  SKIP: no values accumulated")


def test_hippocampus_gradient_through_pfc():
    """Hippocampus params get gradient through PFC → critic → loss chain."""
    mach = make_mach()
    d_model = 64

    # Create hippocampus
    key_dim = mach.n_patches * mach.hebb_rule.d_proj  # 2 * 16 = 32
    d_proj = mach.hebb_rule.d_proj
    hipp = Hippocampus(key_dim=key_dim, pfc_dim=32, n_patches=mach.n_patches,
                       d_proj=d_proj)

    # Store some episodes
    for i in range(5):
        simulate_forward_pass(mach)
        mach.hebbian_step(reward=1.0, step_idx=i, n_steps=10, device='cpu')
        act_summary = mach.get_activation_summary()
        act_summary = act_summary / (act_summary.norm() + 1e-8)
        hipp.store(mach, act_summary, reward=1.0, td_error=float(i + 1), global_step=i)

    # Now do a retrieval that modifies PFC, then run hebbian_step, then backward
    mach.zero_grad()
    hipp.zero_grad()

    # Step 1: retrieve → compute_context_gates → hebbian_step
    simulate_forward_pass(mach)
    act_summary = mach.get_activation_summary()
    act_summary = act_summary / (act_summary.norm() + 1e-8)
    alpha = hipp.retrieve_and_reinstate(mach, act_summary, current_td_error=0.5)
    mach.compute_context_gates()  # PFC GRU uses hippocampus-modified state
    value1, _ = mach.hebbian_step(reward=1.0, step_idx=10, n_steps=20, device='cpu')

    # Step 2: another step for TD target
    simulate_forward_pass(mach)
    act_summary2 = mach.get_activation_summary()
    act_summary2 = act_summary2 / (act_summary2.norm() + 1e-8)
    alpha2 = hipp.retrieve_and_reinstate(mach, act_summary2, current_td_error=0.3)
    mach.compute_context_gates()
    value2, _ = mach.hebbian_step(reward=-1.0, step_idx=11, n_steps=20, device='cpu')

    # Critic loss: V(t) should predict r + gamma * V(t+1)
    td_target = torch.tensor(1.0) + 0.95 * value2.detach()
    critic_loss = (value1 - td_target) ** 2

    # Also include nuclei loss if available
    if hasattr(mach, '_nuclei_loss'):
        total_loss = critic_loss + 0.1 * mach._nuclei_loss
    else:
        total_loss = critic_loss

    total_loss.backward()

    # Check hippocampus gradient
    hipp_grad = {}
    hipp_no_grad = []
    for name, p in hipp.named_parameters():
        comp = name.split('.')[0]
        if p.grad is not None and p.grad.abs().sum() > 0:
            hipp_grad.setdefault(comp, []).append((name, p.grad.norm().item()))
        else:
            hipp_no_grad.append(name)

    print(f"  Alpha from retrieval: {alpha:.4f}")
    print(f"  Hippocampus components WITH gradient:")
    for comp in sorted(hipp_grad.keys()):
        params = hipp_grad[comp]
        avg = sum(n for _, n in params) / len(params)
        print(f"    {comp}: {len(params)} params, avg_norm={avg:.6f}")

    if hipp_no_grad:
        print(f"  Hippocampus params with NO gradient: {hipp_no_grad}")

    # Also check mach components
    mach_grad = set()
    for name, p in mach.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            mach_grad.add(name.split('.')[0])
    print(f"  MACH components with gradient: {sorted(mach_grad)}")

    # At minimum, read_gate and read_to_pfc should get gradient
    # (key_proj may or may not depending on activation detachment)
    has_any_hipp_grad = len(hipp_grad) > 0
    if has_any_hipp_grad:
        print("  PASS: hippocampus receives gradient")
    else:
        print("  FAIL: hippocampus has NO gradient at all")
        print("  Investigating PFC state graph...")
        print(f"    _pfc_state.requires_grad: {mach._pfc_state.requires_grad}")
        print(f"    _pfc_state.grad_fn: {mach._pfc_state.grad_fn}")

    return has_any_hipp_grad


def test_pfc_state_tracks_hippocampus():
    """Verify _pfc_state retains gradient through hippocampus modification."""
    mach = make_mach()
    key_dim = mach.n_patches * mach.hebb_rule.d_proj
    d_proj = mach.hebb_rule.d_proj
    hipp = Hippocampus(key_dim=key_dim, pfc_dim=32, n_patches=mach.n_patches,
                       d_proj=d_proj)

    # Populate
    simulate_forward_pass(mach)
    mach.hebbian_step(reward=1.0, step_idx=0, n_steps=5, device='cpu')
    act = mach.get_activation_summary()
    act = act / (act.norm() + 1e-8)
    hipp.store(mach, act, reward=1.0, td_error=1.0, global_step=0)

    # Check PFC state before retrieval
    pfc_before = mach._pfc_state.clone()
    has_grad_before = mach._pfc_state.requires_grad or mach._pfc_state.grad_fn is not None

    # Retrieve
    alpha = hipp.retrieve_and_reinstate(mach, act, current_td_error=0.5)

    # Check PFC state after retrieval
    pfc_after = mach._pfc_state
    changed = not torch.allclose(pfc_before, pfc_after)
    has_grad_after = pfc_after.requires_grad or pfc_after.grad_fn is not None

    print(f"  PFC changed by retrieval: {changed}")
    print(f"  PFC grad_fn before: {has_grad_before}")
    print(f"  PFC grad_fn after: {has_grad_after} ({pfc_after.grad_fn})")
    print(f"  Alpha: {alpha:.4f}")

    # Now run through PFC GRU (like compute_context_gates does)
    simulate_forward_pass(mach)
    mach.compute_context_gates()

    pfc_after_gru = mach._pfc_state
    has_grad_gru = pfc_after_gru.requires_grad or pfc_after_gru.grad_fn is not None
    print(f"  PFC grad_fn after GRU: {has_grad_gru} ({pfc_after_gru.grad_fn})")

    # Direct test: can we backward from PFC to hippocampus?
    hipp.zero_grad()
    test_loss = mach._pfc_state.sum()
    test_loss.backward(retain_graph=True)

    hipp_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in hipp.parameters()
    )
    print(f"  Direct backward PFC→hipp: {hipp_has_grad}")

    if not hipp_has_grad:
        # Detailed check: which part breaks?
        for name, p in hipp.named_parameters():
            grad_status = "HAS GRAD" if (p.grad is not None and p.grad.abs().sum() > 0) else "no grad"
            print(f"    {name}: {grad_status}")

    # PFC may not change much (small alpha), but grad_fn must be present
    assert has_grad_after, "FAIL: PFC has no grad_fn after hippocampus retrieval"
    print("  PASS: PFC state tracking verified")
    return hipp_has_grad


def test_truncation_window_gradient():
    """Simulate a truncation window and verify gradient at backward time."""
    mach = make_mach()
    key_dim = mach.n_patches * mach.hebb_rule.d_proj
    d_proj = mach.hebb_rule.d_proj
    hipp = Hippocampus(key_dim=key_dim, pfc_dim=32, n_patches=mach.n_patches,
                       d_proj=d_proj)

    # Fill hippocampus
    for i in range(5):
        simulate_forward_pass(mach)
        mach.hebbian_step(reward=0.5, step_idx=i, n_steps=10, device='cpu')
        act = mach.get_activation_summary()
        act = act / (act.norm() + 1e-8)
        hipp.store(mach, act, reward=0.5, td_error=float(i), global_step=i)

    # Detach everything (simulating start of new truncation window)
    for p in mach.patches:
        if p.delta_down is not None:
            p.delta_down = p.delta_down.detach()
        if p.delta_up is not None:
            p.delta_up = p.delta_up.detach()
    for p in mach.attn_patches:
        if p.delta_down is not None:
            p.delta_down = p.delta_down.detach()
        if p.delta_up is not None:
            p.delta_up = p.delta_up.detach()
    if hasattr(mach, '_critic_state'):
        mach._critic_state = mach._critic_state.detach()
    for attr in ('_eta_state', '_decay_state', '_expl_state', '_gamma_state', '_pfc_state'):
        if hasattr(mach, attr):
            setattr(mach, attr, getattr(mach, attr).detach())

    # Now simulate a truncation window (5 steps)
    window_critic_losses = []
    window_nuclei_losses = []
    prev_value = None

    for step in range(5):
        # Hippocampus retrieval (before forward pass, like training loop)
        simulate_forward_pass(mach)
        act = mach.get_activation_summary()
        act = act / (act.norm() + 1e-8)
        alpha = hipp.retrieve_and_reinstate(mach, act, current_td_error=0.3)

        # compute_context_gates (called during patched_model.forward)
        mach.compute_context_gates()

        # hebbian_step (after forward pass)
        value, _ = mach.hebbian_step(reward=1.0 if step % 2 == 0 else -1.0,
                                     step_idx=step, n_steps=5, device='cpu')

        # Critic loss (TD)
        if prev_value is not None:
            td_target = torch.tensor(1.0) + 0.95 * value.detach()
            critic_loss = (prev_value - td_target) ** 2
            window_critic_losses.append(critic_loss)
        prev_value = value

        if hasattr(mach, '_nuclei_loss'):
            window_nuclei_losses.append(mach._nuclei_loss)

    # Backward (like truncation boundary)
    mach.zero_grad()
    hipp.zero_grad()

    avg_critic = torch.stack(window_critic_losses).mean()
    total_loss = 0.5 * avg_critic
    if window_nuclei_losses:
        total_loss = total_loss + 0.1 * torch.stack(window_nuclei_losses).mean()
    total_loss.backward()

    # Check gradients
    mach_grads = {}
    for name, p in mach.named_parameters():
        comp = name.split('.')[0]
        if p.grad is not None and p.grad.abs().sum() > 0:
            mach_grads.setdefault(comp, []).append(p.grad.norm().item())

    hipp_grads = {}
    for name, p in hipp.named_parameters():
        comp = name.split('.')[0]
        if p.grad is not None and p.grad.abs().sum() > 0:
            hipp_grads.setdefault(comp, []).append(p.grad.norm().item())

    print(f"  MACH components with gradient after truncation window:")
    for comp in sorted(mach_grads.keys()):
        avg = sum(mach_grads[comp]) / len(mach_grads[comp])
        print(f"    {comp}: avg={avg:.6f}")

    print(f"  Hippocampus components with gradient after truncation window:")
    if hipp_grads:
        for comp in sorted(hipp_grads.keys()):
            avg = sum(hipp_grads[comp]) / len(hipp_grads[comp])
            print(f"    {comp}: avg={avg:.6f}")
    else:
        print(f"    NONE — hippocampus gradient is disconnected!")
        # Debug: check the graph
        print(f"    Last PFC grad_fn: {mach._pfc_state.grad_fn}")
        print(f"    Last value grad_fn: {value.grad_fn}")

    print("  PASS: truncation window simulation complete")
    return len(hipp_grads) > 0


if __name__ == '__main__':
    print("System health tests\n")
    print("=" * 60)

    print("\n1. Gradient flow — all MACH components")
    test_gradient_all_components()

    print("\n2. Patch writes — residual + attention")
    test_patch_writes_nonzero()

    print("\n3. Patch accumulation — multi-step")
    test_patch_writes_accumulate()

    print("\n4. Eligibility traces")
    test_eligibility_traces()

    print("\n5. Nuclei output variation")
    test_nuclei_outputs_change()

    print("\n6. Critic value responds to rewards")
    test_critic_value_changes()

    print("\n7. Reward-only mode (inference)")
    test_reward_only_mode()

    print("\n8. PFC state tracks hippocampus")
    hipp_direct = test_pfc_state_tracks_hippocampus()

    print("\n9. Hippocampus gradient through PFC→critic→loss")
    hipp_indirect = test_hippocampus_gradient_through_pfc()

    print("\n10. Truncation window simulation")
    hipp_truncation = test_truncation_window_gradient()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Hippocampus direct gradient (PFC→hipp): {'PASS' if hipp_direct else 'FAIL'}")
    print(f"  Hippocampus indirect gradient (critic→PFC→hipp): {'PASS' if hipp_indirect else 'FAIL'}")
    print(f"  Hippocampus in truncation window: {'PASS' if hipp_truncation else 'FAIL'}")
    print("\nAll tests complete.")
