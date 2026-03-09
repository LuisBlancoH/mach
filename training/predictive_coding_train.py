"""
Direct backprop training for predictive coding patches.

Stripped-down training loop — no Hebbian, no nuclei, no PFC, no hippocampus.
Pure gradient descent on CE loss + prediction loss, to verify the predictive
coding architecture works before adding brain machinery.
"""

import copy
import random
import time

import torch


def train_predictive_coding(patched_model, pc_network, tokenizer,
                            train_problems, test_problems, device,
                            epochs=20, lr=1e-4, patience=5,
                            prediction_weight=0.1, grad_clip=1.0,
                            log_interval=50, eval_fn=None):
    """Train predictive coding patches via direct backpropagation.

    Args:
        patched_model: PredictiveCodingPatchedModel
        pc_network: PredictiveCodingNetwork
        tokenizer: Qwen tokenizer
        train_problems: list of {prompt, answer} dicts
        test_problems: list of {prompt, answer} dicts
        device: torch device
        epochs: number of training epochs
        lr: learning rate
        patience: early stopping patience
        prediction_weight: weight of auxiliary prediction loss
        grad_clip: max gradient norm
        eval_fn: function(model, tokenizer, problems, label) → accuracy
    """
    optimizer = torch.optim.Adam(pc_network.parameters(), lr=lr)
    global_step = 0

    best_accuracy = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        random.shuffle(train_problems)
        total_ce_loss = 0.0
        total_pred_loss = 0.0
        n_problems = 0

        pc_network.train()
        epoch_start = time.time()

        for i, p in enumerate(train_problems):
            full_text = p["prompt"] + p["answer"]
            encoding = tokenizer(full_text, return_tensors="pt").to(device)
            input_ids = encoding.input_ids

            prompt_len = len(tokenizer(p["prompt"]).input_ids)
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100

            # Forward pass (hooks apply predictive coding corrections)
            outputs = patched_model(input_ids=input_ids, labels=labels)
            ce_loss = outputs.loss

            # Auxiliary prediction loss: drive hierarchy to predict accurately
            pred_loss = pc_network.get_prediction_loss()
            total_loss = ce_loss + prediction_weight * pred_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(pc_network.parameters(), grad_clip)
            optimizer.step()

            total_ce_loss += ce_loss.item()
            total_pred_loss += pred_loss.item()
            n_problems += 1
            global_step += 1

            if (i + 1) % log_interval == 0 or (i + 1) == len(train_problems):
                avg_ce = total_ce_loss / n_problems
                avg_pred = total_pred_loss / n_problems
                elapsed = time.time() - epoch_start
                rate = n_problems / elapsed if elapsed > 0 else 0
                print(
                    f"\r  [epoch {epoch}] {i+1}/{len(train_problems)} "
                    f"ce={avg_ce:.4f} pred={avg_pred:.4f} "
                    f"[{rate:.1f} p/s]",
                    end="", flush=True
                )

        print()  # newline after progress

        avg_ce = total_ce_loss / n_problems
        avg_pred = total_pred_loss / n_problems

        # Diagnostics
        diag = pc_network.get_diagnostics()
        print(f"  Epoch {epoch}: ce={avg_ce:.4f} pred={avg_pred:.4f}")
        for key in sorted(diag.keys()):
            print(f"    {key}: {diag[key]:.4f}")

        # Gradient norms
        for name, param in pc_network.named_parameters():
            if param.grad is not None:
                print(f"    grad/{name}: {param.grad.norm().item():.6f}")

        # Evaluate
        if eval_fn is not None:
            pc_network.eval()
            accuracy = eval_fn(patched_model, tokenizer, test_problems,
                               label=f"ep{epoch}")
            print(f"  Accuracy: {accuracy:.2%}")
            pc_network.train()

            # Early stopping
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = copy.deepcopy(pc_network.state_dict())
                epochs_without_improvement = 0
                print(f"  ** New best: {best_accuracy:.2%}")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement ({epochs_without_improvement}/{patience})")
                if epochs_without_improvement >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    # Restore best
    if best_state is not None:
        pc_network.load_state_dict(best_state)
        print(f"  Restored best (accuracy: {best_accuracy:.2%})")

    return best_accuracy


def train_predictive_coding_continuous(patched_model, pc_network, tokenizer,
                                       device, n_steps=10000, lr=1e-4,
                                       prediction_weight=0.1, grad_clip=1.0,
                                       checkpoint_path=None, save_path=None):
    """Continuous training on diverse operations (no episodes).

    Like meta_train_continuous but without Hebbian/nuclei/PFC/hippocampus.
    Problems arrive in a stream, PC patches learn via direct backprop.
    """
    from training.two_channel_train import (
        DIVERSE_TRAIN_OPS, generate_few_shot_episode
    )
    from data.arithmetic import extract_number

    optimizer = torch.optim.Adam(pc_network.parameters(), lr=lr)

    # Load checkpoint if exists
    start_step = 0
    if checkpoint_path and save_path:
        import os
        if os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location=device, weights_only=False)
            pc_network.load_state_dict(ckpt['pc_network'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_step = ckpt.get('step', 0)
            print(f"  Resumed from step {start_step}")

    pc_network.train()
    current_op = random.choice(DIVERSE_TRAIN_OPS)
    op_step_count = 0
    op_switch_interval = 20

    all_correct = []
    step_timer = time.time()

    for step in range(start_step, n_steps):
        # Switch operation periodically
        op_step_count += 1
        if op_step_count >= op_switch_interval:
            current_op = random.choice(DIVERSE_TRAIN_OPS)
            op_step_count = 0

        # Generate one problem
        problems = generate_few_shot_episode(1, n_demos=0, op_type=current_op)
        p = problems[0]

        full_text = p["prompt"] + p["answer"]
        encoding = tokenizer(full_text, return_tensors="pt").to(device)
        input_ids = encoding.input_ids

        prompt_len = len(tokenizer(p["prompt"]).input_ids)
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100

        # Forward + backward
        outputs = patched_model(input_ids=input_ids, labels=labels)
        ce_loss = outputs.loss
        pred_loss = pc_network.get_prediction_loss()
        total_loss = ce_loss + prediction_weight * pred_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(pc_network.parameters(), grad_clip)
        optimizer.step()

        # Check accuracy (greedy generation)
        with torch.no_grad():
            prompt_ids = tokenizer(p["prompt"], return_tensors="pt").input_ids.to(device)
            gen = patched_model.generate(
                prompt_ids, max_new_tokens=20,
                do_sample=False, temperature=None, top_p=None
            )
            gen_text = tokenizer.decode(gen[0][prompt_ids.shape[1]:], skip_special_tokens=True)
            pred_num = extract_number(gen_text)
            correct = (pred_num == p.get("numerical_answer"))
            all_correct.append(1.0 if correct else 0.0)

        # Logging
        if (step + 1) % 100 == 0:
            recent = all_correct[-100:]
            acc = sum(recent) / len(recent)
            elapsed = time.time() - step_timer
            rate = 100 / elapsed if elapsed > 0 else 0
            step_timer = time.time()

            print(
                f"Step {step+1:5d} | op={current_op:<10} | "
                f"acc(100)={acc:.0%} ce={ce_loss.item():.3f} "
                f"pred={pred_loss.item():.4f} [{rate:.1f} st/s]"
            )

        # Diagnostics + checkpoint
        if (step + 1) % 2000 == 0:
            diag = pc_network.get_diagnostics()
            print("  Diagnostics:")
            for key in sorted(diag.keys()):
                print(f"    {key}: {diag[key]:.4f}")

            # Gradient norms
            print("  Gradient norms:")
            for name, param in pc_network.named_parameters():
                if param.grad is not None:
                    print(f"    {name}: {param.grad.norm().item():.6f}")

            # Save checkpoint
            if save_path:
                torch.save({
                    'pc_network': pc_network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step + 1,
                }, save_path)
                print(f"  Checkpoint saved to {save_path}")

            # Quick eval
            print(f"  --- Validation (step {step+1}) ---")
            eval_ops = DIVERSE_TRAIN_OPS[:6]
            pc_network.eval()
            for eval_op in eval_ops:
                eval_problems = generate_few_shot_episode(
                    20, n_demos=0, op_type=eval_op
                )
                n_correct = 0
                for ep in eval_problems:
                    with torch.no_grad():
                        prompt_ids = tokenizer(
                            ep["prompt"], return_tensors="pt"
                        ).input_ids.to(device)
                        gen = patched_model.generate(
                            prompt_ids, max_new_tokens=20,
                            do_sample=False, temperature=None, top_p=None
                        )
                        gen_text = tokenizer.decode(
                            gen[0][prompt_ids.shape[1]:],
                            skip_special_tokens=True
                        )
                        pred_num = extract_number(gen_text)
                        if pred_num == ep.get("numerical_answer"):
                            n_correct += 1
                print(f"  EVAL {eval_op:<10} | {n_correct}/{len(eval_problems)} = {n_correct/len(eval_problems):.0%}")
            pc_network.train()
