import copy
import random

import torch
import wandb

from evaluation.baseline import evaluate_model


def train_patches_direct(patched_model, tokenizer, train_problems, test_problems,
                         device, difficulty, epochs=20, lr=1e-4, patience=3):
    """Train cortical patches via direct backpropagation on arithmetic problems."""
    optimizer = torch.optim.Adam(patched_model.patches.parameters(), lr=lr)
    global_step = 0

    best_accuracy = 0.0
    best_patch_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        random.shuffle(train_problems)
        total_loss = 0.0
        n_problems = 0

        n_total = len(train_problems)
        for i, p in enumerate(train_problems):
            full_text = p["prompt"] + p["answer"]
            encoding = tokenizer(full_text, return_tensors="pt").to(device)
            input_ids = encoding.input_ids

            prompt_len = len(tokenizer(p["prompt"]).input_ids)
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100

            outputs = patched_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_problems += 1
            global_step += 1

            if (i + 1) % 50 == 0 or (i + 1) == n_total:
                avg_so_far = total_loss / n_problems
                print(f"\r  [train d{difficulty} ep{epoch}] {i+1}/{n_total}  loss={avg_so_far:.4f}", end="", flush=True)

            if global_step % 100 == 0:
                wandb.log({
                    f"diff{difficulty}/train_loss_avg": total_loss / n_problems,
                    f"diff{difficulty}/train_loss_step": loss.item(),
                    "global_step": global_step,
                })

        print()

        avg_loss = total_loss / n_problems
        accuracy = evaluate_model(patched_model, tokenizer, test_problems, label=f"d{difficulty} ep{epoch}")

        print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}, accuracy = {accuracy:.2%}")

        # Log epoch metrics
        log_dict = {
            f"diff{difficulty}/avg_loss": avg_loss,
            f"diff{difficulty}/accuracy": accuracy,
            f"diff{difficulty}/epoch": epoch,
        }

        # Diagnostics: patch gradient norms
        for name, param in patched_model.patches.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name}: grad norm = {grad_norm:.6f}")
                log_dict[f"diff{difficulty}/grad_norm/{name}"] = grad_norm

        wandb.log(log_dict)

        # Early stopping: save best, stop after `patience` epochs without improvement
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_patch_state = copy.deepcopy(patched_model.patches.state_dict())
            epochs_without_improvement = 0
            print(f"  ** New best accuracy: {best_accuracy:.2%}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best patches
    if best_patch_state is not None:
        patched_model.patches.load_state_dict(best_patch_state)
        print(f"  Restored best patches (accuracy: {best_accuracy:.2%})")

    return patched_model
