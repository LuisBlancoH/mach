import random

import torch

from evaluation.baseline import evaluate_model


def train_patches_direct(patched_model, tokenizer, train_problems, test_problems,
                         device, epochs=20, lr=1e-4):
    """Train cortical patches via direct backpropagation on arithmetic problems."""
    optimizer = torch.optim.Adam(patched_model.patches.parameters(), lr=lr)

    for epoch in range(epochs):
        random.shuffle(train_problems)
        total_loss = 0.0
        n_problems = 0

        for p in train_problems:
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

        avg_loss = total_loss / n_problems
        accuracy = evaluate_model(patched_model, tokenizer, test_problems)

        print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}, accuracy = {accuracy:.2%}")

        # Diagnostics: patch gradient norms
        for name, param in patched_model.patches.named_parameters():
            if param.grad is not None:
                print(f"  {name}: grad norm = {param.grad.norm():.6f}")

    return patched_model
