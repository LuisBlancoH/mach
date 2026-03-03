import sys

import torch

from data.arithmetic import extract_number


def evaluate_model(model, tokenizer, problems, max_new_tokens=10, label=None):
    """Evaluate a model on arithmetic problems. Returns accuracy as a float."""
    correct = 0
    total = len(problems)
    prefix = f"  [{label}]" if label else "  [eval]"

    for i, p in enumerate(problems):
        input_ids = tokenizer(p["prompt"], return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        predicted = extract_number(response)
        if predicted == p["answer"]:
            correct += 1

        if (i + 1) % 50 == 0 or (i + 1) == total:
            acc_so_far = correct / (i + 1)
            print(f"\r{prefix} {i+1}/{total}  acc={acc_so_far:.2%}", end="", flush=True)

    print()
    return correct / total
