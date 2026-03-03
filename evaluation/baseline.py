import sys

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from data.arithmetic import extract_number


class StopOnNonDigit(StoppingCriteria):
    """Stop generation as soon as a non-digit token is produced."""
    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] <= self.prompt_len:
            return False
        last_token = self.tokenizer.decode(input_ids[0, -1])
        return not last_token.strip().isdigit()


def evaluate_model(model, tokenizer, problems, max_new_tokens=10, label=None):
    """Evaluate a model on arithmetic problems. Returns accuracy as a float."""
    correct = 0
    total = len(problems)
    prefix = f"  [{label}]" if label else "  [eval]"

    for i, p in enumerate(problems):
        encoding = tokenizer(p["prompt"], return_tensors="pt").to(model.device)
        prompt_len = encoding.input_ids.shape[1]
        with torch.no_grad():
            output = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([StopOnNonDigit(tokenizer, prompt_len)]),
            )
        response = tokenizer.decode(output[0][encoding.input_ids.shape[1]:], skip_special_tokens=True).strip()
        predicted = extract_number(response)
        if predicted == p["answer"]:
            correct += 1

        if (i + 1) % 50 == 0 or (i + 1) == total:
            acc_so_far = correct / (i + 1)
            print(f"\r{prefix} {i+1}/{total}  acc={acc_so_far:.2%}", end="", flush=True)

    print()
    return correct / total
