#!/usr/bin/env python3
"""
Minimal viable brain: can a predictive coding network learn arithmetic?

No Qwen. No transformers. No backprop in the core.
Just a hierarchy of cortical layers learning through prediction errors
and reward modulation.

Usage:
    # Single-digit addition (easiest test)
    python scripts/run_brain.py

    # Harder
    python scripts/run_brain.py --ops add,sub --max-num 20 --n-steps 50000

    # All operations
    python scripts/run_brain.py --ops add,sub,mul --n-steps 100000
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from models.brain import Brain


# === Simple tokenizer for arithmetic ===

class ArithmeticTokenizer:
    """Minimal tokenizer for arithmetic expressions.

    Tokens: 0-9, +, -, *, /, =, <pad>, <eos>
    """

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

        # Digits
        for i in range(10):
            self.token_to_id[str(i)] = i
            self.id_to_token[i] = str(i)

        # Operators and special tokens
        specials = ['+', '-', '*', '/', '=', ' ', '<pad>', '<eos>']
        for i, s in enumerate(specials):
            idx = 10 + i
            self.token_to_id[s] = idx
            self.id_to_token[idx] = s

        self.pad_id = self.token_to_id['<pad>']
        self.eos_id = self.token_to_id['<eos>']
        self.vocab_size = len(self.token_to_id)

    def encode(self, text):
        """Encode text to token IDs."""
        ids = []
        for char in text:
            if char in self.token_to_id:
                ids.append(self.token_to_id[char])
        ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        """Decode token IDs to text."""
        chars = []
        for i in ids:
            if i == self.eos_id:
                break
            if i == self.pad_id:
                continue
            if i in self.id_to_token:
                chars.append(self.id_to_token[i])
        return ''.join(chars)


# === Problem generation ===

def generate_problem(op='add', max_num=9):
    """Generate a single arithmetic problem.

    Returns:
        prompt_str: e.g. "3 + 4 = "
        answer_str: e.g. "7"
        numerical_answer: 7
    """
    a = random.randint(1, max_num)
    b = random.randint(1, max_num)

    if op == 'add':
        result = a + b
        prompt = f"{a} + {b} = "
    elif op == 'sub':
        # Ensure non-negative
        if a < b:
            a, b = b, a
        result = a - b
        prompt = f"{a} - {b} = "
    elif op == 'mul':
        result = a * b
        prompt = f"{a} * {b} = "
    else:
        raise ValueError(f"Unknown op: {op}")

    return prompt, str(result), result


# === Training ===

def main():
    parser = argparse.ArgumentParser(description="Minimal Viable Brain")
    parser.add_argument("--ops", type=str, default="add",
                        help="Comma-separated operations: add,sub,mul")
    parser.add_argument("--max-num", type=int, default=9,
                        help="Maximum number in problems")
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--d-cortical", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-settle", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for genome (embed/readout)")
    parser.add_argument("--hebbian-eta", type=float, default=0.001,
                        help="Hebbian learning rate for cortical layers")
    args = parser.parse_args()

    ops = args.ops.split(',')
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    print(f"=== Minimal Viable Brain ===")
    print(f"  Device: {device}")
    print(f"  Ops: {ops}, max_num: {args.max_num}")
    print(f"  d_cortical: {args.d_cortical}, n_layers: {args.n_layers}")
    print(f"  n_settle: {args.n_settle}")

    # Build
    tokenizer = ArithmeticTokenizer()
    brain = Brain(
        vocab_size=tokenizer.vocab_size,
        d_embed=64,
        d_cortical=args.d_cortical,
        n_layers=args.n_layers,
        n_settle=args.n_settle,
        max_seq_len=16,  # "99 + 99 = " is 10 chars + eos
    ).to(device)

    n_params = sum(p.numel() for p in brain.parameters())
    n_buffers = sum(b.numel() for b in brain.buffers())
    print(f"  Params (genome): {n_params:,}")
    print(f"  Buffers (plastic brain): {n_buffers:,}")
    print()

    # Optimizer for genome only (embed, readout, critic, norms, precision)
    genome_params = list(brain.embed.parameters()) + \
                    list(brain.pos_embed.parameters()) + \
                    list(brain.seq_compress.parameters()) + \
                    list(brain.readout.parameters()) + \
                    list(brain.critic.parameters())
    # Also include layer norms and precision (they're Parameters)
    for layer in brain.layers:
        genome_params.append(layer.precision_logit)
        genome_params.extend(layer.norm.parameters())

    optimizer = torch.optim.Adam(genome_params, lr=args.lr)

    # Training loop
    all_correct = []
    step_timer = time.time()

    for step in range(args.n_steps):
        op = random.choice(ops)
        prompt_str, answer_str, numerical_answer = generate_problem(
            op=op, max_num=args.max_num
        )

        # Encode prompt and answer
        prompt_ids = tokenizer.encode(prompt_str)
        answer_ids = tokenizer.encode(answer_str)  # includes <eos>

        # Train on each token of the answer autoregressively
        # Context grows: prompt → prompt+digit1 → prompt+digit1+digit2 → ...
        ce_loss = torch.tensor(0.0, device=device)
        context_ids = list(prompt_ids)
        all_pred_chars = []

        for target_id in answer_ids:
            context_tensor = torch.tensor([context_ids], device=device)
            brain.reset()
            logits = brain(context_tensor)
            ce_loss = ce_loss + F.cross_entropy(logits, torch.tensor([target_id], device=device))

            pred_id = logits.argmax(dim=-1).item()
            all_pred_chars.append(tokenizer.id_to_token.get(pred_id, '?'))

            # Teacher forcing: append true token for next step
            context_ids.append(target_id)

        ce_loss = ce_loss / len(answer_ids)

        # Gradient step for genome
        optimizer.zero_grad()
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(genome_params, 1.0)
        optimizer.step()

        # Check if full answer is correct
        pred_answer = ''.join(all_pred_chars).replace('<eos>', '')
        correct = (pred_answer == answer_str)
        all_correct.append(1.0 if correct else 0.0)

        # Reward: graded — partial credit for getting digits right
        if correct:
            reward = 1.0
        elif len(pred_answer) > 0 and pred_answer[0] == answer_str[0]:
            reward = 0.0  # first digit right, not punished
        else:
            reward = -0.5

        # Hebbian update (the brain learning)
        brain.hebbian_step(reward)

        # Logging
        if (step + 1) % 100 == 0:
            recent = all_correct[-100:]
            acc = sum(recent) / len(recent)
            elapsed = time.time() - step_timer
            rate = 100 / elapsed if elapsed > 0 else 0
            step_timer = time.time()

            print(
                f"Step {step+1:5d} | op={op:<5} | "
                f"acc(100)={acc:.0%} ce={ce_loss.item():.3f} "
                f"[{rate:.0f} st/s]"
            )

        # Diagnostics
        if (step + 1) % 2000 == 0:
            diag = brain.get_diagnostics()
            print("  Diagnostics:")
            for key in sorted(diag.keys()):
                val = diag[key]
                print(f"    {key}: {val:.4f}")

            # Eval: test on all problems for each op
            print(f"\n  --- Eval (step {step+1}) ---")
            for eval_op in ops:
                n_correct = 0
                n_total = 0
                examples = []
                for a in range(1, args.max_num + 1):
                    for b in range(1, args.max_num + 1):
                        if eval_op == 'sub' and a < b:
                            continue
                        if eval_op == 'add':
                            ans = a + b
                            p = f"{a} + {b} = "
                        elif eval_op == 'sub':
                            ans = a - b
                            p = f"{a} - {b} = "
                        elif eval_op == 'mul':
                            ans = a * b
                            p = f"{a} * {b} = "

                        ids = tokenizer.encode(p)
                        t = torch.tensor([ids], device=device)
                        gen_ids = brain.generate(t, max_new_tokens=4)
                        pred_str = tokenizer.decode(gen_ids)
                        ans_str = str(ans)
                        if pred_str == ans_str:
                            n_correct += 1
                        n_total += 1

                        # Collect a few examples
                        if len(examples) < 5:
                            mark = "✓" if pred_str == ans_str else "✗"
                            examples.append(f"    {p}{ans_str} → {pred_str} {mark}")

                print(f"  EVAL {eval_op:<5} | {n_correct}/{n_total} = "
                      f"{n_correct/n_total:.0%}")
                for ex in examples:
                    print(ex)
            print()


if __name__ == "__main__":
    main()
