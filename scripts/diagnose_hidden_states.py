#!/usr/bin/env python3
"""
Diagnostic: do Qwen's hidden states at the last demo token encode
coefficient magnitude, or just operation type?

Compares UNPROMPTED (raw demos) vs PROMPTED ("The pattern is:") to see
if prompting Qwen to reason about the rule improves coefficient encoding.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = {
    "unprompted": "",
    "pattern": "\nThe pattern is:",
    "rule": "\nRule: output =",
    "next": "\nFollowing the same rule,",
}


def generate_demo_text(c1, c2, n_demos=5):
    """Generate demo string for a given coefficient pair."""
    lines = []
    for _ in range(n_demos):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        answer = c1 * a + c2 * b
        lines.append(f"{a} ? {b} = {answer}")
    return "\n".join(lines)


def extract_hidden_states(model, tokenizer, text, layer_indices, device):
    """Run text through Qwen and extract last-token hidden states."""
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    hidden_states = {}
    hooks = []

    with torch.no_grad():
        for layer_idx in layer_indices:
            def make_hook(idx):
                def hook(module, input, output):
                    t = output[0] if isinstance(output, tuple) else output
                    hidden_states[idx] = t[:, -1, :].cpu()
                return hook
            h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(h)
        model(input_ids=input_ids)
        for h in hooks:
            h.remove()

    return hidden_states


def analyze(X, all_c1, all_c2, all_labels, n_samples_per_coeff):
    """Run linear probe and cosine similarity analysis. Returns dict of metrics."""
    n_total = len(all_c1)

    # PCA
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)

    # Train/test split
    train_mask = np.array([i % n_samples_per_coeff < 15 for i in range(n_total)])
    test_mask = ~train_mask

    X_train_pca = X_pca[train_mask, :10]
    X_test_pca = X_pca[test_mask, :10]

    reg_c1 = Ridge(alpha=1.0).fit(X_train_pca, all_c1[train_mask])
    reg_c2 = Ridge(alpha=1.0).fit(X_train_pca, all_c2[train_mask])

    r2_c1 = r2_score(all_c1[test_mask], reg_c1.predict(X_test_pca))
    r2_c2 = r2_score(all_c2[test_mask], reg_c2.predict(X_test_pca))

    mae_c1 = np.mean(np.abs(all_c1[test_mask] - reg_c1.predict(X_test_pca)))
    mae_c2 = np.mean(np.abs(all_c2[test_mask] - reg_c2.predict(X_test_pca)))

    # Cosine similarity
    cos_sim = cosine_similarity(X)
    within_sims = []
    across_sims = []
    for i in range(n_total):
        for j in range(i + 1, n_total):
            if all_labels[i] == all_labels[j]:
                within_sims.append(cos_sim[i, j])
            else:
                across_sims.append(cos_sim[i, j])

    # Scaling variant similarity
    scale_sims = {}
    for l1, l2 in [("1a+0b", "2a+0b"), ("1a+0b", "3a+0b"), ("2a+0b", "3a+0b"),
                    ("0a+1b", "0a+2b"), ("0a+1b", "0a+3b"), ("0a+2b", "0a+3b")]:
        idx1 = [i for i, l in enumerate(all_labels) if l == l1]
        idx2 = [i for i, l in enumerate(all_labels) if l == l2]
        if idx1 and idx2:
            sims = [cos_sim[i, j] for i in idx1 for j in idx2]
            scale_sims[f"{l1}_vs_{l2}"] = np.mean(sims)

    return {
        "r2_c1": r2_c1, "r2_c2": r2_c2,
        "mae_c1": mae_c1, "mae_c2": mae_c2,
        "within_sim": np.mean(within_sims),
        "across_sim": np.mean(across_sims),
        "gap": np.mean(within_sims) - np.mean(across_sims),
        "scale_sims": scale_sims,
        "pca_var_10": sum(pca.explained_variance_ratio_[:10]),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading Qwen3-4B...")
    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    coeff_pairs = [
        (0, 1), (0, 2), (0, 3),
        (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 2), (1, 2), (2, 1),
    ]

    layer_indices = [9, 18, 27, 34]
    n_samples_per_coeff = 20

    # Fix random seed for fair comparison across prompt types
    base_seed = 42

    for prompt_name, prompt_suffix in PROMPTS.items():
        print(f"\n{'#'*70}")
        print(f"# PROMPT: {prompt_name!r}  suffix={prompt_suffix!r}")
        print(f"{'#'*70}")

        all_hidden = {layer_idx: [] for layer_idx in layer_indices}
        all_c1 = []
        all_c2 = []
        all_labels = []

        random.seed(base_seed)  # same demos for each prompt type

        for c1, c2 in coeff_pairs:
            for _ in range(n_samples_per_coeff):
                demo_text = generate_demo_text(c1, c2, n_demos=5)
                full_text = demo_text + prompt_suffix

                hidden_states = extract_hidden_states(
                    model, tokenizer, full_text, layer_indices, device
                )
                for layer_idx in layer_indices:
                    all_hidden[layer_idx].append(
                        hidden_states[layer_idx].float().squeeze(0)
                    )
                all_c1.append(c1)
                all_c2.append(c2)
                all_labels.append(f"{c1}a+{c2}b")

        all_c1 = np.array(all_c1)
        all_c2 = np.array(all_c2)

        for layer_idx in layer_indices:
            X = torch.stack(all_hidden[layer_idx]).numpy()
            metrics = analyze(X, all_c1, all_c2, all_labels, n_samples_per_coeff)

            print(f"\n  Layer {layer_idx}:")
            print(f"    R²  c1={metrics['r2_c1']:.4f}  c2={metrics['r2_c2']:.4f}  "
                  f"| MAE  c1={metrics['mae_c1']:.4f}  c2={metrics['mae_c2']:.4f}")
            print(f"    Cosine gap={metrics['gap']:.4f}  "
                  f"(within={metrics['within_sim']:.4f}  across={metrics['across_sim']:.4f})")
            print(f"    PCA top-10 var={metrics['pca_var_10']:.4f}")

            # Compact scaling variant display
            ss = metrics['scale_sims']
            a_sims = [ss.get(k, 0) for k in ["1a+0b_vs_2a+0b", "1a+0b_vs_3a+0b", "2a+0b_vs_3a+0b"]]
            b_sims = [ss.get(k, 0) for k in ["0a+1b_vs_0a+2b", "0a+1b_vs_0a+3b", "0a+2b_vs_0a+3b"]]
            print(f"    Scale sims: a-variants={np.mean(a_sims):.4f}  b-variants={np.mean(b_sims):.4f}")

    # Bonus: check what Qwen actually generates after each prompt
    print(f"\n{'#'*70}")
    print(f"# GENERATION CHECK: what does Qwen say after each prompt?")
    print(f"{'#'*70}")

    random.seed(99)
    for c1, c2 in [(1, 0), (2, 0), (0, 1), (1, 1)]:
        demo_text = generate_demo_text(c1, c2, n_demos=5)
        print(f"\n  Demos ({c1}a+{c2}b):")
        for line in demo_text.split("\n"):
            print(f"    {line}")

        for prompt_name, prompt_suffix in PROMPTS.items():
            if not prompt_suffix:
                continue
            full_text = demo_text + prompt_suffix
            input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model.generate(
                    input_ids, max_new_tokens=30, do_sample=False,
                    temperature=1.0
                )
            generated = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            print(f"    {prompt_name}: {generated.strip()[:80]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
