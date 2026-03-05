#!/usr/bin/env python3
"""
Diagnostic: do Qwen's hidden states at the last demo token encode
coefficient magnitude, or just operation type?

Generates demos for several coefficient pairs, runs them through frozen Qwen,
extracts last-token hidden states from multiple layers, and checks:
1. Do hidden states cluster by (c1,c2) pair? (PCA + cosine similarity)
2. Can a linear probe predict c1 and c2 from the hidden state?
3. How much variance is explained by operation type vs coefficient magnitude?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_demo_text(c1, c2, n_demos=5):
    """Generate demo string for a given coefficient pair."""
    import random
    lines = []
    for _ in range(n_demos):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        answer = c1 * a + c2 * b
        lines.append(f"{a} ? {b} = {answer}")
    return "\n".join(lines)


def extract_hidden_states(model, tokenizer, demo_text, layer_indices, device):
    """Run demo text through Qwen and extract last-token hidden states."""
    input_ids = tokenizer(demo_text, return_tensors="pt").input_ids.to(device)
    hidden_states = {}
    full_seqs = {}
    hooks = []

    with torch.no_grad():
        for layer_idx in layer_indices:
            def make_hook(idx):
                def hook(module, input, output):
                    t = output[0] if isinstance(output, tuple) else output
                    hidden_states[idx] = t[:, -1, :].cpu()  # last token
                    full_seqs[idx] = t[0].cpu()  # full sequence
                return hook
            h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(h)
        model(input_ids=input_ids)
        for h in hooks:
            h.remove()

    return hidden_states, full_seqs, input_ids.shape[1]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("Loading Qwen3-4B...")
    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    # Coefficient pairs to test
    coeff_pairs = [
        (0, 1), (0, 2), (0, 3),  # b-only with different scales
        (1, 0), (2, 0), (3, 0),  # a-only with different scales
        (1, 1), (2, 2), (1, 2), (2, 1),  # mixed
    ]

    layer_indices = [9, 18, 27, 34]  # standard patch layers
    n_samples_per_coeff = 20  # different random demos per coeff pair

    print(f"\nGenerating {len(coeff_pairs)} coeff pairs x {n_samples_per_coeff} samples each")
    print(f"Layers: {layer_indices}\n")

    # Collect hidden states
    all_hidden = {layer_idx: [] for layer_idx in layer_indices}
    all_c1 = []
    all_c2 = []
    all_labels = []  # string label for each sample

    for c1, c2 in coeff_pairs:
        for _ in range(n_samples_per_coeff):
            demo_text = generate_demo_text(c1, c2, n_demos=5)
            hidden_states, _, seq_len = extract_hidden_states(
                model, tokenizer, demo_text, layer_indices, device
            )
            for layer_idx in layer_indices:
                all_hidden[layer_idx].append(hidden_states[layer_idx].float().squeeze(0))
            all_c1.append(c1)
            all_c2.append(c2)
            all_labels.append(f"{c1}a+{c2}b")

    all_c1 = np.array(all_c1)
    all_c2 = np.array(all_c2)
    n_total = len(all_c1)

    print(f"Collected {n_total} samples\n")

    # Analysis per layer
    for layer_idx in layer_indices:
        print(f"{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        X = torch.stack(all_hidden[layer_idx]).numpy()  # (n_total, d_model)

        # 1. PCA — how much variance in first few components?
        pca = PCA(n_components=10)
        X_pca = pca.fit_transform(X)
        print(f"\nPCA explained variance (first 10):")
        for i, v in enumerate(pca.explained_variance_ratio_[:10]):
            print(f"  PC{i+1}: {v:.4f} ({sum(pca.explained_variance_ratio_[:i+1]):.4f} cumulative)")

        # 2. Linear probe: can we predict c1, c2 from hidden state?
        # Train/test split: first 15 samples per coeff = train, last 5 = test
        train_mask = np.array([i % n_samples_per_coeff < 15 for i in range(n_total)])
        test_mask = ~train_mask

        X_train, X_test = X[train_mask], X[test_mask]
        c1_train, c1_test = all_c1[train_mask], all_c1[test_mask]
        c2_train, c2_test = all_c2[train_mask], all_c2[test_mask]

        # Use PCA features for stability
        X_train_pca = X_pca[train_mask, :10]
        X_test_pca = X_pca[test_mask, :10]

        reg_c1 = Ridge(alpha=1.0).fit(X_train_pca, c1_train)
        reg_c2 = Ridge(alpha=1.0).fit(X_train_pca, c2_train)

        c1_pred = reg_c1.predict(X_test_pca)
        c2_pred = reg_c2.predict(X_test_pca)

        r2_c1 = r2_score(c1_test, c1_pred)
        r2_c2 = r2_score(c2_test, c2_pred)

        mae_c1 = np.mean(np.abs(c1_test - c1_pred))
        mae_c2 = np.mean(np.abs(c2_test - c2_pred))

        print(f"\nLinear probe (Ridge on top-10 PCs):")
        print(f"  c1: R²={r2_c1:.4f}, MAE={mae_c1:.4f}")
        print(f"  c2: R²={r2_c2:.4f}, MAE={mae_c2:.4f}")

        # 3. Cosine similarity within vs across coefficient pairs
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim = cosine_similarity(X)

        within_sims = []
        across_sims = []
        for i in range(n_total):
            for j in range(i+1, n_total):
                sim = cos_sim[i, j]
                if all_labels[i] == all_labels[j]:
                    within_sims.append(sim)
                else:
                    across_sims.append(sim)

        print(f"\nCosine similarity:")
        print(f"  Within same (c1,c2):  mean={np.mean(within_sims):.4f}, std={np.std(within_sims):.4f}")
        print(f"  Across diff (c1,c2):  mean={np.mean(across_sims):.4f}, std={np.std(across_sims):.4f}")
        print(f"  Separation gap:       {np.mean(within_sims) - np.mean(across_sims):.4f}")

        # 4. Check: does magnitude info exist beyond type?
        # Compare samples with same "type" but different magnitude
        # e.g., (1,0) vs (2,0) vs (3,0)
        print(f"\n  Similarity between scaling variants:")
        for pairs in [
            ("1a+0b", "2a+0b", "3a+0b"),
            ("0a+1b", "0a+2b", "0a+3b"),
            ("1a+1b", "2a+2b"),
        ]:
            indices = {label: [i for i, l in enumerate(all_labels) if l == label]
                      for label in pairs}
            for l1 in pairs:
                for l2 in pairs:
                    if l1 >= l2:
                        continue
                    sims = [cos_sim[i, j]
                           for i in indices[l1] for j in indices[l2]]
                    print(f"    {l1} vs {l2}: {np.mean(sims):.4f}")

    # 5. Full-sequence analysis at middle layer (18)
    print(f"\n{'='*60}")
    print(f"Full-sequence analysis (layer 18)")
    print(f"{'='*60}")

    # Re-extract with full sequences for a subset
    test_pairs = [(1, 0), (2, 0), (3, 0), (0, 1), (0, 2)]
    for c1, c2 in test_pairs:
        demo_text = generate_demo_text(c1, c2, n_demos=5)
        _, full_seqs, seq_len = extract_hidden_states(
            model, tokenizer, demo_text, [18], device
        )
        seq = full_seqs[18]  # (seq_len, d_model)
        norms = seq.float().norm(dim=-1)
        print(f"\n  {c1}a+{c2}b: seq_len={seq_len}, "
              f"hidden norm range=[{norms.min():.1f}, {norms.max():.1f}], "
              f"mean={norms.mean():.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
