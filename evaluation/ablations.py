import torch


def random_writes_baseline(n_patches=4, n_basis=8, device=None):
    """
    Generate random patch writes for ablation comparison.
    If learned writes don't beat random, the meta-learner isn't learning.
    """
    writes = []
    for patch_idx in range(n_patches):
        for weight_name in ["down", "up"]:
            coefficients = torch.randn(n_basis, device=device) * 0.01
            gate = torch.tensor(0.05, device=device)
            writes.append((patch_idx, weight_name, coefficients, gate))
    return writes
