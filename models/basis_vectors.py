import torch
import torch.nn as nn


class BasisVectors(nn.Module):
    """
    Learnable basis directions for patch weight modification.
    Each direction is a rank-1 outer product U_k V_k^T.
    The meta-learner produces coefficients and a gate; the weight delta is:
        delta_W = gate * sum_k(coeff_k * U_k V_k^T)
    """

    def __init__(self, d_model, hidden_dim=256, n_patches=4, n_basis=8,
                 n_gain_basis=4):
        super().__init__()
        self.n_patches = n_patches
        self.n_basis = n_basis
        self.n_gain_basis = n_gain_basis

        # For each patch: down weight (hidden_dim, d_model), up weight (d_model, hidden_dim)
        # Basis: U (n_basis, rows), V (n_basis, cols) per weight matrix
        self.down_U = nn.ParameterList([
            nn.Parameter(torch.randn(n_basis, hidden_dim) * 0.01)
            for _ in range(n_patches)
        ])
        self.down_V = nn.ParameterList([
            nn.Parameter(torch.randn(n_basis, d_model) * 0.01)
            for _ in range(n_patches)
        ])
        self.up_U = nn.ParameterList([
            nn.Parameter(torch.randn(n_basis, d_model) * 0.01)
            for _ in range(n_patches)
        ])
        self.up_V = nn.ParameterList([
            nn.Parameter(torch.randn(n_basis, hidden_dim) * 0.01)
            for _ in range(n_patches)
        ])

        # Gain basis: per-dimension multiplicative scaling vectors
        # Simpler than outer products — just n_gain_basis vectors of d_model
        self.gain_basis = nn.ParameterList([
            nn.Parameter(torch.randn(n_gain_basis, d_model) * 0.01)
            for _ in range(n_patches)
        ])

    def compute_delta_W(self, patch_idx, weight_name, coefficients, gate):
        """
        Compute weight update as gated sum of rank-1 outer products.

        coefficients: (n_basis,) for down/up, (n_gain_basis,) for gain
        gate: scalar (sigmoid * GATE_SCALE)
        Returns: delta_W (matrix for down/up, vector for gain)
        """
        if weight_name == "gain":
            # Gain: weighted sum of basis vectors (no outer product)
            B = self.gain_basis[patch_idx]  # (n_gain_basis, d_model)
            return gate * torch.einsum('k,ki->i', coefficients, B)

        if weight_name == "down":
            U = self.down_U[patch_idx]  # (n_basis, hidden_dim)
            V = self.down_V[patch_idx]  # (n_basis, d_model)
        else:
            U = self.up_U[patch_idx]    # (n_basis, d_model)
            V = self.up_V[patch_idx]    # (n_basis, hidden_dim)

        # gate * sum_k(coeff_k * U_k outer V_k)
        delta_W = gate * torch.einsum('k,ki,kj->ij', coefficients, U, V)
        return delta_W
