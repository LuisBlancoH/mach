import torch
import torch.nn as nn

from config import PATCH_INIT_STD


class CorticalPatch(nn.Module):
    """Small MLP that adds to the residual stream at a specific layer."""

    def __init__(self, d_model, hidden_dim=256):
        super().__init__()
        self.down = nn.Linear(d_model, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.up = nn.Linear(hidden_dim, d_model, bias=False)

        # Initialize near zero so patches start as identity
        nn.init.normal_(self.down.weight, std=PATCH_INIT_STD)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        return self.up(self.act(self.down(hidden_states)))


class PatchedModel(nn.Module):
    """Wraps a frozen base model with cortical patches hooked into the residual stream."""

    def __init__(self, base_model, d_model, patch_layers, hidden_dim=256):
        super().__init__()
        self.base_model = base_model
        self.patches = nn.ModuleDict({
            str(layer): CorticalPatch(d_model, hidden_dim)
            for layer in patch_layers
        })
        self.patch_layers = patch_layers
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for layer_idx in self.patch_layers:
            layer = self.base_model.model.layers[layer_idx]
            handle = layer.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(handle)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Patches operate in float32 for stable gradients
                patch_input = hidden_states.float()
                patch_output = self.patches[str(layer_idx)](patch_input)
                return (hidden_states + patch_output.to(hidden_states.dtype),) + output[1:]
            else:
                patch_input = output.float()
                patch_output = self.patches[str(layer_idx)](patch_input)
                return output + patch_output.to(output.dtype)
        return hook

    @property
    def device(self):
        return self.base_model.device

    def forward(self, input_ids, labels=None, attention_mask=None):
        return self.base_model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
