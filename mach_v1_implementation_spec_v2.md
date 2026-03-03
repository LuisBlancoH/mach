# MACH v1 Implementation Spec — Phase 1 & Phase 2

## Overview

MACH (Meta-learned Adaptive Cognitive Hierarchy) is a small neural network wrapper (~5M params) around a frozen language model that gives it the ability to learn new computation from user feedback at inference time.

The core building block is the **Universal Module** — a brain-inspired unit containing a GRU (thalamic gating + cortical integration), cerebellum (prediction + correction), meta-learner transformer (cortex), and critic (basal ganglia). The same architecture is used at every level. Multiple modules stack vertically: each module's input stream is the output of the module below it. Higher modules also write into lower modules' weights. Timescales emerge from learned GRU gate dynamics, not fixed schedules.

This spec covers the first two phases of validation:

- **Phase 1**: Can cortical patches (small MLPs added to a frozen LLM's residual stream) learn arithmetic via direct backprop?
- **Phase 2**: Can a simplified universal module learn to write useful patches via basis vector outer products?

Phase 2 uses a stripped-down universal module: no cerebellum, no planning loop, no critic-triggered firing. Fixed cadence. The goal is to validate the core write mechanism before adding the full module's components.

If Phase 1 fails, patches cannot learn — rethink the substrate.
If Phase 2 fails, the core meta-learning mechanism doesn't work — the hardest research question.

---

## The Universal Module (Full Design)

This section describes the complete universal module for reference. Phase 2 implements a simplified subset. Later phases add components incrementally until the full module is realized.

### Brain Mapping

| Component | Brain Analogue | Function |
|-----------|---------------|----------|
| GRU | Thalamus + cortical recurrence | Gates and integrates input stream |
| Cerebellum | Cerebellum | Predicts next input, provides surprise gating and learned corrections |
| Meta-learner | Cortex | Processes integrated state, proposes actions via planning |
| Critic | Basal ganglia | Evaluates meta-learner proposals, gates action release |

### Information Flow

```
BETWEEN FIRINGS (runs every input):

    New input arrives from stream
                |
                v
    Cerebellum compares input to its prediction
                |
                +--> Surprise = ||error|| -----> Gates GRU update
                |
                +--> Correction = learned_proj(error) --> accumulates
                |
                +--> Cerebellum trains itself (supervised, SGD)
                |
                +--> Cerebellum predicts NEXT input (from GRU memory + last action)
                |
                v
    GRU integrates input (surprise-gated)
                |
                v
    Check firing condition:
        surprise_sum > learned_threshold  OR  inputs > max_interval
                |
                v  (if condition met)

ON FIRING:

    Assemble meta-learner input tokens:
        [GRU_memory, accumulated_correction, critic_signals,
         tf_memory, think_0, think_1, think_2]
                |
                v
    Planning loop:
        Iteration i:
            Meta-learner forward pass --> hidden states
            Critic evaluates hidden states --> value
            If value > commit_threshold: commit this proposal
            Else: update critic signals + tf_memory, iterate
                |
                v
    Execute committed proposal:
        think_0 --> action head --> patch writes (via basis vectors)
        think_1 --> memory head --> update transformer memory
        think_2 --> output representation --> input stream for module above
                |
                v
    Reset: accumulated correction = 0, surprise_sum = 0
```

### Why the Critic Evaluates Meta-Learner Hidden States (Not GRU Memory)

The basal ganglia receive projections from all of cortex — sensory, motor, prefrontal. They evaluate cortical activity patterns that encode both perception AND planned action. By the time cortex has processed its inputs, the hidden states contain the full situation: world state + candidate plan.

If the critic evaluated GRU memory directly, it would know the state but not the plan. It would be a state value function. By evaluating meta-learner hidden states, it becomes an action-conditional value function — "this state with this proposed action, how good?" That's what enables the planning loop. The critic evaluates proposals, not just situations.

### Why the Cerebellum Triggers Firing (Not the Critic)

The critic requires meta-learner hidden states, which only exist when the meta-learner fires. The critic cannot trigger something that hasn't happened yet.

The cerebellum runs every input. Its prediction error is computed every input. Cumulative surprise is always available. The cerebellum is the alarm ("something unexpected is happening, wake up cortex"). The critic is the judge ("this plan is good enough, execute it"). Different roles, different components.

In the brain: the cerebellum detects when ongoing programs fail — sensory feedback doesn't match predictions — triggering cortical replanning. The basal ganglia gate the output of the replanning.

### Module Stacking (Multi-Module Architecture)

```
Fast module:
    Input stream: Qwen hidden states (last token, per token)
    Writes: Qwen patches
    Output: think_2 representations (per firing)
    max_interval: ~16 tokens

Slow module:
    Input stream: fast module's think_2 outputs (per fast firing)
    Writes: Qwen patches AND fast module weights
    Output: think_2 representations (per firing)
    max_interval: ~10 fast firings

Each module sees only the output of the module below it (vertical).
Higher modules see cortical output (think_2), not thalamic state (GRU memory).
This matches the brain: prefrontal cortex sees processed representations
from association cortex, not raw sensory data.
```

### Universal Module Components (Full Spec)

```python
class UniversalModule:
    # Cerebellum
    cerebellum_predictor:    MLP(d_meta + action_dim, input_dim)  # predicts next input
    cerebellum_correction:   Linear(input_dim, d_meta)            # error -> correction vector
    # Trained: predictor online (supervised), correction in meta-training

    # GRU
    gru:                     GRUCell(input_dim, d_meta)           # surprise-gated integration
    # update_gate includes: sigmoid(W @ [memory, input] + bias + alpha * surprise)
    # Trained: meta-training

    # Meta-learner (cortex)
    transformer:             Transformer(d_meta, n_layers=2, n_heads=2)
    # Input tokens: [gru_mem, correction, critic_signals, tf_mem, think_0, think_1, think_2]
    # Trained: meta-training

    # Heads
    action_head:             MLP(d_meta, n_writes * (n_basis + 1))  # think_0 -> writes
    memory_head:             MLP(d_meta, n_mem_slots * d_meta)      # think_1 -> tf_mem update
    output_proj:             Linear(d_meta, d_meta)                 # think_2 -> upstream signal
    # Trained: meta-training

    # Critic (basal ganglia)
    critic:                  MLP(d_meta, 1)  # meta-learner hidden states -> value
    # Input: mean-pooled meta-learner hidden states
    # Trained: TD learning online + hindsight in sleep

    # Basis vectors
    basis_vectors:           Per patch, per weight matrix: U(n_basis, rows), V(n_basis, cols)
    # Trained: meta-training

    # Learned thresholds
    surprise_threshold:      scalar  # when to fire
    commit_threshold:        scalar  # when to commit in planning loop
    # Trained: meta-training
```

---

## Hardware & Environment

- **GPU**: NVIDIA A100 (80GB)
- **Base model**: Qwen2.5-4B (or latest Qwen ~4B variant available on HuggingFace)
- **Framework**: PyTorch
- **Python**: 3.10+
- **Key dependencies**: `transformers`, `torch`, `accelerate`, `safetensors`

---

## Phase 1: Direct Patch Training

### Goal

Prove that small MLPs injected into a frozen LLM's residual stream can learn arithmetic that the base model cannot reliably do. This validates the substrate the meta-learner will write into.

### 1.1 Load and Characterize Base Model

Load Qwen2.5-4B. Determine its architecture:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-4B", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-4B")

# Record these values — they parameterize everything else:
d_model = model.config.hidden_size        # e.g., 2560 or 3584
n_layers = model.config.num_hidden_layers  # e.g., 36 or 40
```

**Freeze all base model parameters:**
```python
for param in model.parameters():
    param.requires_grad = False
```

### 1.2 Baseline Evaluation

Before adding patches, measure Qwen's native arithmetic ability. This is the baseline that patches must beat.

**Dataset generation:**

```python
import random

def generate_arithmetic_problems(n, difficulty):
    """
    difficulty levels:
        1: single digit addition (3 + 5)
        2: two digit addition (23 + 45)
        3: three digit addition (347 + 589)
        4: two digit subtraction (67 - 23)
        5: three digit subtraction (523 - 178)
        6: single × single multiplication (7 × 8)
        7: two digit × single digit multiplication (23 × 7)
    """
    problems = []
    for _ in range(n):
        if difficulty == 1:
            a, b = random.randint(1, 9), random.randint(1, 9)
            op, answer = "+", a + b
        elif difficulty == 2:
            a, b = random.randint(10, 99), random.randint(10, 99)
            op, answer = "+", a + b
        elif difficulty == 3:
            a, b = random.randint(100, 999), random.randint(100, 999)
            op, answer = "+", a + b
        elif difficulty == 4:
            a = random.randint(10, 99)
            b = random.randint(1, a)  # ensure positive result
            op, answer = "-", a - b
        elif difficulty == 5:
            a = random.randint(100, 999)
            b = random.randint(1, a)
            op, answer = "-", a - b
        elif difficulty == 6:
            a, b = random.randint(2, 9), random.randint(2, 9)
            op, answer = "×", a * b
        elif difficulty == 7:
            a, b = random.randint(10, 99), random.randint(2, 9)
            op, answer = "×", a * b

        prompt = f"What is {a} {op} {b}? "
        problems.append({"prompt": prompt, "answer": str(answer), "a": a, "b": b, "op": op})
    return problems
```

**Evaluation function:**

```python
def evaluate_model(model, tokenizer, problems, max_new_tokens=10):
    correct = 0
    for p in problems:
        input_ids = tokenizer(p["prompt"], return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        predicted = extract_number(response)
        if predicted == p["answer"]:
            correct += 1
    return correct / len(problems)
```

**Run baseline for each difficulty level. Record results. Expect Qwen to succeed on difficulty 1-2 and start failing on 3+.**

### 1.3 Cortical Patch Architecture

A cortical patch is a small MLP that adds to the residual stream at a specific layer.

```python
class CorticalPatch(nn.Module):
    def __init__(self, d_model, hidden_dim=256):
        super().__init__()
        self.down = nn.Linear(d_model, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, d_model, bias=False)

        # Initialize near zero so patches start as identity
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        return self.up(self.act(self.down(hidden_states)))
```

**Placement**: Insert patches at approximately the quarter points of the network.

```python
patch_layers = [
    n_layers // 4,          # ~25%
    n_layers // 2,          # ~50%
    3 * n_layers // 4,      # ~75%
    n_layers - 2            # near final layer
]
# Example for 36 layers: [9, 18, 27, 34]
```

**Integration**: Hook into the model's forward pass. After each target layer's computation, add the patch output to the residual stream.

```python
class PatchedModel(nn.Module):
    def __init__(self, base_model, d_model, patch_layers, hidden_dim=256):
        super().__init__()
        self.base_model = base_model
        self.patches = nn.ModuleDict({
            str(layer): CorticalPatch(d_model, hidden_dim)
            for layer in patch_layers
        })
        self.patch_layers = patch_layers
        self._register_hooks()

    def _register_hooks(self):
        for layer_idx in self.patch_layers:
            layer = self.base_model.model.layers[layer_idx]
            layer.register_forward_hook(self._make_hook(layer_idx))

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                patch_output = self.patches[str(layer_idx)](hidden_states)
                return (hidden_states + patch_output,) + output[1:]
            else:
                patch_output = self.patches[str(layer_idx)](output)
                return output + patch_output
        return hook

    def forward(self, input_ids, labels=None):
        return self.base_model(input_ids=input_ids, labels=labels)
```

**IMPORTANT**: The exact hook mechanism depends on how Qwen2.5 structures its forward pass. Inspect `model.model.layers[0]` to understand the output format. The hook must add the patch output to the residual stream (the hidden_states tensor), not to attention outputs or other intermediate values.

### 1.4 Training Patches with Direct Backprop

**Training data**: Generate thousands of arithmetic problems. Format each as:

```
Input:  "What is 347 + 589? "
Target: "936"
```

The loss is cross-entropy on the target tokens only.

```python
def train_patches_direct(patched_model, tokenizer, problems, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(patched_model.patches.parameters(), lr=lr)

    for epoch in range(epochs):
        random.shuffle(problems)
        total_loss = 0

        for p in problems:
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

        print(f"Epoch {epoch}: avg loss = {total_loss / len(problems):.4f}")
        accuracy = evaluate_model(patched_model, tokenizer, test_problems)
        print(f"  Accuracy: {accuracy:.2%}")
```

**Training configuration:**
- Optimizer: Adam, lr=1e-4
- Batch size: 1 to start, increase to 8-16 if memory allows
- Training problems: 5000 per difficulty level
- Test problems: 500 per difficulty level (separate from training)
- Epochs: 10-20
- Use float32 for patch parameters even if base model is float16

### 1.5 Phase 1 Success Criteria

For each difficulty level, compare baseline accuracy (Qwen alone) vs patched accuracy (Qwen + trained patches).

**Pass**: Patches improve accuracy by >10 percentage points on at least one difficulty level where baseline is below 80%.

**Strong pass**: Patches achieve >90% accuracy on a difficulty level where baseline is below 50%.

**Fail**: No measurable improvement at any difficulty level.

### 1.6 Phase 1 Diagnostics

If patches show small or no improvement:

1. **Check patch output norms**: Are patches producing meaningful output or still near zero?

2. **Check gradient flow**: Are gradients reaching the patches?
   ```python
   for name, param in patched_model.patches.named_parameters():
       print(f"{name}: grad norm = {param.grad.norm():.6f}")
   ```

3. **Try different patch sizes**: hidden_dim 128, 256, 512.

4. **Try different layer placements**: Every-other-layer instead of quarter points.

5. **Try larger patches**: 2-layer MLPs with skip connections.

6. **Probe Qwen's hidden states**: Train a linear classifier on hidden states to predict answers. If information isn't there, patches can't extract it.

---

## Phase 2: Meta-Learner Training (Simplified Universal Module)

### Goal

Prove that a meta-learner can learn to write patches that create new arithmetic computation in a frozen LLM, using basis vector outer products.

### Prerequisites

Phase 1 must pass. If patches can't learn arithmetic with direct backprop, a meta-learner can't teach them.

### What Phase 2 Implements vs Full Module

Phase 2 uses a simplified universal module to isolate the core write mechanism:

| Component | Full Module | Phase 2 |
|-----------|------------|---------|
| GRU | Surprise-gated, learned timescale | Simple fixed GRU, no surprise gating |
| Cerebellum | Predicts input, provides surprise + correction | **Not included** |
| Meta-learner | Full transformer with planning loop | Transformer, single forward pass (no planning) |
| Critic | Evaluates meta-learner hidden states, gates planning | **Not included** (use raw reward signal) |
| Firing trigger | Cerebellum surprise-triggered | Fixed cadence (every problem) |
| Think tokens | think_0 (action), think_1 (memory), think_2 (output) | think_0 (action), think_1 (memory), think_2 (unused placeholder) |

### 2.1 Architecture Components

All components operate at d_meta=128. Interfaces to Qwen scale with d_model.

#### Observation Projection

```python
class ObservationProjection(nn.Module):
    """Projects Qwen hidden states to meta-learner dimension."""
    def __init__(self, d_model, d_meta=128):
        super().__init__()
        self.proj = nn.Linear(d_model, d_meta, bias=False)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, d_model)
        # Take last token's hidden state
        return self.proj(hidden_states[:, -1, :])  # (batch, d_meta)
```

#### GRU (Simplified for Phase 2)

In Phase 2, the GRU integrates Qwen hidden states without surprise gating. This is a placeholder for the full surprise-gated GRU added in later phases. Even in simplified form, the GRU provides temporal integration — each firing, the meta-learner sees a state that reflects accumulated tokens, not just the latest one.

```python
class SimpleGRU(nn.Module):
    """
    Integrates projected Qwen hidden states over tokens.
    Phase 2 simplified: no surprise gating.
    Full version adds: surprise = ||cerebellar_error|| modulating update gate.
    """
    def __init__(self, d_meta=128):
        super().__init__()
        self.gru_cell = nn.GRUCell(d_meta, d_meta)
        self.memory = None  # (d_meta,) — call reset() before each episode

    def reset(self):
        self.memory = None

    def integrate(self, projected_input):
        """
        projected_input: (d_meta,) — one projected Qwen hidden state
        Updates internal memory. Call once per token or per observation.
        """
        if self.memory is None:
            self.memory = torch.zeros_like(projected_input)
        self.memory = self.gru_cell(projected_input.unsqueeze(0), self.memory.unsqueeze(0)).squeeze(0)
        return self.memory

    def get_memory(self):
        return self.memory
```

#### Basis Vectors

Define 8 directions per patch weight matrix. Each direction is a rank-1 outer product.

```python
class BasisVectors(nn.Module):
    """Learnable basis directions for patch weight modification."""
    def __init__(self, d_model, hidden_dim=256, n_patches=4, n_basis=8):
        super().__init__()
        self.n_patches = n_patches
        self.n_basis = n_basis

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

    def compute_delta_W(self, patch_idx, weight_name, coefficients, gate):
        """
        Compute weight update as gated sum of rank-1 outer products.

        coefficients: (n_basis,)
        gate: scalar
        Returns: delta_W, same shape as target weight matrix
        """
        if weight_name == "down":
            U = self.down_U[patch_idx]  # (n_basis, hidden_dim)
            V = self.down_V[patch_idx]  # (n_basis, d_model)
        else:
            U = self.up_U[patch_idx]    # (n_basis, d_model)
            V = self.up_V[patch_idx]    # (n_basis, hidden_dim)

        delta_W = gate * torch.einsum('k,ki,kj->ij', coefficients, U, V)
        return delta_W
```

#### Meta-Learner Transformer

```python
class MetaLearnerTransformer(nn.Module):
    """
    Small transformer — the cortex of the universal module.

    Input tokens (7 for Phase 2):
        [gru_memory]              1 token: integrated world state
        [reward_signal]           1 token: projected reward (placeholder for critic signals)
        [zero_placeholder]        1 token: placeholder for accumulated cerebellar correction
        [tf_mem]                  1 token: transformer memory (persists across firings)
        [think_0]                 1 token: action output
        [think_1]                 1 token: memory update output
        [think_2]                 1 token: upstream output (unused in Phase 2)

    Full module adds:
        - Accumulated cerebellar correction replaces zero_placeholder
        - Critic signals (value, TD error) replace raw reward
        - Planning loop iterates this forward pass multiple times
    """
    def __init__(self, d_meta=128, n_heads=2, n_layers=2, mlp_dim=256, n_tokens=7):
        super().__init__()
        self.d_meta = d_meta

        # Learned initial vectors
        self.think_0_init = nn.Parameter(torch.randn(d_meta) * 0.01)
        self.think_1_init = nn.Parameter(torch.randn(d_meta) * 0.01)
        self.think_2_init = nn.Parameter(torch.randn(d_meta) * 0.01)
        self.tf_mem_init = nn.Parameter(torch.randn(d_meta) * 0.01)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(n_tokens, d_meta) * 0.01)

        # Transformer layers (pre-norm)
        self.layers = nn.ModuleList([
            TransformerBlock(d_meta, n_heads, mlp_dim)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_meta)

    def forward(self, tokens):
        """
        tokens: (n_tokens, d_meta)
        Returns: (n_tokens, d_meta) hidden states at all positions
        """
        x = tokens + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model)
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
```

#### Action Head

```python
class ActionHead(nn.Module):
    """
    Takes think_0 hidden state, produces patch write coefficients.

    Output: 72 values
        4 patches × 2 weight matrices = 8 writes
        8 basis coefficients + 1 gate per write = 9 values per write
        Total: 8 × 9 = 72
    """
    def __init__(self, d_meta=128, n_patches=4, n_basis=8):
        super().__init__()
        self.n_patches = n_patches
        self.n_basis = n_basis
        n_writes = n_patches * 2
        n_outputs = n_writes * (n_basis + 1)

        self.head = nn.Sequential(
            nn.Linear(d_meta, 64),
            nn.GELU(),
            nn.Linear(64, n_outputs)
        )

    def forward(self, think_0_hidden):
        """
        Returns: list of (patch_idx, weight_name, coefficients, gate)
        """
        raw = self.head(think_0_hidden)

        writes = []
        idx = 0
        for patch_i in range(self.n_patches):
            for weight_name in ["down", "up"]:
                coefficients = raw[idx:idx + self.n_basis]
                gate = torch.sigmoid(raw[idx + self.n_basis]) * GATE_SCALE
                idx += self.n_basis + 1
                writes.append((patch_i, weight_name, coefficients, gate))

        return writes
```

**IMPORTANT**: The gate uses sigmoid scaled by GATE_SCALE (default 0.1). This keeps initial writes small. The meta-learner should start with near-zero modifications. Tune this if patches change too fast (reduce) or too slow (increase).

#### Memory Head

```python
class MemoryHead(nn.Module):
    """Updates transformer memory from think_1 hidden state."""
    def __init__(self, d_meta=128):
        super().__init__()
        self.gate = nn.Linear(d_meta, d_meta)
        self.candidate = nn.Linear(d_meta, d_meta)

    def forward(self, think_1_hidden, tf_mem):
        """
        Gated update of transformer memory.
        """
        g = torch.sigmoid(self.gate(think_1_hidden))
        c = torch.tanh(self.candidate(think_1_hidden))
        return (1 - g) * tf_mem + g * c
```

#### Reward Projection (placeholder for critic in Phase 2)

```python
class RewardProjection(nn.Module):
    """Projects raw reward signals into a token for the meta-learner.
    In the full module, this is replaced by critic signals (value, TD error)."""
    def __init__(self, n_signals=3, d_meta=128):
        super().__init__()
        # Phase 2 signals: [last_reward, cumulative_reward, firing_index]
        self.proj = nn.Linear(n_signals, d_meta)

    def forward(self, signals):
        return self.proj(signals)
```

### 2.2 Differentiable Patches (Critical)

For gradient to flow from Qwen's output loss back through patches into the meta-learner, patch writes must be differentiable. The delta_W must be part of the computational graph, not stored in `.data`.

```python
class DifferentiablePatch(nn.Module):
    """
    Cortical patch with differentiable accumulated writes.
    Base weights are fixed at zero. Meta-learner writes accumulate as delta tensors
    that remain in the computational graph.
    """
    def __init__(self, d_model, hidden_dim=256):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.act = nn.GELU()

        # Base weights (fixed at zero, not parameters)
        self.register_buffer('down_base', torch.zeros(hidden_dim, d_model))
        self.register_buffer('up_base', torch.zeros(d_model, hidden_dim))

        # Accumulated deltas — set externally, part of computational graph
        self.delta_down = None
        self.delta_up = None

    def reset_deltas(self):
        """Call at start of each episode."""
        self.delta_down = torch.zeros(self.hidden_dim, self.d_model,
                                       device=self.down_base.device)
        self.delta_up = torch.zeros(self.d_model, self.hidden_dim,
                                     device=self.up_base.device)

    def accumulate_write(self, weight_name, delta_W):
        """Add a differentiable delta. delta_W must be in the computational graph."""
        if weight_name == "down":
            self.delta_down = self.delta_down + delta_W
        else:
            self.delta_up = self.delta_up + delta_W

    def forward(self, hidden_states):
        """Forward with base + accumulated deltas."""
        W_down = self.down_base + (self.delta_down if self.delta_down is not None else 0)
        W_up = self.up_base + (self.delta_up if self.delta_up is not None else 0)

        h = torch.nn.functional.linear(hidden_states, W_down)
        h = self.act(h)
        h = torch.nn.functional.linear(h, W_up)
        return h
```

This is the critical design. The delta tensors are computed by `basis.compute_delta_W()` which uses `torch.einsum` on learnable basis vectors and meta-learner-produced coefficients. The gradient flows:

```
loss → Qwen output → patch forward (using down_base + delta_down)
     → delta_down → basis.compute_delta_W → einsum(coefficients, U, V)
     → coefficients and gate → action_head → meta-learner transformer
```

The observation path (Qwen hidden states → obs_proj → GRU) is **detached**. The meta-learner gets no gradient about what to observe, only about what to write.

### 2.3 Simplified Universal Module for Phase 2

```python
class MACHPhase2(nn.Module):
    """
    Simplified universal module for Phase 2.
    No cerebellum. No critic. No planning loop. No surprise-gated GRU.
    Fixed firing cadence (once per problem).
    """
    def __init__(self, d_model, n_layers, patch_layers, hidden_dim=256,
                 d_meta=128, n_basis=8):
        super().__init__()
        self.d_model = d_model
        self.d_meta = d_meta
        self.n_patches = len(patch_layers)
        self.patch_layers = patch_layers

        # Observation projection (Qwen -> d_meta)
        self.obs_proj = ObservationProjection(d_model, d_meta)

        # GRU (simplified, no surprise gating)
        self.gru = SimpleGRU(d_meta)

        # Basis vectors
        self.basis = BasisVectors(d_model, hidden_dim, len(patch_layers), n_basis)

        # Meta-learner transformer
        self.transformer = MetaLearnerTransformer(d_meta, n_tokens=7)

        # Heads
        self.action_head = ActionHead(d_meta, len(patch_layers), n_basis)
        self.memory_head = MemoryHead(d_meta)

        # Reward projection (placeholder for critic)
        self.reward_proj = RewardProjection(n_signals=3, d_meta=d_meta)

        # Differentiable patches
        self.patches = nn.ModuleList([
            DifferentiablePatch(d_model, hidden_dim) for _ in patch_layers
        ])

    def reset_episode(self):
        """Call at the start of each episode."""
        self.gru.reset()
        for patch in self.patches:
            patch.reset_deltas()
        self._tf_mem = self.transformer.tf_mem_init.clone()

    def observe(self, base_model, input_ids):
        """
        Run Qwen forward (detached), extract last-token hidden state,
        integrate through GRU.
        """
        with torch.no_grad():
            # Get hidden state from a middle layer for observation
            hidden_state = None
            target_layer = self.patch_layers[len(self.patch_layers) // 2]

            def hook(module, input, output):
                nonlocal hidden_state
                if isinstance(output, tuple):
                    hidden_state = output[0][:, -1, :]
                else:
                    hidden_state = output[:, -1, :]

            h = base_model.model.layers[target_layer].register_forward_hook(hook)
            base_model(input_ids=input_ids)
            h.remove()

        # Project and integrate (detached — no gradient through observation)
        projected = self.obs_proj(hidden_state.unsqueeze(0)).squeeze(0).detach()
        gru_memory = self.gru.integrate(projected)

        return gru_memory

    def fire(self, gru_memory, reward_signals):
        """
        One meta-learner firing. Produces patch writes and memory update.

        gru_memory: (d_meta,)
        reward_signals: (3,) — [last_reward, cumulative_reward, firing_index]

        Returns: writes (list of tuples)
        """
        # Assemble input tokens
        reward_token = self.reward_proj(reward_signals)
        zero_placeholder = torch.zeros(self.d_meta, device=gru_memory.device)

        tokens = torch.stack([
            gru_memory,                             # world state
            reward_token,                           # reward signals
            zero_placeholder,                       # cerebellar correction (Phase 2: zeros)
            self._tf_mem,                           # transformer memory
            self.transformer.think_0_init,          # action
            self.transformer.think_1_init,          # memory update
            self.transformer.think_2_init,          # upstream output (unused Phase 2)
        ])  # (7, d_meta)

        # Forward through meta-learner
        hidden = self.transformer(tokens)  # (7, d_meta)

        # Action from think_0 (position 4)
        writes = self.action_head(hidden[4])

        # Memory update from think_1 (position 5)
        self._tf_mem = self.memory_head(hidden[5], self._tf_mem)

        return writes

    def apply_writes(self, writes):
        """Apply differentiable patch weight modifications."""
        for (patch_idx, weight_name, coefficients, gate) in writes:
            delta_W = self.basis.compute_delta_W(patch_idx, weight_name, coefficients, gate)
            self.patches[patch_idx].accumulate_write(weight_name, delta_W)
```

### 2.4 Hooking Differentiable Patches into Qwen

```python
class MACHPatchedModel(nn.Module):
    """Wraps Qwen with MACH differentiable patches hooked into residual stream."""
    def __init__(self, base_model, mach):
        super().__init__()
        self.base_model = base_model
        self.mach = mach
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for i, layer_idx in enumerate(self.mach.patch_layers):
            layer = self.base_model.model.layers[layer_idx]

            def make_hook(patch_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        patch_out = self.mach.patches[patch_idx](h)
                        return (h + patch_out,) + output[1:]
                    else:
                        patch_out = self.mach.patches[patch_idx](output)
                        return output + patch_out
                return hook

            handle = layer.register_forward_hook(make_hook(i))
            self._hooks.append(handle)

    def forward(self, input_ids, labels=None):
        return self.base_model(input_ids=input_ids, labels=labels)
```

### 2.5 Meta-Training Episode

```python
def run_episode(base_model, mach, patched_model, tokenizer, problems, device):
    """
    One meta-training episode.

    Episode = sequence of arithmetic problems.
    Each problem: observe → fire → write patches → forward Qwen → get reward.
    Patches accumulate writes across the episode.

    Returns:
        total_loss: differentiable, for meta-training backprop
        rewards: list of reward scalars (for logging)
    """
    mach.reset_episode()

    rewards = []
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    last_reward = 0.0
    cumulative_reward = 0.0

    for i, problem in enumerate(problems):
        input_ids = tokenizer(problem["prompt"], return_tensors="pt").input_ids.to(device)

        # Step 1: Observe (detached — GRU integrates Qwen hidden state)
        gru_memory = mach.observe(base_model, input_ids)

        # Step 2: Fire meta-learner
        reward_signals = torch.tensor(
            [last_reward, cumulative_reward, float(i)],
            device=device
        )
        writes = mach.fire(gru_memory, reward_signals)

        # Step 3: Apply writes (differentiable — delta_W stays in graph)
        mach.apply_writes(writes)

        # Step 4: Forward Qwen WITH modified patches, compute loss
        full_text = problem["prompt"] + problem["answer"]
        encoding = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_len = len(tokenizer(problem["prompt"]).input_ids)
        labels = encoding.input_ids.clone()
        labels[0, :prompt_len] = -100

        output = patched_model(input_ids=encoding.input_ids, labels=labels)

        # Step 5: Compute reward (not differentiable, just for logging/signals)
        with torch.no_grad():
            generated = base_model.generate(input_ids, max_new_tokens=10, do_sample=False)
            response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
            predicted = extract_number(response)
            correct = (predicted == problem["answer"])
            reward = 1.0 if correct else -1.0

        rewards.append(reward)
        last_reward = reward
        cumulative_reward += reward

        # Accumulate loss (differentiable path)
        total_loss = total_loss + output.loss

    return total_loss, rewards
```

### 2.6 Meta-Training Outer Loop

```python
def meta_train(base_model, mach, patched_model, tokenizer, device,
               n_episodes=2000, lr=3e-4):
    """
    Train the meta-learner to produce useful patch writes.

    Trainable: obs_proj, gru, basis vectors, transformer, action_head,
               memory_head, reward_proj
    Frozen: base_model (Qwen), patches (written by meta-learner, not by gradient)
    """
    meta_params = []
    meta_params += list(mach.obs_proj.parameters())
    meta_params += list(mach.gru.parameters())
    meta_params += list(mach.basis.parameters())
    meta_params += list(mach.transformer.parameters())
    meta_params += list(mach.action_head.parameters())
    meta_params += list(mach.memory_head.parameters())
    meta_params += list(mach.reward_proj.parameters())

    optimizer = torch.optim.Adam(meta_params, lr=lr)

    # Curriculum
    curriculum = [
        (0, 500, 1),       # single digit addition
        (500, 1000, 2),    # two digit addition
        (1000, 1500, 3),   # three digit addition
        (1500, 2000, 2),   # mixed
    ]

    for episode_idx in range(n_episodes):
        difficulty = 1
        for start, end, diff in curriculum:
            if start <= episode_idx < end:
                difficulty = diff

        problems = generate_arithmetic_problems(20, difficulty)

        optimizer.zero_grad()
        loss, rewards = run_episode(
            base_model, mach, patched_model, tokenizer, problems, device
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(meta_params, max_norm=1.0)
        optimizer.step()

        if episode_idx % 10 == 0:
            avg_reward = sum(rewards) / len(rewards)
            early_acc = sum(1 for r in rewards[:5] if r > 0) / 5
            late_acc = sum(1 for r in rewards[-5:] if r > 0) / 5
            print(f"Episode {episode_idx}: loss={loss.item():.4f} "
                  f"avg_reward={avg_reward:.2f} "
                  f"early_acc={early_acc:.0%} late_acc={late_acc:.0%}")
```

### 2.7 Memory Management

Each episode maintains a computational graph across multiple Qwen forward passes. Memory optimization strategies:

1. **Gradient checkpointing**: `model.gradient_checkpointing_enable()` — recompute Qwen activations during backward.

2. **Shorter episodes**: Start with 5 problems per episode. Increase to 20 once mechanism is validated.

3. **Per-problem backward** (memory-saving fallback): If full-episode backprop exceeds memory, backward per problem instead. This gives gradient through the current problem's writes but not across problems.
   ```python
   for problem in problems:
       # ... observe, fire, write, forward ...
       output.loss.backward()  # accumulate gradients
   optimizer.step()
   ```

4. **Mixed precision**: `torch.cuda.amp` for Qwen forward passes.

### 2.8 Phase 2 Success Criteria

**Primary metric**: Within-episode learning.

```
early_accuracy = mean(correct for problems[0:5])
late_accuracy = mean(correct for problems[15:20])
```

**Pass**: `late_accuracy > early_accuracy` by ≥10 percentage points, consistently after meta-training converges.

**Strong pass**: `late_accuracy > 80%` on difficulty where base model < 50%.

**Ablation**: Compare to random writes (random coefficients instead of meta-learner output). If learned writes don't beat random, meta-learner isn't learning.

```python
def random_writes_baseline(n_patches=4, n_basis=8):
    writes = []
    for patch_idx in range(n_patches):
        for weight_name in ["down", "up"]:
            coefficients = torch.randn(n_basis) * 0.01
            gate = torch.tensor(0.05)
            writes.append((patch_idx, weight_name, coefficients, gate))
    return writes
```

### 2.9 Phase 2 Diagnostics

If the meta-learner doesn't learn:

1. **Basis vector norms**: Growing or stuck at init?
2. **Gate values**: Near zero (not writing) or saturated (writing blindly)?
3. **Action head outputs**: Diverse or collapsed?
4. **Gradient norms per component**: Is gradient flowing everywhere?
   ```python
   for name, param in mach.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad norm = {param.grad.norm():.6f}")
   ```
5. **Patch delta norms over episode**: Should increase (patches accumulating writes).
6. **Try 16 or 32 basis directions** instead of 8.
7. **Try d_meta=256**, 3 transformer layers.
8. **Try REINFORCE** as alternative to middle-ground backprop.
9. **Initialize basis vectors from Phase 1 gradients**: PCA of the gradient directions that direct backprop used. This gives the basis vectors a head start.

---

## Phase 2+ Roadmap: Completing the Universal Module

After Phase 2, add components one at a time. Each addition is a separate test.

### Phase 3: Add Critic (~1-2 days)

```
Add:
    Critic MLP: meta-learner hidden states → scalar value
    TD learning: critic trains online within episodes
    Replace raw reward signal with critic signals (value, TD error)

Meta-training change:
    Inner loop: critic SGD within episodes
    Outer loop: backprop through inner loop

Test: Phase 3 accuracy > Phase 2 accuracy
```

### Phase 4: Add Cerebellum (~1-2 days)

```
Add:
    Cerebellum predictor: predicts next Qwen hidden state
    Surprise gating: ||prediction error|| modulates GRU update gate
    Correction projection: learned projection of error → d_meta vector
    Accumulated correction as meta-learner input token (replaces zero placeholder)

Test: Phase 4 accuracy > Phase 3 accuracy
```

### Phase 5: Surprise-Triggered Firing (~1-2 days)

```
Change:
    Replace fixed firing cadence with:
        surprise_sum > learned_threshold OR tokens > max_interval
    Learned surprise threshold (meta-trained)

Test: Does variable cadence help? Compare to fixed cadence.
```

### Phase 6: Planning Loop (~1-2 days)

```
Add:
    Planning loop: 1-3 iterations
    Critic gates: evaluates meta-learner hidden states per iteration
    Commit threshold (meta-trained)
    Best-valued proposal committed

Test: Phase 6 accuracy > Phase 5 accuracy
Bonus: harder problems use more iterations?
```

### Phase 7: Sleep (~2-3 days)

```
Add:
    Trajectory storage during episodes
    Hindsight critic retraining (true discounted returns)
    Cerebellar replay
    Patch consolidation (weighted average toward high-return states)

Test: session N+1 starts better than session N
```

### Phase 8: Slow Module (~1 week)

```
Add:
    Second universal module
    Input stream: fast module's think_2 outputs
    Writes: patches AND fast module weights
    Fast module meta-trained with weight noise for robustness

Test: multi-session with slow module > without
```

---

## File Structure

```
mach/
├── README.md
├── requirements.txt
├── config.py                    # All hyperparameters
├── models/
│   ├── __init__.py
│   ├── patches.py               # CorticalPatch, DifferentiablePatch
│   ├── basis_vectors.py         # BasisVectors
│   ├── gru.py                   # SimpleGRU (Phase 2), SurpriseGatedGRU (Phase 4+)
│   ├── cerebellum.py            # Cerebellum (Phase 4+)
│   ├── meta_learner.py          # MetaLearnerTransformer, TransformerBlock
│   ├── action_head.py           # ActionHead
│   ├── memory_head.py           # MemoryHead
│   ├── critic.py                # Critic (Phase 3+)
│   ├── observation.py           # ObservationProjection
│   ├── reward_projection.py     # RewardProjection (Phase 2 only)
│   └── universal_module.py      # MACHPhase2, full UniversalModule (later)
├── data/
│   └── arithmetic.py            # generate_arithmetic_problems, extract_number
├── training/
│   ├── phase1_direct.py         # Direct backprop patch training
│   ├── phase2_meta_train.py     # Meta-training loop
│   └── episode.py               # run_episode
├── evaluation/
│   ├── baseline.py              # Evaluate base model
│   ├── evaluate.py              # evaluate_model, compare results
│   └── ablations.py             # Random writes, no patches, etc.
├── utils/
│   ├── logging.py               # Training metrics
│   └── checkpointing.py         # Save/load state
└── scripts/
    ├── run_phase1.py            # Entry point for Phase 1
    └── run_phase2.py            # Entry point for Phase 2
```

---

## Hyperparameters Summary

```python
# config.py

# Base model
BASE_MODEL = "Qwen/Qwen2.5-4B"
DEVICE = "cuda"
DTYPE = torch.float16

# Universal module dimensions
D_META = 128               # internal dimension for all module components
PATCH_HIDDEN_DIM = 256     # patch MLP hidden dimension
N_BASIS = 8                # basis directions per weight matrix
N_META_LAYERS = 2          # meta-learner transformer layers
N_META_HEADS = 2           # meta-learner attention heads
META_MLP_DIM = 256         # meta-learner MLP hidden dimension

# Write mechanism
GATE_SCALE = 0.1           # sigmoid output multiplier for write gates
PATCH_INIT_STD = 0.01      # std for patch weight initialization

# Phase 1
PHASE1_LR = 1e-4
PHASE1_EPOCHS = 20
PHASE1_TRAIN_PROBLEMS = 5000
PHASE1_TEST_PROBLEMS = 500

# Phase 2
PHASE2_LR = 3e-4
PHASE2_EPISODES = 2000
PHASE2_PROBLEMS_PER_EPISODE = 20  # reduce to 5-10 if memory constrained
PHASE2_GRAD_CLIP = 1.0

# Future phases
MAX_INTERVAL = 16          # max tokens between firings (Phase 5+)
MAX_PLANNING_ITERS = 3     # planning loop ceiling (Phase 6+)
```

---

## Execution Order

1. **Install dependencies**: `pip install torch transformers accelerate safetensors`
2. **Download model**: `python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-4B')"`
3. **Run Phase 1 baseline**: `python scripts/run_phase1.py --baseline-only`
4. **Run Phase 1 patch training**: `python scripts/run_phase1.py`
5. **Analyze Phase 1 results**: Compare accuracy per difficulty level
6. **If Phase 1 passes, run Phase 2**: `python scripts/run_phase2.py`
7. **Analyze Phase 2 results**: early vs late accuracy, learning curves, ablations

---

## What Success Looks Like

**Phase 1 success**: Clear graph showing baseline Qwen accuracy (flat, low) vs patched Qwen accuracy (high) on multi-digit arithmetic. Proof that small MLPs in the residual stream can add computation the base model doesn't have.

**Phase 2 success**: Clear graph showing within-episode accuracy improving from problem 1 to problem 20. The meta-learner writes patches that make Qwen better at arithmetic as the episode progresses. Proof that a learned controller can produce useful weight modifications through basis vector outer products.

If both pass: the core MACH mechanism works. The write path is validated. Proceed to Phase 3+ to build out the full universal module.
