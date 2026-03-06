# Dual Hebbian — Fair Generalization Test (ep400)

## Architecture
- MACHDualHebbian: residual patches + attention output patches
- 4 patch layers, n_rank=2, d_proj=32, attn_hidden=64
- 920K trainable params

## Training
- Task: diverse_ops (9 train ops, 3 held-out, 3 impossible)
- Train ops: add, sub, mul, div, gcd, abs_diff, avg, first, second
- Held-out ops (Qwen CAN do): mod, max, min
- Impossible ops (Qwen CAN'T do): digit_sum_add, bitwise_and, bitwise_xor

## Key Result: GENERALIZATION CONFIRMED

Hebbian learning generalizes to unseen operations that Qwen has latent capability for.
It does NOT help with operations Qwen fundamentally cannot compute.

## Ablation at ep400

```
op       |  with_hebb |  no_update |    no_init
---------+------------+------------+-----------
add      |       30%  |       23%  |       32%
sub      |       62%  |       11%  |       13%
mul      |       50%  |        0%  |        1%
div      |       48%  |       62%  |       59%
gcd      |       57%  |       57%  |       54%
abs_diff |       48%  |       13%  |        5%
avg      |        7%  |        0%  |        2%
first    |       45%  |       19%  |       22%
second   |       50%  |       14%  |       14%
tavg     |       44%  |       22%  |       22%
---------+------------+------------+----------- (held-out: Qwen CAN do)
mod      |       32%  |       12%  |        7%
max      |       18%  |        3%  |        3%
min      |       31%  |       33%  |       32%
havg     |       27%  |       16%  |       14%
---------+------------+------------+----------- (impossible: Qwen CAN'T do)
digit_sum|        6%  |        8%  |        8%
bit_and  |       10%  |       10%  |        9%
bit_xor  |        3%  |        6%  |        6%
iavg     |        6%  |        8%  |        8%
```

## Diagnostics at ep400

```
attn_hebb_rule grad: 0.000953
hebb_rule grad:      0.025562
eta_head grad:       0.005514

Residual patch deltas: 0.9-1.2
Attention patch deltas: 0.5-1.2
```

## Operation eval at ep400
- add=20%, sub=60%, mul=46%, div=41%, gcd=52%, abs_diff=42%

## Interpretation
- tavg 44% vs 22%: Hebbian works on trained ops
- havg 27% vs 16%: Hebbian GENERALIZES to unseen-but-possible ops
- iavg 6% vs 8%: Hebbian can't create impossible computation
- mod 32% vs 12% (2.7x): strongest generalization signal
- max 18% vs 3% (6x): clear generalization
- min 31% vs 33%: Qwen defaults to min-like behavior, no help needed
- Attention patch gradient (0.001) much lower than residual (0.026)
