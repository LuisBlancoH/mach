import torch

# Base model
BASE_MODEL = "Qwen/Qwen3-4B"
DTYPE = torch.bfloat16

# Device auto-detection: cuda > mps > cpu
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Universal module dimensions
D_META = 128
PATCH_HIDDEN_DIM = 256
N_BASIS = 8
N_META_LAYERS = 2
N_META_HEADS = 2
META_MLP_DIM = 256

# Write mechanism
GATE_SCALE = 0.1
PATCH_INIT_STD = 0.01

# Phase 1
PHASE1_LR = 1e-4
PHASE1_EPOCHS = 20
PHASE1_TRAIN_PROBLEMS = 5000
PHASE1_TEST_PROBLEMS = 500

# Phase 2
PHASE2_LR = 3e-4
PHASE2_EPISODES = 2000
PHASE2_PROBLEMS_PER_EPISODE = 20
PHASE2_GRAD_CLIP = 1.0

# Phase 3
PHASE3_LR = 1e-4
PHASE3_EPISODES = 2000
PHASE3_PROBLEMS_PER_EPISODE = 20
PHASE3_GRAD_CLIP = 1.0
PHASE3_CRITIC_LOSS_WEIGHT = 0.5
PHASE3_GAMMA = 0.95

# Phase 4
PHASE4_LR = 1e-4
PHASE4_EPISODES = 2000
PHASE4_PROBLEMS_PER_EPISODE = 20
PHASE4_GRAD_CLIP = 1.0
PHASE4_CRITIC_LOSS_WEIGHT = 0.5
PHASE4_GAMMA = 0.5
PHASE4_CEREBELLUM_LR = 1e-3
PHASE4_SURPRISE_SCALE = 2.0
PHASE4_TD_MODULATION = 0.5    # CE loss weighted by |reward - V(s)|; dopamine-like
PHASE4_RECENCY_ALPHA = 1.0    # Progressive weighting: problem 0 → 1.0, last → 2.0

# Phase 6
PHASE6_LR = 1e-4
PHASE6_EPISODES = 2000
PHASE6_PROBLEMS_PER_EPISODE = 20
PHASE6_GRAD_CLIP = 1.0
PHASE6_CRITIC_LOSS_WEIGHT = 0.5
PHASE6_GAMMA = 0.95

# Phase 5 (brain-like)
PHASE5_D_OBS = 64       # observation projection output
PHASE5_D_GRU = 64       # GRU hidden state (hippocampal memory)
PHASE5_D_TASK = 32      # task state (PFC working memory) — bottleneck
PHASE5_SPARSITY_BETA = 0.01  # L1 penalty on task state
PHASE5_LR = 3e-4
PHASE5_EPISODES = 2000
PHASE5_PROBLEMS_PER_EPISODE = 20
PHASE5_GRAD_CLIP = 1.0
PHASE5_N_DELIBERATION_STEPS = 0  # 0 = no deliberation, 3+ = iterative refinement
PHASE5_DECORR_BETA = 0.01        # lateral inhibition: decorrelation loss weight
PHASE5_TASK_NOISE = 0.0          # forgetting: noise std on task state (0 = off)
PHASE5_ENERGY_BETA = 0.0         # free energy: unified metabolic cost (0 = use separate losses)
PHASE5_N_SELF_EVAL_STEPS = 0    # self-evaluation: observe own output on demos (0 = off)
PHASE5_TD_MODULATION = 0.0      # TD-weighted CE loss (0 = off, 0.5 = moderate)
PHASE5_CRITIC_BETA = 0.1        # critic loss weight
PHASE5_GAMMA = 0.0              # discount factor (0 = no discounting, each problem independent)
PHASE5_SATISFACTION_THRESHOLD = 0.5  # critic value above which self-eval stops early
PHASE5_CONSOLIDATION = False         # cross-episode slow memory (off by default)
PHASE5_EMA_DECAY = 0.95              # slow memory EMA decay rate
PHASE5_N_PLANNING_STEPS = 0         # 0 = no planning, 3+ = critic-gated deliberation
PHASE5_PLANNING_TEMPERATURE = 1.0   # softmax temperature for candidate selection
PHASE5_MULTI_LAYER_OBS = False  # observe all 4 patch layers (vs just middle)
PHASE5_N_PATCH_LAYERS = 4      # how many evenly-spaced layers (4=default, 12=searchable)

# Future phases
MAX_INTERVAL = 16
MAX_PLANNING_ITERS = 3
