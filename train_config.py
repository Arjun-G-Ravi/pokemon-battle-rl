# ──────────────────────────────────────────────────────────────────────────────
#  train_config.py  –  All training hyperparameters live here
# ──────────────────────────────────────────────────────────────────────────────

# ── Battle setup ──────────────────────────────────────────────────────────────

# Total number of battles to train for
N_BATTLES = 5000

# Battles per PPO weight update (effective "batch size" in episodes)
UPDATE_EVERY = 50

# Battle format (must match your Showdown server)
# e.g. "gen1randombattle", "gen2randombattle", "gen3randombattle"
BATTLE_FORMAT = "gen1randombattle"

# Opponent to train against. Options:
#   "random"          →  RandomModel   (uniformly random moves)
#   "strongest_move"  →  StrongestMoveModel  (greedy highest base-power)
#   "model2"          →  Model2  (pre-trained PPO from random_battle_trained_model.pt)
OPPONENT = "random"

# ── PPO hyperparameters ───────────────────────────────────────────────────────

# Adam learning rate
LR = 3e-4

# Discount factor for future rewards
GAMMA = 0.99

# GAE lambda (bias-variance trade-off for advantage estimation)
GAE_LAMBDA = 0.95

# PPO clipping epsilon
CLIP_EPS = 0.2

# Value function loss coefficient
VF_COEF = 0.5

# Entropy bonus coefficient (encourages exploration)
ENT_COEF = 0.01

# Number of gradient update epochs per PPO update
N_EPOCHS = 4

# Mini-batch size for each gradient step
MINI_BATCH = 256

# Maximum gradient norm for clipping
MAX_GRAD_NORM = 0.5

# ── Network ───────────────────────────────────────────────────────────────────

# Hidden layer size for the actor-critic network
HIDDEN_SIZE = 256

# Device: "cpu" or "cuda"
DEVICE = "cpu"
