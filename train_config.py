# ──────────────────────────────────────────────────────────────────────────────
#  train_config.py  –  Edit this file to configure training
# ──────────────────────────────────────────────────────────────────────────────

# Number of battles to train for
N_BATTLES = 4000

# How many battles to play before each PPO weight update
UPDATE_EVERY = 10

# Adam learning rate
LR = 3e-4

# Discount factor for future rewards
GAMMA = 0.99

# Battle format (must match what's supported by your Showdown server)
# e.g. "gen1randombattle", "gen2randombattle", "gen3randombattle"
BATTLE_FORMAT = "gen1randombattle"

# ── Opponent for training ─────────────────────────────────────────────────────
# Which model to train against. Options:
#   "random"          →  RandomModel   (uniformly random moves)
#   "strongest_move"  →  StrongestMoveModel  (greedy highest base-power)
#   "model2"          →  Model2  (pre-trained PPO from random_battle_trained_model.pt)
OPPONENT = "random"
