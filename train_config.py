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
