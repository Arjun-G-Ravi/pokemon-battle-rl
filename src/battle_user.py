"""
battle_user.py  –  Challenge the trained PPO bot

Loads the latest Model1 checkpoint and lets the local trainer account
challenge it for one battle.

Usage:
    source pokemon_env/bin/activate.fish
    python src/battle_user.py

The .env file must contain:
    trainer_name=<your Showdown username>

Make sure the Pokemon Showdown server is running:
    node pokemon-showdown start --no-security
"""

import asyncio
import os
import sys

from poke_env import AccountConfiguration
from poke_env.player import Player
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from model1 import Model1, action_idx_to_choice

# ──────────────────────────────────────────────────────────────────────────────
#  Load trained model
# ──────────────────────────────────────────────────────────────────────────────

MODEL = Model1()

if os.path.exists(MODEL.CHECKPOINT):
    MODEL.load()
    print(f"✓ Loaded checkpoint from {MODEL.CHECKPOINT}")
else:
    print(f"⚠  No checkpoint found at {MODEL.CHECKPOINT}")
    print(f"   Tip: run  python src/train.py  first to train the model.")
    print(f"   The bot will use its untrained (random-ish) policy.\n")

# ──────────────────────────────────────────────────────────────────────────────
#  Player wrapper
# ──────────────────────────────────────────────────────────────────────────────

class PPOBotPlayer(Player):
    """poke-env Player that uses the trained PPO policy (no exploration / no buffer writes)."""

    def __init__(self, model: Model1, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def choose_move(self, battle):
        # predict_rl with store=False  →  pure inference, no rollout saved
        choice = self.model.predict_rl(battle, store=False)
        if choice is not None:
            return self.create_order(choice)
        return self.choose_random_move(battle)


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()
trainer_name = os.getenv("trainer_name")
if not trainer_name:
    raise SystemExit(
        "ERROR: 'trainer_name' not set in .env\n"
        "Create a .env file in the project root with:\n"
        "    trainer_name=YourShowdownUsername"
    )

bot_account = AccountConfiguration("PPO_Bot", None)
bot = PPOBotPlayer(
    model=MODEL,
    account_configuration=bot_account,
    battle_format="gen1randombattle",
)

print(f"Challenging {trainer_name} …  (format: gen1randombattle)\n")
asyncio.run(bot.send_challenges(trainer_name, n_challenges=1))

print("\nBattle finished!")
for tag, battle in bot.battles.items():
    winner = "PPO Bot" if battle.won else trainer_name
    print(f"  {tag}  →  {winner} won")