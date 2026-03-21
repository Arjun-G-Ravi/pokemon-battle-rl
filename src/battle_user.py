'''
This script allows a user to play a battle against a model.
The model is selected here and used to drive the bot's decisions.
'''
import asyncio
import os
import sys

from poke_env import AccountConfiguration
from poke_env.player import Player
from dotenv import load_dotenv

# ── Choose your model here ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model1 import Model1
from model2 import Model2

MODEL = Model1()   # swap to Model2() to use the first-move model
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()
trainer_name = os.getenv('trainer_name')


class ModelPlayer(Player):
    """A poke-env Player that delegates move choice to a Model instance."""

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def choose_move(self, battle):
        choice = self.model.predict(battle)
        if choice is not None:
            return self.create_order(choice)
        # fallback: let poke-env pick (should rarely happen)
        return self.choose_random_move(battle)


bot_account = AccountConfiguration("ModelBot", None)
bot = ModelPlayer(
    model=MODEL,
    account_configuration=bot_account,
    battle_format="gen1randombattle",
)

print(f"Challenging {trainer_name} with {type(MODEL).__name__}....")
asyncio.run(bot.send_challenges(trainer_name, n_challenges=1))

print("\nBattle finished!")
for tag, battle in bot.battles.items():
    if battle.won:
        print("Bot won the match")
    else:
        print(f"{trainer_name} won the match")