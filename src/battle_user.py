'''
This script will allow a user to play a battle against a model.
'''
import asyncio
import sys
import uuid
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
import os
from dotenv import load_dotenv
# from model1 import Model # choose the model here

load_dotenv()
trainer_name = os.getenv('trainer_name')
# model = Model()

bot_account = AccountConfiguration(f"RandBot", None)
bot = RandomPlayer(
        account_configuration=bot_account,
        battle_format="gen1randombattle",
    )
print(f'Challenging {trainer_name}....')
asyncio.run(bot.send_challenges(trainer_name, n_challenges=1))

print("Battle finished!")
for tag, battle in bot.battles.items():
    if battle.won:
        print('Bot won the match')
    else:
        print(f'{trainer_name} won the match')