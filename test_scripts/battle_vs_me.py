import asyncio
import sys
import uuid
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
import os
from dotenv import load_dotenv

load_dotenv()


async def main(human_username: str):
    uid = uuid.uuid4().hex[:6]
    bot_account = AccountConfiguration(f"RandBot_{uid}", None)

    bot = RandomPlayer(
        account_configuration=bot_account,
        battle_format="gen8randombattle",
    )

    print(f"🤖 Bot connected as: {bot.username}")
    print(f"👤 Challenging you ({human_username}) to a Gen 8 random battle...")
    print(f"   ➡️  Go to http://localhost:8000 and ACCEPT the challenge!\n")

    # Challenge the human player; wait until 1 battle is finished
    await bot.send_challenges(human_username, n_challenges=1)

    # --- Results ---
    print("\n" + "=" * 50)
    print("Battle finished!")
    for tag, battle in bot.battles.items():
        bot_result   = "WIN ✅" if battle.won     else "LOSS ❌"
        human_result = "WIN ✅" if not battle.won else "LOSS ❌"
        print(f"  Bot   ({bot.username}): {bot_result}")
        print(f"  You   ({human_username}):  {human_result}")
    print("=" * 50)


if __name__ == "__main__":
    human = os.getenv('trainer_name')
    asyncio.run(main(human))
