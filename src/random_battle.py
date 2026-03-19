"""
Starts a Gen 8 random battle between two RandomPlayers on the local
Pokemon Showdown server (localhost:8000).

Run the server first:
    node pokemon-showdown start --no-security

Then run this script:
    python random_battle.py

You can watch the battle live at http://localhost:8000
"""

import asyncio
import uuid
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer


def make_player(name: str) -> RandomPlayer:
    """Create a RandomPlayer with a unique username to avoid nametaken errors."""
    uid = uuid.uuid4().hex[:6]
    account = AccountConfiguration(f"{name}_{uid}", None)
    return RandomPlayer(
        account_configuration=account,
        battle_format="gen8randombattle",
    )


async def main():
    player1 = make_player("Bot1")
    player2 = make_player("Bot2")

    print(f"⚔️  Starting battle: {player1.username} vs {player2.username}")
    print("   Watch live at: http://localhost:8000\n")

    await player1.battle_against(player2, n_battles=1)

    # --- Results ---
    print("=" * 50)
    print("Battle finished!")
    print(
        f"  {player1.username}: "
        f"{player1.n_won_battles}W / {player1.n_finished_battles} played"
    )
    print(
        f"  {player2.username}: "
        f"{player2.n_won_battles}W / {player2.n_finished_battles} played"
    )
    print("=" * 50)

    for tag, battle in player1.battles.items():
        result = "WIN ✅" if battle.won else "LOSS ❌"
        print(f"  {tag}: {player1.username} → {result}")


if __name__ == "__main__":
    asyncio.run(main())
