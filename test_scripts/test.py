"""
Basic script to connect to the local Pokemon Showdown server using poke-env.
Server must be running at localhost:8000 (node pokemon-showdown start --no-security)

By default, Player uses LocalhostServerConfiguration which targets localhost:8000.
With --no-security, no credentials are needed.
"""

import asyncio
from poke_env.player import RandomPlayer


async def main():
    # No args needed — LocalhostServerConfiguration is the default (localhost:8000)
    # With --no-security, no password is required either.
    player = RandomPlayer(battle_format="gen8randombattle")

    print(f"✅ Connected to Pokemon Showdown at localhost:8000")
    print(f"   Player username: {player.username}")

    # Give the websocket a moment to fully connect, then disconnect cleanly
    await asyncio.sleep(3)
    await player.ps_client.stop_listening()
    print("Connection closed.")


if __name__ == "__main__":
    asyncio.run(main())
