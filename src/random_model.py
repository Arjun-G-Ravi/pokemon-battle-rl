"""
random_model.py  –  Baseline: uniformly random move selection.
"""

import random


class RandomModel:
    """Picks a random move or switch every turn."""

    def predict(self, battle):
        choices = battle.available_moves + battle.available_switches
        return random.choice(choices) if choices else None
