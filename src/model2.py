"""
model2.py  –  Pretrained PPO bot loaded from the random-battle checkpoint.

Uses the same Model1 architecture but loads a specific checkpoint so it
acts as a fixed, stable opponent during training.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model1 import Model1

_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "random_battle_trained_model.pt"
)


class Model2:
    """PPO bot pre-trained on gen1randombattle against a random opponent.

    Loads the checkpoint at ``checkpoints/random_battle_trained_model.pt``
    and runs pure inference (no training / no buffer writes).
    """

    def __init__(self):
        self._model = Model1()
        if os.path.exists(_CHECKPOINT):
            self._model.load(_CHECKPOINT)
            print(f"[Model2] Loaded checkpoint ← {_CHECKPOINT}")
        else:
            print(f"[Model2] ⚠  Checkpoint not found at {_CHECKPOINT}")
            print(f"         Running with untrained weights.")

    def predict(self, battle):
        """Inference-only prediction (no exploration, no buffer)."""
        choice = self._model.predict_rl(battle, store=False)
        if choice is not None:
            return choice
        # fallback
        choices = battle.available_moves + battle.available_switches
        import random
        return random.choice(choices) if choices else None
