"""
strongest_move_model.py  –  Greedy baseline: always uses the highest base-power move.

If no damaging move is available it falls back to the first available move/switch.
"""


class StrongestMoveModel:
    """Always picks the move with the highest base power.

    Tie-breaking order: base_power desc → first in list.
    Falls back to a switch (or any choice) when no damaging move exists.
    """

    def predict(self, battle):
        moves = battle.available_moves

        # Filter to moves that actually do damage
        damaging = [m for m in moves if m.base_power > 0]

        if damaging:
            return max(damaging, key=lambda m: m.base_power)

        # No damaging move available – try any move, then switches
        if moves:
            return moves[0]
        if battle.available_switches:
            return battle.available_switches[0]
        return None
