'''
Model2: Always chooses the first available move each turn.
'''


class Model2:
    """First-move model — always picks the first available move each turn."""

    def predict(self, battle):
        """
        Given a poke-env Battle object, return a move or switch.
        Returns the first available move, or first available switch if no moves.
        """
        if battle.available_moves:
            return battle.available_moves[0]
        if battle.available_switches:
            return battle.available_switches[0]
        return None
