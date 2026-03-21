
import random


class Model1:


    def predict(self, battle):

        # self._print_state(battle)

        choices = battle.available_moves + battle.available_switches
        if not choices:
            return None
        return random.choice(choices)
    
    def train_rl(self):
        pass

    def predict_rl(self):
        pass



    def _print_state(self, battle):
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  TURN {battle.turn}  |  {battle.battle_tag}")
        print(sep)

        # ── My active pokemon ────────────────────────────────────────────────
        my = battle.active_pokemon
        print(f"\n[MY ACTIVE]  {my.species.upper()}")
        print(f"  HP        : {my.current_hp_fraction * 100:.1f}%")
        print(f"  Status    : {my.status.name if my.status else 'none'}")
        print(f"  Types     : {[t.name for t in my.types if t]}")
        print(f"  Boosts    : {dict(my.boosts)}")

        print(f"\n  Available moves:")
        for m in battle.available_moves:
            acc = f"{m.accuracy * 100:.0f}%" if isinstance(m.accuracy, float) else str(m.accuracy)
            print(
                f"    • {m.id:<18} type={m.type.name:<10} "
                f"bp={m.base_power:<5} acc={acc:<6} pp={m.current_pp}/{m.max_pp}"
            )

        # ── Opponent active pokemon ──────────────────────────────────────────
        opp = battle.opponent_active_pokemon
        print(f"\n[OPP ACTIVE] {opp.species.upper() if opp else '???'}")
        if opp:
            print(f"  HP        : {opp.current_hp_fraction * 100:.1f}%")
            print(f"  Status    : {opp.status.name if opp.status else 'none'}")
            print(f"  Types     : {[t.name for t in opp.types if t]}")
            print(f"  Boosts    : {dict(opp.boosts)}")

        # ── Available switches ───────────────────────────────────────────────
        if battle.available_switches:
            print(f"\n  Available switches:")
            for p in battle.available_switches:
                print(
                    f"    • {p.species:<18} hp={p.current_hp_fraction * 100:.1f}%  "
                    f"status={p.status.name if p.status else 'none'}"
                )

        # ── Full team overview ───────────────────────────────────────────────
        print(f"\n[MY TEAM]")
        for p in battle.team.values():
            fainted = "FAINTED" if p.fainted else f"hp={p.current_hp_fraction * 100:.1f}%"
            print(f"    • {p.species:<18} {fainted}  status={p.status.name if p.status else 'none'}")

        print(f"\n[OPP TEAM]")
        for p in battle.opponent_team.values():
            fainted = "FAINTED" if p.fainted else f"hp={p.current_hp_fraction * 100:.1f}%"
            print(f"    • {p.species:<18} {fainted}  status={p.status.name if p.status else 'none'}")

        # ── Weather / field ──────────────────────────────────────────────────
        print(f"\n[FIELD]")
        print(f"  Weather   : {battle.weather}")
        print(f"  My side   : {[e.name for e in battle.side_conditions]}")
        print(f"  Opp side  : {[e.name for e in battle.opponent_side_conditions]}")
        print(sep)