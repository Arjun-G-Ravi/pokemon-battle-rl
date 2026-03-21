"""
train.py  –  PPO training loop for Model1

Edit train_config.py (in the project root) to change hyperparameters,
then run:

    source pokemon_env/bin/activate.fish
    python src/train.py

Make sure the Pokemon Showdown server is running first:
    node pokemon-showdown start --no-security
"""

import asyncio
import os
import sys
from datetime import datetime
import matplotlib
matplotlib.use("Agg")   # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from poke_env import AccountConfiguration
from poke_env.player import Player, RandomPlayer

sys.path.insert(0, os.path.dirname(__file__))
from model1 import Model1, build_obs, _build_action_mask, action_idx_to_choice
from random_model import RandomModel
from strongest_move_model import StrongestMoveModel
from model2 import Model2

# ──────────────────────────────────────────────────────────────────────────────
#  Opponent model → poke-env Player wrapper
# ──────────────────────────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

class OpponentPlayer(Player):
    """Wraps any model with a .predict(battle) interface into a poke-env Player."""

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def choose_move(self, battle):
        choice = self.model.predict(battle)
        if choice is not None:
            return self.create_order(choice)
        return self.choose_random_move(battle)


def _make_opponent(entry: str, battle_format: str) -> Player:
    """Instantiate one opponent Player.

    `entry` can be:
      - "random"          → RandomModel
      - "strongest_move"  → StrongestMoveModel
      - "model2"          → Model2 (fixed pretrained PPO)
      - any path ending in .pt  → Model1 loaded from that checkpoint
    """
    key = entry.strip()

    # Checkpoint path?
    if key.endswith(".pt") or os.path.isfile(key):
        ckpt_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", key)
            if not os.path.isabs(key) else key
        )
        m = Model1()
        if os.path.exists(ckpt_path):
            m.load(ckpt_path)
            label = f"Model1 ← {ckpt_path}"
        else:
            label = f"Model1 (untrained, path not found: {ckpt_path})"

        class _PPOInference:
            """Thin wrapper that calls predict_rl in inference-only mode."""
            def __init__(self, model): self._m = model
            def predict(self, battle): return self._m.predict_rl(battle, store=False)

        model = _PPOInference(m)
    elif key.lower() == "random":
        model = RandomModel()
        label = "RandomModel"
    elif key.lower() == "strongest_move":
        model = StrongestMoveModel()
        label = "StrongestMoveModel"
    elif key.lower() == "model2":
        model = Model2()
        label = "Model2 (pretrained PPO)"
    else:
        raise ValueError(
            f"Unknown OPPONENT entry '{key}'. "
            "Use 'random', 'strongest_move', 'model2', or a path to a .pt file."
        )

    print(f"  [Opponent pool] {label}")
    return OpponentPlayer(
        model=model,
        account_configuration=AccountConfiguration("Opp_Player", None),
        battle_format=battle_format,
    )


def _build_opponent_pool(
    opponents: list | str,
    battle_format: str,
) -> list:
    """Build a list of OpponentPlayers from the config OPPONENT value.

    Accepts a single string or a list of strings/paths.
    """
    if isinstance(opponents, str):
        opponents = [opponents]
    return [_make_opponent(entry, battle_format) for entry in opponents]

# ──────────────────────────────────────────────────────────────────────────────
#  Reward shaping
# ──────────────────────────────────────────────────────────────────────────────

_DAMAGING_STATUSES = {"brn", "par", "psn", "tox"}
_SLEEP_STATUS      = {"slp"}


class BattleSnapshot:
    """Lightweight snapshot of trackable per-turn battle state."""
    __slots__ = ("opp_fainted", "my_fainted", "opp_hp_fracs", "opp_statuses")

    def __init__(self, battle):
        self.opp_fainted  = sum(1 for p in battle.opponent_team.values() if p.fainted)
        self.my_fainted   = sum(1 for p in battle.team.values() if p.fainted)
        self.opp_hp_fracs = {
            ident: p.current_hp_fraction
            for ident, p in battle.opponent_team.items()
        }
        self.opp_statuses = {
            ident: (p.status.name.lower() if p.status else None)
            for ident, p in battle.opponent_team.items()
        }


def _turn_reward(prev: BattleSnapshot, curr: BattleSnapshot) -> float:
    """Shaped per-turn reward:
      +1.0   opponent pokemon fainted
      -0.5   my pokemon fainted
      +0.1   dealt > 50% HP to any opponent pokemon this turn
      +0.2   inflicted sleep on opponent
      +0.05  inflicted burn / paralysis / poison on opponent
    """
    r = 0.0

    r += (curr.opp_fainted - prev.opp_fainted) * 1.0
    r -= (curr.my_fainted  - prev.my_fainted)  * 0.5

    for ident, prev_hp in prev.opp_hp_fracs.items():
        curr_hp = curr.opp_hp_fracs.get(ident, prev_hp)
        if (prev_hp - curr_hp) > 0.5:
            r += 0.1

        prev_status = prev.opp_statuses.get(ident)
        curr_status = curr.opp_statuses.get(ident)
        if prev_status is None and curr_status is not None:
            if curr_status in _SLEEP_STATUS:
                r += 0.2
            elif curr_status in _DAMAGING_STATUSES:
                r += 0.05

    return r


# ──────────────────────────────────────────────────────────────────────────────
#  PPO Player
# ──────────────────────────────────────────────────────────────────────────────

class PPOPlayer(Player):
    """poke-env Player that drives decisions through Model1 (PPO)."""

    def __init__(self, model: Model1, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self._prev_snap: dict[str, BattleSnapshot] = {}
        self.wins   = 0
        self.losses = 0

    def choose_move(self, battle):
        tag  = battle.battle_tag
        curr = BattleSnapshot(battle)

        if battle.turn > 1 and tag in self._prev_snap:
            r = _turn_reward(self._prev_snap[tag], curr)
            self.model.step_reward(r, done=False)

        self._prev_snap[tag] = curr

        choice = self.model.predict_rl(battle, store=True)
        if choice is not None:
            return self.create_order(choice)
        return self.choose_random_move(battle)

    def register_result(self, battle) -> None:
        """Apply terminal reward: +10 win / -10 loss + surviving HP advantage."""
        won = battle.won is True
        if won:
            self.wins += 1
        else:
            self.losses += 1

        my_hp  = sum(p.current_hp_fraction for p in battle.team.values())
        opp_hp = sum(p.current_hp_fraction for p in battle.opponent_team.values())
        terminal_r = (10.0 if won else -10.0) + (my_hp - opp_hp)

        self.model.finish_episode(won=won)
        if self.model.buffer.rewards:
            self.model.buffer.rewards[-1] = terminal_r

        self._prev_snap.pop(battle.battle_tag, None)


# ──────────────────────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────────────────────

async def train(
    n_battles: int       = 500,
    update_every: int    = 10,
    lr: float            = 3e-4,
    gamma: float         = 0.99,
    gae_lambda: float    = 0.95,
    clip_eps: float      = 0.2,
    vf_coef: float       = 0.5,
    ent_coef: float      = 0.01,
    n_epochs: int        = 4,
    mini_batch: int      = 64,
    max_grad_norm: float = 0.5,
    hidden_size: int     = 256,
    device: str          = "cpu",
    battle_format: str   = "gen1randombattle",
    opponents: list      = None,
    save_name: str       = "model1_ppo",
):
    if opponents is None:
        opponents = ["random"]

    # Derive checkpoint path from save_name
    _ckpt_dir  = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    _ckpt_path = os.path.normpath(os.path.join(_ckpt_dir, f"{save_name}.pt"))

    model = Model1(
        lr=lr, gamma=gamma, gae_lambda=gae_lambda,
        clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
        n_epochs=n_epochs, mini_batch=mini_batch,
        max_grad_norm=max_grad_norm, hidden_size=hidden_size,
        device=device,
    )
    # Override the default checkpoint path with the configured save name
    model.CHECKPOINT = _ckpt_path

    if os.path.exists(_ckpt_path):
        model.load(_ckpt_path)

    ppo_player = PPOPlayer(
        model=model,
        account_configuration=AccountConfiguration("PPO_Trainer", None),
        battle_format=battle_format,
    )

    print(f"\n{'='*60}")
    print(f"  PPO Training  |  {n_battles} battles  |  update every {update_every}")
    print(f"  Checkpoint   →  {_ckpt_path}")
    opp_pool = _build_opponent_pool(opponents, battle_format)
    print(f"{'='*60}\n")

    completed  = 0
    batch_num  = 0
    n_batches  = (n_battles + update_every - 1) // update_every

    # ── Metric history (one entry per batch) ─────────────────────────────────
    history: dict[str, list] = {
        "episode":     [],
        "win_rate":    [],
        "policy_loss": [],
        "value_loss":  [],
        "entropy":     [],
    }

    while completed < n_battles:
        batch_size = min(update_every, n_battles - completed)
        batch_num += 1

        # Pick a random opponent from the pool each batch
        import random as _rnd
        opp_player = _rnd.choice(opp_pool)

        # Run the full batch in one shot
        await ppo_player.battle_against(opp_player, n_battles=batch_size)

        # ── Collect terminal rewards for every battle in this batch ──────────
        for battle in ppo_player.battles.values():
            if battle.finished and battle.battle_tag not in getattr(ppo_player, "_processed", set()):
                ppo_player.register_result(battle)
                if not hasattr(ppo_player, "_processed"):
                    ppo_player._processed = set()
                ppo_player._processed.add(battle.battle_tag)

        completed += batch_size

        # ── PPO update ───────────────────────────────────────────────────────
        metrics = model.update()
        model.save()

        total = ppo_player.wins + ppo_player.losses
        win_rate = ppo_player.wins / total if total > 0 else 0.0
        pol_loss = metrics.get("policy_loss", float("nan"))
        val_loss = metrics.get("value_loss",  float("nan"))
        entropy  = metrics.get("entropy",      float("nan"))

        # record for plot
        history["episode"].append(completed)
        history["win_rate"].append(win_rate * 100)
        history["policy_loss"].append(pol_loss)
        history["value_loss"].append(val_loss)
        history["entropy"].append(entropy)

        print(
            f"[batch {batch_num:>3}/{n_batches}  ep {completed:>5}/{n_battles}]  "
            f"win%={win_rate*100:.1f}  "
            f"pol_loss={pol_loss:.4f}  val_loss={val_loss:.4f}  entropy={entropy:.4f}"
        )

        # update the plot every time the checkpoint is saved
        plot_training(history, model.CHECKPOINT)

    print(f"\n{'='*60}")
    total = ppo_player.wins + ppo_player.losses
    print(f"  Training complete!  Final win rate: {ppo_player.wins}/{total} = {ppo_player.wins/total*100:.1f}%")
    print(f"  Checkpoint: {model.CHECKPOINT}")
    print(f"{'='*60}\n")




# ──────────────────────────────────────────────────────────────────────────────
#  Plot helper
# ──────────────────────────────────────────────────────────────────────────────

def plot_training(history: dict, checkpoint_path: str) -> None:
    """Save a 2×2 training dashboard to the checkpoints directory."""
    eps       = history["episode"]
    win_rate  = history["win_rate"]
    pol_loss  = history["policy_loss"]
    val_loss  = history["value_loss"]
    entropy   = history["entropy"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle("PPO Training Dashboard", fontsize=16, fontweight="bold",
                 color="white", y=0.98)

    _PANEL_BG   = "#1a1a2e"
    _GRID_COLOR = "#2a2a4a"
    _TEXT_COLOR = "#c8c8e8"

    panel_cfg = [
        (axes[0, 0], win_rate,  "Win Rate (%)",    "#4ade80", (0, 100)),
        (axes[0, 1], pol_loss,  "Policy Loss",     "#f97316", None),
        (axes[1, 0], val_loss,  "Value Loss",      "#38bdf8", None),
        (axes[1, 1], entropy,   "Entropy",         "#a78bfa", None),
    ]

    for ax, data, title, color, ylim in panel_cfg:
        ax.set_facecolor(_PANEL_BG)
        ax.plot(eps, data, color=color, linewidth=1.8, alpha=0.9, zorder=3)

        # smoothed trend (window = 10% of data)
        w = max(3, len(data) // 10)
        if len(data) >= w:
            import numpy as np
            kernel  = np.ones(w) / w
            smoothed = np.convolve(data, kernel, mode="valid")
            valid_eps = eps[w - 1:]
            ax.plot(valid_eps, smoothed, color="white", linewidth=2.5,
                    alpha=0.6, linestyle="--", zorder=4, label="Moving avg")

        ax.set_title(title, color=_TEXT_COLOR, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Episode", color=_TEXT_COLOR, fontsize=9)
        ax.tick_params(colors=_TEXT_COLOR, labelsize=8)
        ax.spines[:].set_color(_GRID_COLOR)
        ax.grid(True, color=_GRID_COLOR, linewidth=0.6, zorder=0)
        if ylim:
            ax.set_ylim(*ylim)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        # fill under the curve
        ax.fill_between(eps, data, alpha=0.15, color=color, zorder=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir  = os.path.dirname(checkpoint_path)
    model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    plot_path = os.path.join(out_dir, f"training_plot_{model_name}_{timestamp}.png")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    # print(f"[Plot] Updated → {plot_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point  –  reads from train_config.py in the project root
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # train_config.py lives one level above src/
    _config_path = os.path.join(os.path.dirname(__file__), "..", "train_config.py")
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("train_config", _config_path)
    cfg   = _ilu.module_from_spec(_spec)  # type: ignore
    _spec.loader.exec_module(cfg)         # type: ignore

    print(f"Config loaded from: {os.path.normpath(_config_path)}")
    print(
        f"  battles={cfg.N_BATTLES}  update_every={cfg.UPDATE_EVERY}  "
        f"format={cfg.BATTLE_FORMAT}  save_name={cfg.SAVE_NAME}\n"
        f"  opponents={cfg.OPPONENT}\n"
        f"  lr={cfg.LR}  gamma={cfg.GAMMA}  gae_lambda={cfg.GAE_LAMBDA}  "
        f"clip_eps={cfg.CLIP_EPS}  vf_coef={cfg.VF_COEF}  ent_coef={cfg.ENT_COEF}\n"
        f"  n_epochs={cfg.N_EPOCHS}  mini_batch={cfg.MINI_BATCH}  "
        f"max_grad_norm={cfg.MAX_GRAD_NORM}  hidden={cfg.HIDDEN_SIZE}  device={cfg.DEVICE}\n"
    )

    asyncio.run(train(
        n_battles=cfg.N_BATTLES,
        update_every=cfg.UPDATE_EVERY,
        lr=cfg.LR,
        gamma=cfg.GAMMA,
        gae_lambda=cfg.GAE_LAMBDA,
        clip_eps=cfg.CLIP_EPS,
        vf_coef=cfg.VF_COEF,
        ent_coef=cfg.ENT_COEF,
        n_epochs=cfg.N_EPOCHS,
        mini_batch=cfg.MINI_BATCH,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        hidden_size=cfg.HIDDEN_SIZE,
        device=cfg.DEVICE,
        battle_format=cfg.BATTLE_FORMAT,
        opponents=cfg.OPPONENT,
        save_name=cfg.SAVE_NAME,
    ))
