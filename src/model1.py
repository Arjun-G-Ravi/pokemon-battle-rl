
import os
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ──────────────────────────────────────────────────────────────────────────────
#  Constants / lookup tables
# ──────────────────────────────────────────────────────────────────────────────

# All 18 Pokemon types (gen-6+).  Index order is fixed so one-hot vectors
# are consistent across every call.
ALL_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]
TYPE_TO_IDX = {t.lower(): i for i, t in enumerate(ALL_TYPES)}
NUM_TYPES = len(ALL_TYPES)   # 18

# Status conditions  (+ "fnt" for fainted bench slots)
ALL_STATUSES = ["brn", "frz", "par", "psn", "tox", "slp", "fnt"]
STATUS_TO_IDX = {s: i for i, s in enumerate(ALL_STATUSES)}
NUM_STATUSES = len(ALL_STATUSES)   # 7

NUM_BENCH = 5   # slots for bench Pokémon (team size 6 minus the active one)

# Boost stats tracked by poke-env  (atk, def, spa, spd, spe, accuracy, evasion)
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
NUM_BOOSTS = len(BOOST_KEYS)   # 7

# Move category one-hot  [physical, special, non-damaging]
MOVE_CATS = ["physical", "special", "status"]
CAT_TO_IDX = {c: i for i, c in enumerate(MOVE_CATS)}
NUM_CATS = len(MOVE_CATS)   # 3

# Pokemon stats columns in the DB
STAT_COLS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
NUM_STATS = len(STAT_COLS)   # 6

STAT_NORM = 200.0   # normalise HP/ATK/… and move base power by this value
MAX_MOVES = 4

# DB_PATH is resolved relative to this file so the project root doesn't matter
_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "..", "pokemon_database.csv")


def _load_pokemon_db(path: str) -> dict:
    """Load pokemon_database.csv and return a name→stats dict.

    The Name column sometimes contains compound strings like
    "VenusaurMega Venusaur".  We keep the *first* occurrence of each
    Pokédex number so that base forms take priority over megas, but we
    also store cleaned up secondary keys for mega look-ups.
    """
    db: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name_raw = row["Name"].strip()
            # normalise to lower-case and strip spaces for matching
            key = name_raw.lower().replace(" ", "").replace("'", "").replace(".", "").replace("-", "").replace("♀", "f").replace("♂", "m")
            if key not in db:
                db[key] = {
                    "hp":    int(row["HP"]),
                    "atk":   int(row["Attack"]),
                    "def":   int(row["Defense"]),
                    "spa":   int(row["Sp. Atk"]),
                    "spd":   int(row["Sp. Def"]),
                    "spe":   int(row["Speed"]),
                }
    return db

_POKEMON_DB: dict | None = None

def get_pokemon_db() -> dict:
    global _POKEMON_DB
    if _POKEMON_DB is None:
        _POKEMON_DB = _load_pokemon_db(DB_PATH)
    return _POKEMON_DB

def lookup_stats(species: str) -> np.ndarray:
    """Return a length-6 array of base stats, normalised by STAT_NORM.
    Falls back to zeros if the species is not in the database.
    """
    db = get_pokemon_db()
    key = species.lower().replace(" ", "").replace("'", "").replace(".", "").replace("-", "").replace("♀", "f").replace("♂", "m")
    row = db.get(key)
    if row is None:
        return np.zeros(NUM_STATS, dtype=np.float32)
    return np.array(
        [row["hp"], row["atk"], row["def"], row["spa"], row["spd"], row["spe"]],
        dtype=np.float32,
    ) / STAT_NORM

# ──────────────────────────────────────────────────────────────────────────────
#  State embedding
# ──────────────────────────────────────────────────────────────────────────────

def _type_one_hot(pokemon) -> np.ndarray:
    """18-dim one-hot for slot-1 type  +  18-dim one-hot for slot-2 type (zeros if None)."""
    vec = np.zeros(NUM_TYPES * 2, dtype=np.float32)
    types = [t for t in pokemon.types if t is not None]
    if len(types) >= 1:
        idx = TYPE_TO_IDX.get(types[0].name.lower(), -1)
        if idx >= 0:
            vec[idx] = 1.0
    if len(types) >= 2:
        idx = TYPE_TO_IDX.get(types[1].name.lower(), -1)
        if idx >= 0:
            vec[NUM_TYPES + idx] = 1.0
    return vec   # shape: (36,)

def _status_one_hot(pokemon, include_fainted: bool = False) -> np.ndarray:
    """One-hot for the primary status condition (all zeros = healthy).

    If `include_fainted` is True the vector has 7 dims (6 conditions + fainted),
    otherwise it has 6 dims.
    """
    n = NUM_STATUSES if include_fainted else NUM_STATUSES - 1
    vec = np.zeros(n, dtype=np.float32)
    if include_fainted and pokemon.fainted:
        vec[STATUS_TO_IDX["fnt"]] = 1.0
    elif pokemon.status is not None:
        idx = STATUS_TO_IDX.get(pokemon.status.name.lower(), -1)
        if 0 <= idx < n:
            vec[idx] = 1.0
    return vec

def _boost_vec(pokemon) -> np.ndarray:
    """7-dim vector of stat boosts, each normalised to [-1, 1] (raw range is [-6,6])."""
    boosts = pokemon.boosts   # dict-like
    return np.array(
        [boosts.get(k, 0) / 6.0 for k in BOOST_KEYS],
        dtype=np.float32,
    )   # shape: (7,)

def _pokemon_features(pokemon) -> np.ndarray:
    """Encode a single active Pokémon into a fixed-length vector.

    Layout:
      base stats (6)  |  status one-hot (6)  |  type one-hot ×2 (36)
    | stat boosts (7)  |  current HP fraction (1)
    Total = 6 + 6 + 36 + 7 + 1 = 56 dims
    """
    stats   = lookup_stats(pokemon.species)                              # (6,)
    status  = _status_one_hot(pokemon, include_fainted=False)            # (6,)
    types   = _type_one_hot(pokemon)                                     # (36,)
    boosts  = _boost_vec(pokemon)                                        # (7,)
    hp_frac = np.array([float(pokemon.current_hp_fraction)], dtype=np.float32)  # (1,)
    return np.concatenate([stats, status, types, boosts, hp_frac])   # (56,)

def _opp_features(pokemon) -> np.ndarray:
    """Same layout as _pokemon_features but we skip our own boosts for the opponent.
    We DO include their boosts (visible from battle log).

    Layout identical: 56 dims.
    """
    return _pokemon_features(pokemon)

def _move_features(move) -> np.ndarray:
    """Encode a single move.

    Layout:
      base_power (1, normalised)  |  move type one-hot (18)
    | category one-hot (3)
    Total = 1 + 18 + 3 = 22 dims
    """
    bp = np.array([move.base_power / STAT_NORM], dtype=np.float32)          # (1,)

    type_vec = np.zeros(NUM_TYPES, dtype=np.float32)
    if move.type is not None:
        idx = TYPE_TO_IDX.get(move.type.name.lower(), -1)
        if idx >= 0:
            type_vec[idx] = 1.0

    cat_vec = np.zeros(NUM_CATS, dtype=np.float32)
    if move.category is not None:
        idx = CAT_TO_IDX.get(move.category.name.lower(), -1)
        if idx >= 0:
            cat_vec[idx] = 1.0

    return np.concatenate([bp, type_vec, cat_vec])   # (22,)

# Observation dimension breakdown:
#   my active pokemon features  : 56
#   opp active pokemon features : 56
#   4 × move features           : 4 × 22 = 88
#   5 × bench slot features     : 5 × 50 = 250
#     each bench slot: hp (1) + status+fainted (7) + type ×2 (36) + stats (6) = 50
# Total                         = 56 + 56 + 88 + 250 = 450
_BENCH_SLOT_DIM = 1 + NUM_STATUSES + NUM_TYPES * 2 + NUM_STATS   # 1+7+36+6 = 50
OBS_DIM = 56 + 56 + MAX_MOVES * 22 + NUM_BENCH * _BENCH_SLOT_DIM   # = 450


def _bench_features(pokemon) -> np.ndarray:
    """Encode a single bench Pokémon into a 50-dim vector.

    Layout:
      HP fraction (1)  |  status + fainted one-hot (7)  |  type ×2 (36)  |  base stats (6)
    Total = 50 dims
    """
    hp_frac = np.array([float(pokemon.current_hp_fraction)], dtype=np.float32)  # (1,)
    status  = _status_one_hot(pokemon, include_fainted=True)                     # (7,)
    types   = _type_one_hot(pokemon)                                             # (36,)
    stats   = lookup_stats(pokemon.species)                                      # (6,)
    return np.concatenate([hp_frac, status, types, stats])   # (50,)


def _bench_vec(team: dict, active_pokemon) -> np.ndarray:
    """Encode up to NUM_BENCH (5) bench Pokémon into a fixed-size vector.

    Bench = team minus the currently active Pokémon, padded with zeros if fewer
    than 5 bench slots are occupied.
    """
    bench = [
        p for ident, p in team.items()
        if p is not active_pokemon
    ][:NUM_BENCH]

    parts = [_bench_features(p) for p in bench]
    while len(parts) < NUM_BENCH:
        parts.append(np.zeros(_BENCH_SLOT_DIM, dtype=np.float32))

    return np.concatenate(parts)   # (250,)


def build_obs(battle) -> np.ndarray:
    """Convert a poke-env AbstractBattle into a flat float32 observation vector."""
    my  = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    my_vec  = _pokemon_features(my)                    # (56,)

    if opp is not None:
        opp_vec = _opp_features(opp)                   # (56,)
    else:
        opp_vec = np.zeros(56, dtype=np.float32)

    # Moves: pad / truncate to exactly MAX_MOVES slots
    moves = battle.available_moves[:MAX_MOVES]
    move_parts = [_move_features(m) for m in moves]
    while len(move_parts) < MAX_MOVES:
        move_parts.append(np.zeros(22, dtype=np.float32))
    move_vec = np.concatenate(move_parts)              # (88,)

    # Bench Pokémon (our team minus the active one)
    bench_vec = _bench_vec(battle.team, my)            # (250,)

    obs = np.concatenate([my_vec, opp_vec, move_vec, bench_vec])
    assert obs.shape == (OBS_DIM,), f"OBS_DIM mismatch: got {obs.shape[0]}, expected {OBS_DIM}"
    return obs.astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
#  PPO network
# ──────────────────────────────────────────────────────────────────────────────

# Action space: up to 4 moves + up to 5 switches  (standard Gen-1 team size 6)
# We use a fixed size of 9 logits and mask invalid actions.
ACT_DIM = 9   # indices 0-3 = moves, 4-8 = switches


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h     = self.shared(obs)
        logits = self.actor(h)
        value  = self.critic(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor,
                             mask: torch.Tensor,
                             action: torch.Tensor | None = None):
        logits, value = self(obs)
        # Apply mask: set invalid logits to -1e9
        logits = logits + (~mask.bool()).float() * -1e9
        dist   = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value


# ──────────────────────────────────────────────────────────────────────────────
#  Rollout buffer
# ──────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores transitions for one PPO update cycle."""

    def __init__(self):
        self.obs:      list[np.ndarray]  = []
        self.masks:    list[np.ndarray]  = []
        self.actions:  list[int]         = []
        self.log_probs: list[float]      = []
        self.rewards:  list[float]       = []
        self.values:   list[float]       = []
        self.dones:    list[bool]        = []

    def store(self, obs, mask, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ──────────────────────────────────────────────────────────────────────────────
#  PPO trainer
# ──────────────────────────────────────────────────────────────────────────────

class PPOTrainer:
    """Proximal Policy Optimisation update logic."""

    def __init__(
        self,
        net: ActorCritic,
        lr: float           = 3e-4,
        gamma: float        = 0.99,
        gae_lambda: float   = 0.95,
        clip_eps: float     = 0.2,
        vf_coef: float      = 0.5,
        ent_coef: float     = 0.01,
        n_epochs: int       = 4,
        mini_batch: int     = 64,
        max_grad_norm: float = 0.5,
        device: str         = "cpu",
    ):
        self.net           = net
        self.optimizer     = optim.Adam(net.parameters(), lr=lr)
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.vf_coef       = vf_coef
        self.ent_coef      = ent_coef
        self.n_epochs      = n_epochs
        self.mini_batch    = mini_batch
        self.max_grad_norm = max_grad_norm
        self.device        = torch.device(device)

    # ------------------------------------------------------------------
    def _compute_gae(
        self,
        rewards: list[float],
        values:  list[float],
        dones:   list[bool],
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalised Advantage Estimates + returns."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0
        next_val   = last_value

        for t in reversed(range(n)):
            next_non_terminal = 1.0 - float(dones[t])
            delta     = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae  = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            next_val  = values[t]

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    # ------------------------------------------------------------------
    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> dict:
        """Run PPO optimisation on the data in `buffer`.

        Returns a dict with scalar training metrics.
        """
        advantages, returns = self._compute_gae(
            buffer.rewards, buffer.values, buffer.dones, last_value
        )
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs_t      = torch.tensor(np.array(buffer.obs),      dtype=torch.float32).to(self.device)
        masks_t    = torch.tensor(np.array(buffer.masks),    dtype=torch.bool).to(self.device)
        actions_t  = torch.tensor(buffer.actions,            dtype=torch.long).to(self.device)
        old_lp_t   = torch.tensor(buffer.log_probs,          dtype=torch.float32).to(self.device)
        adv_t      = torch.tensor(advantages,                dtype=torch.float32).to(self.device)
        returns_t  = torch.tensor(returns,                   dtype=torch.float32).to(self.device)

        n = len(buffer)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.mini_batch):
                idx = indices[start: start + self.mini_batch]

                _, new_lp, ent, new_val = self.net.get_action_and_value(
                    obs_t[idx], masks_t[idx], actions_t[idx]
                )

                ratio = torch.exp(new_lp - old_lp_t[idx])
                adv_mb = adv_t[idx]

                # Clipped surrogate objective
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = 0.5 * (new_val - returns_t[idx]).pow(2).mean()

                entropy_loss = -ent.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"]  += value_loss.item()
                metrics["entropy"]     += (-entropy_loss).item()

        return metrics


# ──────────────────────────────────────────────────────────────────────────────
#  ACTION → poke-env choice helper
# ──────────────────────────────────────────────────────────────────────────────

def _build_action_mask(battle) -> np.ndarray:
    """Boolean mask of length ACT_DIM marking valid actions."""
    mask = np.zeros(ACT_DIM, dtype=bool)
    for i, _ in enumerate(battle.available_moves[:4]):
        mask[i] = True
    for j, _ in enumerate(battle.available_switches[:5]):
        mask[4 + j] = True
    return mask

def action_idx_to_choice(action_idx: int, battle):
    """Convert an integer action index to a poke-env order object, or None."""
    moves    = battle.available_moves[:4]
    switches = battle.available_switches[:5]

    if action_idx < 4:
        if action_idx < len(moves):
            return moves[action_idx]
    else:
        switch_idx = action_idx - 4
        if switch_idx < len(switches):
            return switches[switch_idx]
    # Fallback: random valid choice
    choices = battle.available_moves + battle.available_switches
    return random.choice(choices) if choices else None


# ──────────────────────────────────────────────────────────────────────────────
#  Model1  –  the class your battle server uses
# ──────────────────────────────────────────────────────────────────────────────

class Model1:
    """PPO-based Pokemon battle agent.

    Usage in a training loop::

        model = Model1()
        # … run battles, calling model.predict_rl(battle, store=True) …
        # … after episode ends call model.finish_episode(won=True/False) …
        # … periodically call model.update() …
    """

    CHECKPOINT = os.path.join(_HERE, "..", "checkpoints", "model1_ppo.pt")

    def __init__(
        self,
        lr: float           = 3e-4,
        gamma: float        = 0.99,
        gae_lambda: float   = 0.95,
        clip_eps: float     = 0.2,
        vf_coef: float      = 0.5,
        ent_coef: float     = 0.01,
        n_epochs: int       = 4,
        mini_batch: int     = 64,
        max_grad_norm: float = 0.5,
        hidden_size: int    = 256,
        device: str         = "cpu",
    ):
        self.device = torch.device(device)
        self.net    = ActorCritic(hidden=hidden_size).to(self.device)
        self.trainer = PPOTrainer(
            self.net,
            lr=lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
            n_epochs=n_epochs, mini_batch=mini_batch,
            max_grad_norm=max_grad_norm,
            device=device,
        )
        self.buffer = RolloutBuffer()
        self._last_obs: np.ndarray | None  = None
        self._last_val: float              = 0.0
        self._pending: tuple | None        = None   # (obs, mask, action, log_prob, value)

    # ------------------------------------------------------------------
    #  Inference helpers
    # ------------------------------------------------------------------

    def predict(self, battle) -> object:
        """Random policy (used when not training with RL)."""
        choices = battle.available_moves + battle.available_switches
        return random.choice(choices) if choices else None

    @torch.no_grad()
    def predict_rl(self, battle, store: bool = True) -> object:
        """Select action using the current policy.

        When ``store=True`` the transition is added to the rollout buffer
        so it can be used for a subsequent PPO update.
        """
        obs  = build_obs(battle)
        mask = _build_action_mask(battle)

        obs_t  = torch.tensor(obs,  dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)

        action, log_prob, _, value = self.net.get_action_and_value(obs_t, mask_t)

        action   = action.item()
        log_prob = log_prob.item()
        value    = value.item()

        if store:
            # reward = 0 at decision time; updated by finish_episode / step_reward
            self._pending = (obs, mask, action, log_prob, value)

        self._last_obs = obs
        self._last_val = value

        return action_idx_to_choice(action, battle)

    def step_reward(self, reward: float, done: bool = False) -> None:
        """Store the reward for the most recently generated action."""
        pending = self._pending
        if pending is not None:
            obs, mask, action, log_prob, value = pending
            self.buffer.store(obs, mask, action, log_prob, reward, value, done)
            self._pending = None

    def finish_episode(self, won: bool) -> None:
        """Call at the end of a battle with the episode outcome."""
        reward = 1.0 if won else -1.0
        self.step_reward(reward, done=True)

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """Run one PPO optimisation pass on the collected rollout buffer.

        Returns training metrics and clears the buffer.
        """
        if len(self.buffer) == 0:
            return {}
        metrics = self.trainer.update(self.buffer, last_value=self._last_val)
        self.buffer.clear()
        return metrics

    def save(self, path: str | None = None) -> None:
        path = path or self.CHECKPOINT
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"net": self.net.state_dict()}, path)
        pass  # checkpoint saved silently

    def load(self, path: str | None = None) -> None:
        path = path or self.CHECKPOINT
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        print(f"[Model1] Loaded checkpoint ← {path}")

    # ------------------------------------------------------------------
    #  Debug
    # ------------------------------------------------------------------

    def _print_state(self, battle) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  TURN {battle.turn}  |  {battle.battle_tag}")
        print(sep)

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

        opp = battle.opponent_active_pokemon
        print(f"\n[OPP ACTIVE] {opp.species.upper() if opp else '???'}")
        if opp:
            print(f"  HP        : {opp.current_hp_fraction * 100:.1f}%")
            print(f"  Status    : {opp.status.name if opp.status else 'none'}")
            print(f"  Types     : {[t.name for t in opp.types if t]}")
            print(f"  Boosts    : {dict(opp.boosts)}")

        if battle.available_switches:
            print(f"\n  Available switches:")
            for p in battle.available_switches:
                print(
                    f"    • {p.species:<18} hp={p.current_hp_fraction * 100:.1f}%  "
                    f"status={p.status.name if p.status else 'none'}"
                )

        print(f"\n[MY TEAM]")
        for p in battle.team.values():
            fainted = "FAINTED" if p.fainted else f"hp={p.current_hp_fraction * 100:.1f}%"
            print(f"    • {p.species:<18} {fainted}  status={p.status.name if p.status else 'none'}")

        print(f"\n[OPP TEAM]")
        for p in battle.opponent_team.values():
            fainted = "FAINTED" if p.fainted else f"hp={p.current_hp_fraction * 100:.1f}%"
            print(f"    • {p.species:<18} {fainted}  status={p.status.name if p.status else 'none'}")

        print(f"\n[FIELD]")
        print(f"  Weather   : {battle.weather}")
        print(f"  My side   : {[e.name for e in battle.side_conditions]}")
        print(f"  Opp side  : {[e.name for e in battle.opponent_side_conditions]}")
        print(sep)

        obs = build_obs(battle)
        print(f"\n[OBS VECTOR]  shape={obs.shape}  min={obs.min():.3f}  max={obs.max():.3f}")
        print(sep)