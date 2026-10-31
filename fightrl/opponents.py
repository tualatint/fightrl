"""Opponents: scripted bots, frozen policies, and a self-play league."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .core.config import GameConfig
from .core.controller import ControllerState
from .core.state import GameState

# Moves are named after the generic action buttons; bots pick from whatever
# the game actually exposes.
_PUNCHES = ("LP", "MP", "HP")
_KICKS = ("LK", "MK", "HK")


class Opponent:
    """Per-frame opponent interface."""

    def reset(self) -> None:
        pass

    def act(self, obs: dict[str, np.ndarray] | None, info: dict[str, Any], state: GameState) -> ControllerState:
        return ControllerState()


class ScriptedBot(Opponent):
    """Heuristic CPU opponent.

    Styles: ``aggressive``, ``defensive``, ``random``, ``passive``.
    Decisions are made every ``interval`` frames and held in between.
    """

    def __init__(self, style: str = "aggressive", buttons: Sequence[str] | None = None,
                 interval: int = 4, seed: int | None = None):
        self.style = style
        self.buttons = list(buttons or ("LP", "HP", "LK", "HK"))
        self.interval = interval
        self.rng = np.random.default_rng(seed)
        self._tick = 0
        self._ctrl = ControllerState()

    def reset(self) -> None:
        self._tick = 0
        self._ctrl = ControllerState()

    def _pick_attack(self) -> ControllerState:
        available = [b for b in self.buttons if b in _PUNCHES + _KICKS] or list(self.buttons)
        if not available:
            return ControllerState()
        weight = np.array([3.0 if b in ("HP", "HK") else 2.0 if b in ("MP", "MK") else 1.0
                           for b in available])
        weight = weight / weight.sum()
        button = str(self.rng.choice(available, p=weight))
        if self.rng.random() < 0.25:
            return ControllerState("down", frozenset({button}))
        return ControllerState("neutral", frozenset({button}))

    def act(self, obs, info, state: GameState) -> ControllerState:
        self._tick += 1
        if self._tick % self.interval != 0:
            return self._ctrl
        facing = -state.p1_facing  # the opponent faces the agent
        fwd = "right" if facing > 0 else "left"
        back = "left" if facing > 0 else "right"
        dist = state.distance if state.distance is not None else 120.0
        rng = self.rng.random()

        if self.style == "random":
            direction = str(self.rng.choice(
                ["neutral", "left", "right", "up", "down", "up_left", "up_right", "down_left", "down_right"]
            ))
            buttons = frozenset(b for b in self.buttons if self.rng.random() < 0.08)
            self._ctrl = ControllerState(direction, buttons)
        elif self.style == "passive":
            self._ctrl = ControllerState(back if rng < 0.6 else "neutral")
        elif self.style == "defensive":
            if dist < 55 and rng < 0.45:
                self._ctrl = self._pick_attack()
            elif rng < 0.55:
                self._ctrl = ControllerState(back)
            elif rng < 0.65:
                self._ctrl = ControllerState(fwd)
            else:
                self._ctrl = ControllerState()
        else:  # aggressive
            if dist > 140:
                self._ctrl = ControllerState(fwd) if rng < 0.85 else ControllerState(f"up_{fwd}")
            elif dist > 70:
                if rng < 0.55:
                    self._ctrl = ControllerState(fwd)
                elif rng < 0.75:
                    self._ctrl = ControllerState(back)
                else:
                    self._ctrl = ControllerState(f"up_{fwd}")
            else:
                if rng < 0.60:
                    self._ctrl = self._pick_attack()
                elif rng < 0.75:
                    self._ctrl = ControllerState(back)
                elif rng < 0.86:
                    self._ctrl = ControllerState(fwd)
                else:
                    self._ctrl = ControllerState(f"up_{back}")
        return self._ctrl


class PolicyOpponent(Opponent):
    """Frozen policy (SB3 model) played on the opposite side."""

    def __init__(self, model: Any, action_decode: Callable[[Any], list[ControllerState]],
                 deterministic: bool = False):
        self.model = model
        self.action_decode = action_decode
        self.deterministic = deterministic
        self._pending: list[ControllerState] = []
        self._last_obs = None

    def reset(self) -> None:
        self._pending = []
        self._last_obs = None

    def set_model(self, model: Any) -> None:
        self.model = model
        self.reset()

    def act(self, obs, info, state: GameState) -> ControllerState:
        if not self._pending:
            if obs is None:
                return ControllerState()
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            self._pending = self.action_decode(action)
        return self._pending.pop(0) if self._pending else ControllerState()


class LeagueOpponent(Opponent):
    """Samples a checkpoint from a directory (or list) at every episode.

    The directory is re-scanned on every reset, so running environments pick
    up checkpoints written by the trainer while it learns.  Until the first
    checkpoint exists the league falls back to a scripted bot, which gives
    PPO a reasonable opponent from step zero.  Even once checkpoints exist a
    fraction of episodes are still played against a scripted style: without
    that, the policy forgets how to chase a retreating opponent.
    """

    def __init__(self, checkpoints: Sequence[str] | str, action_decode, deterministic: bool = False,
                 seed: int | None = None, cache_size: int = 8,
                 bootstrap_style: str | Sequence[str] = ("aggressive", "random", "defensive", "passive"),
                 bootstrap_prob: float = 0.25,
                 ):
        self._source = checkpoints
        self._decode = action_decode
        self._deterministic = deterministic
        self._cache: dict[str, Any] = {}
        self._order: list[str] = []
        self._cache_size = cache_size
        self.bootstrap_prob = float(bootstrap_prob)
        self.rng = np.random.default_rng(seed)
        styles = [bootstrap_style] if isinstance(bootstrap_style, str) else list(bootstrap_style)
        # A mix of styles matters: an all-aggressive bootstrap teaches the agent
        # to stand still and punish approaches, never to walk in itself.
        self._bootstrap_styles = styles or ["aggressive"]
        self._bootstrap: dict[str, ScriptedBot] = {
            style: ScriptedBot(style, seed=seed) for style in set(self._bootstrap_styles)
        }
        self._bootstrap_active = self._bootstrap[self._bootstrap_styles[0]]
        self.paths: list[str] = self._discover()
        self._player: PolicyOpponent | None = None
        if self.paths:
            self._player = PolicyOpponent(self._load(self.paths[0]), action_decode, deterministic)

    def _discover(self) -> list[str]:
        if isinstance(self._source, str):
            return sorted(glob.glob(os.path.join(self._source, "*.zip")))
        return list(self._source)

    def _load(self, path: str):
        if path in self._cache:
            return self._cache[path]
        from stable_baselines3 import PPO

        model = PPO.load(path, device="cpu")
        self._cache[path] = model
        self._order.append(path)
        while len(self._order) > self._cache_size:
            self._cache.pop(self._order.pop(0), None)
        return model

    def reset(self) -> None:
        discovered = self._discover()
        if discovered:
            self.paths = discovered
        use_scripted = not self.paths or self.rng.random() < self.bootstrap_prob
        if use_scripted:
            style = str(self.rng.choice(self._bootstrap_styles))
            self._bootstrap_active = self._bootstrap[style]
            self._bootstrap_active.reset()
            self._player = None
            return
        path = str(self.rng.choice(self.paths))
        if self._player is None:
            self._player = PolicyOpponent(self._load(path), self._decode, self._deterministic)
        else:
            self._player.set_model(self._load(path))
        self._player.reset()

    def act(self, obs, info, state: GameState) -> ControllerState:
        if self._player is None:
            return self._bootstrap_active.act(obs, info, state)
        return self._player.act(obs, info, state)


def _load_policy(path: str, action_decode, deterministic: bool) -> PolicyOpponent:
    from stable_baselines3 import PPO

    if not Path(path).exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return PolicyOpponent(PPO.load(path, device="cpu"), action_decode, deterministic)


def make_opponent(spec: Any, config: GameConfig, env: Any) -> Opponent | None:
    """Build an opponent from a short spec string.

    ``aggressive`` / ``defensive`` / ``random`` / ``passive`` -- scripted bot
    ``bot:<style>`` -- scripted bot, explicit
    ``checkpoint:<file.zip>`` -- frozen SB3 policy
    ``league:<dir>`` -- sample a policy per episode from a directory
    ``none`` -- no opponent (dummy inputs)
    """
    if spec is None:
        return None
    if not isinstance(spec, str):
        return spec
    spec = spec.strip()
    if spec.lower() in ("none", "noop", "off"):
        return None
    decode = lambda action: env.action_to_controllers(action, side=2)  # noqa: E731
    if spec.startswith("bot:"):
        spec = spec.split(":", 1)[1]
    if spec in ("aggressive", "defensive", "random", "passive"):
        return ScriptedBot(spec, buttons=config.action_buttons)
    if spec.startswith("checkpoint:"):
        return _load_policy(spec.split(":", 1)[1], decode, deterministic=False)
    if spec.startswith("league:"):
        return LeagueOpponent(spec.split(":", 1)[1], decode, deterministic=False)
    raise ValueError(f"unknown opponent spec {spec!r}")
