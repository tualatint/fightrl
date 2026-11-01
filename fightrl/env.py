"""Gymnasium environment for two-player fighting games.

Observation is a dict:
    pixels: uint8 (frame_stack, H, W) grayscale, resized
    state:  float32 (12,) normalized game state

Action space depends on ``action_mode``:
    "macro":  Discrete(len(macros)) -- scripted moves (recommended)
    "hybrid": Discrete(n_primitives + len(macros) + 1) -- raw stick/button
              primitives held for ``frameskip`` frames, skill macros, plus an
              explicit "empty" action (all inputs released)
    "full":   MultiDiscrete([9, 2, 2, ...]) -- 8-way stick + button presses
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .backends import make_backend
from .backends.base import Backend
from .core.config import GameConfig, load_game_config
from .core.controller import DIRECTIONS, ControllerState
from .core.state import GameState, _is_flat, make_state_reader
from .opponents import make_opponent


try:  # OpenCV is much faster than a numpy resize when present
    import cv2 as _cv2
except Exception:  # pragma: no cover - optional dependency
    _cv2 = None

# Label of the trailing "empty" hybrid action (all inputs released).
EMPTY_ACTION = "empty"

# Decisions remembered in the ``history`` observation, plus the state each
# decision started from.  Without them the agent cannot tell what it just
# committed to: one step is 3 frames or a 28-frame jump, so stacking four
# frames alone encodes a window anywhere between 0.2 s and 1.4 s.
HISTORY_ACTIONS = 4


def _mirror_state(features: np.ndarray) -> np.ndarray:
    """Flip a state vector so a P1 policy can play as P2."""
    mirrored = features.copy()
    mirrored[0], mirrored[1] = features[1], features[0]
    mirrored[4], mirrored[5] = features[5], features[4]
    mirrored[2] = -features[2]
    mirrored[7] = -features[7]
    return mirrored


def _mirror_history(history: np.ndarray) -> np.ndarray:
    """Mirror the state block of ``history``; action one-hots stay put.

    The one-hots index the acting side's own decisions, so they mean the same
    thing in either view; only the state the decision started from flips.
    """
    mirrored = history.copy()
    mirrored[:12] = _mirror_state(history[:12])
    return mirrored


def action_count_for(config: Any, action_mode: str | None = None) -> int:
    """Number of discrete decisions for ``config``; 0 means MultiDiscrete."""
    mode = action_mode or config.action_mode_default
    if mode == "macro":
        return len(config.macros)
    if mode == "hybrid":
        return config.hybrid_action_count
    if mode == "full":
        return 0
    raise ValueError(f"unknown action_mode {mode!r}")


def action_names_for(config: Any, action_mode: str | None = None) -> list[str]:
    """Readable name per action index, without needing an emulator."""
    mode = action_mode or config.action_mode_default
    macros = [m.name for m in config.macros]
    if mode != "hybrid":
        return macros
    explicit = config.primitive_actions
    if explicit is not None:
        primitives = [
            str(entry["name"]) or f"{entry['dir']}+{'+'.join(entry['buttons']) or 'none'}"
            for entry in explicit
        ]
        extra: list[str] = []
    else:
        primitives = [
            f"{direction}+{'+'.join(buttons) if buttons else 'none'}"
            for direction in DIRECTIONS
            for buttons in config.primitive_button_sets()
        ]
        extra = [EMPTY_ACTION]
    return primitives + macros + extra


def observation_space_for(
    config: Any,
    *,
    obs_size: tuple[int, int] = (84, 84),
    frame_stack: int = 4,
    history_actions: int = HISTORY_ACTIONS,
    action_mode: str | None = None,
) -> spaces.Dict:
    """Observation space for ``config``, without needing an emulator."""
    n_actions = action_count_for(config, action_mode)
    h, w = obs_size
    return spaces.Dict(
        {
            "pixels": spaces.Box(0, 255, (frame_stack, h, w), dtype=np.uint8),
            "state": spaces.Box(-2.0, 2.0, (12,), dtype=np.float32),
            "history": spaces.Box(
                -2.0, 2.0, (12 + history_actions * n_actions,), dtype=np.float32
            ),
        }
    )


_RESIZE_CACHE: dict[tuple[int, int, int, int], tuple[np.ndarray, ...]] = {}


def _resize_gray(frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize an RGB/gray uint8 image to (out_h, out_w) uint8."""
    if _cv2 is not None:
        gray = _cv2.cvtColor(frame, _cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        return _cv2.resize(gray, (out_w, out_h), interpolation=_cv2.INTER_AREA)
    if frame.ndim == 3:
        gray = (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2])
    else:
        gray = frame.astype(np.float32)
    h, w = gray.shape
    if h == out_h and w == out_w:
        return gray.astype(np.uint8)
    key = (h, w, out_h, out_w)
    cached = _RESIZE_CACHE.get(key)
    if cached is None:
        ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (h / out_h) - 0.5
        xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (w / out_w) - 0.5
        y0 = np.clip(np.floor(ys).astype(np.int32), 0, h - 1)
        x0 = np.clip(np.floor(xs).astype(np.int32), 0, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        x1 = np.minimum(x0 + 1, w - 1)
        wy = np.clip(ys - y0, 0.0, 1.0).astype(np.float32)[:, None]
        wx = np.clip(xs - x0, 0.0, 1.0).astype(np.float32)[None, :]
        cached = (y0, y1, x0, x1, wy, wx)
        if len(_RESIZE_CACHE) > 16:
            _RESIZE_CACHE.clear()
        _RESIZE_CACHE[key] = cached
    y0, y1, x0, x1, wy, wx = cached
    top = gray[y0][:, x0] * (1 - wx) + gray[y0][:, x1] * wx
    bottom = gray[y1][:, x0] * (1 - wx) + gray[y1][:, x1] * wx
    out = top * (1 - wy) + bottom * wy
    return np.clip(out, 0, 255).astype(np.uint8)


class FightEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        game: str | GameConfig = "mock",
        *,
        opponent: Any = "aggressive",
        action_mode: str | None = None,
        frameskip: int | None = None,
        obs_size: tuple[int, int] = (84, 84),
        frame_stack: int = 4,
        history_actions: int = HISTORY_ACTIONS,
        grayscale: bool = True,
        max_episode_frames: int | None = None,
        backend: Backend | None = None,
        capture_video: bool = True,
        render_mode: str | None = None,
        ram_map: Any = None,
        state_reader: Any = None,
    ):
        super().__init__()
        self.config = load_game_config(game) if isinstance(game, str) else game
        if ram_map is not None:
            from pathlib import Path

            from .core.config import load_ram_map
            from .core.state import RamMap

            if isinstance(ram_map, RamMap):
                self.config.ram_map = ram_map
            elif isinstance(ram_map, str):
                self.config.ram_map = load_ram_map(ram_map)
            elif isinstance(ram_map, dict):
                self.config.ram_map = RamMap.from_dict(ram_map)
            else:
                raise TypeError("ram_map must be a RamMap, dict or JSON path")
        self.render_mode = render_mode
        self.capture_video = capture_video
        self.frameskip = int(frameskip or self.config.frameskip)
        self.action_mode = action_mode or self.config.action_mode_default
        self.opponent_side = 2  # DuelEnv overrides: the learner plays P2 there
        self._primitive_sets = self.config.primitive_button_sets()
        self._explicit = self.config.primitive_actions
        self._n_primitives = self.config.n_primitives
        self.max_episode_frames = int(max_episode_frames or self.config.max_episode_frames)
        self.obs_size = tuple(obs_size)
        self.frame_stack = int(frame_stack)
        self.history_actions = int(history_actions)
        self.grayscale = grayscale

        self.backend = backend if backend is not None else make_backend(self.config, capture_video)
        self.reader = state_reader if state_reader is not None else make_state_reader(self.backend, self.config)
        self._start_state: bytes | None = None
        if self.config.start_state:
            from pathlib import Path

            path = Path(self.config.start_state)
            if path.exists():
                self._start_state = path.read_bytes()
            else:
                raise FileNotFoundError(
                    f"{self.config.id}: start_state not found at {path}. "
                    "Record one with `python -m fightrl.capture --game "
                    f"{self.config.id}`."
                )

        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._last_obs: dict[str, np.ndarray] = {}
        self.state = GameState()
        self.frame_count = 0
        self.episode_count = 0
        self._prev_state: GameState | None = None
        self._winner = 0
        self._damaged_streak = 0

        self._build_spaces()
        # Actions whose damage earns ``reward.weapon_break_deal`` on top of the
        # normal damage term -- the rage-gauge special is otherwise invisible
        # to the agent, so connecting with it needs an explicit incentive.
        break_names = set(self.config.weapon_break)
        self._weapon_break_indices = {
            index for index, name in enumerate(self.action_names) if name in break_names
        }
        self.opponent = (
            make_opponent(opponent, config=self.config, env=self)
            if isinstance(opponent, str)
            else opponent
        )
        self._raw_frame: np.ndarray | None = None
        self.on_frame: Any = None  # optional callback(raw_frame) for video capture

    # -- spaces --------------------------------------------------------- #
    def _build_spaces(self) -> None:
        self.observation_space = observation_space_for(
            self.config,
            obs_size=self.obs_size,
            frame_stack=self.frame_stack,
            history_actions=self.history_actions,
            action_mode=self.action_mode,
        )
        n_actions = action_count_for(self.config, self.action_mode)
        self._history_dim = self.observation_space["history"].shape[0]
        self._hist = np.zeros(self._history_dim, np.float32)
        self._opp_hist = np.zeros(self._history_dim, np.float32)
        self._opp_action: int | None = None

        if self.action_mode == "macro":
            self.action_space = spaces.Discrete(len(self.config.macros))
        elif self.action_mode == "hybrid":
            self.action_space = spaces.Discrete(n_actions)
        elif self.action_mode == "full":
            self.action_space = spaces.MultiDiscrete([9] + [2] * len(self.config.action_buttons))
        else:
            raise ValueError(f"unknown action_mode {self.action_mode!r}")

    @property
    def macro_names(self) -> list[str]:
        return [m.name for m in self.config.macros]

    @property
    def primitive_names(self) -> list[str]:
        """Labels for the raw stick+button half of a hybrid action space."""
        if self._explicit is not None:
            return [
                str(entry["name"]) or f"{entry['dir']}+{'+'.join(entry['buttons']) or 'none'}"
                for entry in self._explicit
            ]
        return [
            f"{direction}+{'+'.join(buttons) if buttons else 'none'}"
            for direction in DIRECTIONS
            for buttons in self._primitive_sets
        ]

    @property
    def action_names(self) -> list[str]:
        """Readable name per action index (macro and hybrid modes)."""
        return action_names_for(self.config, self.action_mode)

    def action_to_controllers(self, action: Any, side: int = 1) -> list[ControllerState]:
        facing = self.state.p1_facing if side == 1 else -self.state.p1_facing
        if self.action_mode == "macro":
            macro = self.config.macros[int(action)]
            return [step.controller(facing) for step in macro.steps for _ in range(step.frames)]
        if self.action_mode == "hybrid":
            index = int(action)
            if index >= self._n_primitives + len(self.config.macros):
                # Explicit empty action: nothing held for the whole window.
                return [ControllerState()] * self.frameskip
            if index < self._n_primitives:
                if self._explicit is not None:
                    entry = self._explicit[index]
                    direction = str(entry["dir"])
                    buttons = frozenset(entry["buttons"])
                    frames = int(entry["frames"] or self.frameskip)
                    if not buttons:
                        return [ControllerState(direction, frozenset())] * frames
                    held = max(1, frames - 1)
                    return ([ControllerState(direction, buttons)] * held
                            + [ControllerState(direction, frozenset())] * (frames - held))
                # Raw input: a stick direction held for the frameskip window,
                # with buttons released on the last frame.  Fighting games
                # edge-trigger attacks, so a button held every frame would
                # fire once and then be ignored no matter how often the action
                # repeats.  Directions are absolute (screen left/right) like in
                # "full" mode -- facing-relative motions belong to skills.
                sets = len(self._primitive_sets)
                direction = DIRECTIONS[index // sets]
                buttons = frozenset(self._primitive_sets[index % sets])
                held = max(1, self.frameskip - 1)
                return ([ControllerState(direction, buttons)] * held
                        + [ControllerState(direction, frozenset())] * (self.frameskip - held))
            macro = self.config.macros[index - self._n_primitives]
            return [step.controller(facing) for step in macro.steps for _ in range(step.frames)]
        direction = DIRECTIONS[int(action[0])]
        buttons = frozenset(
            b for b, on in zip(self.config.action_buttons, action[1:]) if on
        )
        return [ControllerState(direction, buttons)] * self.frameskip

    # -- episode lifecycle ---------------------------------------------- #
    def _apply_start(self) -> None:
        # Boot the core before unserializing: FBNeo (and some other cores)
        # crash or produce a black frame if a state is loaded first.
        self.backend.reset()
        if self._start_state is not None:
            self.backend.load_state(self._start_state)
        elif self.config.startup:
            for step in self.config.startup:
                ctrl = step.controller(1)
                for _ in range(step.frames):
                    self.backend.step([ctrl, ctrl])

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.backend.reset(seed=seed)
        self._apply_start()
        # Warm up the core: FBNeo renders one black frame after unserialize,
        # so vision calibration would otherwise see an empty picture.  Skip
        # flat frames too (e.g. a full-screen hit flash), which carry no HUD.
        for _ in range(12):
            self.backend.step([ControllerState(), ControllerState()])
            screen = self.backend.screen()
            if screen.max() > 0 and not _is_flat(screen):
                break
        self.frame_count = 0
        self.episode_count += 1
        self._winner = 0
        if self.opponent is not None:
            self.opponent.reset()
        self.reader.calibrate()
        self.state = self.reader.read()
        if not self.state.valid:
            raise RuntimeError(
                f"{self.config.id}: could not read game state. "
                f"source={self.reader.last_source}. Provide RAM addresses "
                "(`python -m fightrl.ramscan`) or HUD rectangles in the game config."
            )
        self._prev_state = self.state
        self._damaged_streak = 0
        self._last_damaged_state = None
        self._reset_pixels()
        self._reset_history()
        obs = self._observe()
        info = self._info()
        return obs, info

    def _reset_pixels(self) -> None:
        self._frames.clear()
        frame = self._preprocess(self.backend.screen())
        for _ in range(self.frame_stack):
            self._frames.append(frame)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        return _resize_gray(frame, self.obs_size[0], self.obs_size[1])

    # -- stepping ------------------------------------------------------- #
    def step(self, action: Any):
        if not self.state.valid:
            raise RuntimeError("episode state is invalid; call reset() first")
        if self._prev_state is None:
            self._prev_state = self.state

        agent_ctrls = self.action_to_controllers(action)
        start_features = self.state.features(12)
        reward = 0.0
        dmg_dealt = dmg_taken = 0.0
        terminated = truncated = False
        frames_run = 0

        opp_obs = self._opponent_obs(self._last_obs)
        for ctrl in agent_ctrls:
            if self.opponent is not None:
                opp_ctrl = self.opponent.act(opp_obs, self._info(), self.state)
                index = getattr(self.opponent, "last_action", None)
                if index is not None and index != self._opp_action:
                    self._opp_action = int(index)
                    self._opp_hist = self._push_history(
                        self._opp_hist, int(index), self._opponent_view()
                    )
            else:
                opp_ctrl = ControllerState()
            self.backend.step([ctrl, opp_ctrl])
            frames_run += 1
            self.frame_count += 1
            if self.on_frame is not None:
                self.on_frame(self.backend.screen())
            self.state = self.reader.read()
            both_full = self.state.p1_frac >= 0.99 and self.state.p2_frac >= 0.99
            if both_full and self._damaged_streak >= 2:
                # The bars are full again after having been damaged: the round
                # ended (usually on time-out) and the next one began.  Vision
                # readers have no timer, so this is the only visible boundary.
                # Score it from the last damaged reading: the current state is
                # already the refilled bars, which would count as a draw.
                last = self._last_damaged_state or self._prev_state
                self._winner = self._winner_by_health(last)
                reward += self._result_reward(self._winner)
                reward -= self.config.reward.get("time", 0.0) * (
                    frames_run / max(self.backend.fps, 1.0)
                )
                self._prev_state = self.state
                terminated = True
                break
            self._damaged_streak = 0 if both_full else self._damaged_streak + 1
            if not both_full:
                self._last_damaged_state = self.state
            # `_reward` runs once per emulated frame; charge one frame of the
            # time penalty, not the cumulative frames_run of this step.
            r, dealt, taken = self._reward(self._prev_state, self.state, 1)
            reward += r
            dmg_dealt += dealt
            dmg_taken += taken
            self._prev_state = self.state
            if self.state.round_over:
                self._winner = self.state.winner
                terminated = True
                break
            if self._timer_expired():
                terminated = True
                break
            if self.frame_count >= self.max_episode_frames:
                truncated = True
                break

        if not terminated and not truncated and self.frame_count >= self.max_episode_frames:
            truncated = True

        if self.action_mode != "full" and int(action) in self._weapon_break_indices:
            # Only damage actually dealt is rewarded, not the motion itself.
            reward += self.config.reward.get("weapon_break_deal", 0.0) * dmg_dealt

        frame = self._preprocess(self.backend.screen())
        self._frames.append(frame)
        if self.action_mode != "full":
            self._hist = self._push_history(self._hist, int(action), start_features)
        obs = self._observe()
        info = self._info()
        info.update({"dmg_dealt": dmg_dealt, "dmg_taken": dmg_taken})
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _timer_expired(self) -> bool:
        tmax = self.state.timer_max
        return bool(tmax and self.state.timer <= 0 and not self.state.round_over)

    @staticmethod
    def _winner_by_health(state: GameState) -> int:
        if state.p1_frac > state.p2_frac + 0.005:
            return 1
        if state.p2_frac > state.p1_frac + 0.005:
            return 2
        return 0

    def _result_reward(self, winner: int) -> float:
        w = self.config.reward
        if winner == 1:
            return w.get("win", 1.0)
        if winner == 2:
            return -w.get("loss", 1.0)
        return 0.0

    def _reward(self, prev: GameState, cur: GameState, frames: int):
        w = self.config.reward
        deal = float(np.clip(prev.p2_frac - cur.p2_frac, 0.0, 1.0))
        take = float(np.clip(prev.p1_frac - cur.p1_frac, 0.0, 1.0))
        lead_prev = prev.p1_frac - prev.p2_frac
        lead_cur = cur.p1_frac - cur.p2_frac
        r = w.get("deal", 1.0) * deal - w.get("take", 1.0) * take
        r += w.get("lead", 0.0) * (lead_cur - lead_prev)
        d_prev, d_cur = prev.distance, cur.distance
        if d_prev is not None and d_cur is not None:
            width = float(cur.stage_width or 320.0)
            r += w.get("approach", 0.0) * ((d_prev - d_cur) / width)
        r -= w.get("time", 0.0) * (frames / max(self.backend.fps, 1.0))
        if cur.round_over:
            r += self._result_reward(cur.winner)
        return r, deal, take

    # -- observation ---------------------------------------------------- #
    def _opponent_obs(self, obs: dict[str, np.ndarray] | None) -> dict[str, np.ndarray] | None:
        """View ``obs`` from the opponent's side; P1 sees the canonical one."""
        if obs is None:
            return None
        state = obs["state"] if self.opponent_side == 1 else _mirror_state(obs["state"])
        return {"pixels": obs["pixels"], "state": state, "history": self._opp_hist}

    def _opponent_view(self) -> np.ndarray:
        """Current state features, oriented the way the opponent sees them."""
        features = self.state.features(12)
        return features if self.opponent_side == 1 else _mirror_state(features)

    def _push_history(
        self, history: np.ndarray, action: int, start_features: np.ndarray
    ) -> np.ndarray:
        """Slide the decision/state history left and append this decision."""
        n_actions = int(self.action_space.n) if self.action_mode != "full" else 0
        if n_actions <= 0:
            return history
        updated = np.zeros_like(history)
        updated[:12] = start_features
        updated[12 + n_actions:] = history[12:-n_actions]
        if 0 <= action < n_actions:
            updated[12 + action] = 1.0
        return updated

    def _reset_history(self) -> None:
        self._hist = np.zeros(self._history_dim, np.float32)
        self._hist[:12] = self.state.features(12)
        self._opp_hist = np.zeros(self._history_dim, np.float32)
        self._opp_hist[:12] = self._opponent_view()
        self._opp_action = None

    def _observe(self) -> dict[str, np.ndarray]:
        pixels = np.stack(self._frames, axis=0) if self._frames else np.zeros(
            (self.frame_stack, *self.obs_size), np.uint8
        )
        obs = {"pixels": pixels, "state": self.state.features(12), "history": self._hist}
        self._last_obs = obs
        return obs

    def _info(self) -> dict[str, Any]:
        return {
            "p1_hp": self.state.p1_hp,
            "p2_hp": self.state.p2_hp,
            "p1_frac": self.state.p1_frac,
            "p2_frac": self.state.p2_frac,
            "winner": self._winner,
            "round_over": self.state.round_over,
            "frames": self.frame_count,
            "state_source": self.reader.last_source,
        }

    # -- misc ----------------------------------------------------------- #
    def render(self):
        self._raw_frame = self.backend.screen()
        return self._raw_frame

    def close(self):
        if getattr(self, "backend", None) is not None:
            self.backend.close()


def make_env(game: str = "mock", **kwargs: Any) -> FightEnv:
    return FightEnv(game, **kwargs)
