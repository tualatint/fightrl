"""Game configuration: ROM, core, controls, macros, state sources."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .actions import Macro, MacroStep, default_macros
from .controller import DIRECTIONS
from .state import RamMap

PACKAGE_GAMES_DIR = Path(__file__).resolve().parent.parent / "games"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_REWARD = {
    "deal": 1.0,       # per unit of damage dealt (scaled by max health)
    "take": 1.0,       # penalty per unit of damage taken
    "win": 1.0,        # round win bonus
    "loss": 1.0,       # round loss penalty
    "time": 0.002,     # per-second penalty (discourage stalling)
    "lead": 0.25,      # potential-based shaping on the health lead
    "approach": 0.6,   # reward per fraction of the stage closed
    "weapon_break_deal": 0.0,  # extra multiplier on damage dealt by a weapon-break
}

# libretro joypad button ids (RETRO_DEVICE_ID_JOYPAD_*).
LIBRETRO_BUTTONS = {
    "B": 0, "Y": 1, "SELECT": 2, "START": 3,
    "UP": 4, "DOWN": 5, "LEFT": 6, "RIGHT": 7,
    "A": 8, "X": 9, "L": 10, "R": 11, "L2": 12, "R2": 13,
}


def _resolve(data: dict[str, Any], value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    for base in (Path(data["_dir"]), PROJECT_ROOT, Path.cwd()):
        candidate = base / path
        if candidate.exists():
            return str(candidate.resolve())
    return str((Path(data["_dir"]) / path).resolve())


def _resolve_core(data: dict[str, Any], value: str | None) -> str | None:
    """Like :func:`_resolve`, but keeps bare core names bare.

    A core is often referred to by name (``fbneo_libretro.so``) and located
    later via ``FIGHRL_CORES``/``cores`` by :func:`resolve_core_path`; turning
    the name into a config-relative path would defeat that lookup.
    """
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    for base in (Path(data["_dir"]), PROJECT_ROOT, Path.cwd(), Path.cwd() / "cores"):
        candidate = base / path
        if candidate.exists():
            return str(candidate.resolve())
    return value


@dataclass
class GameConfig:
    id: str
    title: str = ""
    backend: str = "libretro"          # "libretro" | "mock"
    core: str | None = None            # path or bare name of the .so
    rom: str | None = None
    system_dir: str | None = None
    start_state: str | None = None
    startup: list[MacroStep] = field(default_factory=list)
    button_map: dict[str, int] = field(default_factory=dict)
    core_options: dict[str, str] = field(default_factory=dict)
    action_buttons: list[str] = field(default_factory=list)
    macros: list[Macro] = field(default_factory=list)
    action_mode: str | None = None
    primitive_buttons: list[tuple[str, ...]] | None = None
    primitive_actions: list[dict[str, object]] | None = None
    weapon_break: list[str] = field(default_factory=list)
    frameskip: int = 4
    state_source: str = "auto"
    ram_map: RamMap | None = None
    hud: dict[str, Any] | None = None
    screen: dict[str, int] = field(default_factory=lambda: {"width": 320, "height": 224})
    max_episode_frames: int = 60 * 100
    reward: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REWARD))
    mock: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    @property
    def action_mode_default(self) -> str:
        return self.action_mode or ("macro" if self.macros else "full")

    def primitive_button_sets(self) -> list[tuple[str, ...]]:
        """Button subsets for the raw-input half of a hybrid action space.

        Defaults to "no button" plus every single button; configs can add
        simultaneous presses (throws, evasions) via ``primitive_buttons``.
        """
        if self.primitive_buttons is not None:
            return [tuple(combo) for combo in self.primitive_buttons]
        return [()] + [(button,) for button in self.action_buttons]

    @property
    def n_primitives(self) -> int:
        """Size of the raw-input half of a hybrid action space."""
        if self.primitive_actions is not None:
            return len(self.primitive_actions)
        return len(DIRECTIONS) * len(self.primitive_button_sets())

    @property
    def hybrid_action_count(self) -> int:
        """Number of hybrid actions; a curated primitive list needs no empty slot."""
        return self.n_primitives + len(self.macros) + (0 if self.primitive_actions is not None else 1)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameConfig":
        data = dict(data)
        data["_dir"] = data.get("_dir", str(PROJECT_ROOT))
        action_buttons = list(data.get("action_buttons", ["LP", "HP", "LK", "HK"]))
        if data.get("macros"):
            macros = [Macro.from_dict(m) for m in data["macros"]]
        else:
            macros = default_macros(action_buttons) + [
                Macro.from_dict(m) for m in data.get("special_macros", [])
            ]
        startup = [MacroStep.from_dict(s) for s in data.get("startup", [])]
        primitive_buttons = None
        if data.get("primitive_buttons") is not None:
            primitive_buttons = [
                tuple(str(b) for b in combo) for combo in data["primitive_buttons"]
            ]
            unknown = {b for combo in primitive_buttons for b in combo} - set(action_buttons)
            if unknown:
                raise ValueError(
                    f"{data['id']}: primitive_buttons has unknown buttons {sorted(unknown)}"
                )
        primitive_actions = None
        if data.get("primitive_actions") is not None:
            primitive_actions = []
            for entry in data["primitive_actions"]:
                direction = str(entry.get("dir", "neutral"))
                if direction not in DIRECTIONS:
                    raise ValueError(
                        f"{data['id']}: primitive_actions has unknown direction {direction!r}"
                    )
                buttons = tuple(str(b) for b in entry.get("buttons", ()))
                unknown = set(buttons) - set(action_buttons)
                if unknown:
                    raise ValueError(
                        f"{data['id']}: primitive_actions has unknown buttons {sorted(unknown)}"
                    )
                frames = entry.get("frames")
                primitive_actions.append({
                    "dir": direction,
                    "buttons": buttons,
                    "frames": int(frames) if frames is not None else None,
                    "name": str(entry.get("name") or ""),
                })
        action_mode = data.get("action_mode")
        if action_mode is not None and action_mode not in ("macro", "full", "hybrid"):
            raise ValueError(f"{data['id']}: unknown action_mode {action_mode!r}")
        reward = dict(DEFAULT_REWARD)
        reward.update(data.get("reward", {}))
        ram_map = RamMap.from_dict(data["ram_map"]) if data.get("ram_map") else None
        return cls(
            id=str(data["id"]),
            title=str(data.get("title", data["id"])),
            backend=str(data.get("backend", "libretro")),
            core=_resolve_core(data, data.get("core")),
            rom=_resolve(data, data.get("rom")),
            system_dir=_resolve(data, data.get("system_dir")),
            start_state=_resolve(data, data.get("start_state")),
            startup=startup,
            button_map={k: int(v) for k, v in data.get("button_map", {}).items()},
            core_options=dict(data.get("core_options", {})),
            action_buttons=action_buttons,
            macros=macros,
            action_mode=action_mode,
            primitive_buttons=primitive_buttons,
            primitive_actions=primitive_actions,
            frameskip=int(data.get("frameskip", 4)),
            state_source=str(data.get("state_source", "auto")),
            ram_map=ram_map,
            hud=data.get("hud"),
            screen={"width": int(data.get("screen", {}).get("width", 320)),
                    "height": int(data.get("screen", {}).get("height", 224))},
            max_episode_frames=int(data.get("max_episode_frames", 60 * 100)),
            reward=reward,
            weapon_break=[str(n) for n in data.get("weapon_break", ())],
            mock=dict(data.get("mock", {})),
            source_path=data.get("source_path"),
        )

    def with_overrides(self, **kwargs: Any) -> "GameConfig":
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
        return self


def load_ram_map(path: str) -> RamMap:
    """Load a ram_map from a JSON file (either ``{"ram_map": {...}}`` or a bare map)."""
    with open(Path(path).expanduser(), "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return RamMap.from_dict(data.get("ram_map", data))


def find_config_path(game: str) -> Path:
    candidate = Path(game).expanduser()
    if candidate.suffix == ".json" or candidate.exists():
        if not candidate.exists():
            raise FileNotFoundError(f"game config not found: {game}")
        return candidate
    for base in (PACKAGE_GAMES_DIR, Path.cwd() / "games"):
        path = base / f"{game}.json"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"unknown game {game!r}; available: "
        + ", ".join(sorted(p.stem for p in PACKAGE_GAMES_DIR.glob("*.json")))
    )


def load_game_config(game: str, **overrides: Any) -> GameConfig:
    path = find_config_path(game)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["_dir"] = str(path.parent)
    data["source_path"] = str(path)
    cfg = GameConfig.from_dict(data)
    return cfg.with_overrides(**overrides)


def list_games() -> list[dict[str, str]]:
    out = []
    for path in sorted(PACKAGE_GAMES_DIR.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            out.append({"id": data.get("id", path.stem), "title": data.get("title", ""),
                        "backend": data.get("backend", "libretro")})
        except Exception:
            continue
    return out


def resolve_core_path(core: str | None, extra_dirs: list[str] | None = None) -> str:
    if not core:
        raise ValueError("this game config has no libretro core configured")
    path = Path(core).expanduser()
    if path.exists():
        return str(path)
    name = path.name if path.name.endswith(".so") else f"{path.name}_libretro.so"
    dirs = [Path(os.environ.get("FIGHRL_CORES", "")) if os.environ.get("FIGHRL_CORES") else None]
    dirs += [Path(d) for d in (extra_dirs or [])]
    dirs += [PROJECT_ROOT / "cores", Path.cwd() / "cores", Path.home() / ".fightrl" / "cores"]
    for base in dirs:
        if base is None:
            continue
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"libretro core {name!r} not found. Put it in ./cores or set FIGHRL_CORES "
        f"(see README: scripts/fetch_cores.sh)."
    )
