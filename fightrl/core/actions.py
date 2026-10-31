"""Macro actions: short scripted input sequences (moves, jumps, blocks).

Fighting-game specials need multi-frame motion inputs (e.g. down, down-forward,
forward + punch).  A per-frame multi-discrete action space makes those very
hard to discover, so the environment can also expose a *macro* action space
where a single action is a short scripted sequence.

Macros are resolved against the character's facing at execution time, so
``fwd`` means "towards the opponent".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .controller import ControllerState, resolve_direction


@dataclass(frozen=True)
class MacroStep:
    direction: str = "neutral"
    buttons: tuple[str, ...] = ()
    frames: int = 1

    def controller(self, facing: int = 1) -> ControllerState:
        return ControllerState(resolve_direction(self.direction, facing), frozenset(self.buttons))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MacroStep":
        return cls(
            direction=str(data.get("dir", "neutral")),
            buttons=tuple(data.get("buttons", ())),
            frames=max(1, int(data.get("frames", 1))),
        )


@dataclass
class Macro:
    name: str
    steps: list[MacroStep] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        return sum(s.frames for s in self.steps)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Macro":
        return cls(
            name=str(data["name"]),
            steps=[MacroStep.from_dict(s) for s in data.get("steps", [])]
            or [MacroStep()],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": [
                {"dir": s.direction, "buttons": list(s.buttons), "frames": s.frames}
                for s in self.steps
            ],
        }


def _pick_button(action_buttons: tuple[str, ...], preferred: tuple[str, ...]) -> str:
    """First button from ``preferred`` that this game actually has.

    Six-button fighters name their strengths (LP/HP/LK/HK); four-button ones
    (Samurai Shodown) use letters, so a hardcoded ``HP`` would silently map to
    no joypad bit and the skill would do nothing.
    """
    for button in preferred:
        if button in action_buttons:
            return button
    return action_buttons[0]


def default_macros(action_buttons: Iterable[str]) -> list[Macro]:
    """Generic move set: movement, blocking, and normals in every stance."""
    buttons = tuple(action_buttons)
    heavy = _pick_button(buttons, ("HK", "HP", "MK", "MP", "B", "A"))
    light = _pick_button(buttons, ("LK", "LP", "MK", "MP", "C", "A", "B"))
    macros = [
        Macro("wait", [MacroStep("neutral", (), 6)]),
        Macro("walk_fwd", [MacroStep("fwd", (), 8)]),
        Macro("walk_back", [MacroStep("back", (), 8)]),
        Macro("crouch", [MacroStep("down", (), 8)]),
        Macro("block", [MacroStep("back", (), 10)]),
        Macro("jump_fwd", [MacroStep("fwd_up", (), 6), MacroStep("neutral", (), 16)]),
        Macro("jump_back", [MacroStep("back_up", (), 6), MacroStep("neutral", (), 16)]),
        Macro("jump_attack", [
            MacroStep("fwd_up", (), 8),
            MacroStep("fwd", (heavy,), 10),
            MacroStep("neutral", (), 10),
        ]),
        Macro("crouch_attack", [
            MacroStep("down", (), 2),
            MacroStep("down", (light,), 10),
            MacroStep("down", (), 6),
        ]),
    ]
    for button in action_buttons:
        macros.append(
            Macro(f"attack_{button}", [
                MacroStep("neutral", (), 2),
                MacroStep("neutral", (button,), 8),
                MacroStep("neutral", (), 4),
            ])
        )
    return macros
