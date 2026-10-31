"""Controller abstraction shared by all backends.

A ``ControllerState`` is what a human or a policy produces for one frame:
an 8-way stick direction plus a set of held buttons.  Backends translate it
to their native input format (libretro joypad bits, mock-engine flags, ...).

Directions are absolute screen directions.  Macro actions may use the
relative tokens ``fwd`` / ``back`` which are resolved against the facing of
the character being controlled (see ``resolve_direction``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable

DIRECTION_VECTORS = {
    "neutral": (0, 0),
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "up_left": (-1, -1),
    "up_right": (1, -1),
    "down_left": (-1, 1),
    "down_right": (1, 1),
}
DIRECTIONS = tuple(DIRECTION_VECTORS)

_RELATIVE = {
    "fwd", "back",
    "fwd_up", "back_up", "up_fwd", "up_back",
    "fwd_down", "back_down", "down_fwd", "down_back",
}


def mirror_direction(direction: str) -> str:
    """Swap left and right for a direction token."""
    if "left" in direction and "right" in direction:
        return direction
    if direction.endswith("_left"):
        return direction[: -len("left")] + "right"
    if direction.endswith("_right"):
        return direction[: -len("right")] + "left"
    if direction == "left":
        return "right"
    if direction == "right":
        return "left"
    return direction


def resolve_direction(token: str, facing: int = 1) -> str:
    """Resolve ``fwd``/``back`` tokens into an absolute direction.

    ``facing`` is +1 when the character looks to the right of the screen.
    """
    if token in DIRECTION_VECTORS:
        return token
    if token not in _RELATIVE:
        raise ValueError(f"unknown direction token: {token!r}")
    forward = "right" if facing >= 0 else "left"
    backward = mirror_direction(forward)
    mapping = {
        "fwd": forward,
        "back": backward,
        "fwd_up": f"up_{forward}",
        "up_fwd": f"up_{forward}",
        "back_up": f"up_{backward}",
        "up_back": f"up_{backward}",
        "fwd_down": f"down_{forward}",
        "down_fwd": f"down_{forward}",
        "back_down": f"down_{backward}",
        "down_back": f"down_{backward}",
    }
    return mapping[token]


@dataclass(frozen=True)
class ControllerState:
    """One frame of input for one player."""

    direction: str = "neutral"
    buttons: FrozenSet[str] = frozenset()

    def __post_init__(self) -> None:
        if self.direction not in DIRECTION_VECTORS:
            raise ValueError(f"unknown direction: {self.direction!r}")

    @classmethod
    def from_components(
        cls, direction: str = "neutral", buttons: Iterable[str] | None = None
    ) -> "ControllerState":
        return cls(direction=direction, buttons=frozenset(buttons or ()))

    def held(self, button: str) -> bool:
        return button in self.buttons

    def vector(self) -> tuple[int, int]:
        return DIRECTION_VECTORS[self.direction]

    def flipped(self) -> "ControllerState":
        return ControllerState(mirror_direction(self.direction), self.buttons)


NEUTRAL = ControllerState()
