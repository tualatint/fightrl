"""Backend interface: something that can run a fighting game frame by frame."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from ..core.controller import ControllerState

NEUTRAL_INPUT = (ControllerState(), ControllerState())


class Backend(ABC):
    """A two-player game emulator.

    ``step`` advances exactly one game frame.  Backends must be driven from a
    single thread.
    """

    name = "base"
    n_ports = 2
    fps = 60.0
    has_ram = False

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        ...

    @abstractmethod
    def step(self, controllers: Sequence[ControllerState]) -> None:
        ...

    @abstractmethod
    def screen(self) -> np.ndarray:
        """Latest frame as HxWx3 uint8 RGB."""

    def read_ram(self, addr: int, size: int) -> bytes:
        raise NotImplementedError(f"{self.name} backend has no RAM access")

    @property
    def ram_size(self) -> int:
        return 0

    def save_state(self) -> bytes:
        raise NotImplementedError(f"{self.name} backend cannot save states")

    def load_state(self, data: bytes) -> None:
        raise NotImplementedError(f"{self.name} backend cannot load states")

    def close(self) -> None:
        pass
