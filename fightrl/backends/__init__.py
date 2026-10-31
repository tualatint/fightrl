"""Backend factory."""

from __future__ import annotations

from pathlib import Path

from ..core.config import GameConfig, resolve_core_path
from .base import Backend
from .libretro import LibretroBackend
from .mock import MockFighter, make_mock_backend


def make_backend(config: GameConfig, capture_video: bool = True) -> Backend:
    if config.backend == "mock":
        backend = make_mock_backend(config)
        backend.reset(seed=0)
        return backend
    if config.backend == "libretro":
        core = resolve_core_path(config.core)
        if not config.rom:
            raise ValueError(f"{config.id}: no ROM configured")
        return LibretroBackend(
            core_path=core,
            rom_path=config.rom,
            system_dir=config.system_dir or str(Path(core).parent),
            button_map=config.button_map,
            core_options=config.core_options,
            capture_video=capture_video,
        )
    raise ValueError(f"unknown backend {config.backend!r}")


__all__ = ["make_backend", "MockFighter", "LibretroBackend", "Backend"]
