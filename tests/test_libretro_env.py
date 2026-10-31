"""Integration: FightEnv driving a real libretro core (free 2048 core, stub HUD).

This exercises the full plumbing -- frame capture, input encoding, save states,
observation preprocessing -- without needing a ROM or a fighting-game HUD.
"""

from pathlib import Path

import numpy as np
import pytest

from fightrl.backends.libretro import LibretroBackend
from fightrl.core.config import GameConfig
from fightrl.core.state import GameState
from fightrl.env import FightEnv

CORE = Path(__file__).resolve().parent.parent / "cores" / "2048_libretro.so"

pytestmark = pytest.mark.skipif(not CORE.exists(), reason="2048 libretro core not installed")


class _StubReader:
    """Pretends the HUD says both fighters are at full health."""

    def __init__(self):
        self.last_source = "stub"

    def calibrate(self):
        pass

    def read(self):
        return GameState(p1_hp=1.0, p2_hp=1.0, p1_hp_max=1.0, p2_hp_max=1.0, valid=True)


def test_env_over_libretro(tmp_path):
    cfg = GameConfig(
        id="libretro_smoke",
        backend="libretro",
        action_buttons=["LP", "HP"],
        button_map={"LP": 0, "HP": 8, "START": 3},
        screen={"width": 376, "height": 464},
        frameskip=2,
        max_episode_frames=100,
    )
    backend = LibretroBackend(CORE, rom_path=None, system_dir=tmp_path)
    env = FightEnv(cfg, backend=backend, opponent="none", action_mode="full",
                   frameskip=2, state_reader=_StubReader())
    obs, info = env.reset(seed=0)
    assert obs["pixels"].shape == (4, 84, 84)
    assert env.observation_space.contains(obs)

    frames = []
    env.on_frame = frames.append
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step([4, 1, 1])  # right + both buttons
        assert env.observation_space.contains(obs)
    assert len(frames) == 5 * 2
    assert frames[0].dtype == np.uint8 and frames[0].ndim == 3

    # Save state: reset restores the same pixels.
    snapshot = env.backend.save_state()
    after_reset = env.reset(seed=0)[0]["pixels"]
    assert after_reset.shape == (4, 84, 84)
    env.backend.load_state(snapshot)
    assert env.backend.screen().shape[2] == 3
    env.close()
