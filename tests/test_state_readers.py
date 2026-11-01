"""State reader tests: vision must agree with ground-truth RAM on the mock game."""

import numpy as np

from fightrl.backends.mock import MockFighter, ADDR_P1_HP, ADDR_P2_HP
from fightrl.core.config import load_game_config
from fightrl.core.state import RamMap, RamStateReader, VisionStateReader
from fightrl.env import FightEnv


def test_ram_reader_reads_health():
    backend = MockFighter()
    backend.reset()
    reader = RamStateReader(RamMap.from_dict({
        "p1_hp": {"addr": ADDR_P1_HP}, "p2_hp": {"addr": ADDR_P2_HP},
        "p1_hp_max": 176, "p2_hp_max": 176,
    }))
    state = reader.read(backend)
    assert state.valid
    assert state.p1_hp == 176 and state.p2_hp == 176
    backend.p1.hp = 90
    backend._write_ram()
    assert reader.read(backend).p1_hp == 90


def test_ram_reader_detects_out_of_range_values():
    backend = MockFighter()
    backend.reset()
    backend.ram[ADDR_P1_HP] = 250  # bogus value above max
    reader = RamStateReader(RamMap.from_dict({
        "p1_hp": {"addr": ADDR_P1_HP}, "p2_hp": {"addr": ADDR_P2_HP},
        "p1_hp_max": 176, "p2_hp_max": 176,
    }))
    assert not reader.read(backend).valid


def test_bar_fill_reads_low_health_red():
    """A depleted bar turns red; the reader must not mistake that for death."""
    from fightrl.core.state import _bar_fill

    roi = (0, 30, 0, 128)
    frame = np.zeros((30, 128, 3), np.uint8)
    frame[:, :26] = (220, 8, 8)  # ~20% low-health red
    red = _bar_fill(frame, roi, (95.0, 150.0))
    assert 0.15 <= red <= 0.25, red
    green = np.zeros_like(frame)
    green[:, :64] = (8, 220, 60)
    assert 0.45 <= _bar_fill(green, roi, (95.0, 150.0)) <= 0.55
    # A wide red area is a stage flash, not a full bar.
    flash = np.zeros((30, 128, 3), np.uint8)
    flash[:, :] = (220, 8, 8)
    assert _bar_fill(flash, roi, (95.0, 150.0)) == 0.0
    empty = np.zeros((30, 128, 3), np.uint8)
    assert _bar_fill(empty, roi, (95.0, 150.0)) == 0.0


def test_vision_matches_ram():
    cfg = load_game_config("mock", state_source="vision")
    env = FightEnv(cfg, opponent="aggressive")
    env.reset(seed=3)
    assert env.reader.vision.calibrated
    errors = []
    for _ in range(160):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        truth = env.reader.ram.read(env.backend)
        seen = env.reader.vision.read(env.backend)
        errors.append((abs(seen.p1_frac - truth.p1_frac), abs(seen.p2_frac - truth.p2_frac)))
        if term or trunc:
            break
    errors = np.array(errors)
    assert errors.mean() < 0.01, errors.mean()
    assert errors.max() < 0.03, errors.max()
    env.close()


def test_vision_calibration_finds_bars():
    backend = MockFighter()
    backend.reset()
    reader = VisionStateReader()
    assert reader.calibrate(backend.screen())
    y0, y1, x0, x1 = reader._p1_roi
    assert 0 <= y0 < y1 <= 224
    assert x0 < 160 < 304


class TimedRefillFighter(MockFighter):
    """A round that ends by time-out: bars refill, ``round_over`` never fires.

    Vision-only games have no timer, so the env must notice the refill itself.
    """

    def reset(self, seed=None):
        super().reset(seed)
        self._n = 0
        self._refilled = False

    def step(self, controllers):
        self._n += 1
        if self._n == 5:
            self.p2.hp = self.health_max * 0.5
        if self._n >= 40 and not self._refilled:
            self.p1.hp = self.p2.hp = self.health_max
            self._refilled = True
        super().step(controllers)


def test_vision_round_refill_terminates():
    backend = TimedRefillFighter()
    env = FightEnv(load_game_config("mock", state_source="vision"),
                   opponent="none", backend=backend)
    env.reset()
    terminated = truncated = False
    dealt = 0.0
    for _ in range(40):
        _, reward, terminated, truncated, info = env.step(env.macro_names.index("wait"))
        dealt += info["dmg_dealt"]
        if terminated or truncated:
            break
    env.close()
    assert terminated and not truncated
    assert info["winner"] == 1  # P1 was ahead when the bars refilled
    assert dealt > 0.4

