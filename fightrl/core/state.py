"""Game-state extraction and reward-relevant bookkeeping.

Fighting-game RL needs to know the health of both fighters.  Two sources are
supported:

* :class:`RamStateReader` -- exact values read from emulated console RAM
  (a "RAM map").  Preferred whenever it is available; use
  ``python -m fightrl.ramscan`` to discover addresses for a ROM.
* :class:`VisionStateReader` -- reads the HUD health bars from pixels.  Works
  on any ROM with no reverse engineering, at the cost of ~1-2% accuracy.

:func:`make_state_reader` combines both ("auto" mode).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np


@dataclass
class GameState:
    p1_hp: float = 0.0
    p2_hp: float = 0.0
    p1_hp_max: float = 1.0
    p2_hp_max: float = 1.0
    timer: float = 0.0
    timer_max: float = 0.0
    p1_x: float | None = None
    p2_x: float | None = None
    stage_width: float | None = None
    valid: bool = True

    @property
    def p1_frac(self) -> float:
        return float(np.clip(self.p1_hp / max(self.p1_hp_max, 1e-9), 0.0, 1.0))

    @property
    def p2_frac(self) -> float:
        return float(np.clip(self.p2_hp / max(self.p2_hp_max, 1e-9), 0.0, 1.0))

    @property
    def p1_dead(self) -> bool:
        return self.p1_hp <= 0.0

    @property
    def p2_dead(self) -> bool:
        return self.p2_hp <= 0.0

    @property
    def winner(self) -> int:
        """1 if P1 wins the round, 2 if P2 wins, 0 otherwise."""
        if self.p1_dead and not self.p2_dead:
            return 2
        if self.p2_dead and not self.p1_dead:
            return 1
        if self.p1_dead and self.p2_dead:
            return 0
        return 0

    @property
    def round_over(self) -> bool:
        return self.p1_dead or self.p2_dead

    @property
    def p1_facing(self) -> int:
        if self.p1_x is not None and self.p2_x is not None:
            return 1 if self.p2_x >= self.p1_x else -1
        return 1

    @property
    def distance(self) -> float | None:
        if self.p1_x is None or self.p2_x is None:
            return None
        return abs(self.p2_x - self.p1_x)

    def features(self, n: int = 12) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        width = float(self.stage_width or 320.0)
        timer_frac = self.timer / self.timer_max if self.timer_max else 0.0
        values = [
            self.p1_frac,
            self.p2_frac,
            self.p1_frac - self.p2_frac,
            float(np.clip(timer_frac, 0.0, 1.0)),
            (self.p1_x / width) if self.p1_x is not None else 0.0,
            (self.p2_x / width) if self.p2_x is not None else 0.0,
            (self.distance / width) if self.distance is not None else 0.0,
            float(self.p1_facing),
            1.0 if self.valid else 0.0,
            1.0 if self.round_over else 0.0,
        ]
        out[: min(n, len(values))] = values[:n]
        return out


# --------------------------------------------------------------------------- #
# RAM based reader
# --------------------------------------------------------------------------- #


@dataclass
class RamField:
    addr: int
    width: int = 1
    endian: str = "little"
    scale: float = 1.0

    def read(self, raw: bytes, offset: int = 0) -> float | None:
        start = self.addr + offset
        if start < 0 or start + self.width > len(raw):
            return None
        chunk = raw[start : start + self.width]
        value = int.from_bytes(chunk, self.endian, signed=False)
        return value * self.scale


@dataclass
class RamMap:
    p1_hp: RamField | None = None
    p2_hp: RamField | None = None
    timer: RamField | None = None
    p1_x: RamField | None = None
    p2_x: RamField | None = None
    p1_hp_max: float = 1.0
    p2_hp_max: float = 1.0
    timer_max: float = 0.0
    stage_width: float | None = None

    @property
    def complete(self) -> bool:
        return self.p1_hp is not None and self.p2_hp is not None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RamMap":
        kwargs: dict[str, Any] = {}
        for key in ("p1_hp", "p2_hp", "timer", "p1_x", "p2_x"):
            value = data.get(key)
            if value is not None:
                kwargs[key] = RamField(
                    addr=int(value["addr"]),
                    width=int(value.get("width", 1)),
                    endian=value.get("endian", "little"),
                    scale=float(value.get("scale", 1.0)),
                )
        for key in ("p1_hp_max", "p2_hp_max", "timer_max", "stage_width"):
            if data.get(key) is not None:
                kwargs[key] = float(data[key])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in ("p1_hp", "p2_hp", "timer", "p1_x", "p2_x"):
            fld = getattr(self, key)
            if fld is not None:
                out[key] = {
                    "addr": fld.addr,
                    "width": fld.width,
                    "endian": fld.endian,
                    "scale": fld.scale,
                }
        for key in ("p1_hp_max", "p2_hp_max", "timer_max", "stage_width"):
            value = getattr(self, key)
            if value:
                out[key] = value
        return out


class RamStateReader:
    """Reads fighter health/positions from emulator RAM."""

    def __init__(self, ram_map: RamMap):
        self.ram_map = ram_map

    @property
    def available(self) -> bool:
        return self.ram_map.complete

    def read(self, backend) -> GameState:
        m = self.ram_map
        state = GameState(
            p1_hp_max=m.p1_hp_max,
            p2_hp_max=m.p2_hp_max,
            timer_max=m.timer_max,
            stage_width=m.stage_width,
        )

        def field(fld: RamField) -> float | None:
            raw = backend.read_ram(fld.addr, fld.width)
            if len(raw) < fld.width:
                return None
            return int.from_bytes(raw, fld.endian) * fld.scale

        try:
            p1 = field(m.p1_hp)
            p2 = field(m.p2_hp)
        except Exception:
            state.valid = False
            return state
        if p1 is None or p2 is None:
            state.valid = False
            return state
        state.p1_hp, state.p2_hp = p1, p2
        if m.timer is not None:
            timer = field(m.timer)
            state.timer = float(timer or 0.0)
        for name in ("p1_x", "p2_x"):
            fld = getattr(m, name)
            if fld is not None:
                setattr(state, name, field(fld))
        # Plausibility check: a broken / stale RAM map must not silently
        # produce a constant reward of zero for the whole run.
        if state.p1_hp_max > 1.0 and state.p1_hp > state.p1_hp_max * 1.01 + 1:
            state.valid = False
        if state.p2_hp_max > 1.0 and state.p2_hp > state.p2_hp_max * 1.01 + 1:
            state.valid = False
        return state


# --------------------------------------------------------------------------- #
# Vision based reader
# --------------------------------------------------------------------------- #


def _mask_from_rgb(rgb: np.ndarray, hue_range: tuple[float, float]) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    delta = np.maximum(mx - mn, 1e-6)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    hue = np.zeros_like(mx)  # degrees: 0 red, 60 yellow, 120 green, 240 blue
    is_r = mx == r
    is_g = (mx == g) & ~is_r
    is_b = ~(is_r | is_g)
    hue[is_r] = (60.0 * (g[is_r] - b[is_r]) / delta[is_r]) % 360.0
    hue[is_g] = 60.0 * (2.0 + (b[is_g] - r[is_g]) / delta[is_g])
    hue[is_b] = 60.0 * (4.0 + (r[is_b] - g[is_b]) / delta[is_b])
    lo, hi = hue_range
    return (hue >= lo) & (hue <= hi) & (sat > 0.35) & (mx > 110.0)


def _bar_mask(frame: np.ndarray, hue_range: tuple[float, float] = (35.0, 150.0)) -> np.ndarray:
    """Mask of HUD health-bar pixels: vivid yellow/green, never blue or cyan.

    Health bars are drawn in a green/yellow family.  Stage art that leaks into
    the HUD band is either dark, red/orange or blue/cyan (e.g. Samurai Shodown
    II's lightning flashes), so filtering by hue keeps those backgrounds out of
    the fill measurement.  Brightness alone made the bar read as full whenever
    the stage flashed.

    The floor is 35 degrees, not 0: damage trails are orange, and their bright
    shades (255, 101, 0) sit at ~24 degrees.  Games whose bar flashes yellow
    while the trail animates through yellow too (Samurai Shodown II) can pass a
    narrower ``hue_range`` such as (95, 150) to keep only the pure green.
    """
    if frame.ndim != 3 or frame.shape[2] < 3:
        return np.zeros(frame.shape[:2], dtype=bool)
    return _mask_from_rgb(frame[..., :3].astype(np.float32), hue_range)


def _bar_fill(
    frame: np.ndarray, roi: tuple[int, int, int, int], hue_range: tuple[float, float]
) -> float:
    """Fill fraction of one bar, computed only inside its ROI.

    Same result as ``_fill(_bar_mask(frame), roi)`` but touches ~3k pixels
    instead of all 71k, which dominates the per-frame cost of vision stepping.
    """
    y0, y1, x0, x1 = roi
    region = frame[y0:y1, x0:x1, :3]
    if region.size == 0:
        return 0.0
    mask = _mask_from_rgb(region.astype(np.float32), hue_range)
    fill = float(np.median(mask.mean(axis=1)))
    # Low health repaints the bar red (Samurai Shodown II also flashes it
    # yellow), so a green-only mask reads a low bar as empty and ends the
    # round early.  Accept those hues only as a *small* run: a wide red area
    # is a stage flash or effect, not a health bar, and must not read as full.
    lo, hi = hue_range
    low = _mask_from_rgb(region.astype(np.float32), (0.0, lo)) if lo > 0.0 else mask
    low |= _mask_from_rgb(region.astype(np.float32), (340.0, 360.0))
    low_fill = float(np.median(low.mean(axis=1)))
    if fill <= 0.01 and low_fill <= 0.35:
        return low_fill
    return fill


def _is_flat(frame: np.ndarray) -> bool:
    """True for full-screen flashes/fades, which carry no HUD information.

    Samurai Shodown II paints the whole screen a solid colour on heavy hits and
    during transitions; reading the bars from such a frame would look like a
    double KO.  Measure the standard deviation of luma, not of the raw RGB
    array: a uniform red screen has channel-wise variation but constant luma.
    """
    if not frame.size:
        return False
    pixels = frame[::4, ::4].astype(np.float32)
    luma = pixels.mean(axis=2) if pixels.ndim == 3 else pixels
    return bool(luma.std() < 3.0)


class VisionStateReader:
    """Locates the two HUD health bars and tracks their fill fraction."""

    def __init__(self, hud: Mapping[str, Any] | None = None, top_fraction: float = 0.30):
        self.hud = dict(hud or {})
        self.top_fraction = top_fraction
        hue = self.hud.get("hue")
        self._hue = (float(hue[0]), float(hue[1])) if hue else (35.0, 150.0)
        self._calibrated = False
        self._p1_roi: tuple[int, int, int, int] | None = None
        self._p2_roi: tuple[int, int, int, int] | None = None
        self._p1_full = 1.0
        self._p2_full = 1.0
        self._last_state: GameState | None = None

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def _roi_from_hud(self, key: str) -> tuple[int, int, int, int] | None:
        rect = self.hud.get(key)
        if not rect:
            return None
        x, y, w, h = (int(v) for v in rect)
        return (y, y + h, x, x + w)

    def calibrate(self, frame: np.ndarray) -> bool:
        """Find the health bars on a frame taken at (near) full health."""
        if _is_flat(frame):
            return False
        mask = _bar_mask(frame, self._hue)
        if mask.size == 0:
            return False
        h, w = mask.shape
        p1 = self._roi_from_hud("p1_bar")
        p2 = self._roi_from_hud("p2_bar")
        if p1 is None or p2 is None:
            p1, p2 = self._detect_bars(mask, w, h)
        if p1 is None or p2 is None:
            return False
        self._p1_roi, self._p2_roi = p1, p2
        self._p1_full = max(self._fill(mask, p1), 1e-6)
        self._p2_full = max(self._fill(mask, p2), 1e-6)
        self._calibrated = True
        return True

    @staticmethod
    def _detect_bars(
        mask: np.ndarray, width: int, height: int
    ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        top = max(4, int(height * 0.30))
        min_len = max(12, int(width * 0.08))
        candidates: list[tuple[int, int, int, int]] = []  # y, x0, x1, length
        for y in range(top):
            row = mask[y]
            if not row.any():
                continue
            diff = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
            starts = np.flatnonzero(diff == 1)
            ends = np.flatnonzero(diff == -1)
            for x0, x1 in zip(starts, ends):
                length = int(x1 - x0)
                if length >= min_len:
                    candidates.append((y, int(x0), int(x1), length))
        best = None
        best_score = 0
        left = [c for c in candidates if c[1] < width * 0.45]
        right = [c for c in candidates if c[1] >= width * 0.55]
        for a in left:
            for b in right:
                if abs(a[0] - b[0]) > 2:
                    continue
                ratio = min(a[3], b[3]) / max(a[3], b[3])
                if ratio < 0.75:
                    continue
                score = a[3] + b[3]
                if score > best_score:
                    best_score = score
                    best = (a, b)
        if best is None:
            return None, None
        (y, x0, x1, _), (y2, u0, u1, _) = best
        return (max(0, y - 1), y + 2, x0, x1), (max(0, y2 - 1), y2 + 2, u0, u1)

    @staticmethod
    def _fill(mask: np.ndarray, roi: tuple[int, int, int, int]) -> float:
        y0, y1, x0, x1 = roi
        region = mask[y0:y1, x0:x1]
        if region.size == 0:
            return 0.0
        return float(np.median(region.mean(axis=1)))

    def read(self, backend) -> GameState:
        frame = backend.screen()
        if _is_flat(frame):
            return self._repeat_last()
        state = GameState()
        if not self._calibrated:
            if not self.calibrate(frame):
                state.valid = False
                return state
        p1_raw = _bar_fill(frame, self._p1_roi, self._hue) / self._p1_full
        p2_raw = _bar_fill(frame, self._p2_roi, self._hue) / self._p2_full
        if p1_raw < 0.02 and p2_raw < 0.02:
            # No bars on screen: round transitions hide the HUD (and a double
            # KO draws an empty one).  Keep the last reading instead of
            # reporting a fake double death.
            return self._repeat_last()
        p1 = float(np.clip(p1_raw, 0.0, 1.0))
        p2 = float(np.clip(p2_raw, 0.0, 1.0))
        # Vision is noisy at the tails; snap clear extremes.
        p1 = 0.0 if p1 < 0.02 else (1.0 if p1 > 0.985 else p1)
        p2 = 0.0 if p2 < 0.02 else (1.0 if p2 > 0.985 else p2)
        state.p1_hp, state.p2_hp = p1, p2
        state.p1_hp_max = state.p2_hp_max = 1.0
        self._last_state = state
        return state

    def _repeat_last(self) -> GameState:
        if self._last_state is None:
            return GameState(valid=False)
        return replace(self._last_state)


class AutoStateReader:
    """Prefers RAM, falls back to vision, and detects implausible RAM maps."""

    def __init__(
        self,
        backend,
        ram: RamStateReader | None = None,
        vision: VisionStateReader | None = None,
        mode: str = "auto",
    ):
        if mode not in ("auto", "ram", "vision"):
            raise ValueError(f"state_source must be auto/ram/vision, got {mode!r}")
        self.backend = backend
        self.ram = ram if (ram and ram.available) else None
        self.vision = vision
        self.mode = mode
        self.ram_ok = mode in ("auto", "ram") and self.ram is not None
        self.vision_ok = mode in ("auto", "vision") and self.vision is not None
        self.last_source = "none"
        self._ram_failures = 0

    def calibrate(self, frame: np.ndarray | None = None) -> None:
        if self.vision is not None and not self.vision.calibrated:
            self.vision.calibrate(frame if frame is not None else self.backend.screen())
        # If the RAM map gives addresses but no max health, take the value at
        # episode start (full health) as the max.
        if self.ram is not None and self.ram.ram_map.p1_hp_max <= 1.0:
            snapshot = self.ram.read(self.backend)
            if snapshot.valid and snapshot.p1_hp > 1:
                self.ram.ram_map.p1_hp_max = float(snapshot.p1_hp)
                self.ram.ram_map.p2_hp_max = float(snapshot.p2_hp)

    def read(self) -> GameState:
        if self.ram_ok:
            state = self.ram.read(self.backend)
            if state.valid:
                self.last_source = "ram"
                self._ram_failures = 0
                return state
            self._ram_failures += 1
            if self.mode == "ram" or self._ram_failures > 30:
                self.ram_ok = False
        if self.vision_ok:
            state = self.vision.read(self.backend)
            self.last_source = "vision" if state.valid else "none"
            return state
        return GameState(valid=False)


def make_state_reader(backend, config) -> AutoStateReader:
    ram = RamStateReader(config.ram_map) if config.ram_map and config.ram_map.complete else None
    vision = VisionStateReader(config.hud)
    return AutoStateReader(backend, ram=ram, vision=vision, mode=config.state_source)
