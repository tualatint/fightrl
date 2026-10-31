"""Discover health / timer RAM addresses for a ROM by watching memory change.

The tool plays the game with two "brawler" bots while reading (a) the whole
emulated RAM and (b) the HUD health bars (vision).  Every RAM address whose
value drops in lock-step with a visible health decrease is reported as a
candidate.  Paste the winner into the game config's ``ram_map`` and the
environment gets exact health values instead of the vision estimate.

Example::

    python -m fightrl.ramscan --game sf2_snes --seconds 45 --out states/sf2_ram.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .backends import make_backend
from .core.config import GameConfig, load_game_config
from .core.controller import ControllerState
from .core.state import VisionStateReader


class BrawlerBot:
    """Mashes attacks while walking forward -- good at generating damage events."""

    def __init__(self, side: int, buttons: Sequence[str], seed: int = 0):
        self.side = side
        self.buttons = list(buttons)
        self.fwd = "right" if side == 1 else "left"
        self.back = "left" if side == 1 else "right"
        self.rng = np.random.default_rng(seed + side)
        self.t = 0
        self.button = self.buttons[0] if self.buttons else "LP"

    def reset(self) -> None:
        self.t = 0

    def act(self, obs, info, state) -> ControllerState:
        self.t += 1
        phase = self.t % 16
        if phase == 0:
            self.button = str(self.rng.choice(self.buttons)) if self.buttons else "LP"
        if phase < 8:
            return ControllerState(self.fwd)
        if phase < 10:
            return ControllerState("neutral", frozenset({self.button}))
        if phase < 12:
            return ControllerState("neutral")
        return ControllerState(self.back)


class _WidthStats:
    """Per-address change statistics for one integer width."""

    def __init__(self, n: int):
        self.init = np.zeros(n, np.int64)
        self.maxv = np.zeros(n, np.int64)
        self.minv = np.full(n, 1 << 30, np.int64)
        self.change_count = np.zeros(n, np.int64)
        self.drop_count = np.zeros(n, np.int64)
        self.events: dict[int, list[tuple[int, int]]] = {}
        # unit-decrement bookkeeping (timer detection)
        self.unit_count = np.zeros(n, np.int64)
        self.last_unit_frame = np.full(n, -1, np.int64)
        self.interval_sum = np.zeros(n, np.int64)
        self.interval_sq = np.zeros(n, np.int64)
        self.intervals = np.zeros(n, np.int64)

    def _on_change(self, addr: int, frame: int, delta: int, value: int) -> None:
        if value > self.maxv[addr]:
            self.maxv[addr] = value
        if value < self.minv[addr]:
            self.minv[addr] = value
        self.change_count[addr] += 1
        if abs(delta) >= 2:
            self.events.setdefault(addr, []).append((frame, delta))
            if delta < 0:
                self.drop_count[addr] += 1
        if delta == -1:
            last = self.last_unit_frame[addr]
            if last >= 0:
                interval = frame - last
                self.interval_sum[addr] += interval
                self.interval_sq[addr] += interval * interval
                self.intervals[addr] += 1
            self.last_unit_frame[addr] = frame
            self.unit_count[addr] += 1


class RamScanner:
    def __init__(self, ram_size: int):
        self.size = ram_size
        self.frame = 0
        self.prev8: np.ndarray | None = None
        self.prev16: np.ndarray | None = None
        self.prev16be: np.ndarray | None = None
        self.w1 = _WidthStats(ram_size)
        self.w2 = _WidthStats(max(1, ram_size - 1))
        # Neo Geo (and other big-endian targets) store 16-bit counters as
        # high-byte-first; scanning only little-endian pairs misses them.
        self.w2be = _WidthStats(max(1, ram_size - 1))
        self._have_init = False

    def update(self, raw: bytes) -> None:
        cur8 = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
        if not self._have_init:
            self.w1.init[:] = cur8
            self.w1.maxv[:] = cur8
            self.w1.minv[:] = cur8
            self.prev8 = cur8
            cur16 = cur8[:-1] | (cur8[1:] << 8)
            self.w2.init[:] = cur16
            self.w2.maxv[:] = cur16
            self.w2.minv[:] = cur16
            self.prev16 = cur16
            cur16be = (cur8[:-1] << 8) | cur8[1:]
            self.w2be.init[:] = cur16be
            self.w2be.maxv[:] = cur16be
            self.w2be.minv[:] = cur16be
            self.prev16be = cur16be
            self._have_init = True
            self.frame += 1
            return

        delta8 = cur8 - self.prev8
        for i in np.flatnonzero(delta8).tolist():
            self.w1._on_change(i, self.frame, int(delta8[i]), int(cur8[i]))
        self.prev8 = cur8

        cur16 = cur8[:-1] | (cur8[1:] << 8)
        delta16 = cur16 - self.prev16
        for i in np.flatnonzero(delta16).tolist():
            self.w2._on_change(i, self.frame, int(delta16[i]), int(cur16[i]))
        self.prev16 = cur16

        cur16be = (cur8[:-1] << 8) | cur8[1:]
        delta16be = cur16be - self.prev16be
        for i in np.flatnonzero(delta16be).tolist():
            self.w2be._on_change(i, self.frame, int(delta16be[i]), int(cur16be[i]))
        self.prev16be = cur16be
        self.frame += 1

    # -- analysis ------------------------------------------------------- #
    @staticmethod
    def _match(events: dict[int, list[tuple[int, int]]], addr: int,
               ref: list[int], tol: int = 4) -> tuple[float, float, float]:
        ev = [f for f, d in events.get(addr, ()) if d < 0]
        if not ev or not ref:
            return 0.0, 0.0, 0.0
        matched_ref = sum(1 for f in ref if any(abs(f - e) <= tol for e in ev))
        matched_ev = sum(1 for e in ev if any(abs(e - f) <= tol for f in ref))
        recall = matched_ref / len(ref)
        precision = matched_ev / len(ev)
        if recall + precision == 0:
            return 0.0, recall, precision
        return 2 * recall * precision / (recall + precision), recall, precision

    def health_candidates(self, ref: list[int], top: int = 8) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for width, stats, endian in ((1, self.w1, "little"),
                                     (2, self.w2, "little"),
                                     (2, self.w2be, "big")):
            for addr, events in stats.events.items():
                if stats.drop_count[addr] < 2:
                    continue
                init = int(stats.init[addr])
                if init < 20 or init > 4096:
                    continue
                if int(stats.maxv[addr]) > init:  # health never exceeds the start value
                    continue
                if int(stats.minv[addr]) >= init:
                    continue
                f1, recall, precision = self._match(stats.events, addr, ref)
                if f1 <= 0:
                    continue
                out.append({
                    "addr": int(addr), "width": width, "endian": endian,
                    "f1": round(f1, 3),
                    "recall": round(recall, 3), "precision": round(precision, 3),
                    "init": init, "min": int(stats.minv[addr]),
                    "drops": int(stats.drop_count[addr]),
                })
        out.sort(key=lambda c: (c["f1"], c["drops"]), reverse=True)
        return out[:top]

    def timer_candidates(self, top: int = 5) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for width, stats, endian in ((1, self.w1, "little"),
                                     (2, self.w2, "little"),
                                     (2, self.w2be, "big")):
            for addr in range(len(stats.unit_count)):
                if stats.unit_count[addr] < 8 or stats.intervals[addr] < 6:
                    continue
                mean = stats.interval_sum[addr] / stats.intervals[addr]
                var = stats.interval_sq[addr] / stats.intervals[addr] - mean * mean
                if not (45.0 <= mean <= 75.0) or var > 36.0:
                    continue
                init = int(stats.init[addr])
                if not (2 <= init <= 300):
                    continue
                out.append({
                    "addr": int(addr), "width": width, "endian": endian,
                    "init": init,
                    "mean_interval": round(float(mean), 2), "std": round(float(var ** 0.5), 2),
                    "decrements": int(stats.unit_count[addr]),
                })
        out.sort(key=lambda c: (c["std"], -c["decrements"]))
        return out[:top]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Correlate RAM changes with HUD health drops")
    p.add_argument("--game", required=True)
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write discovered ram_map JSON here")
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--tol", type=int, default=4, help="frame tolerance when matching drops")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    config: GameConfig = load_game_config(args.game)
    backend = make_backend(config, capture_video=True)
    backend.reset()
    start_state = None
    if config.start_state and Path(config.start_state).exists():
        start_state = Path(config.start_state).read_bytes()
    if start_state:
        backend.load_state(start_state)
    else:
        print("warning: no start_state configured; scanning from the current game state")
    for _ in range(4):
        backend.step([ControllerState(), ControllerState()])
        if backend.screen().max() > 0:
            break

    vision = VisionStateReader(config.hud)
    if not vision.calibrate(backend.screen()):
        print("error: could not locate the health bars. Add a \"hud\" section to the game "
              "config with p1_bar/p2_bar rectangles and retry.")
        backend.close()
        return 1

    if backend.ram_size == 0:
        print("error: this core does not expose system RAM; vision state is the only option")
        backend.close()
        return 1

    scanner = RamScanner(backend.ram_size)
    p1_bot = BrawlerBot(1, config.action_buttons, args.seed)
    p2_bot = BrawlerBot(2, config.action_buttons, args.seed)
    prev = vision.read(backend)
    ref_p1: list[int] = []
    ref_p2: list[int] = []
    total = int(args.seconds * 60)
    round_ended = False

    for frame in range(total):
        raw = backend.read_ram(0, backend.ram_size)
        scanner.update(raw)
        state = vision.read(backend)
        if state.valid:
            if prev.valid:
                if state.p1_frac < prev.p1_frac - 0.01:
                    ref_p1.append(scanner.frame)
                if state.p2_frac < prev.p2_frac - 0.01:
                    ref_p2.append(scanner.frame)
            prev = state
            if state.p1_dead or state.p2_dead:
                round_ended = True
        backend.step([p1_bot.act(None, {}, state), p2_bot.act(None, {}, state)])
        if round_ended:
            print(f"round ended after {frame} frames; stopping scan")
            break
        if (frame + 1) % 600 == 0:
            print(f"  {frame + 1}/{total} frames, p1 damage events={len(ref_p1)}, "
                  f"p2 damage events={len(ref_p2)}")
    backend.close()

    if len(ref_p1) < 3 or len(ref_p2) < 3:
        print("not enough damage events were observed; try a longer --seconds or a "
              "different start state")

    p1_candidates = scanner.health_candidates(ref_p1, args.top)
    p2_candidates = scanner.health_candidates(ref_p2, args.top)
    timers = scanner.timer_candidates()

    def show(title: str, rows: list[dict[str, Any]]) -> None:
        print(f"\n{title}")
        if not rows:
            print("  (none)")
        for row in rows:
            print(f"  0x{row['addr']:06x} width={row['width']} endian={row['endian']} "
                  f"f1={row['f1']:.2f} "
                  f"recall={row['recall']:.2f} precision={row['precision']:.2f} "
                  f"init={row['init']} min={row['min']} drops={row['drops']}")

    show("P1 health candidates", p1_candidates)
    show("P2 health candidates", p2_candidates)
    print("\nTimer candidates")
    for row in timers:
        print(f"  0x{row['addr']:06x} width={row['width']} endian={row['endian']} "
              f"init={row['init']} "
              f"interval={row['mean_interval']}±{row['std']} decrements={row['decrements']}")
    if not timers:
        print("  (none)")

    def field(best: dict[str, Any]) -> dict[str, Any]:
        out = {"addr": best["addr"], "width": best["width"]}
        if best["endian"] != "little":
            out["endian"] = best["endian"]
        return out

    ram_map: dict[str, Any] = {}
    if p1_candidates and p1_candidates[0]["f1"] >= 0.5:
        best = p1_candidates[0]
        ram_map["p1_hp"] = field(best)
        ram_map["p1_hp_max"] = best["init"]
    if p2_candidates and p2_candidates[0]["f1"] >= 0.5:
        best = p2_candidates[0]
        if ram_map.get("p1_hp", {}).get("addr") != best["addr"]:
            ram_map["p2_hp"] = field(best)
            ram_map["p2_hp_max"] = best["init"]
    if timers:
        best = timers[0]
        ram_map["timer"] = field(best)
        ram_map["timer_max"] = best["init"]

    if ram_map:
        print("\nAdd this to the game config (verify against a few rounds first):")
        print(json.dumps({"ram_map": ram_map}, indent=2))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({"ram_map": ram_map}, indent=2) + "\n")
            print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
