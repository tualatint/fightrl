"""A small deterministic 2D fighting game used for development and testing.

It is *not* Street Fighter or Samurai Shodown: it is a compact stand-in with
the same interface (health bars, rounds, hitstun/blockstun, jumps, 60 fps,
320x224 screen) and a game-RAM image, so the state readers, reward, RAM
scanner, replay tooling and training loop can be exercised end to end without
requiring any ROM.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.controller import ControllerState
from .base import Backend

SCREEN_W = 320
SCREEN_H = 224
STAGE_W = 480
GROUND_SCREEN_Y = 200

GRAVITY = 0.5
JUMP_VY = 9.0
WALK_SPEED = 2.4
HALF_WIDTH = 14.0
BODY_HEIGHT = 92.0
MIN_SEPARATION = 30.0

# RAM addresses of the mock game (mirrors how a real ROM layout looks).
RAM_SIZE = 0x2000
ADDR_P1_HP = 0x0A00
ADDR_P2_HP = 0x0A01
ADDR_TIMER = 0x0A02
ADDR_P1_X = 0x0A10
ADDR_P2_X = 0x0A12
ADDR_ROUND_OVER = 0x0A14

MOVES: dict[str, dict[str, float]] = {
    "LP": dict(startup=3, active=2, recovery=6, dmg=5, reach=30, hitstun=12, blockstun=8, push=2.0),
    "MP": dict(startup=5, active=3, recovery=8, dmg=9, reach=34, hitstun=14, blockstun=10, push=2.5),
    "HP": dict(startup=8, active=3, recovery=12, dmg=14, reach=40, hitstun=18, blockstun=12, push=3.5),
    "LK": dict(startup=5, active=3, recovery=8, dmg=7, reach=34, hitstun=13, blockstun=9, push=2.5),
    "MK": dict(startup=7, active=3, recovery=10, dmg=11, reach=38, hitstun=16, blockstun=11, push=3.0),
    "HK": dict(startup=10, active=4, recovery=14, dmg=17, reach=46, hitstun=20, blockstun=14, push=4.0),
}
DEFAULT_MOVE = MOVES["MP"]

CHARACTER_COLORS = [(70, 150, 240), (235, 120, 70)]


@dataclass
class Fighter:
    x: float
    facing: int
    hp: float
    color: tuple[int, int, int]
    y: float = 0.0
    vy: float = 0.0
    vx: float = 0.0
    move: str | None = None
    move_frame: int = 0
    hit_consumed: bool = False
    hitstun: int = 0
    blockstun: int = 0
    blocking: bool = False
    crouching: bool = False
    air_attack_used: bool = False
    ko: bool = False
    hit_flash: int = 0
    prev_buttons: frozenset[str] = field(default_factory=frozenset)

    @property
    def airborne(self) -> bool:
        return self.y > 0.0

    @property
    def busy(self) -> bool:
        return self.hitstun > 0 or self.blockstun > 0 or self.ko

    @property
    def attacking(self) -> bool:
        return self.move is not None

    def move_phase(self) -> str:
        if self.move is None:
            return "idle"
        params = MOVES.get(self.move, DEFAULT_MOVE)
        if self.move_frame < params["startup"]:
            return "startup"
        if self.move_frame < params["startup"] + params["active"]:
            return "active"
        return "recovery"


class MockFighter(Backend):
    name = "mock"
    has_ram = True
    fps = 60.0

    def __init__(self, health_max: int = 176, round_seconds: int = 99, seed: int | None = None,
                 p1_color: tuple[int, int, int] = CHARACTER_COLORS[0],
                 p2_color: tuple[int, int, int] = CHARACTER_COLORS[1]):
        self.health_max = float(health_max)
        self.round_frames = int(round_seconds * 60)
        self.p1_color = tuple(p1_color)
        self.p2_color = tuple(p2_color)
        self.rng = np.random.default_rng(seed)
        self.ram = bytearray(RAM_SIZE)
        self._frame: np.ndarray | None = None
        self._dirty = True
        self.p1 = Fighter(x=150.0, facing=1, hp=self.health_max, color=self.p1_color)
        self.p2 = Fighter(x=330.0, facing=-1, hp=self.health_max, color=self.p2_color)
        self.timer_frames = self.round_frames
        self.round_over = False
        self.kos = 0
        self.round_wins = [0, 0]
        self._noise = 0

    # -- Backend API -------------------------------------------------- #
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.p1 = Fighter(x=150.0, facing=1, hp=self.health_max, color=self.p1_color)
        self.p2 = Fighter(x=330.0, facing=-1, hp=self.health_max, color=self.p2_color)
        self.timer_frames = self.round_frames
        self.round_over = False
        self._write_ram()
        self._dirty = True

    def step(self, controllers: Sequence[ControllerState]) -> None:
        if self.round_over:
            return
        c1, c2 = controllers[0], controllers[1]
        self._update_fighter(self.p1, c1, self.p2)
        self._update_fighter(self.p2, c2, self.p1)
        self._resolve_hits(self.p1, self.p2)
        self._resolve_hits(self.p2, self.p1)
        self._separate()
        self._physics(self.p1)
        self._physics(self.p2)
        self._update_facing()
        self.timer_frames = max(0, self.timer_frames - 1)
        for fighter in (self.p1, self.p2):
            if fighter.hit_flash > 0:
                fighter.hit_flash -= 1
            if fighter.hitstun > 0:
                fighter.hitstun -= 1
            if fighter.blockstun > 0:
                fighter.blockstun -= 1
        if self.p1.hp <= 0 or self.p2.hp <= 0 or self.timer_frames == 0:
            self.round_over = True
            self.p1.ko = self.p1.hp <= 0
            self.p2.ko = self.p2.hp <= 0
        self._noise = (self._noise + 1) % 256
        self._write_ram()
        self._dirty = True

    def screen(self) -> np.ndarray:
        if self._frame is None or self._dirty:
            self._frame = self._render()
            self._dirty = False
        return self._frame

    def read_ram(self, addr: int, size: int) -> bytes:
        return bytes(self.ram[addr : addr + size])

    @property
    def ram_size(self) -> int:
        return RAM_SIZE

    def save_state(self) -> bytes:
        return pickle.dumps({
            "p1": self.p1, "p2": self.p2, "timer": self.timer_frames,
            "over": self.round_over, "ram": bytes(self.ram),
        })

    def load_state(self, data: bytes) -> None:
        payload = pickle.loads(data)
        self.p1, self.p2 = payload["p1"], payload["p2"]
        self.timer_frames, self.round_over = payload["timer"], payload["over"]
        self.ram = bytearray(payload["ram"])
        self._dirty = True

    # -- simulation ---------------------------------------------------- #
    def _update_fighter(self, f: Fighter, ctrl: ControllerState, opp: Fighter) -> None:
        if f.ko:
            return
        pressed = ctrl.buttons
        if f.busy:
            f.blocking = False
            f.prev_buttons = pressed
            return
        dx, dy = ctrl.vector()
        back_dir = -f.facing

        # begin an attack on a fresh button press
        newly = [b for b in pressed if b not in f.prev_buttons]
        if f.move is None and not (f.airborne and f.air_attack_used):
            for button in ("HP", "MP", "LP", "HK", "MK", "LK"):
                if button in newly or (button in pressed and not f.airborne and not f.prev_buttons):
                    if button in pressed:
                        f.move, f.move_frame, f.hit_consumed = button, 0, False
                        f.crouching = dy > 0 and not f.airborne
                        if f.airborne:
                            f.air_attack_used = True
                        break
        f.prev_buttons = pressed

        if f.move is not None:
            f.move_frame += 1
            params = MOVES.get(f.move, DEFAULT_MOVE)
            if f.move_frame >= params["startup"] + params["active"] + params["recovery"]:
                f.move = None
                f.move_frame = 0
            # attacking characters keep their horizontal momentum in the air
            if not f.airborne:
                f.blocking = False
                f.crouching = dy > 0
            return

        if f.airborne:
            f.blocking = False
            return

        f.crouching = dy > 0
        f.blocking = (dx == back_dir) and not f.crouching
        if f.blocking or dy > 0:
            f.vx = 0.0
        else:
            f.vx = dx * WALK_SPEED
        if dy < 0 and f.y <= 0.0:
            f.vy = JUMP_VY
            f.y = 0.01
            f.vx = dx * WALK_SPEED * 1.1
            f.air_attack_used = False

    def _resolve_hits(self, attacker: Fighter, defender: Fighter) -> None:
        if attacker.move is None or attacker.hit_consumed or defender.ko:
            return
        if attacker.move_phase() != "active":
            return
        params = MOVES.get(attacker.move, DEFAULT_MOVE)
        reach = params["reach"] + (18 if attacker.airborne else 0)
        gap = abs(defender.x - attacker.x)
        if gap > reach + HALF_WIDTH:
            return
        if defender.airborne != attacker.airborne and abs(attacker.y - defender.y) > 70:
            return
        attacker.hit_consumed = True
        if defender.blocking and not defender.airborne:
            defender.blockstun = int(params["blockstun"])
            defender.vx = -attacker.facing * 1.2
            return
        defender.hp = max(0.0, defender.hp - params["dmg"])
        defender.hitstun = int(params["hitstun"])
        defender.vx = attacker.facing * params["push"]
        defender.vy = max(defender.vy, 1.5) if defender.airborne else defender.vy
        defender.move = None
        defender.move_frame = 0
        defender.hit_flash = 4
        defender.blocking = False

    def _separate(self) -> None:
        gap = self.p2.x - self.p1.x
        if abs(gap) < MIN_SEPARATION:
            push = (MIN_SEPARATION - abs(gap)) / 2.0
            sign = 1.0 if gap >= 0 else -1.0
            self.p1.x -= push * sign
            self.p2.x += push * sign

    def _physics(self, f: Fighter) -> None:
        if f.airborne:
            f.y += f.vy
            f.vy -= GRAVITY
            if f.y <= 0.0:
                f.y = 0.0
                f.vy = 0.0
                f.air_attack_used = False
                f.move = None if f.move and f.move_frame > 4 else f.move
                f.vx = 0.0
        else:
            if f.hitstun > 0 or f.blockstun > 0:
                f.vx *= 0.85
            f.x += f.vx
            if not (f.hitstun > 0 or f.blockstun > 0 or f.move):
                f.vx = 0.0
        f.x = float(np.clip(f.x, 30.0, STAGE_W - 30.0))

    def _update_facing(self) -> None:
        if not self.p1.attacking and not self.p2.attacking and not self.p1.hitstun and not self.p2.hitstun:
            if self.p2.x >= self.p1.x:
                self.p1.facing, self.p2.facing = 1, -1
            else:
                self.p1.facing, self.p2.facing = -1, 1

    # -- RAM shadow ---------------------------------------------------- #
    def _write_ram(self) -> None:
        ram = self.ram
        ram[ADDR_P1_HP] = int(np.clip(self.p1.hp, 0, 255))
        ram[ADDR_P2_HP] = int(np.clip(self.p2.hp, 0, 255))
        ram[ADDR_TIMER] = int(self.timer_frames / 60)
        ram[ADDR_P1_X:ADDR_P1_X + 2] = int(self.p1.x).to_bytes(2, "little")
        ram[ADDR_P2_X:ADDR_P2_X + 2] = int(self.p2.x).to_bytes(2, "little")
        ram[ADDR_ROUND_OVER] = 1 if self.round_over else 0
        # Decorative memory activity so a RAM scanner has realistic noise.
        ram[0x0B00] = (self._noise + 37) % 256
        ram[0x0B01] = int(self.p1.y) % 256
        ram[0x0B02] = int(self.p2.y) % 256

    # -- rendering ----------------------------------------------------- #
    def _render(self) -> np.ndarray:
        frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        frame[:, :] = (12, 14, 34)
        frame[GROUND_SCREEN_Y:, :] = (38, 34, 30)
        for y in range(GROUND_SCREEN_Y, SCREEN_H, 8):
            frame[y : y + 2, :] = (52, 46, 40)

        mid = (self.p1.x + self.p2.x) / 2.0
        cam = float(np.clip(mid - SCREEN_W / 2, 0, STAGE_W - SCREEN_W))
        for fighter in (self.p1, self.p2):
            self._draw_fighter(frame, fighter, cam)
        self._draw_hud(frame)
        if self.round_over:
            frame[100:118, 120:200] = (240, 220, 60)
        return frame

    @staticmethod
    def _draw_fighter(frame: np.ndarray, f: Fighter, cam: float) -> None:
        x = int(f.x - cam)
        feet = int(GROUND_SCREEN_Y - f.y)
        height = int(BODY_HEIGHT * (0.62 if f.crouching else 1.0))
        color = (255, 255, 255) if f.hit_flash else f.color
        body_top = feet - height
        x0, x1 = x - 13, x + 13
        frame[body_top:feet, x0:x1] = color
        head_y0, head_y1 = body_top - 14, body_top - 2
        frame[head_y0:head_y1, x - 7 : x + 7] = color
        if f.attacking:
            params = MOVES.get(f.move, DEFAULT_MOVE)
            phase = f.move_phase()
            reach = int(params["reach"] * (0.55 if phase == "startup" else 1.0))
            arm_y = body_top + (18 if f.airborne else 30)
            if f.facing >= 0:
                frame[arm_y : arm_y + 7, x + 13 : x + 13 + reach] = color
            else:
                frame[arm_y : arm_y + 7, x - 13 - reach : x - 13] = color
        if f.blocking:
            frame[body_top + 12 : body_top + 36, x + (16 if f.facing > 0 else -20) : x + (20 if f.facing > 0 else -16)] = (220, 220, 230)

    def _draw_hud(self, frame: np.ndarray) -> None:
        p1_frac = float(np.clip(self.p1.hp / self.health_max, 0, 1))
        p2_frac = float(np.clip(self.p2.hp / self.health_max, 0, 1))
        # Dark backing so the bars read as bars.
        frame[8:22, 12:308] = (20, 20, 26)
        self._draw_bar(frame, 16, 10, 120, p1_frac, left_to_right=True)
        self._draw_bar(frame, 184, 10, 120, p2_frac, left_to_right=False)
        # round timer pips
        frac = self.timer_frames / self.round_frames
        frame[10:20, 148:172] = (40, 40, 48)
        frame[10:20, 148 : 148 + int(24 * frac)] = (230, 210, 90)

    @staticmethod
    def _draw_bar(frame: np.ndarray, x: int, y: int, width: int, frac: float, left_to_right: bool) -> None:
        color = (60, 220, 70) if frac > 0.30 else (235, 200, 45)
        fill = int(round(width * frac))
        if left_to_right:
            frame[y : y + 10, x : x + fill] = color
        else:
            frame[y : y + 10, x + width - fill : x + width] = color


def make_mock_backend(config: Any = None) -> MockFighter:
    params = dict(getattr(config, "mock", None) or {})
    return MockFighter(
        health_max=int(params.get("health", 176)),
        round_seconds=int(params.get("round_seconds", 99)),
        p1_color=tuple(params.get("p1_color", CHARACTER_COLORS[0])),
        p2_color=tuple(params.get("p2_color", CHARACTER_COLORS[1])),
    )
