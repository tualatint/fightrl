"""Play the game yourself with the keyboard and save a start state.

Real fighting games start with menus and character select.  Rather than
scripting that, record a save state at the first frame of a match and point
``start_state`` in the game config at it.  Every episode then begins exactly
there.

Controls
--------
arrows / WASD : move
J K L         : first three action buttons (e.g. LP MP HP on SF2)
U I O         : next three action buttons (e.g. LK MK HK on SF2)
Enter         : START          RightShift / N : SELECT
F5            : save state     F12 : screenshot
Space         : pause          Esc : quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .backends import make_backend
from .core.config import load_game_config
from .core.controller import ControllerState


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Keyboard-play a game and capture a save state")
    p.add_argument("--game", required=True)
    p.add_argument("--out", default=None, help="state path (default: states/<game>.state)")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--from", dest="from_state", default=None,
                   help="load this state at startup (defaults to the configured start_state)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    import pygame

    config = load_game_config(args.game)
    backend = make_backend(config)
    load_path = args.from_state or config.start_state
    if load_path and Path(load_path).exists():
        backend.load_state(Path(load_path).read_bytes())
        print(f"loaded {load_path}")

    frame = backend.screen()
    h, w = frame.shape[:2]
    pygame.init()
    pygame.display.set_caption(f"fightrl capture - {config.title or config.id} "
                               f"(S save, P screenshot, Esc quit)")
    screen = pygame.display.set_mode((w * args.scale, h * args.scale))
    clock = pygame.time.Clock()

    out_path = Path(args.out or f"states/{config.id}.state")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    buttons = list(config.action_buttons)
    key_map = {}
    for key, btn in zip((pygame.K_j, pygame.K_k, pygame.K_l, pygame.K_u, pygame.K_i, pygame.K_o), buttons):
        key_map[key] = btn
    key_map[pygame.K_RETURN] = "START"
    key_map[pygame.K_RSHIFT] = "SELECT"

    running = True
    paused = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F5:
                    data = backend.save_state()
                    out_path.write_bytes(data)
                    print(f"saved state ({len(data)} bytes) -> {out_path}")
                elif event.key == pygame.K_F12:
                    shot = f"states/{config.id}_screenshot.png"
                    pygame.image.save(screen, shot)
                    print(f"screenshot -> {shot}")
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("paused" if paused else "running")

        keys = pygame.key.get_pressed()
        if not paused:
            dx = int(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - int(keys[pygame.K_LEFT] or keys[pygame.K_a])
            dy = int(keys[pygame.K_DOWN] or keys[pygame.K_s]) - int(keys[pygame.K_UP] or keys[pygame.K_w])
            direction = {
                (0, 0): "neutral", (0, -1): "up", (0, 1): "down",
                (-1, 0): "left", (1, 0): "right",
                (-1, -1): "up_left", (1, -1): "up_right",
                (-1, 1): "down_left", (1, 1): "down_right",
            }[(dx, dy)]
            held = frozenset(btn for key, btn in key_map.items() if keys[key])
            backend.step([ControllerState(direction, held), ControllerState()])

        frame = backend.screen()
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(np.transpose(frame, (1, 0, 2))))
        if args.scale != 1:
            surf = pygame.transform.scale(surf, (w * args.scale, h * args.scale))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)

    backend.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
