"""Round-robin Elo ladder for checkpoints and scripted bots.

Every policy pair plays one game per side (removing P1/P2 start-state bias);
each policy also plays a few games against the scripted bots.  Ratings are
fitted with the Bradley-Terry model via iterative Elo updates, so the result
does not depend on the order games were played.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from .env import FightEnv
from .opponents import make_opponent


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def fit_ratings(results: list[tuple[str, str, float]], names: list[str],
                k: float = 32.0, passes: int = 6, seed: int = 0) -> dict[str, float]:
    """Bradley-Terry ratings from (player_a, player_b, score_a) games."""
    ratings = {name: 1500.0 for name in names}
    rng = np.random.default_rng(seed)
    for _ in range(passes):
        for index in rng.permutation(len(results)):
            a, b, score = results[int(index)]
            delta = k * (score - expected_score(ratings[a], ratings[b]))
            ratings[a] += delta
            ratings[b] -= delta
    return ratings


def play_game(env: FightEnv, model: PPO, opponent: str, seed: int) -> int:
    """Play one episode with ``model`` on P1 and return the winner (1/2/0)."""
    env.opponent = make_opponent(opponent, config=env.config, env=env)
    obs, _ = env.reset(seed=seed)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return int(info["winner"])


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Elo ladder over checkpoints and scripted bots")
    p.add_argument("--game", default="samsho2_neogeo")
    p.add_argument("--models", nargs="*", default=None, help="checkpoint .zip paths")
    p.add_argument("--model-dir", default=None, help="directory of .zip checkpoints")
    p.add_argument("--bots", default="aggressive,defensive,random,passive")
    p.add_argument("--games-per-pair", type=int, default=2, help="policy-vs-policy games (even)")
    p.add_argument("--bot-games", type=int, default=2, help="games per policy-vs-bot pair")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out", default=None, help="write ratings and games as JSON")
    p.add_argument("--action-mode", default=None, choices=[None, "macro", "hybrid", "full"])
    p.add_argument("--frameskip", type=int, default=None)
    p.add_argument("--obs-size", type=int, default=84)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--max-episode-frames", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    paths: list[str] = list(args.models or [])
    if args.model_dir:
        paths.extend(sorted(str(p) for p in Path(args.model_dir).glob("*.zip")))

    env = FightEnv(
        args.game,
        opponent="none",
        action_mode=args.action_mode,
        frameskip=args.frameskip,
        obs_size=(args.obs_size, args.obs_size),
        frame_stack=args.frame_stack,
        max_episode_frames=args.max_episode_frames,
    )
    bot_names = [b.strip() for b in args.bots.split(",") if b.strip()]
    models = {Path(p).stem: PPO.load(p, device=device) for p in paths}
    names = list(models) + bot_names
    specs = {name: f"checkpoint:{p}" for name, p in zip(models, paths)}

    results: list[tuple[str, str, float]] = []
    game_index = 0
    try:
        for i, a in enumerate(names):
            if a in bot_names:
                continue
            for b in names[i + 1:]:
                if b in bot_names:
                    continue
                orient = max(args.games_per_pair // 2, 1)
                for _ in range(orient):
                    for first, second in ((a, b), (b, a)):
                        winner = play_game(env, models[first], specs[second], args.seed + game_index)
                        game_index += 1
                        score = 0.5 if winner == 0 else float(winner == 1)
                        results.append((first, second, score))
        for a in models:
            for bot in bot_names:
                for _ in range(args.bot_games):
                    winner = play_game(env, models[a], bot, args.seed + game_index)
                    game_index += 1
                    results.append((a, bot, 0.5 if winner == 0 else float(winner == 1)))
    finally:
        env.close()

    ratings = fit_ratings(results, names, seed=args.seed)
    table = sorted(ratings.items(), key=lambda kv: -kv[1])
    for name, rating in table:
        played = sum(1 for a, b, _ in results if a == name or b == name)
        wins = sum(s if a == name else 1.0 - s for a, b, s in results if name in (a, b))
        print(f"{name:<22} elo={rating:6.1f}  games={played:3d}  wins={wins:.1f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "game": args.game,
            "ratings": {n: round(r, 2) for n, r in ratings.items()},
            "games_played": len(results),
            "results": [{"p1": a, "p2": b, "score_p1": s} for a, b, s in results],
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
