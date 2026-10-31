"""Evaluate a trained policy and optionally export a video of it playing."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from .env import FightEnv


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained fighting-game agent")
    p.add_argument("--game", default="mock")
    p.add_argument("--model", required=True, help="path to a .zip PPO model")
    p.add_argument("--opponent", default="aggressive")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    p.add_argument("--video", default=None, help="write an mp4 replay here")
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--action-mode", default=None, choices=[None, "macro", "hybrid", "full"])
    p.add_argument("--frameskip", type=int, default=None)
    p.add_argument("--obs-size", type=int, default=84)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--max-episode-frames", type=int, default=None)
    p.add_argument("--json-out", default=None, help="write summary metrics as JSON")
    p.add_argument("--ram-map", default=None, help="JSON ram_map to use instead of the game config's")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO.load(args.model, device=device)
    env_kwargs = dict(
        action_mode=args.action_mode,
        frameskip=args.frameskip,
        obs_size=(args.obs_size, args.obs_size),
        frame_stack=args.frame_stack,
        max_episode_frames=args.max_episode_frames,
        ram_map=args.ram_map,
    )
    env = FightEnv(args.game, opponent=args.opponent, **env_kwargs)

    writer = None
    if args.video:
        import imageio.v2 as imageio

        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=args.fps, macro_block_size=1)
        env.on_frame = lambda frame: writer.append_data(frame)

    actions = Counter()
    results = []
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            dealt = taken = 0.0
            while True:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                if env.action_mode in ("macro", "hybrid"):
                    actions[env.action_names[int(action)]] += 1
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                dealt += info.get("dmg_dealt", 0.0)
                taken += info.get("dmg_taken", 0.0)
                if terminated or truncated:
                    break
            row = {
                "episode": episode + 1,
                "winner": int(info["winner"]),
                "result": {1: "win", 2: "loss", 0: "draw"}[int(info["winner"])],
                "reward": round(total_reward, 3),
                "frames": info["frames"],
                "damage_dealt": round(dealt, 3),
                "damage_taken": round(taken, 3),
                "final_hp": [round(info["p1_frac"], 3), round(info["p2_frac"], 3)],
            }
            results.append(row)
            print(json.dumps(row))
    finally:
        if writer is not None:
            writer.close()
        env.close()

    wins = sum(r["winner"] == 1 for r in results)
    losses = sum(r["winner"] == 2 for r in results)
    summary = {
        "model": args.model,
        "game": args.game,
        "opponent": args.opponent,
        "episodes": len(results),
        "wins": wins,
        "losses": losses,
        "draws": len(results) - wins - losses,
        "win_rate": round(wins / max(len(results), 1), 3),
        "mean_reward": round(float(np.mean([r["reward"] for r in results])), 3),
        "mean_damage_dealt": round(float(np.mean([r["damage_dealt"] for r in results])), 3),
        "mean_damage_taken": round(float(np.mean([r["damage_taken"] for r in results])), 3),
        "mean_frames": round(float(np.mean([r["frames"] for r in results])), 1),
        "top_actions": actions.most_common(10),
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"summary": summary, "episodes": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
