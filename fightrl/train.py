"""PPO training for fightrl environments (with optional self-play league).

Examples
--------
Train on the built-in mock game against a scripted CPU::

    python -m fightrl.train --game mock --opponent aggressive --steps 300000

Train on a real ROM (needs a start-state capture, see README)::

    python -m fightrl.train --game sf2_snes --opponent league:runs/sf2/league \
        --steps 5000000 --n-envs 8 --device cuda

Adding ``--league <dir>`` writes a frozen copy of the model into ``<dir>``
every ``--save-every`` timesteps; the environments re-scan that directory at
every episode, so the opponent pool grows while training proceeds.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .core.config import list_games, load_game_config
from .env import FightEnv


class LeagueCheckpointCallback(BaseCallback):
    """Periodically save the policy and publish a frozen copy to the league."""

    def __init__(self, save_dir: Path, league_dir: Path | None, every: int):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.league_dir = Path(league_dir) if league_dir else None
        self.every = int(every)
        self.last = 0
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.league_dir:
            self.league_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        steps = self.model.num_timesteps
        if steps - self.last < self.every:
            return True
        self.last = steps
        name = f"ppo_{steps:09d}"
        path = self.save_dir / name
        self.model.save(str(path))
        if self.league_dir:
            shutil.copyfile(f"{path}.zip", self.league_dir / f"{name}.zip")
            print(f"[league] published {name} -> {self.league_dir}")
        return True


class WinRateCallback(BaseCallback):
    """Logs win/loss rate and keeps the best win-rate checkpoint."""

    def __init__(self, save_dir: Path, min_episodes: int = 8, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.min_episodes = min_episodes
        self.best = float("-inf")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buffer = list(self.model.ep_info_buffer)
        if len(buffer) < self.min_episodes:
            return
        winners = np.array([e.get("winner", 0) for e in buffer])
        win_rate = float(np.mean(winners == 1))
        loss_rate = float(np.mean(winners == 2))
        draw_rate = float(np.mean(winners == 0))
        self.logger.record("rollout/win_rate", win_rate)
        self.logger.record("rollout/loss_rate", loss_rate)
        self.logger.record("rollout/draw_rate", draw_rate)
        if self.verbose:
            print(f"[league] win_rate={win_rate:.2f} loss_rate={loss_rate:.2f} "
                  f"draw_rate={draw_rate:.2f} episodes={len(buffer)}")
        # Draws must not look as good as wins: a stalling policy that times out
        # every round would otherwise earn a perfect win rate.
        score = win_rate - loss_rate + 0.25 * draw_rate
        if score > self.best + 1e-6:
            self.best = score
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.save_dir / "best"))


def make_thunk(game: str, opponent: str, seed: int, kwargs: dict[str, Any]) -> Callable[[], Any]:
    def thunk() -> Any:
        torch.set_num_threads(1)
        env = FightEnv(game, opponent=opponent, **kwargs)
        env = Monitor(env, info_keywords=("winner", "p1_hp", "p2_hp"))
        env.reset(seed=seed)
        return env

    return thunk


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a PPO agent to play a fighting game")
    p.add_argument("--game", default="mock", help="game id (see --list-games) or path to a game JSON")
    p.add_argument("--opponent", default="aggressive",
                   help="aggressive|defensive|random|passive|checkpoint:<zip>|league:<dir>|none")
    p.add_argument("--league", default=None,
                   help="enable self-play league: directory where frozen checkpoints are published")
    p.add_argument("--steps", type=int, default=300_000, help="total environment steps")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=256, help="rollout length per env")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out", default="runs/ppo", help="output directory")
    p.add_argument("--save-every", type=int, default=50_000, help="checkpoint interval (env steps)")
    p.add_argument("--action-mode", default=None, choices=[None, "macro", "hybrid", "full"])
    p.add_argument("--frameskip", type=int, default=None)
    p.add_argument("--obs-size", type=int, default=84)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--max-episode-frames", type=int, default=None)
    p.add_argument("--dummy", action="store_true", help="use DummyVecEnv (no subprocesses)")
    p.add_argument("--resume", default=None, help="resume from a saved .zip model")
    p.add_argument("--ram-map", default=None, help="JSON ram_map to use instead of the game config's")
    p.add_argument("--tensorboard", action="store_true", help="write tensorboard logs")
    p.add_argument("--list-games", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.list_games:
        for game in list_games():
            print(f"{game['id']:<16} {game['backend']:<9} {game['title']}")
        return 0

    config = load_game_config(args.game)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    league_dir = Path(args.league) if args.league else None
    opponent = f"league:{league_dir}" if league_dir else args.opponent

    env_kwargs = dict(
        action_mode=args.action_mode,
        frameskip=args.frameskip,
        frame_stack=args.frame_stack,
        obs_size=(args.obs_size, args.obs_size),
        max_episode_frames=args.max_episode_frames,
        ram_map=args.ram_map,
    )
    print(f"game={config.id} backend={config.backend} opponent={opponent} "
          f"action_mode={args.action_mode or config.action_mode_default} "
          f"n_envs={args.n_envs}")

    thunks = [make_thunk(args.game, opponent, args.seed + i, env_kwargs)
              for i in range(args.n_envs)]
    if args.dummy or args.n_envs == 1:
        venv = DummyVecEnv(thunks)
    else:
        venv = SubprocVecEnv(thunks)

    print(f"observation space: {venv.observation_space}")
    mode = args.action_mode or config.action_mode_default
    if mode == "hybrid":
        print(f"action space: {venv.action_space} "
              f"({config.n_primitives} primitives + {len(config.macros)} macros)")
    else:
        print(f"action space: {venv.action_space} ({len(config.macros)} macros)")

    if args.n_steps * args.n_envs < args.batch_size:
        print(f"warning: n_steps*n_envs={args.n_steps * args.n_envs} < batch_size={args.batch_size}",
              file=sys.stderr)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.resume:
        model = PPO.load(args.resume, env=venv, device=device)
        # With reset_num_timesteps=False SB3 adds the checkpoint's counter to
        # the budget itself, so ``--steps`` always means "additional steps".
        print(f"resumed from {args.resume} at {model.num_timesteps} steps")
    else:
        model = PPO(
            "MultiInputPolicy",
            venv,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            n_epochs=args.n_epochs,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=str(out_dir / "tb") if args.tensorboard else None,
            device=device,
            seed=args.seed,
            verbose=1,
        )

    model.learn(
        total_timesteps=args.steps,
        callback=[LeagueCheckpointCallback(out_dir / "checkpoints", league_dir, args.save_every),
                  WinRateCallback(out_dir, verbose=1)],
        reset_num_timesteps=False,
        progress_bar=False,
    )
    model.save(str(out_dir / "final"))
    print(f"saved model to {out_dir / 'final.zip'}")

    # Round-robin sanity check against the scripted bots.
    if config.backend == "mock":
        print("evaluating against scripted bots ...")
        for style in ("passive", "random", "aggressive"):
            wins, episodes = evaluate_quickly(args.game, model, style, env_kwargs, args.seed)
            print(f"  vs {style:<10} win_rate={wins / max(episodes, 1):.2f} ({episodes} episodes)")
    venv.close()
    return 0


def evaluate_quickly(game: str, model: Any, opponent: str, env_kwargs: dict[str, Any],
                     seed: int, episodes: int = 5) -> tuple[float, int]:
    env = FightEnv(game, opponent=opponent, **env_kwargs)
    wins = 0
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + 1000 + episode)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        wins += int(info["winner"] == 1)
    env.close()
    return wins, episodes


if __name__ == "__main__":
    raise SystemExit(main())
