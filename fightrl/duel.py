"""Two-agent (P1 + P2) PPO training: a port of Lexer04's Samurai Shodown setup.

Adapted from
https://github.com/Lexer04/Samurai-Shodown-with-Reinforcement-Learning-PPO
("Train 2 agent to fight one another in Samurai Shodown with PPO", Yoshua
Kaleb Purwanto).  That project trains P1 against the CPU, then P2 against the
CPU, then pits the two learned policies against each other in a 2-player
environment.  Here the same idea runs on any FightRL game (the mock engine,
SF2, or Samurai Shodown II on FBNeo) and the two policies co-evolve instead of
being trained only once.

The key piece is :class:`P2PerspectiveEnv`: a :class:`FightEnv` in which the
*learner* controls P2.  It swaps the controller ports at the backend and
mirrors the state vector + reward, so a policy always sees itself in slot 0.
The frozen/policy opponent drives P1 and always sees the unmirrored view.

Layout produced by ``python -m fightrl.duel``::

    runs/duel/
      p1/final.zip  p1/best.zip  p1/checkpoints/
      p2/final.zip  p2/best.zip  p2/checkpoints/
      duel_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .backends.base import Backend
from .core.config import load_game_config
from .core.controller import ControllerState
from .env import FightEnv
from .opponents import Opponent, PolicyOpponent, ScriptedBot
from .train import LeagueCheckpointCallback, WinRateCallback


class FlippedBot(ScriptedBot):
    """Scripted bot driving P1; its heuristics assume it is P2."""

    def act(self, obs: Any, info: dict[str, Any], state: Any) -> ControllerState:
        return super().act(obs, info, state).flipped()


class SwappedBackend(Backend):
    """Present a two-port backend with P1/P2 swapped (learner = P2)."""

    name = "swapped"

    def __init__(self, inner: Backend):
        self.inner = inner

    @property
    def n_ports(self) -> int:
        return self.inner.n_ports

    @property
    def fps(self) -> float:
        return self.inner.fps

    @property
    def has_ram(self) -> bool:
        return self.inner.has_ram

    @property
    def ram_size(self) -> int:
        return self.inner.ram_size

    def reset(self, seed: int | None = None) -> None:
        self.inner.reset(seed=seed)

    def step(self, controllers: Sequence[ControllerState]) -> None:
        self.inner.step([controllers[1], controllers[0]])

    def screen(self) -> np.ndarray:
        return self.inner.screen()

    def read_ram(self, addr: int, size: int) -> bytes:
        return self.inner.read_ram(addr, size)

    def save_state(self) -> bytes:
        return self.inner.save_state()

    def load_state(self, data: bytes) -> None:
        self.inner.load_state(data)

    def close(self) -> None:
        self.inner.close()

    def __getattr__(self, item: str) -> Any:
        # Expose core-specific attributes (mock_backend.p1, ...) for debugging
        # without recursing before ``inner`` exists.
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(item)
        return getattr(inner, item)


class P2PerspectiveEnv(FightEnv):
    """FightEnv where the learner plays P2 and ``opponent`` plays P1.

    Observations are P2-centric (the state vector is mirrored and not the
    pixels, matching how a frozen policy is fed to P2 elsewhere in FightRL),
    rewards are computed from P2's damage/win perspective, and ``info`` is
    learner-centric: ``winner == 1`` always means "the learner won".
    ``env.state`` itself stays canonical (P1/P2 as the emulator sees them).
    """

    def __init__(self, game: Any = "mock", *, opponent: Any = "aggressive", **kwargs: Any):
        super().__init__(game, opponent="none", **kwargs)
        self.backend = SwappedBackend(self.backend)
        self.opponent = build_p1_opponent(opponent, self)

    # The learner is P2, so macro directions resolve against P2's facing.
    def action_to_controllers(self, action: Any, side: int = 2) -> list[ControllerState]:
        return super().action_to_controllers(action, side=2)

    def _observe(self) -> dict[str, np.ndarray]:
        obs = super()._observe()  # canonical; also sets self._last_obs
        mirrored = FightEnv._opponent_obs(obs)
        # _last_obs feeds the P1 opponent via _opponent_obs, which mirrors
        # again -- the mirror is an involution, so P1 sees the canonical view.
        self._last_obs = mirrored
        return mirrored

    def _reward(self, prev: Any, cur: Any, frames: int) -> tuple[float, float, float]:
        w = self.config.reward
        p1_damaged = float(np.clip(prev.p2_frac - cur.p2_frac, 0.0, 1.0))
        p2_damaged = float(np.clip(prev.p1_frac - cur.p1_frac, 0.0, 1.0))
        dealt, taken = p2_damaged, p1_damaged
        lead_prev = prev.p1_frac - prev.p2_frac
        lead_cur = cur.p1_frac - cur.p2_frac
        r = w.get("deal", 1.0) * dealt - w.get("take", 1.0) * taken
        r -= w.get("lead", 0.0) * (lead_cur - lead_prev)
        d_prev, d_cur = prev.distance, cur.distance
        if d_prev is not None and d_cur is not None:
            width = float(cur.stage_width or 320.0)
            r += w.get("approach", 0.0) * ((d_prev - d_cur) / width)
        r -= w.get("time", 0.0) * (frames / max(self.backend.fps, 1.0))
        if cur.round_over:
            r += self._result_reward(cur.winner)
        return r, dealt, taken

    def _result_reward(self, winner: int) -> float:
        # Canonical winner 2 means the learner (P2) won the round.
        w = self.config.reward
        if winner == 2:
            return w.get("win", 1.0)
        if winner == 1:
            return -w.get("loss", 1.0)
        return 0.0

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        info["p1_hp"], info["p2_hp"] = info["p2_hp"], info["p1_hp"]
        info["p1_frac"], info["p2_frac"] = info["p2_frac"], info["p1_frac"]
        info["winner"] = {0: 0, 1: 2, 2: 1}[int(info["winner"])]
        info["side"] = 2
        return info


def build_p1_opponent(spec: Any, env: FightEnv) -> Opponent | None:
    """Build the P1 side of a :class:`P2PerspectiveEnv` from a spec string.

    Supports the scripted styles, ``checkpoint:<zip>`` and ``none``.  Unlike
    :func:`fightrl.opponents.make_opponent`, the decode is facing-correct for
    P1 (``side=1``), and scripted bots are flipped because their heuristics
    assume they are P2.
    """
    if spec is None:
        return None
    if not isinstance(spec, str):
        return spec
    spec = spec.strip()
    if spec.lower() in ("none", "noop", "off"):
        return None
    if spec.startswith("bot:"):
        spec = spec.split(":", 1)[1]
    if spec in ("aggressive", "defensive", "random", "passive"):
        return FlippedBot(spec, buttons=env.config.action_buttons)
    if spec.startswith("checkpoint:"):
        path = spec.split(":", 1)[1]
        if not Path(path).exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        decode = lambda action: FightEnv.action_to_controllers(env, action, side=1)  # noqa: E731
        return PolicyOpponent(PPO.load(path, device="cpu"), decode, deterministic=False)
    raise ValueError(f"unsupported P1 opponent spec {spec!r} (use a scripted style, checkpoint:<zip> or none)")


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def _make_env(game: str, side: int, opponent: str, kwargs: dict[str, Any]) -> FightEnv:
    cls = P2PerspectiveEnv if side == 2 else FightEnv
    return cls(game, opponent=opponent, **kwargs)


def _thunk(game: str, side: int, opponent: str, seed: int, kwargs: dict[str, Any]) -> Callable[[], Any]:
    def thunk() -> Any:
        torch.set_num_threads(1)
        env = _make_env(game, side, opponent, kwargs)
        env = Monitor(env, info_keywords=("winner", "p1_hp", "p2_hp"))
        env.reset(seed=seed)
        return env

    return thunk


def train_side(
    *,
    game: str,
    side: int,
    opponent: str,
    steps: int,
    out_dir: str | Path,
    n_envs: int = 8,
    n_steps: int = 256,
    batch_size: int = 2048,
    lr: float = 3e-4,
    ent_coef: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    n_epochs: int = 10,
    seed: int = 0,
    device: str = "auto",
    save_every: int = 50_000,
    resume: str | Path | None = None,
    dummy: bool = False,
    tensorboard: bool = False,
    env_kwargs: Mapping[str, Any] | None = None,
) -> Path:
    """Train one side of the duel and return the path to ``final.zip``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(env_kwargs or {})

    thunks = [_thunk(game, side, opponent, seed + i, kwargs) for i in range(n_envs)]
    venv = DummyVecEnv(thunks) if dummy or n_envs == 1 else SubprocVecEnv(thunks)

    effective_batch = min(batch_size, n_steps * n_envs)
    if effective_batch != batch_size:
        print(f"note: batch_size {batch_size} > n_steps*n_envs; using {effective_batch}",
              file=sys.stderr)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if resume and Path(resume).exists():
        model = PPO.load(str(resume), env=venv, device=device)
        print(f"P{side}: resumed {resume} at {model.num_timesteps} steps")
    else:
        model = PPO(
            "MultiInputPolicy",
            venv,
            n_steps=n_steps,
            batch_size=effective_batch,
            learning_rate=lr,
            ent_coef=ent_coef,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_epochs=n_epochs,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=str(out / "tb") if tensorboard else None,
            device=device,
            seed=seed,
            verbose=1,
        )

    print(f"P{side}: {steps} steps vs {opponent} ({n_envs} envs, {device}) -> {out}")
    model.learn(
        total_timesteps=steps,
        callback=[
            LeagueCheckpointCallback(out / "checkpoints", None, save_every),
            WinRateCallback(out, verbose=1),
        ],
        reset_num_timesteps=False,
        progress_bar=False,
    )
    model.save(str(out / "final"))
    venv.close()
    print(f"P{side}: saved {out / 'final.zip'}")
    return out / "final.zip"


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


def head_to_head(
    *,
    game: str,
    p1_model: str | Path,
    p2_model: str | Path,
    episodes: int = 10,
    seed: int = 0,
    env_kwargs: Mapping[str, Any] | None = None,
    video: str | Path | None = None,
    deterministic: bool = True,
    device: str = "auto",
) -> dict[str, Any]:
    """Play two live policies against each other, P1 as the learner.

    Both networks act every step (no frozen scripted opponent), which is the
    "P1 agent vs P2 agent" phase of the Lexer04 project.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    env = FightEnv(game, opponent="none", **dict(env_kwargs or {}))
    p1 = PPO.load(str(p1_model), device=device)
    p2 = PPO.load(str(p2_model), device=device)
    env.opponent = PolicyOpponent(
        p2,
        lambda action: env.action_to_controllers(action, side=2),
        deterministic=deterministic,
    )

    writer = None
    if video:
        import imageio.v2 as imageio

        Path(video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video), fps=env.backend.fps, macro_block_size=1)
        env.on_frame = lambda frame: writer.append_data(frame)

    rows = []
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            total_reward = 0.0
            dealt = taken = 0.0
            while True:
                action, _ = p1.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                dealt += info.get("dmg_dealt", 0.0)
                taken += info.get("dmg_taken", 0.0)
                if terminated or truncated:
                    break
            rows.append({
                "episode": episode + 1,
                "winner": int(info["winner"]),
                "result": {1: "p1", 2: "p2", 0: "draw"}[int(info["winner"])],
                "reward": round(total_reward, 3),
                "frames": info["frames"],
                "damage_dealt": round(dealt, 3),
                "damage_taken": round(taken, 3),
                "final_hp": [round(info["p1_frac"], 3), round(info["p2_frac"], 3)],
            })
            print(json.dumps(rows[-1]))
    finally:
        if writer is not None:
            writer.close()
        env.close()

    p1_wins = sum(r["winner"] == 1 for r in rows)
    p2_wins = sum(r["winner"] == 2 for r in rows)
    draws = len(rows) - p1_wins - p2_wins
    summary = {
        "game": game,
        "p1_model": str(p1_model),
        "p2_model": str(p2_model),
        "episodes": len(rows),
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "p1_win_rate": round(p1_wins / max(len(rows), 1), 3),
        "mean_reward": round(float(np.mean([r["reward"] for r in rows])), 3),
        "mean_frames": round(float(np.mean([r["frames"] for r in rows])), 1),
    }
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "episodes": rows}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train two PPO agents to fight each other")
    p.add_argument("--game", default="mock", help="game id (see --list-games) or path to a game JSON")
    p.add_argument("--phase", default="duel", choices=["p1", "p2", "duel", "head2head"],
                   help="p1: train P1 only; p2: train P2 vs P1; "
                        "duel: alternate both, then play head-to-head; head2head: evaluate only")
    p.add_argument("--rounds", type=int, default=2, help="alternating rounds for --phase duel")
    p.add_argument("--steps", type=int, default=200_000, help="env steps per training stage")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out", default="runs/duel")
    p.add_argument("--save-every", type=int, default=50_000)
    p.add_argument("--dummy", action="store_true", help="use DummyVecEnv (no subprocesses)")
    p.add_argument("--tensorboard", action="store_true")
    # opponents / models
    p.add_argument("--p1-opponent", default="aggressive",
                   help="P1's opponent while P1 trains (scripted style, checkpoint:<zip>, none)")
    p.add_argument("--p2-opponent", default=None,
                   help="P2's opponent while P2 trains (default: the latest P1 checkpoint)")
    p.add_argument("--p1-model", default=None, help="P1 model for p2/head2head or resume for p1")
    p.add_argument("--p2-model", default=None, help="P2 model for head2head or resume for p2")
    # evaluation
    p.add_argument("--episodes", type=int, default=10, help="head-to-head episodes")
    p.add_argument("--stochastic", action="store_true", help="sample actions during head-to-head")
    p.add_argument("--video", default=None, help="write an mp4 of the head-to-head here")
    p.add_argument("--json-out", default=None, help="write head-to-head metrics here")
    # env
    p.add_argument("--action-mode", default=None, choices=[None, "macro", "hybrid", "full"])
    p.add_argument("--frameskip", type=int, default=None)
    p.add_argument("--obs-size", type=int, default=84)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--max-episode-frames", type=int, default=None)
    p.add_argument("--ram-map", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    config = load_game_config(args.game)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        action_mode=args.action_mode,
        frameskip=args.frameskip,
        frame_stack=args.frame_stack,
        obs_size=(args.obs_size, args.obs_size),
        max_episode_frames=args.max_episode_frames,
        ram_map=args.ram_map,
    )
    common = dict(
        game=args.game,
        steps=args.steps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        n_epochs=args.n_epochs,
        seed=args.seed,
        device=args.device,
        save_every=args.save_every,
        dummy=args.dummy,
        tensorboard=args.tensorboard,
        env_kwargs=env_kwargs,
    )

    def need(path: Path, hint: str) -> Path:
        if not path.exists():
            raise SystemExit(f"missing model {path}; {hint}")
        return path

    p1_path = Path(args.p1_model) if args.p1_model else out / "p1" / "final.zip"
    p2_path = Path(args.p2_model) if args.p2_model else out / "p2" / "final.zip"

    print(f"game={config.id} action_mode={args.action_mode or config.action_mode_default} "
          f"n_envs={args.n_envs}")

    if args.phase in ("p1", "duel"):
        if args.phase == "p1" or not p1_path.exists():
            p1_path = train_side(
                side=1,
                opponent=args.p1_opponent,
                out_dir=out / "p1",
                resume=args.p1_model,
                **common,
            )

    if args.phase == "p2":
        p1_path = need(p1_path, "train P1 first with --phase p1")
        opponent = args.p2_opponent or f"checkpoint:{p1_path}"
        train_side(
            side=2,
            opponent=opponent,
            out_dir=out / "p2",
            resume=args.p2_model,
            **common,
        )
        return 0

    if args.phase == "duel":
        for round_index in range(args.rounds):
            p1_path = need(p1_path, "train P1 first")
            p2_path = train_side(
                side=2,
                opponent=f"checkpoint:{p1_path}",
                out_dir=out / "p2",
                resume=p2_path if p2_path.exists() else None,
                **common,
            )
            p1_path = train_side(
                side=1,
                opponent=f"checkpoint:{p2_path}",
                out_dir=out / "p1",
                resume=p1_path,
                **common,
            )
            print(f"duel round {round_index + 1}/{args.rounds} done")

    p1_path = need(p1_path, "nothing to evaluate")
    p2_path = need(p2_path, "train P2 first with --phase p2 or --phase duel")

    result = head_to_head(
        game=args.game,
        p1_model=p1_path,
        p2_model=p2_path,
        episodes=args.episodes,
        seed=args.seed,
        env_kwargs=env_kwargs,
        video=args.video,
        deterministic=not args.stochastic,
        device=args.device,
    )
    payload = {"summary": result["summary"], "episodes": result["episodes"]}
    (out / "duel_eval.json").write_text(json.dumps(payload, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
