"""Realtime model-vs-model arena.

Serves a small web page where two checkpoints are picked and watched fighting
in the browser.  Frames are streamed as MJPEG; sides alternate every match so
both policies play P1 and P2.

    python -m fightrl.arena --game samsho2_neogeo --port 8765
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import random
import re
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel
from stable_baselines3 import PPO

from .core.config import load_game_config
from .env import FightEnv, observation_space_for
from .opponents import make_opponent

ROOT = Path(__file__).resolve().parent.parent
GAME = "samsho2_neogeo"
RUNS = ROOT / "runs"
PAGE = Path(__file__).with_name("arena.html")

app = FastAPI(title="FightRL Arena")
STATE: dict[str, object] = {"session": None, "id": 0}
LOCK = threading.Lock()
ENGINE: dict[str, object] = {"env": None}
MODEL_CACHE: dict[str, PPO] = {}
MODEL_ORDER: list[str] = []


def _get_env(game: str) -> FightEnv:
    """One libretro core per server process: tearing one down mid-flight is unsafe."""
    env = ENGINE["env"]
    if not isinstance(env, FightEnv):
        env = FightEnv(game, opponent="none")
        ENGINE["env"] = env
    return env


def _load_model(path: str) -> PPO:
    if path in MODEL_CACHE:
        return MODEL_CACHE[path]
    model = PPO.load(path, device="cuda" if torch.cuda.is_available() else "cpu")
    MODEL_CACHE[path] = model
    MODEL_ORDER.append(path)
    while len(MODEL_ORDER) > 8:
        MODEL_CACHE.pop(MODEL_ORDER.pop(0), None)
    return model


def _shutdown(session: "Session") -> None:
    """Stop a session and wait for its loop to leave the shared core alone."""
    session.stop_run()
    session.thread.join(timeout=8.0)


def _space_signature(space: object) -> dict[str, tuple]:
    """Keys, shapes and dtypes of a (possibly dict) space, for comparison."""
    subspaces = getattr(space, "spaces", None)
    if isinstance(subspaces, dict):
        return {
            key: (tuple(sub.shape), str(sub.dtype))
            for key, sub in subspaces.items()
        }
    return {"": (tuple(getattr(space, "shape", ())), str(getattr(space, "dtype", "")))}


def _stored_spaces(path: Path) -> tuple[int | None, dict[str, tuple] | None]:
    """Action count and observation signature of a checkpoint, weightless.

    SB3 2.x stores both spaces in its JSON ``data`` entry as base64 pickles.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read("data"))
        action = pickle.loads(base64.b64decode(data["action_space"][":serialized:"]))
        observation = pickle.loads(base64.b64decode(data["observation_space"][":serialized:"]))
    except Exception:
        return None, None
    return int(action.n), _space_signature(observation)


def _expected_spaces() -> tuple[int, dict[str, tuple]]:
    config = load_game_config(GAME)
    return config.hybrid_action_count, _space_signature(observation_space_for(config))


def _list_models() -> list[dict[str, object]]:
    expected_actions, expected_obs = _expected_spaces()
    items = []
    for path in sorted(list(RUNS.glob("*/checkpoints/*.zip")) + list(RUNS.glob("*/final*.zip"))):
        # Checkpoints with a different action space or observation would index
        # into the wrong inputs, so they are hidden rather than offered and broken.
        if _stored_spaces(path) != (expected_actions, expected_obs):
            continue
        match = re.search(r"(\d{6,})", path.stem)
        steps = int(match.group(1)) if match else 0
        items.append({
            "id": str(path),
            "label": f"{path.parent.parent.name}/{path.stem} · {steps:,} steps",
            "steps": steps,
        })
    return sorted(items, key=lambda item: -int(item["steps"]))


class StartRequest(BaseModel):
    p1: str
    p2: str
    seed: int = 0


class Session:
    """Owns one FightEnv, two frozen policies and the latest JPEG frame."""

    def __init__(self, game: str, path_a: str, path_b: str, seed: int, session_id: int = 0):
        self.sid = session_id
        self.game = game
        self.paths = {"a": path_a, "b": path_b}
        self.labels = {"a": Path(path_a).stem, "b": Path(path_b).stem}
        self.models = {key: _load_model(path) for key, path in self.paths.items()}
        self.env = _get_env(game)
        self.env.on_frame = self._on_frame
        self.seed = seed
        self.series = {"a": 0, "b": 0}
        self.swapped = False
        self.rounds = {"a": 0, "b": 0}
        self.round_no = 1
        rng_seed = time.time_ns() & 0xFFFFFFFF
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        torch.manual_seed(rng_seed)
        self.frame_count = 0
        self.jpeg: bytes = b""
        self.jpeg_lock = threading.Lock()
        self.status = {
            "phase": "starting", "winner": None, "reason": None, "hp1": 1.0, "hp2": 1.0,
            "p1": "", "p2": "", "frames": 0,
            "series": {"a": 0, "b": 0}, "round": 1, "best_of": 3,
            "labels": self.labels,
        }
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self._next_frame = time.perf_counter()

    def _on_frame(self, frame: np.ndarray) -> None:
        """Pace every emulated frame to 60 fps so motion is smooth.

        env.step() runs a whole macro as one call; pacing here (inside the
        frame callback) keeps frames evenly spaced instead of bursting them
        and then sleeping.
        """
        now = time.perf_counter()
        delay = self._next_frame - now
        if delay > 0:
            time.sleep(delay)
        else:
            self._next_frame = now
        self._next_frame += 1.0 / 60.0
        self.frame_count += 1
        image = Image.fromarray(frame).resize((480, 336), Image.NEAREST)
        buffer = __import__("io").BytesIO()
        image.save(buffer, format="JPEG", quality=72)
        with self.jpeg_lock:
            self.jpeg = buffer.getvalue()

    def _start_match(self) -> dict:
        p1_key, p2_key = ("b", "a") if self.swapped else ("a", "b")
        self.env.opponent = make_opponent(
            f"checkpoint:{self.paths[p2_key]}", config=self.env.config, env=self.env
        )
        self.status.update({"phase": "fight", "winner": None, "reason": None, "p1": p1_key,
                            "p2": p2_key, "round": self.round_no,
                            "series": dict(self.rounds)})
        self._next_frame = time.perf_counter() + 0.05
        return self.env.reset(seed=self.seed + self.series["a"] + self.series["b"])

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            obs, _ = self._start_match()
            while not self.stop.is_set():
                model = self.models[self.status["p1"]]
                action, _ = model.predict(obs, deterministic=False)
                obs, _, terminated, truncated, info = self.env.step(action)
                self.status.update({
                    "hp1": float(info.get("p1_frac", 0.0)),
                    "hp2": float(info.get("p2_frac", 0.0)),
                    "frames": int(info.get("frames", 0)),
                })
                if terminated or truncated:
                    winner = int(info["winner"])
                    reason = (
                        "time_limit" if truncated
                        else "ko" if info.get("round_over") else "hud_reset"
                    )
                    print(
                        f"[arena] end session={self.sid} reason={reason} winner={winner} "
                        f"frames={info.get('frames')} hp1={info.get('p1_frac'):.3f} "
                        f"hp2={info.get('p2_frac'):.3f}",
                        flush=True,
                    )
                    if winner:
                        key = self.status["p1"] if winner == 1 else self.status["p2"]
                        self.rounds[key] += 1
                        self.status["winner"] = key
                    self.status["series"] = dict(self.rounds)
                    if self.rounds["a"] < 2 and self.rounds["b"] < 2:
                        self.status["phase"] = "intermission"
                        self.status["reason"] = reason
                        if self.stop.wait(2.5):
                            break
                        self.round_no += 1
                    else:
                        self.status["phase"] = "match_over"
                        self.status["reason"] = reason
                        if self.stop.wait(4.0):
                            break
                        self.rounds = {"a": 0, "b": 0}
                        self.round_no = 1
                        self.swapped = not self.swapped
                    obs, _ = self._start_match()
                    continue
        finally:
            self.stop.set()
            if self.env.on_frame == self._on_frame:
                self.env.on_frame = None

    def stop_run(self) -> None:
        self.stop.set()


def _encode_frame(session: Session | None) -> bytes | None:
    if session is None:
        return None
    with session.jpeg_lock:
        return session.jpeg or None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE.read_text()


@app.get("/api/models")
def models() -> dict:
    return {"models": _list_models()}


@app.post("/api/start")
def start(request: StartRequest) -> dict:
    expected = _expected_spaces()
    for path in (request.p1, request.p2):
        if not path.startswith(str(RUNS)) or not Path(path).exists():
            raise HTTPException(status_code=400, detail=f"unknown model: {path}")
        if _stored_spaces(Path(path)) != expected:
            raise HTTPException(
                status_code=400,
                detail=f"model was trained on a different action/observation space: {path}",
            )
    with LOCK:
        old = STATE["session"]
        if isinstance(old, Session):
            _shutdown(old)
        STATE["id"] = int(STATE["id"]) + 1
        session = Session(GAME, request.p1, request.p2, request.seed, int(STATE["id"]))
        STATE["session"] = session
        session.start()
        return {"session": STATE["id"], "labels": session.labels}


@app.post("/api/stop")
def stop() -> dict:
    with LOCK:
        session = STATE["session"]
        if isinstance(session, Session):
            _shutdown(session)
        STATE["session"] = None
    return {"ok": True}


@app.get("/api/status")
def status() -> dict:
    session = STATE["session"]
    if not isinstance(session, Session):
        return {"active": False}
    data = dict(session.status)
    data.update({"active": True, "session": STATE["id"], "labels": session.labels})
    return data


@app.get("/stream")
def stream(session: int = 0) -> StreamingResponse:
    def frames():
        while True:
            current = STATE["session"]
            if not isinstance(current, Session) or STATE["id"] != session:
                break
            jpeg = _encode_frame(current)
            if jpeg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(1 / 60)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Realtime model arena")
    parser.add_argument("--game", default="samsho2_neogeo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
