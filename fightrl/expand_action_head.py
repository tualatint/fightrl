"""Adapt a saved PPO checkpoint to a different discrete action space.

Two modes:

``--copy`` (default)
    Grow the action head and copy the trained rows, filling the new rows with
    zero weights plus a strongly negative bias so they start out essentially
    never chosen.  Use this when actions were only *appended*.

``--fresh-head``
    Rebuild the policy for a reordered/curated action space: the CNN features,
    both 256-wide trunks and the value head are carried over (they do not
    depend on which action is taken), while the action head is reinitialised.
    Use this when action indices changed meaning.

    python -m fightrl.expand_action_head --src runs/.../ppo_x.zip \
        --dst runs/.../ppo_x_new.zip --n-actions 47 --fresh-head
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from torch import nn

from .core.config import load_game_config
from .env import action_names_for

DEFAULT_ACTIONS = 63
# -10 keeps the new logit far below anything a tanh-bounded trunk produces,
# so the extra action starts with a share of ~e^-10 of the probability mass.
DEFAULT_BIAS = -10.0


class _StubEnv(gym.Env):
    """Spaces-only env so a fresh policy can be built without the emulator."""

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = {
            key: np.zeros(space.shape, space.dtype)
            for key, space in self.observation_space.spaces.items()
        }
        return obs, {}

    def step(self, action):
        return self.reset()[0], 0.0, True, False, {}


def expand_action_head(model: PPO, n_actions: int, bias: float = DEFAULT_BIAS) -> PPO:
    """Return ``model`` with an ``action_net`` of ``n_actions`` outputs (rows copied)."""
    net: nn.Linear = model.policy.action_net
    if net.out_features == n_actions:
        return model
    if net.out_features > n_actions:
        raise ValueError(
            f"model has {net.out_features} actions; cannot copy rows into {n_actions}"
        )

    widened = nn.Linear(net.in_features, n_actions).to(net.weight.device)
    with torch.no_grad():
        widened.weight[: net.out_features].copy_(net.weight)
        widened.bias[: net.out_features].copy_(net.bias)
        widened.weight[net.out_features:].zero_()
        widened.bias[net.out_features:].fill_(bias)

    model.policy.action_net = widened
    model.action_space = spaces.Discrete(n_actions)
    model.policy.action_space = model.action_space
    return model


def reinit_action_head(
    model: PPO,
    n_actions: int,
    net_arch: list[int],
    observation_space: spaces.Space | None = None,
) -> PPO:
    """Build a new policy with a fresh head, keeping whatever still fits.

    Tensors are kept only when their shapes still match, so the CNN and its
    state embedding survive an observation change while the trunks, which grow
    with the input width, are rebuilt.
    """
    stub = _StubEnv(observation_space or model.policy.observation_space,
                    spaces.Discrete(n_actions))
    fresh = PPO("MultiInputPolicy", stub, device="cpu",
                policy_kwargs=dict(net_arch=list(net_arch)))

    target = fresh.policy.state_dict()
    state = {
        key: value for key, value in model.policy.state_dict().items()
        if key in target and target[key].shape == value.shape and not key.startswith("action_net.")
    }
    fresh.policy.load_state_dict(state, strict=False)
    kept = set(state)
    conv = {key for key in target if key.startswith(("features_extractor.cnn.", "features_extractor.linear."))}
    lost = conv - kept
    if lost:
        raise RuntimeError(f"warm start dropped CNN tensors: {sorted(lost)}")
    fresh.num_timesteps = model.num_timesteps
    print(f"kept {len(kept)} tensors, rebuilt {len(target) - len(kept)}")
    return fresh


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="checkpoint to adapt")
    ap.add_argument("--dst", type=Path, required=True, help="where to write the adapted checkpoint")
    ap.add_argument("--n-actions", type=int, default=DEFAULT_ACTIONS)
    ap.add_argument("--bias", type=float, default=DEFAULT_BIAS,
                    help="copy mode: bias for the new rows; more negative means rarer")
    ap.add_argument("--fresh-head", action="store_true",
                    help="reinitialise the action head instead of copying rows")
    ap.add_argument("--history-actions", type=int, default=0,
                    help="add a decision-history key of this depth to the observation")
    ap.add_argument("--bias-action", default=None,
                    help="action name whose logit bias is nudged (needs --game)")
    ap.add_argument("--bias-delta", type=float, default=0.0,
                    help="amount added to that action's bias; positive favours it")
    ap.add_argument("--game", default="samsho2_neogeo",
                    help="game config used to resolve --bias-action")
    ap.add_argument("--net-arch", default="256,256")
    args = ap.parse_args(argv)

    arch = [int(n) for n in args.net_arch.split(",") if n.strip()]
    model = PPO.load(args.src, device="cpu")
    before = model.policy.action_net.out_features
    print(f"{args.src}: {before} actions, {model.num_timesteps} steps")

    observation_space = None
    if args.history_actions:
        dim = 12 + args.history_actions * args.n_actions
        observation_space = spaces.Dict({
            **model.policy.observation_space.spaces,
            "history": spaces.Box(-2.0, 2.0, (dim,), dtype=np.float32),
        })
        print(f"observation: {model.policy.observation_space} -> {observation_space}")

    if args.fresh_head:
        model = reinit_action_head(model, args.n_actions, arch, observation_space)
        print(f"fresh action head: {args.n_actions} rows")
    else:
        if before >= args.n_actions:
            if not args.bias_action:
                print("nothing to do: head is already that wide")
                return 0
            print("head is already that wide; only applying the bias nudge")
        else:
            expand_action_head(model, args.n_actions, args.bias)

    if args.bias_action:
        names = action_names_for(load_game_config(args.game))
        if len(names) != before:
            raise SystemExit(f"head has {before} actions but {args.game} defines {len(names)}")
        if args.bias_action not in names:
            raise SystemExit(f"{args.bias_action!r} is not an action of {args.game}")
        index = names.index(args.bias_action)
        with torch.no_grad():
            model.policy.action_net.bias[index] += args.bias_delta
        print(f"bias[{index}] ({args.bias_action}) += {args.bias_delta:+.3f} "
              f"-> {model.policy.action_net.bias[index].item():+.3f}")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.dst))
    print(f"wrote {args.dst} ({args.n_actions} actions, {model.num_timesteps} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
