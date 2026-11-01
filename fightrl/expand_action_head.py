"""Widen a saved PPO policy's discrete action head without changing behaviour.

Growing ``action_mode="hybrid"`` from ``Discrete(62)`` to ``Discrete(63)``
invalidates older checkpoints: their ``action_net`` was saved with 62 rows and
``PPO.load(..., env=env)`` rebuilds the layer for the new space, so the shapes
no longer match.  This copies the trained rows over and initialises the new
rows with zero weights plus a strongly negative bias, which makes the extra
actions start out essentially never selected.  ``V(s)``, the feature extractor
and the old action preferences are untouched, so the adapted policy plays
exactly like the original until it decides to use the new option.

    python -m fightrl.expand_action_head --src runs/.../ppo_x.zip --dst runs/.../ppo_x_63.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from torch import nn

DEFAULT_ACTIONS = 63
# -10 keeps the new logit far below anything a tanh-bounded trunk produces,
# so the extra action starts with a share of ~e^-10 of the probability mass.
DEFAULT_BIAS = -10.0


def expand_action_head(model: PPO, n_actions: int, bias: float = DEFAULT_BIAS) -> PPO:
    """Return ``model`` with an ``action_net`` of ``n_actions`` outputs."""
    net: nn.Linear = model.policy.action_net
    if net.out_features == n_actions:
        return model
    if net.out_features > n_actions:
        raise ValueError(
            f"model already has {net.out_features} actions; cannot shrink to {n_actions}"
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="checkpoint to adapt")
    ap.add_argument("--dst", type=Path, required=True, help="where to write the adapted checkpoint")
    ap.add_argument("--n-actions", type=int, default=DEFAULT_ACTIONS)
    ap.add_argument("--bias", type=float, default=DEFAULT_BIAS,
                    help="bias for the new rows; more negative means rarely chosen")
    args = ap.parse_args(argv)

    model = PPO.load(args.src, device="cpu")
    before = model.policy.action_net.out_features
    if before >= args.n_actions:
        print(f"{args.src} already has {before} actions; nothing to do")
        return 0

    expand_action_head(model, args.n_actions, args.bias)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.dst))
    print(f"{args.src} ({before} actions) -> {args.dst} ({args.n_actions} actions, "
          f"new rows bias={args.bias})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
