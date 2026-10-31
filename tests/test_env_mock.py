"""Tests for the mock backend and Gymnasium environment (no ROM required)."""

import numpy as np
import pytest

from fightrl import make_env
from fightrl.core.config import load_game_config
from fightrl.core.controller import ControllerState, resolve_direction


@pytest.fixture
def env():
    env = make_env("mock", opponent="aggressive", frame_stack=4)
    yield env
    env.close()


def test_spaces(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs["pixels"].shape == (4, 84, 84)
    assert obs["state"].shape == (12,)
    assert info["p1_hp"] == 176
    assert env.action_space.n == len(env.macro_names)


def test_step_contract(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "p1_hp" in info and "winner" in info


def test_full_action_mode():
    cfg = load_game_config("mock")
    env = make_env(cfg, opponent="none", action_mode="full")
    env.reset(seed=0)
    assert env.action_space.nvec.tolist() == [9, 2, 2, 2, 2, 2, 2]
    ctrls = env.action_to_controllers([4, 1, 0, 0, 1, 0, 0])
    assert len(ctrls) == env.frameskip
    assert ctrls[0].direction == "right"
    assert ctrls[0].buttons == frozenset({"LP", "LK"})
    env.close()


def test_hybrid_action_mode_mixes_primitives_and_skills():
    cfg = load_game_config("mock")
    env = make_env(cfg, opponent="none", action_mode="hybrid")
    env.reset(seed=0)
    n_primitives = 9 * len(cfg.primitive_button_sets())
    assert cfg.primitive_button_sets()[0] == ()  # "no button" comes first
    assert env.action_space.n == n_primitives + len(cfg.macros)
    assert len(env.action_names) == env.action_space.n

    # Low indices are raw input, held for the frameskip window.
    sets = len(cfg.primitive_button_sets())
    held_hp = env.action_to_controllers(8 * sets + 3)  # down_right + HP
    assert len(held_hp) == env.frameskip
    assert held_hp[0].direction == "down_right"
    assert held_hp[0].buttons == frozenset({"HP"})
    # Attacks edge-trigger, so the window ends on a release: repeating the same
    # action presses again instead of holding the button dead.
    assert held_hp[-1].buttons == frozenset()

    # Higher indices are skills, which keep their scripted length and facing.
    walk = env.action_to_controllers(n_primitives + env.macro_names.index("walk_fwd"))
    assert len(walk) > env.frameskip
    assert all(c.direction == "right" for c in walk)
    env.close()


def test_hybrid_primitive_reaches_the_backend():
    cfg = load_game_config("mock")
    env = make_env(cfg, opponent="none", action_mode="hybrid")
    env.reset(seed=0)
    # Close the distance by hand, then hold a raw button instead of a skill.
    env.backend.p1.x, env.backend.p2.x = 180.0, 205.0
    env.state = env.reader.read()
    start_hp = env.state.p2_hp
    hp_index = env.action_names.index("neutral+HP")
    dealt = 0.0
    for _ in range(10):
        _, _, term, trunc, info = env.step(hp_index)
        dealt += info["dmg_dealt"]
        if term or trunc:
            break
    assert env.state.p2_hp < start_hp
    assert dealt > 0
    env.close()


def test_hybrid_primitive_buttons_are_configurable():
    cfg = load_game_config("mock")
    cfg.primitive_buttons = [(), ("LP",), ("LP", "HP")]
    env = make_env(cfg, opponent="none", action_mode="hybrid")
    env.reset(seed=0)
    assert env.action_space.n == 9 * 3 + len(cfg.macros)
    both = env.action_to_controllers(env.action_names.index("neutral+LP+HP"))
    assert both[0].buttons == frozenset({"LP", "HP"})
    env.close()


def test_macro_resolution_respects_facing():
    cfg = load_game_config("mock")
    env = make_env(cfg, opponent="none")
    env.reset(seed=0)
    env.state.p1_x, env.state.p2_x = 200.0, 120.0  # opponent is to the left
    ctrls = env.action_to_controllers(env.macro_names.index("walk_fwd"))
    assert ctrls[0].direction == "left"
    assert resolve_direction("fwd", 1) == "right"
    assert resolve_direction("back", 1) == "left"
    assert resolve_direction("fwd_up", -1) == "up_left"
    env.close()


def test_episode_terminates_and_rewards_are_finite(env):
    total = 0.0
    for episode in range(2):
        env.reset(seed=episode)
        steps = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total += reward
            steps += 1
            assert np.isfinite(reward)
            if terminated or truncated:
                break
            assert steps < 2000
        assert info["round_over"] or truncated or env._timer_expired()
    assert np.isfinite(total)


def test_determinism_without_opponent():
    cfg = load_game_config("mock")
    def rollout():
        env = make_env(cfg, opponent="none")
        env.reset(seed=42)
        rng = np.random.default_rng(0)
        rewards = []
        for _ in range(40):
            _, r, term, trunc, info = env.step(int(rng.integers(env.action_space.n)))
            rewards.append(r)
            if term or trunc:
                break
        env.close()
        return rewards, info["p1_hp"], info["p2_hp"]

    assert rollout() == rollout()


def test_blocking_and_damage_shape_reward():
    cfg = load_game_config("mock")
    env = make_env(cfg, opponent="none")
    env.reset(seed=0)
    # P2 never acts; the agent mashes HP at close range and must deal damage.
    env.backend.p1.x, env.backend.p2.x = 180.0, 205.0
    env.state = env.reader.read()
    start_hp = env.state.p2_hp
    dealt = 0.0
    for _ in range(10):
        _, _, term, trunc, info = env.step(env.macro_names.index("attack_HP"))
        dealt += info["dmg_dealt"]
        if term or trunc:
            break
    assert env.state.p2_hp < start_hp
    assert dealt > 0
    env.close()
