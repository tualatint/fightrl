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
    assert obs["history"].shape == (12 + 4 * len(env.macro_names),)
    assert info["p1_hp"] == 176
    assert env.action_space.n == len(env.macro_names)


def test_history_records_decisions():
    env = make_env("mock", opponent="none")
    env.reset(seed=0)
    n = env.action_space.n
    actions = [3, 1, 4, 0, 2]
    before = env.state.features(12)
    obs = None
    for action in actions:
        obs, *_ = env.step(action)
        assert obs["history"][12 + action] == 1.0
        assert obs["history"][:12].tolist() == before.tolist()
        before = env.state.features(12)
    # Only the last four decisions are kept, newest first.
    for offset, action in enumerate(reversed(actions[-4:])):
        assert obs["history"][12 + offset * n + action] == 1.0
    for slot in range(4):
        block = obs["history"][12 + slot * n: 12 + (slot + 1) * n]
        assert block.sum() == 1.0
    env.close()


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
    assert env.action_space.n == n_primitives + len(cfg.macros) + 1
    assert len(env.action_names) == env.action_space.n
    assert env.action_names[-1] == "empty"
    # The trailing empty action releases everything for the whole window.
    empty = env.action_to_controllers(env.action_space.n - 1)
    assert len(empty) == env.frameskip
    assert all(c.direction == "neutral" and not c.buttons for c in empty)

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
    assert env.action_space.n == 9 * 3 + len(cfg.macros) + 1
    both = env.action_to_controllers(env.action_names.index("neutral+LP+HP"))
    assert both[0].buttons == frozenset({"LP", "HP"})
    env.close()


def test_explicit_primitive_actions_replace_the_grid():
    cfg = load_game_config("mock")
    cfg.primitive_actions = [
        {"name": "idle", "dir": "neutral", "buttons": (), "frames": None},
        {"name": "slash_heavy", "dir": "neutral", "buttons": ("LP", "HP"), "frames": None},
        {"name": "walk_fwd", "dir": "right", "buttons": (), "frames": 5},
    ]
    env = make_env(cfg, opponent="none", action_mode="hybrid")
    env.reset(seed=0)
    assert env.action_space.n == 3 + len(cfg.macros)
    assert env.action_names[:3] == ["idle", "slash_heavy", "walk_fwd"]
    # An explicit list is exhaustive: the grid's empty slot is no longer added.
    assert "empty" not in env.action_names
    assert env.action_to_controllers(1)[0].buttons == frozenset({"LP", "HP"})
    assert len(env.action_to_controllers(2)) == 5
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


def test_weapon_break_bonus_rewards_damage_dealt():
    results = {}
    attack = None
    for bonus in (0.0, 2.0):
        cfg = load_game_config("mock")
        cfg.reward = {**cfg.reward, "weapon_break_deal": bonus}
        cfg.weapon_break = ["attack_HP"]
        env = make_env(cfg, opponent="none")
        env.reset(seed=0)
        env.backend.p1.x, env.backend.p2.x = 180.0, 205.0
        env.state = env.reader.read()
        attack = env.action_names.index("attack_HP")
        _, reward, _, _, info = env.step(attack)
        results[bonus] = (reward, info["dmg_dealt"], env._weapon_break_indices)
        env.close()

    plain, damage, indices = results[0.0]
    boosted, other_damage, _ = results[2.0]
    assert damage == other_damage > 0  # the mock is deterministic
    assert boosted - plain == pytest.approx(2.0 * damage)
    assert indices == {attack}  # only the configured action carries the bonus


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
