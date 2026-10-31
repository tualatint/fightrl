"""Tests for the two-agent duel (P1/P2) training, on the mock game."""

import numpy as np
import pytest

from fightrl.duel import P2PerspectiveEnv, head_to_head, train_side


@pytest.fixture
def p2_env():
    env = P2PerspectiveEnv("mock", opponent="none")
    yield env
    env.close()


def test_p2_observation_and_info_are_mirrored(p2_env):
    obs, info = p2_env.reset(seed=0)
    state = p2_env.state
    features = obs["state"]
    assert features[0] == pytest.approx(state.p2_frac)
    assert features[1] == pytest.approx(state.p1_frac)
    assert features[2] == pytest.approx(-(state.p1_frac - state.p2_frac))
    assert features[7] == pytest.approx(-state.p1_facing)
    assert info["p1_hp"] == state.p2_hp
    assert info["p2_hp"] == state.p1_hp
    assert obs["pixels"].shape == (4, 84, 84)


def test_p2_controls_p2_and_earns_damage(p2_env):
    p2_env.reset(seed=0)
    inner = p2_env.backend.inner
    inner.p1.x, inner.p2.x = 180.0, 205.0
    inner._update_facing()
    p2_env.state = p2_env.reader.read()
    start_hp = p2_env.state.p1_hp
    dealt = 0.0
    for _ in range(12):
        _, _, terminated, truncated, info = p2_env.step(
            p2_env.macro_names.index("attack_HP")
        )
        dealt += info["dmg_dealt"]
        if terminated or truncated:
            break
    # The learner is P2, so a successful attack must damage canonical P1.
    assert p2_env.state.p1_hp < start_hp
    assert dealt > 0


def test_p2_winner_is_learner_centric(p2_env):
    p2_env.reset(seed=0)
    inner = p2_env.backend.inner
    inner.p1.hp = 0.0
    inner._write_ram()
    _, reward, terminated, _, info = p2_env.step(p2_env.macro_names.index("wait"))
    assert terminated
    assert info["winner"] == 1
    assert reward > 0  # KO damage + round win, not a loss


def test_p2_rollouts_are_deterministic():
    def rollout():
        env = P2PerspectiveEnv("mock", opponent="none")
        env.reset(seed=42)
        rng = np.random.default_rng(0)
        rewards = []
        for _ in range(20):
            _, reward, terminated, truncated, _ = env.step(int(rng.integers(env.action_space.n)))
            rewards.append(reward)
            if terminated or truncated:
                break
        env.close()
        return rewards

    assert rollout() == rollout()


def test_duel_training_and_head_to_head(tmp_path):
    p1 = train_side(
        game="mock", side=1, opponent="random", steps=64, out_dir=tmp_path / "p1",
        n_envs=1, n_steps=32, batch_size=32, device="cpu", dummy=True,
    )
    p2 = train_side(
        game="mock", side=2, opponent=f"checkpoint:{p1}", steps=64, out_dir=tmp_path / "p2",
        n_envs=1, n_steps=32, batch_size=32, device="cpu", dummy=True,
    )
    assert p1.exists() and p2.exists()

    result = head_to_head(
        game="mock", p1_model=p1, p2_model=p2, episodes=1, seed=0, device="cpu",
    )
    assert result["summary"]["episodes"] == 1
    assert result["summary"]["p1_wins"] + result["summary"]["p2_wins"] + result["summary"]["draws"] == 1
