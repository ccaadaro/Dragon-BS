import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from dragon_bs.envs import BandSelectionEnv, band_statistics


@pytest.fixture
def statistics():
    return (np.array([0.8, 0.6, 0.3]),
            np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.9], [0.2, 0.9, 1.0]]))


@pytest.mark.parametrize("algo", ["a2c", "dqn", "ppo"])
def test_gymnasium_contract(algo, statistics):
    env = BandSelectionEnv(algo=algo, n_bands=2, statistics=statistics)
    check_env(env, skip_render_check=True)


def test_dqn_mean_redundancy_and_duplicate(statistics):
    env = BandSelectionEnv(algo="dqn", n_bands=3, statistics=statistics)
    env.reset(seed=7)
    assert env.step(0)[1] == pytest.approx(0.8)
    assert env.step(0)[1] == -1
    assert env.selected_bands == [0]
    assert env.step(1)[1] == pytest.approx(0.6 - 0.2 * 0.4)
    obs, reward, terminated, truncated, info = env.step(2)
    assert reward == pytest.approx(0.3 - 0.2 * (0.2 + 0.9) / 2)
    assert terminated and not truncated
    assert info["selected_bands"] == [0, 1, 2]
    np.testing.assert_array_equal(obs[:3], np.ones(3))


def test_a2c_max_redundancy_and_terminal_bonus(statistics):
    env = BandSelectionEnv(algo="a2c", n_bands=3, statistics=statistics)
    env.reset()
    env.step(0)
    obs, reward, *_ = env.step(1)
    assert reward == pytest.approx(0.6 - 0.5 * 0.4)
    assert obs[-1] == pytest.approx(1 / 3)
    _, reward, terminated, *_ = env.step(2)
    assert reward == pytest.approx(0.3 - 0.5 * 0.9 + (0.8 + 0.6 + 0.3) / 3)
    assert terminated


def test_ppo_masks_preferences_and_retains_terminal_transition(statistics):
    env = BandSelectionEnv(algo="ppo", n_bands=2, statistics=statistics)
    env.reset()
    action = np.array([1, 0.5, 0], dtype=np.float32)
    assert env.step(action)[1] == pytest.approx(0.8)
    _, reward, terminated, *_ = env.step(action)
    assert reward == pytest.approx(0.6) and not terminated
    np.testing.assert_array_equal(action, [1, 0.5, 0])
    assert env.step(action)[1:4] == (1.0, True, False)
    with pytest.raises(RuntimeError):
        env.step(action)


def test_constant_spectra_remain_finite():
    ent, corr = band_statistics(np.ones((20, 3)))
    assert np.isfinite(ent).all() and np.isfinite(corr).all()
    np.testing.assert_array_equal(ent, np.zeros(3))


@pytest.mark.parametrize("n_bands", [0, 4])
def test_invalid_budget_rejected(n_bands, statistics):
    with pytest.raises(ValueError):
        BandSelectionEnv(n_bands=n_bands, statistics=statistics)
