"""End-to-end tests for the M0 policy comparison on synthetic topology."""

from __future__ import annotations

from neuroflynav.experiments.run_spike import main, run_spike


def test_run_spike_is_reproducible_and_bounded() -> None:
    first = run_spike(episodes=15, grid_size=5, max_steps=60, base_seed=123)
    second = run_spike(episodes=15, grid_size=5, max_steps=60, base_seed=123)
    assert first == second
    assert first.episodes == 15
    assert first.base_seed == 123
    assert first.episode_seeds == tuple(range(123, 138))
    assert first.graph_nodes > 0
    assert first.graph_edges > 0
    assert [summary.policy for summary in first.policies] == [
        "RandomAgent",
        "FlyConnectome",
        "ShuffledFly",
    ]
    assert {summary.graph for summary in first.policies} == {
        "synthetic",
        "shuffled(seed=123)",
    }
    for summary in first.policies:
        assert 0 <= summary.successes <= first.episodes
        assert 0.0 <= summary.success_rate <= 1.0
        assert summary.mean_steps > 0.0
        assert isinstance(summary.mean_reward, float)


def test_run_spike_rejects_non_positive_episode_count() -> None:
    import pytest

    with pytest.raises(ValueError, match="at least 1"):
        run_spike(episodes=0)


def test_cli_runs_and_returns_success() -> None:
    assert main(["--episodes", "3", "--grid-size", "4", "--seed", "1"]) == 0


def test_run_spike_rejects_invalid_environment_configuration() -> None:
    import pytest

    with pytest.raises(ValueError, match="grid_size"):
        run_spike(grid_size=1)
    with pytest.raises(ValueError, match="max_steps"):
        run_spike(max_steps=0)
