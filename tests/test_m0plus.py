"""End-to-end tests for the M0+ reservoir comparison (spike)."""

from __future__ import annotations

import numpy as np
import pytest

from neuroflynav.experiments.run_m0plus import main, run_m0plus
from neuroflynav.training.imitation import fit_ridge


def test_run_m0plus_is_reproducible_and_bounded() -> None:
    first = run_m0plus(train_episodes=20, eval_episodes=10, base_seed=0)
    second = run_m0plus(train_episodes=20, eval_episodes=10, base_seed=0)
    assert first == second
    names = [metric.policy for metric in first.policies]
    assert names == ["FlyConnectome", "ShuffledFly", "RandomRecurrent", "Reactive"]
    for metric in first.policies:
        assert 0.0 <= metric.success_rate <= 1.0
        assert 0.0 <= metric.robust_success_rate <= 1.0
        assert metric.mean_steps > 0.0


def test_run_m0plus_rejects_non_positive_counts() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_m0plus(train_episodes=0)


def test_fit_ridge_returns_readout_shape() -> None:
    features = np.ones((5, 3), dtype=np.float64)
    targets = np.zeros((5, 4), dtype=np.float64)
    readout = fit_ridge(features, targets, ridge_lambda=1e-2)
    assert readout.shape == (4, 3)


def test_cli_runs_and_returns_success() -> None:
    assert main(["--train", "10", "--eval", "5", "--seed", "1"]) == 0
