"""Tests for the discrete-grid navigation environment (M0 spike)."""

from __future__ import annotations

import pytest

from neuroflynav.env.nav import (
    Action,
    GridNavConfig,
    GridNavEnv,
    GridObservation,
    NavError,
)


def _greedy_action(observation: GridObservation) -> Action:
    delta_row, delta_column = observation.goal_delta
    if delta_row < 0:
        return Action.UP
    if delta_row > 0:
        return Action.DOWN
    if delta_column < 0:
        return Action.LEFT
    return Action.RIGHT


def test_reset_is_deterministic_and_places_distinct_goal() -> None:
    env = GridNavEnv(GridNavConfig(size=5, max_steps=10))
    first, _ = env.reset(seed=42)
    second, _ = env.reset(seed=42)
    assert first == second
    assert first.agent != first.goal


def test_step_requires_reset() -> None:
    env = GridNavEnv()
    with pytest.raises(NavError):
        env.step(Action.UP)


def test_agent_stays_within_bounds() -> None:
    env = GridNavEnv(GridNavConfig(size=3, max_steps=20))
    env.reset(seed=0)
    for _ in range(20):
        observation, _, terminated, truncated, _ = env.step(Action.UP)
        assert 0 <= observation.agent[0] < 3
        assert 0 <= observation.agent[1] < 3
        if terminated or truncated:
            break


def test_greedy_policy_reaches_goal_with_terminal_reward() -> None:
    env = GridNavEnv(GridNavConfig(size=4, max_steps=50))
    observation, _ = env.reset(seed=0)
    terminated = False
    for _ in range(50):
        observation, reward, terminated, truncated, _ = env.step(
            _greedy_action(observation)
        )
        if terminated:
            assert reward == 1.0
            break
        if truncated:
            break
    assert terminated


def test_truncation_happens_without_reaching_goal() -> None:
    env = GridNavEnv(GridNavConfig(size=5, max_steps=1))
    first, _ = env.reset(seed=1)
    # Step away from the goal so a single step cannot reach it.
    away = Action.UP if first.goal_delta[0] >= 0 else Action.DOWN
    _, _, terminated, truncated, _ = env.step(away)
    assert not terminated
    assert truncated
