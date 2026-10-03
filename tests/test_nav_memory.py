"""Tests for the hidden-cue navigation environment (M0+ spike)."""

from __future__ import annotations

import pytest

from neuroflynav.env.nav import Action, NavError
from neuroflynav.env.nav_memory import (
    FEATURE_DIM,
    MemoryGridNavConfig,
    MemoryGridNavEnv,
    encode_observation,
)


def test_reset_is_deterministic_for_a_seed() -> None:
    env = MemoryGridNavEnv(MemoryGridNavConfig(size=7, max_steps=30, cue_steps=3))
    first, _ = env.reset(seed=1)
    second, _ = env.reset(seed=1)
    assert first == second


def test_cue_is_visible_then_hidden() -> None:
    env = MemoryGridNavEnv(MemoryGridNavConfig(size=7, max_steps=30, cue_steps=2))
    observation, _ = env.reset(seed=0)
    assert observation.cue_visible
    observation, _, _, _, _ = env.step(Action.UP)
    assert observation.cue_visible
    observation, _, _, _, _ = env.step(Action.UP)
    assert not observation.cue_visible
    assert observation.cue_delta == (0, 0)


def test_step_requires_reset() -> None:
    with pytest.raises(NavError):
        MemoryGridNavEnv().step(Action.UP)


def test_optimal_action_reaches_goal() -> None:
    env = MemoryGridNavEnv(MemoryGridNavConfig(size=6, max_steps=50, cue_steps=3))
    env.reset(seed=3)
    terminated = False
    for _ in range(50):
        _, _, terminated, truncated, _ = env.step(env.optimal_action())
        if terminated or truncated:
            break
    assert terminated


def test_encode_observation_length_and_hidden_cue() -> None:
    env = MemoryGridNavEnv(MemoryGridNavConfig(size=5, max_steps=10, cue_steps=0))
    observation, _ = env.reset(seed=0)
    features = encode_observation(observation, 5)
    assert features.shape == (FEATURE_DIM,)
    assert not observation.cue_visible
    assert features[3] == 0.0
    assert features[4] == 0.0
    assert features[5] == 0.0
