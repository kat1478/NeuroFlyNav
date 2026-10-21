import numpy as np

from neuroflynav.tasks.heading import (
    INPUT_DIM,
    HeadingTaskConfig,
    generate_sequence,
    heading_sector,
    hold_last_landmark_accuracy,
    scored_steps,
)


def test_sequence_shapes_and_cue_masking() -> None:
    task = HeadingTaskConfig(steps=20, cue_steps=3, landmark_every=8)
    inputs, sectors = generate_sequence(task, seed=1)
    assert inputs.shape == (20, INPUT_DIM)
    assert sectors.shape == (20,)
    shown = inputs[:, 2] == 1.0
    assert shown.tolist() == [s < 3 or s % 8 == 0 for s in range(20)]
    assert np.all(inputs[~shown, 3:] == 0.0)
    assert not np.any(scored_steps(task) & shown)


def test_heading_sector_wraps_angles() -> None:
    theta = np.array([0.0, np.pi / 2, np.pi, -0.1, 2 * np.pi + 0.1])
    assert heading_sector(theta, 4).tolist() == [0, 1, 2, 3, 0]


def test_hold_last_landmark_is_perfect_without_turning() -> None:
    task = HeadingTaskConfig(turn_std=1e-9, landmark_every=8)
    assert hold_last_landmark_accuracy(task, range(5)) == 1.0
