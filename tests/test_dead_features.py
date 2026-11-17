"""Dead-ball signals on crafted arrays, positions in metres."""

import numpy as np

from mtah.dead.features import (
    ball_outside,
    ball_still,
    median_speed,
    speeds,
)

FPS = 25.0
PITCH = (105.0, 68.0)
NAN = np.nan


def test_speeds_and_median_with_missing_players():
    # Three players moving at 1, 2 and 3 m/s along x; the fastest vanishes on rows 5-9.
    t = np.arange(20)[:, None] / FPS
    players = np.stack(
        [np.column_stack([v * t[:, 0], np.zeros(20)]) for v in (1, 2, 3)], 1
    )
    players[5:10, 2] = NAN
    s = speeds(players, FPS)
    assert np.isnan(s[0]).all() and np.isnan(s[5:11, 2]).all()
    np.testing.assert_allclose(s[1], [1, 2, 3])
    m = median_speed(players, FPS, window_s=0.04)  # one frame: no averaging
    assert np.isnan(m[0])
    np.testing.assert_allclose(m[1], 2.0)
    np.testing.assert_allclose(m[6], 1.5)  # median of 1 and 2


def test_median_speed_window_ignores_nan():
    players = np.zeros((10, 1, 2))
    players[:, 0, 0] = np.arange(10) * 0.08  # 2 m/s
    m = median_speed(players, FPS, window_s=0.2)  # 5 frames
    # Row 0 has no speed of its own, but its window has some.
    np.testing.assert_allclose(m[:3], 2.0)


def test_ball_outside_margin():
    ball = np.array(
        [[-0.5, 30], [-1.5, 30], [106.5, 30], [50, 69.2], [50, -2], [NAN, NAN]]
    )
    assert ball_outside(ball, PITCH, margin=1.0).tolist() == [
        False,
        True,
        True,
        True,
        True,
        False,
    ]


def test_ball_still_against_rolling():
    still = np.tile([[50.0, 30.0]], (40, 1))
    rolling = np.column_stack([50 + np.arange(40) * 0.04, np.full(40, 30.0)])  # 1 m/s
    # A 10-frame window around row 5 still reaches row 0, which has no speed.
    assert ball_still(still, FPS, window_s=0.4)[6:35].all()
    assert not ball_still(rolling, FPS, window_s=0.4).any()
    gap = still.copy()
    gap[20] = NAN  # a missing frame breaks stillness around it
    assert not ball_still(gap, FPS, window_s=0.4)[18:23].any()
