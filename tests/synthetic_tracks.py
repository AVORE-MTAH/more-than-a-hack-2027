"""Synthetic Tracking scenes for the cleaning tests: no match data, positions in metres."""

import numpy as np

from mtah.io.tracking import PITCH_SIZE, Tracking

# Params
FPS = 25.0
LENGTH, WIDTH = PITCH_SIZE
# Four home players around x = 35 and four away around x = 70 in period 1, mirrored in period 2,
# so home defends goal 0 (x = 0) in period 1 and goal 1 in period 2.
N_OUTFIELD = 8
OUTFIELD_TEAM = (0, 0, 0, 0, 1, 1, 1, 1)


def scene(extra_teams, seconds=20, periods=1):
    """(Tracking, t): outfield players moving smoothly, plus NaN slots labelled `extra_teams`.

    The extra slots follow the outfield ones, at index N_OUTFIELD onwards; `t` is the time in
    seconds within each period.
    """
    per_period = round(seconds * FPS)
    n_frames = per_period * periods
    period = np.repeat(np.arange(1, periods + 1), per_period).astype(np.int8)
    t = np.tile(np.arange(per_period) / FPS, periods)

    players = np.full((n_frames, N_OUTFIELD + len(extra_teams), 2), np.nan)
    for i, team in enumerate(OUTFIELD_TEAM):
        x = (35.0 if team == 0 else 70.0) + 6 * np.sin(0.5 * t + i)
        y = 10.0 + 15.0 * (i % 4) + 4 * np.cos(0.3 * t + i)
        players[:, i, 0] = np.where(period == 2, LENGTH - x, x)
        players[:, i, 1] = y

    tracking = Tracking(
        players=players,
        ball=np.full((n_frames, 2), np.nan),
        team=np.array(OUTFIELD_TEAM + tuple(extra_teams), dtype=np.int8),
        period=period,
        video_frame=np.arange(n_frames),
        fps=FPS,
        pitch_size=PITCH_SIZE,
    )
    return tracking, t


def keeper_path(t, goal, sway=0.15):
    """(len(t), 2): a keeper 5 m off goal `goal`'s line, swaying a little."""
    x = 5.0 + sway * np.sin(1.0 * t)
    y = WIDTH / 2 + sway * np.cos(0.7 * t)
    return np.column_stack([x if goal == 0 else LENGTH - x, y])
