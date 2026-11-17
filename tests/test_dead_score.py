import numpy as np
import pytest

import mtah.dead.score as dead_score
from mtah.dead.score import (
    bridge_dribbles,
    dribbling,
)
from mtah.io.tracking import PITCH_SIZE, Tracking


@pytest.fixture(autouse=True)
def mechanics(monkeypatch):
    """The scenes are timed for these values: the rules' behaviour is tested, not the values
    fitted on external data (mtah.tune.restarts), which are the defaults."""
    for name, value in (("MOVE_M", 1.0), ("LINE_M", 0.5), ("WHISTLE_MAX_S", 60.0)):
        monkeypatch.setattr(dead_score, name, value)


FPS = 25.0
FAST, SLOW = 3.0, 0.3  # m/s, either side of the 1.5 m/s boundary


def tracking_of(speed, ball_missing, ball_outside=None):
    """Three players all moving at `speed` (per row) along x; the ball mid-pitch or missing."""
    n = len(speed)
    x = 10 + np.cumsum(np.asarray(speed) / FPS)
    players = np.stack([np.column_stack([x, np.full(n, y)]) for y in (20, 34, 48)], 1)
    ball = np.tile([[52.0, 34.0]], (n, 1))
    if ball_outside is not None:
        ball[ball_outside] = [52.0, -3.0]  # 3 m beyond the touchline
    ball[ball_missing] = np.nan
    return Tracking(
        players=players,
        ball=ball,
        team=np.array([0, 0, 1], dtype=np.int8),
        period=np.ones(n, dtype=np.int8),
        video_frame=np.arange(n),
        fps=FPS,
        pitch_size=PITCH_SIZE,
    )


def scene(player_speed, ball, period=None):
    """Three players at their own per-row speeds (rows, 3) along x, and a given ball path."""
    player_speed = np.asarray(player_speed, dtype=float)
    n = len(player_speed)
    x = np.array([10.0, 30.0, 50.0]) + np.cumsum(player_speed / FPS, axis=0)
    y = np.broadcast_to([20.0, 34.0, 48.0], (n, 3))
    return Tracking(
        players=np.stack([x, y], axis=2),
        ball=np.asarray(ball, dtype=float),
        team=np.array([0, 0, 1], dtype=np.int8),
        period=np.ones(n, dtype=np.int8) if period is None else period,
        video_frame=np.arange(n),
        fps=FPS,
        pitch_size=PITCH_SIZE,
    )


def all_at(speed, n):
    return np.full((n, 3), speed)


def test_bridge_dribbles_fills_a_lull_inside_one_gap():
    missing = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0], dtype=bool)
    dribbled = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
    # Rows 2-3 lie between two dribbled rows of the first gap; 5-6 come after its last one, and
    # the second gap has only one dribbled row.
    np.testing.assert_array_equal(
        bridge_dribbles(dribbled, missing),
        np.array([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0], dtype=bool),
    )


def test_a_dribbler_slowing_for_a_moment_stays_dribbled():
    # Player 0 runs at 4 m/s with the ball unseen from row 100, stops for rows 150-189 and runs
    # on; the others stand still.
    n = 300
    speed = np.zeros((n, 3))
    speed[:, 0] = 4.0
    speed[150:190, 0] = 0.0
    ball = scene(speed, np.full((n, 2), np.nan)).players[:, 0] + [0.5, 0.0]
    ball[100:] = np.nan
    players = scene(speed, ball).players
    raw = dribbling(players, ball, FPS)
    assert not raw[160:180].any()  # the flag drops in the lull
    assert bridge_dribbles(raw, np.isnan(ball).any(axis=1))[100:250].all()


def ball_moving(speed_by_row):
    """(rows, 2) ball positions along x moving at the given speed (m/s) on each row."""
    x = 30.0 + np.cumsum(np.asarray(speed_by_row, dtype=float) / FPS)
    return np.column_stack([x, np.full(len(x), 34.0)])


def test_a_goal_kick_ends_at_the_kick_not_at_the_ball_thrown_back():
    from mtah.dead.score import out_stoppages

    # The ball runs over the goal line (x = 105.3) and lies 3 m behind it on rows 100-150 while
    # the players jog back (2.5 m/s); it is thrown back into the box on 150-160, rolls on, is
    # set down still on 175-250, and the goal kick goes at 15 m/s from row 250. The players
    # stand (1 m/s) from 200 and jog (1.8 m/s) after the kick.
    n = 500
    x = np.empty(n)
    x[:80] = 90.0 + 0.2 * np.arange(80)
    x[80:100] = 106.0 + 0.1 * np.arange(20)
    x[100:150] = 108.0
    x[150:160] = np.linspace(108.0, 100.0, 10)
    x[160:175] = np.linspace(100.0, 99.4, 15)
    x[175:250] = 99.4
    x[250:270] = 99.4 - 0.6 * np.arange(1, 21)
    x[270:] = x[269] - 0.1 * np.arange(1, n - 269)
    ball = np.column_stack([x, np.full(n, 34.0)])
    speed = np.select(
        [np.arange(n) < 80, np.arange(n) < 200, np.arange(n) < 255],
        [3.0, 2.5, 1.0],
        1.8,
    )
    period = np.ones(n, dtype=np.int8)

    stopped = out_stoppages(ball, speed, period, FPS, PITCH_SIZE)
    first = int(np.argmax(stopped))
    assert 80 < first < 90  # beyond the line by 1 m, three rows running
    assert stopped[first:250].all() and not stopped[250:].any()


def test_a_nudge_while_the_players_jog_back_is_not_a_goal_kick():
    from mtah.dead.score import out_stoppages

    # As the goal kick test, but the ball, set down still in the box on 175-220, is nudged a
    # metre into place on 220-230 and lies still again; the players jog back at 1.5 m/s all the
    # while, and the goal kick at 15 m/s from row 250 sets them going at 2.2 m/s.
    n = 500
    x = np.empty(n)
    x[:80] = 90.0 + 0.2 * np.arange(80)
    x[80:100] = 106.0 + 0.1 * np.arange(20)
    x[100:150] = 108.0
    x[150:160] = np.linspace(108.0, 100.0, 10)
    x[160:175] = np.linspace(100.0, 99.4, 15)
    x[175:220] = 99.4
    x[220:230] = 99.4 - 0.1 * np.arange(1, 11)
    x[230:250] = 98.4
    x[250:270] = 98.4 - 0.6 * np.arange(1, 21)
    x[270:] = x[269] - 0.1 * np.arange(1, n - 269)
    ball = np.column_stack([x, np.full(n, 34.0)])
    speed = np.select([np.arange(n) < 80, np.arange(n) < 252], [3.0, 1.5], 2.2)
    period = np.ones(n, dtype=np.int8)
    stopped = out_stoppages(ball, speed, period, FPS, PITCH_SIZE)
    first = int(np.argmax(stopped))
    assert stopped[first:250].all() and not stopped[250:].any()


def test_the_players_must_speed_up_not_just_move():
    from mtah.dead.score import restart_signals

    n = 300
    ball = ball_moving(np.zeros(n))
    rows = np.arange(n)
    # Jogging back into position at a steady 1.5 m/s: above 1.2, but not speeding up.
    _, _, going = restart_signals(ball, np.full(n, 1.5), FPS)
    assert not going[100:200].any()
    # Standing (1 m/s), then jogging (1.8 m/s) from row 150: going from a little before it.
    _, _, going = restart_signals(ball, np.where(rows < 150, 1.0, 1.8), FPS)
    assert going[150] and not going[80]


def test_a_throw_in_ends_at_the_throw():
    from mtah.dead.score import out_stoppages

    # The ball runs over the touchline (y = 0) and is held 2 m beyond it from row 100, out of
    # sight on 185-203 as it goes overhead, and thrown in: seen 1.1 m inside on row 204. The
    # players move at a steady 2 m/s throughout: never "getting going".
    n = 400
    y = np.empty(n)
    y[:80] = 10.0 - 0.15 * np.arange(80)
    y[80:100] = -2.0
    y[100:200] = -2.0
    y[200:210] = np.linspace(-2.0, 5.0, 10)
    y[210:] = 5.0 + 0.2 * np.arange(1, n - 209)
    ball = np.column_stack([np.full(n, 50.0), y])
    speed = np.full(n, 2.0)
    period = np.ones(n, dtype=np.int8)
    stopped = out_stoppages(ball, speed, period, FPS, PITCH_SIZE)
    first = int(np.argmax(stopped))
    assert stopped[first:204].all() and not stopped[204:].any()
    # The same with the ball unseen through the throw.
    hidden = ball.copy()
    hidden[185:204] = np.nan
    stopped = out_stoppages(hidden, speed, period, FPS, PITCH_SIZE)
    assert stopped[first:185].all() and not stopped[204:].any()


def test_a_throw_coming_in_over_several_frames():
    from mtah.dead.score import throw_ins

    # Held 2 m beyond the touchline (y = 0), then thrown in at 0.3 m a frame from row 100.
    y = np.concatenate([np.full(100, -2.0), -2.0 + 0.3 * np.arange(1, 61)])
    ball = np.column_stack([np.full(len(y), 50.0), y])
    thrown = throw_ins(ball, PITCH_SIZE, FPS)
    # The ball is last on the line (within 0.5 m) on row 107 (0.4 m inside), and reaches 1 m
    # inside on row 109: the throw is dated on row 108, the first row after the line.
    assert list(np.flatnonzero(thrown)) == [108]
