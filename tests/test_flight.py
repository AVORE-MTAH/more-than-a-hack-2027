"""The ball in flight (mtah.predict.flight) on synthetic ball paths, in metres at 25 fps."""

import numpy as np

from mtah.dead.score import runs
from mtah.predict.flight import in_flight

FPS = 25.0
# The paths are timed for these thresholds: the detector's behaviour is tested, not the values
# fitted on external data (scripts/weights_report.py), which are the defaults.
MECHANICS = {"speed": 8.0, "shortest_s": 0.3}


def path(*legs, start=(30.0, 30.0)):
    """A ball path from (frames, velocity (vx, vy) in m/s) legs, one row per frame."""
    rows = [np.asarray(start, dtype=float)]
    for frames, velocity in legs:
        for _ in range(frames):
            rows.append(rows[-1] + np.asarray(velocity, dtype=float) / FPS)
    return np.array(rows)


def test_a_pass_is_in_flight_and_the_ball_at_rest_is_not():
    # At rest for 2 s, a 15 m/s pass for 1 s, at rest again.
    ball = path((50, (0, 0)), (25, (15, 0)), (50, (0, 0)))
    got = in_flight(ball, FPS, **MECHANICS)
    # The ball leaves row 50 and arrives on row 75: the 0.2 s spans reach 2 rows past each end.
    assert runs(got) == [(48, 77)]


def test_a_dribble_and_a_hard_touch_are_not():
    dribble = path((50, (0, 0)), (100, (5, 1)), (50, (0, 0)))
    assert not in_flight(dribble, FPS, **MECHANICS).any()
    touch = path((50, (0, 0)), (3, (12, 0)), (50, (0, 0)))  # 0.12 s, 1.4 m
    assert not in_flight(touch, FPS, **MECHANICS).any()


def test_tracker_noise_does_not_break_a_flight_or_make_one():
    rng = np.random.default_rng(0)
    ball = path((50, (0, 0)), (40, (14, 3)), (50, (0, 0)))
    ball += rng.normal(0, 0.05, ball.shape)
    assert len(runs(in_flight(ball, FPS, **MECHANICS))) == 1
    dribble = path((200, (4, 2))) + rng.normal(0, 0.1, (201, 2))
    assert not in_flight(dribble, FPS, **MECHANICS).any()


def test_a_touch_in_flight_ends_it():
    # A pass along x, headed on at 90 degrees: two flights with a break at the header.
    ball = path((50, (0, 0)), (25, (15, 0)), (25, (0, 15)), (50, (0, 0)))
    got = in_flight(ball, FPS, **MECHANICS)
    assert len(runs(got)) == 2
    assert not got[75]  # the header
    # A gentle curve (10 degrees over the flight) stays one flight.
    curve = path((50, (0, 0)), (12, (15, 0)), (13, (15, 2.6)), (50, (0, 0)))
    assert len(runs(in_flight(curve, FPS, **MECHANICS))) == 1


def test_a_ball_lost_in_flight_stays_in_flight():
    ball = path((50, (0, 0)), (50, (15, 0)), (50, (0, 0)))
    ball[70:85] = np.nan  # 0.6 s unseen, mid-flight
    got = in_flight(ball, FPS, **MECHANICS)
    assert len(runs(got)) == 1 and got[70:85].all()


def test_a_gap_after_a_reception_or_too_long_is_not_bridged():
    # The ball vanishes in flight and reappears 1 m on, 1 s later: received and kept.
    ball = path((50, (0, 0)), (20, (15, 0)), (25, (1, 0)), (50, (0, 0)))
    ball[71:95] = np.nan
    assert not in_flight(ball, FPS, **MECHANICS)[71:95].any()
    # Reappearing far off but at a right angle to the flight: not the same flight.
    turned = path((50, (0, 0)), (20, (15, 0)), (25, (0, 15)), (50, (0, 0)))
    turned[71:95] = np.nan
    assert not in_flight(turned, FPS, **MECHANICS)[71:95].any()
    # 2 s unseen: too long to say.
    long = path((50, (0, 0)), (80, (12, 0)), (50, (0, 0)))
    long[60:110] = np.nan
    assert not in_flight(long, FPS, **MECHANICS)[60:110].any()


def test_a_flight_does_not_cross_periods():
    ball = path((50, (0, 0)), (50, (15, 0)), (50, (0, 0)))
    period = np.where(np.arange(len(ball)) < 75, 1, 2)
    got = in_flight(ball, FPS, period, **MECHANICS)
    assert not got[75]
    assert all(period[a] == period[b] for a, b in runs(got))


def test_short_input():
    assert not in_flight(np.zeros((3, 2)), FPS).any()
    assert len(in_flight(np.zeros((0, 2)), FPS)) == 0


def test_each_flight_belongs_to_the_player_it_left():
    from mtah.predict.flight import kickers

    flight = np.zeros(100, dtype=bool)
    flight[20:40] = flight[60:80] = True
    codes = np.full(100, -1, dtype=np.int8)
    distance = np.full(100, np.nan)
    # Home kicks the first: at its feet before it, then it flies past away players.
    codes[15:22], distance[15:22] = 0, 0.4
    codes[22:40], distance[22:40] = 1, 1.0
    # Nobody within KICKER_M (3 m) of the second one's start.
    codes[56:65], distance[56:65] = 1, 4.0
    got = kickers(flight, codes, distance, FPS)
    assert (got[20:40] == 0).all()
    assert (got[60:80] == -1).all() and (got[~flight] == -1).all()


def test_each_flight_has_the_player_it_reached():
    from mtah.predict.flight import receivers

    flight = np.zeros(100, dtype=bool)
    flight[20:40] = True
    codes = np.full(100, -1, dtype=np.int8)
    distance = np.full(100, np.nan)
    codes[15:22], distance[15:22] = 0, 0.4  # home kicks it
    codes[37:45], distance[37:45] = 1, 0.6  # away takes it at its end
    got = receivers(flight, codes, distance, FPS)
    assert (got[20:40] == 1).all() and (got[~flight] == -1).all()
