"""The nearest players by team, and possession lengths (mtah.predict.possession)."""

import numpy as np

from mtah.predict.possession import AWAY, HOME, nearest, possession_lengths

H, A = HOME, AWAY


def test_nearest_with_opponent():
    team = np.array([H, H, A], dtype=np.int8)
    players = np.array([[[10.0, 0.0], [3.0, 0.0], [5.0, 0.0]]] * 2)
    players[1, 2] = np.nan  # no away player tracked on row 1
    ball = np.array([[0.0, 0.0], [np.nan, np.nan]])
    ball[1] = [4.0, 0.0]
    codes, distance, opponent = nearest(players, team, ball)
    assert codes.tolist() == [H, H]
    np.testing.assert_allclose(distance, [3.0, 1.0])
    np.testing.assert_allclose(opponent, [5.0, np.inf])


def test_possession_lengths():
    labels = np.array(list("XHHHDHHAAX"))
    assert possession_lengths(labels).tolist() == [3, 2, 2]
    assert possession_lengths(np.array([], dtype="<U1")).tolist() == []


# A player within 1 m of the ball controls it; NEAR_HOME and NEAR_AWAY are 1 m from a player.
LOOSE_BALL = [45.0, 30.0]  # far from everyone
