"""Who restarts after a stoppage (mtah.predict.restart) on small synthetic cases."""

import numpy as np

from mtah.predict.restart import (
    offline_team,
)

FPS = 25.0


def test_offline_team():
    players = np.full((3, 2, 2), 30.0)
    players[1, 0, 1] = -1.0  # home's player beyond the top touchline
    players[2, 0, 1] = 69.0  # both teams beyond one
    players[2, 1, 1] = -0.8
    assert offline_team(players, np.array([0, 1]), (105.0, 68.0)).tolist() == [
        -1,
        0,
        -1,
    ]
