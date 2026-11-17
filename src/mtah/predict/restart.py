"""A team beyond a touchline: who is about to take a throw-in.

A player more than OFFLINE_M beyond a touchline, from one team only, is a throw-in's taker
(right on 91-98% of the stoppages it shows on in SkillCorner's and DFL's data). It is one input
of the per-frame classifier (mtah.predict.classifier).
"""

import numpy as np

# Params
OFFLINE_M = 0.5  # beyond the touchline: where a throw-in is taken from

H, A = 0, 1


def offline_team(players, team, pitch_size, margin=OFFLINE_M):
    """Per row: the team code of the players beyond a touchline by more than margin, when only
    one team has any; -1 otherwise."""
    width = pitch_size[1]
    y = players[..., 1]
    with np.errstate(invalid="ignore"):
        off = (y < -margin) | (y > width + margin)
    team = np.asarray(team)
    home = off[:, team == H].any(axis=1)
    away = off[:, team == A].any(axis=1)
    return np.where(home & ~away, H, np.where(away & ~home, A, -1)).astype(np.int8)
