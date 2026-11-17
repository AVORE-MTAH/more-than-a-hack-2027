"""The players nearest the ball, by team, and the possessions in a labelling.

nearest             per tracking row, the nearest player's team, their distance to the
                    ball, and the other team's nearest distance: the team evidence of the
                    model (mtah.predict.hmm) and of the frame classifier
possession_lengths  the lengths of the possessions in a run of labels, for the counts
                    make_submission.py prints
"""

import numpy as np

HOME, AWAY, UNKNOWN = 0, 1, -1


def nearest(players, team, ball):
    """(codes, distance, opponent) per tracking row.

    codes and distance are the nearest tracked player's team and distance to the ball, opponent
    the distance of the other team's nearest player; UNKNOWN and inf where the ball or every
    player is missing, inf for opponent where the other team has nobody tracked. Positions must
    be in metres: on the 0-1 scale the short axis is stretched and the wrong player wins near
    the touchlines.
    """
    offset = players - ball[:, None, :]
    distance = np.hypot(offset[..., 0], offset[..., 1])
    distance[np.isnan(distance)] = np.inf
    by_team = [distance[:, team == t].min(axis=1, initial=np.inf) for t in (HOME, AWAY)]
    best = np.minimum(*by_team)
    observed = np.isfinite(best)

    codes = np.full(len(distance), UNKNOWN, dtype=np.int8)
    codes[observed] = np.where(by_team[HOME] <= by_team[AWAY], HOME, AWAY)[observed]
    opponent = np.where(codes == AWAY, by_team[HOME], by_team[AWAY])
    return codes, best, opponent


def possession_lengths(labels):
    """Lengths in frames of the maximal H or A runs in `labels` (D, X and gaps break them)."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return np.zeros(0, dtype=np.int64)
    edges = np.flatnonzero(np.concatenate([[True], labels[1:] != labels[:-1], [True]]))
    starts, stops = edges[:-1], edges[1:]
    return (stops - starts)[np.isin(labels[starts], ("H", "A"))]
