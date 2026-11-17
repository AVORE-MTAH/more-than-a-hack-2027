"""The stale and coasting thresholds of the cleaning (mtah.clean.tracks), from external data.

A leftover track is either still (stale) or carried on in a perfectly straight line
(coasting). Neither the DFL nor SkillCorner data marks such ghosts: SkillCorner's extrapolated
players follow a model, not a straight line. What the DFL data does give is real players, none of
whom should be removed. So each threshold sits at QUANTILE (1%) of real players' values over the
same 4 s windows: at most about one real window in a hundred looks like a ghost, on tracking as
smooth as DFL's. Noisier tracking moves real players further from a ghost, never closer.

    stale_max_spread_m   the QUANTILE of a real player's 4 s spread (bounding-box diagonal)
    coast_max_excess_m   the QUANTILE of a real player's 4 s path minus straight move, over the
                         windows slower than coast_max_speed_mps (a lost track drifts slowly;
                         a straight line faster than that is play), which stays as set

Windows start every STEP_S (1 s), to keep the pooled sample small.
"""

import numpy as np

from mtah.clean.tracks import DEFAULT_CONFIG, window_spread

# Params
QUANTILE = 0.01
STEP_S = 1.0


def window_values(players, fps, window_s, step_s=STEP_S):
    """(spread, excess, move) in metres of every complete window, one per slot and start."""
    w = max(2, round(window_s * fps))
    step = max(1, round(step_s * fps))
    n = len(players)
    if n < w:
        empty = np.zeros(0)
        return empty, empty, empty
    spread = window_spread(players, w)[::step]
    starts = np.arange(0, n - w + 1, step)
    ends = starts + w - 1
    moves = np.hypot(*np.diff(players, axis=0).transpose(2, 0, 1))
    walked = np.concatenate(
        [np.zeros((1, players.shape[1])), np.cumsum(np.nan_to_num(moves), axis=0)]
    )
    gaps = np.concatenate(
        [np.zeros((1, players.shape[1])), np.cumsum(np.isnan(moves), axis=0)]
    )
    path = walked[ends] - walked[starts]  # the steps into rows start+1..end
    complete = (gaps[ends] - gaps[starts] == 0) & np.isfinite(spread)
    move = np.hypot(*(players[ends] - players[starts]).transpose(2, 0, 1))
    return spread[complete], (path - move)[complete], move[complete]


def fit(trackings, config=DEFAULT_CONFIG, quantile=QUANTILE):
    """{stale_max_spread_m, coast_max_excess_m, windows} over real players' windows."""
    spreads, excesses = [], []
    for t in trackings:
        spread, excess, move = window_values(t.players, t.fps, config.stale_window_s)
        slowest = config.coast_max_speed_mps * config.stale_window_s
        spreads.append(spread)
        excesses.append(excess[move <= slowest])
    spread, excess = np.concatenate(spreads), np.concatenate(excesses)
    return {
        "stale_max_spread_m": float(np.quantile(spread, quantile)),
        "coast_max_excess_m": float(np.quantile(excess, quantile)),
        "windows": len(spread),
    }
