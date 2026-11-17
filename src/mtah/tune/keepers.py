"""The keeper identification's zone and limits (mtah.clean.tracks), fitted on the DFL data.

The DFL data names its keepers (TW) and labels every player's team correctly, so on it an
identification can be scored directly:

    coverage   the share of a keeper's tracked frames on which he holds the identity: what
               matters on tracking that mislabels keepers
    flips      frames on which the identity sits on a player of the other team, moving him into
               the defending team's keeper slot: the one error that changes a team

fit() takes the grid point (the seed zone's radius, how long a track must hold it alone,
how far from his line a keeper may stay) with the most coverage among those whose flips stay
under MAX_FLIPS (0.1%) of the keepers' frames (coverage to a tenth of a percent, then
the fewer flips), or the fewest flips if none does.
"""

import itertools
from dataclasses import replace

import numpy as np

from mtah.clean.tracks import DEFAULT_CONFIG, clean_tracking

# Params
MAX_FLIPS = 0.001
GRID = {
    "seed_radius_m": (12.0, 16.0, 20.0),
    "seed_s": (1.0, 3.0, 6.0, 10.0),
    "far_from_line_m": (40.0,),
}


def score(matches, config):
    """(coverage, flips as a share of keeper frames) over DflMatch-like matches."""
    covered = total = flips = 0
    for m in matches:
        t = m.tracking
        result = clean_tracking(t, config)
        for k in m.keepers:
            seen = ~np.isnan(t.players[:, k, 0])
            covered += int(((result.keeper_track == k).any(axis=1) & seen).sum())
            total += int(seen.sum())
        for goal in (0, 1):
            slot = result.keeper_track[:, goal]
            for p, teams in result.defends.items():
                rows = (t.period == p) & (slot >= 0)
                flips += int(np.sum(t.team[slot[rows]] != teams[goal]))
    return covered / max(total, 1), flips / max(total, 1)


def fit(matches, base=DEFAULT_CONFIG, grid=GRID):
    """(best config, [(values dict, coverage, flips)]) over the grid's combinations."""
    names = list(grid)
    results = []
    for values in itertools.product(*(grid[name] for name in names)):
        chosen = dict(zip(names, values, strict=True))
        coverage, flips = score(matches, replace(base, **chosen))
        results.append((chosen, coverage, flips))
    allowed = [r for r in results if r[2] <= MAX_FLIPS]
    # Coverage to a tenth of a percent, then the fewer team-changing errors.
    best = (
        max(allowed, key=lambda r: (round(r[1], 3), -r[2]))
        if allowed
        else min(results, key=lambda r: r[2])
    )
    return replace(base, **best[0]), results
