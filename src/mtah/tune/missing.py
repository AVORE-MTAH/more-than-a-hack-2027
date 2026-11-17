"""A missing ball like a tactical camera's, fitted on Metrica's sample games and applied to others.

Metrica's tracking loses the ball as a tactical feed does: on about 40% of frames, mostly while
the ball is dead, the stoppages timed by its events. fit() measures, separately for live and for
dead play, the lengths of the runs with the ball seen and with it missing (QUANTILES of each, in
seconds) and the share of frames missing. degrade() then removes the ball from another match
(the DFL data's, never missing) the same way: within each stretch of live or dead play it
alternates seen and missing runs drawn from that state's lengths, starting missing with that
state's share, so each state ends up missing about as often as in Metrica's games.

The labels say which frames are live or dead (H, A or D); frames labelled X are left as they are.
"""

import numpy as np

from mtah.dead.score import runs

# Params
QUANTILES = np.linspace(0.025, 0.975, 39)


def _state(labels):
    labels = np.asarray(labels).astype(str)
    return np.where(np.isin(labels, ("H", "A")), 0, np.where(labels == "D", 1, -1))


def fit(pairs, fps):
    """{state: {share, seen_s, missing_s}} for state "live" and "dead", from (ball, labels)
    pairs: ball (rows, 2) with NaN where missing, labels per row."""
    out = {}
    for code, name in ((0, "live"), (1, "dead")):
        seen_runs, missing_runs, n_missing, n_total = [], [], 0, 0
        for ball, labels in pairs:
            state = _state(labels)
            missing = np.isnan(np.asarray(ball, float)).any(axis=1)
            for first, last in runs(state == code):
                part = missing[first : last + 1]
                n_missing += int(part.sum())
                n_total += len(part)
                seen_runs += [(b - a + 1) / fps for a, b in runs(~part)]
                missing_runs += [(b - a + 1) / fps for a, b in runs(part)]
        out[name] = {
            "share": n_missing / max(n_total, 1),
            "seen_s": np.quantile(seen_runs, QUANTILES).tolist() if seen_runs else [],
            "missing_s": np.quantile(missing_runs, QUANTILES).tolist()
            if missing_runs
            else [],
        }
    return out


def degrade(ball, labels, fps, fitted, rng):
    """A copy of ball with the ball removed in runs, live and dead play as fitted."""
    out = np.array(ball, dtype=float, copy=True)
    state = _state(labels)
    for code, name in ((0, "live"), (1, "dead")):
        params = fitted[name]
        if not params["seen_s"] or not params["missing_s"]:
            continue
        for first, last in runs(state == code):
            row = first
            missing = rng.random() < params["share"]
            while row <= last:
                lengths = params["missing_s"] if missing else params["seen_s"]
                length = max(1, round(float(rng.choice(lengths)) * fps))
                if missing:
                    out[row : min(row + length, last + 1)] = np.nan
                row += length
                missing = not missing
    return out
