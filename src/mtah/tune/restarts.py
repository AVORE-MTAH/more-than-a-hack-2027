"""The ball-out stoppage and restart thresholds (mtah.dead.score), fitted on the DFL data.

The DFL data marks the ball dead or alive on every frame, so each stoppage's true restart is the
first live frame after it. Scored are the stoppages in which the ball went out (seen beyond a
line, dead.features.ball_outside): the ones out_stoppages is for; a foul or an offside stops play
with the ball on the pitch and is no ball-out.

    error      for each such stoppage our ball-out stoppage overlaps, where ours ends against the
               true restart, in seconds (+ late), capped at CAP_S
    missed     such stoppages ours does not overlap (counted as CAP_S late)
    false      our stoppages overlapping no dead frame at all

The thresholds are module constants of mtah.dead.score; patched() sets them for a run. Fitted
freely (fit(), coordinate descent toward the lowest mean absolute error) on one Metrica game, they
get worse on the other, both ways: two games do not pin 17 values down. fit_consistent() takes a
change only when it lowers the error in every game separately, so what it keeps holds across
matches; a value it does not move stays at its physical default.
"""

import contextlib

import numpy as np

import mtah.dead.score as dead_score
from mtah.dead.features import ball_outside, median_speed

# Params
CAP_S = 10.0
MIN_DEAD_S = 1.0  # shorter dead runs are not stoppages to score
STEPS = (0.5, 0.75, 1.5, 2.0)  # tried multiples of each value
NAMES = (
    "STILL_S",
    "STILL_M",
    "MOVE_S",
    "MOVE_M",
    "OFF_M",
    "RECENT_S",
    "GOING_S",
    "GOING_SPEED",
    "GOING_RISE",
    "AFTER_S",
    "REST_S",
    "REST_M",
    "IN_M",
    "LINE_M",
    "THROW_GAP_S",
    "SETTLE_S",
    "WHISTLE_MAX_S",
)


@contextlib.contextmanager
def patched(**values):
    """Set mtah.dead.score's constants for the duration."""
    old = {name: getattr(dead_score, name) for name in values}
    try:
        for name, value in values.items():
            setattr(dead_score, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(dead_score, name, value)


def _runs(mask):
    return dead_score.runs(np.asarray(mask, dtype=bool))


def prepared(match):
    """What score() needs of a DflMatch, computed once: (tracking, pace, dead, out)."""
    return prepared_from(match.tracking, match.labels == "D")


def prepared_from(tracking, dead):
    """(tracking, pace, dead, out) for any tracking with a dead mask per row."""
    t = tracking
    return (
        t,
        median_speed(t.players, t.fps),
        np.asarray(dead, dtype=bool),
        ball_outside(t.ball, t.pitch_size),
    )


def metrica_dead(tracking, intervals):
    """Per row of a Metrica tracking: inside one of dead_intervals' video-frame ranges."""
    dead = np.zeros(len(tracking.ball), dtype=bool)
    frame = tracking.video_frame
    for first, last in intervals:
        dead |= (frame >= first) & (frame <= last)
    return dead


def score(prepared_matches):
    """{mae_s, median_s, scored, missed, false} over prepared matches."""
    errors, missed, false, scored = [], 0, 0, 0
    for t, pace, dead, out in prepared_matches:
        ours = dead_score.out_stoppages(t.ball, pace, t.period, t.fps, t.pitch_size)
        cap = round(CAP_S * t.fps)
        shortest = round(MIN_DEAD_S * t.fps)
        for first, last in _runs(dead):
            if last - first + 1 < shortest or not out[first : last + 1].any():
                continue
            scored += 1
            overlap = np.flatnonzero(ours[first : last + 1])
            if not len(overlap):
                missed += 1
                errors.append(CAP_S)
                continue
            end = first + overlap[-1]
            while end + 1 < len(ours) and ours[end + 1]:
                end += 1
            errors.append(float(np.clip(end - last, -cap, cap)) / t.fps)
        for first, last in _runs(ours):
            if not dead[first : last + 1].any():
                false += 1
    e = np.abs(np.array(errors))
    return {
        "mae_s": float(e.mean()) if len(e) else 0.0,
        "median_s": float(np.median(errors)) if errors else 0.0,
        "scored": scored,
        "missed": missed,
        "false": false,
    }


def fit(prepared_matches, names=NAMES, steps=STEPS, sweeps=2):
    """(chosen values, best score, log of (name, value, mae)) by coordinate descent."""
    chosen = {name: getattr(dead_score, name) for name in names}
    with patched(**chosen):
        best = score(prepared_matches)
    log = []
    for _ in range(sweeps):
        improved = False
        for name in names:
            for step in steps:
                trial = {**chosen, name: type(chosen[name])(chosen[name] * step)}
                with patched(**trial):
                    result = score(prepared_matches)
                log.append((name, trial[name], result["mae_s"]))
                if result["mae_s"] < best["mae_s"] - 1e-6:
                    chosen, best, improved = trial, result, True
        if not improved:
            break
    return chosen, best, log


def fit_consistent(per_game, names=NAMES, steps=STEPS, sweeps=2):
    """(chosen values, [score per game], log): a change is kept only if it lowers the mean
    absolute error in every game of per_game (a list of prepared-match lists)."""
    chosen = {name: getattr(dead_score, name) for name in names}
    with patched(**chosen):
        best = [score(game)["mae_s"] for game in per_game]
    log = []
    for _ in range(sweeps):
        improved = False
        for name in names:
            for step in steps:
                trial = {**chosen, name: type(chosen[name])(chosen[name] * step)}
                with patched(**trial):
                    maes = [score(game)["mae_s"] for game in per_game]
                log.append((name, trial[name], maes))
                if all(m < b - 1e-6 for m, b in zip(maes, best, strict=True)):
                    chosen, best, improved = trial, maes, True
        if not improved:
            break
    return chosen, best, log
