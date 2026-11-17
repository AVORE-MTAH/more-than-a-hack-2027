"""The ball in flight: between a kick and its reception, nobody can take possession.

The tracking is flat, so a ball passing over or past players reads as near them, and the nearest
player's team (the model's team evidence) can flip possession in mid-pass. The challenge's
definition (CHALLENGE.md) keeps a loose ball with the last team that had it, so possession can only change where the ball
is received. A flight is where the ball travels fast and straight:

    fast        the ball covers FLIGHT_SPEED (4 m/s) or more over SPAN_S (0.2 s); every frame
                of such a span counts
    touch       the path turns by more than KINK_DEG (30 degrees) between the SPAN_S before a
                frame and the SPAN_S after it, both at TURN_SPEED (3 m/s) or more: a header, a
                deflection, a first-time pass. The flight ends there, so a switch can happen
    lost        the ball goes missing in flight (high balls do): a gap of up to GAP_MAX_S (1.5 s)
                right after a fast frame stays in flight if the ball reappears at GAP_SPEED
                (5 m/s) or more from where it vanished, within KINK_DEG of the line it was on
    minimum     a flight lasts FLIGHT_S (0.5 s) or more; shorter fast stretches are a hard
                dribble touch or noise

A flight never crosses a change of period. All of it is per tracking row.

Until the reception the ball still belongs to whoever kicked it: kickers() gives each flight
the team of the nearest player within KICKER_M (3 m) of the ball around its start (KICK_S,
0.12 s, either side), the player it left; receivers() the team of the nearest player within
KICKER_M around its end, the player it reached (none for a ball out of play or lost).
"""

import numpy as np

from mtah.dead.score import runs

# Params
# m/s over SPAN_S; with FLIGHT_S, fitted by scripts/weights_report.py --skillcorner.
FLIGHT_SPEED = 4.0
SPAN_S = 0.2
KINK_DEG = 30.0
TURN_SPEED = 3.0
GAP_MAX_S = 1.5
GAP_SPEED = 5.0  # m/s, from where the ball vanished to where it reappears
FLIGHT_S = 0.5
KICKER_M = 3.0
KICK_S = 0.12  # fitted with KICKER_M and TURN_SPEED: DFL, spell F1


def in_flight(
    ball,
    fps,
    period=None,
    speed=FLIGHT_SPEED,
    shortest_s=FLIGHT_S,
    kink_deg=KINK_DEG,
):
    """Per row of ball (rows, 2) in metres, NaN where unseen: whether the ball is in flight.

    period (per row), if given, keeps a flight within one period; speed, shortest_s and
    kink_deg default to FLIGHT_SPEED, FLIGHT_S and KINK_DEG (scripts/weights_report.py chose them).
    """
    ball = np.asarray(ball, dtype=float)
    n = len(ball)
    k = max(1, round(SPAN_S * fps))
    if n <= k:
        return np.zeros(n, dtype=bool)

    # step[t]: the move over the span ending at row t (NaN for the first k rows).
    step = np.full((n, 2), np.nan)
    step[k:] = ball[k:] - ball[:-k]
    with np.errstate(invalid="ignore"):
        span_speed = np.hypot(step[:, 0], step[:, 1]) * fps / k
        fast_end = span_speed >= speed
    if period is not None:
        period = np.asarray(period)
        fast_end[k:] &= period[k:] == period[:-k]
    # Every row of a fast span: rows t-k..t for each fast span ending at t.
    total = np.concatenate([[0], np.cumsum(fast_end)])
    upto = np.minimum(np.arange(n) + k + 1, n)
    fast = total[upto] - total[np.arange(n)] > 0

    fast &= ~_touches(step, span_speed, k, fps, kink_deg)
    if period is not None:
        fast[1:] &= period[1:] == period[:-1]  # the first row of a period starts afresh
    flight = fast | _lost_in_flight(ball, step, fast, fps, period, kink_deg)

    shortest = max(1, round(shortest_s * fps))
    out = np.zeros(n, dtype=bool)
    for first, last in runs(flight):
        if last - first + 1 >= shortest:
            out[first : last + 1] = True
    return out


def kickers(flight, codes, distance, fps):
    """Per row: the kicking team's code on each flight's rows, -1 elsewhere.

    flight from in_flight; codes and distance the nearest player's team (-1 with the ball
    unseen) and distance per row (mtah.predict.possession.nearest). A flight with nobody
    within KICKER_M around its start has no kicker.
    """
    return _nearest_team(flight, codes, distance, fps, at_end=False)


def receivers(flight, codes, distance, fps):
    """Per row: the receiving team's code on each flight's rows, -1 elsewhere: as kickers,
    around the flight's end."""
    return _nearest_team(flight, codes, distance, fps, at_end=True)


def _nearest_team(flight, codes, distance, fps, at_end):
    """On each flight's rows, the nearest player's team within KICKER_M of the ball within
    KICK_S of the flight's first row (or last, at_end); -1 elsewhere or with nobody there."""
    out = np.full(len(flight), -1, dtype=np.int8)
    reach = max(1, round(KICK_S * fps))
    for first, last in runs(np.asarray(flight, dtype=bool)):
        anchor = last if at_end else first
        lo, hi = max(0, anchor - reach), min(len(flight), anchor + reach + 1)
        near = np.where(codes[lo:hi] >= 0, distance[lo:hi], np.inf)
        best = int(np.argmin(near))
        if near[best] <= KICKER_M:
            out[first : last + 1] = codes[lo + best]
    return out


def _turn_too_sharp(before, after, kink_deg=KINK_DEG):
    """Per row: the angle between the moves before and after exceeds kink_deg."""
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(before * after, axis=1) / (
            np.hypot(*before.T) * np.hypot(*after.T)
        )
        return cos < np.cos(np.radians(kink_deg))


def _touches(step, span_speed, k, fps, kink_deg=KINK_DEG):
    """Rows where the ball's path turns sharply at speed: someone touched it."""
    n = len(step)
    before, after = step, np.full_like(step, np.nan)
    after[:-k] = step[k:]  # the span starting at t is the one ending at t + k
    speed_after = np.full(n, np.nan)
    speed_after[:-k] = span_speed[k:]
    with np.errstate(invalid="ignore"):
        quick = (span_speed >= TURN_SPEED) & (speed_after >= TURN_SPEED)
    return quick & _turn_too_sharp(before, after, kink_deg)


def _lost_in_flight(ball, step, fast, fps, period, kink_deg=KINK_DEG):
    """Rows of the gaps the ball crosses in flight: vanished fast, reappeared further on."""
    n = len(ball)
    missing = np.isnan(ball).any(axis=1)
    out = np.zeros(n, dtype=bool)
    longest = round(GAP_MAX_S * fps)
    for first, last in runs(missing):
        before, after = first - 1, last + 1
        if before < 0 or after >= n or last - first + 1 > longest or not fast[before]:
            continue
        if period is not None and period[before] != period[after]:
            continue
        crossed = ball[after] - ball[before]
        speed = np.hypot(*crossed) * fps / (after - before)
        heading = step[before][None]
        straight = not _turn_too_sharp(heading, crossed[None], kink_deg)[0]
        if speed >= GAP_SPEED and straight:
            out[first : last + 1] = True
    return out
