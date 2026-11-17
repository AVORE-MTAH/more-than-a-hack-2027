"""Dead ball from the tracking: the signals of a stoppage and of its restart, per tracking row.

    out_stoppages     the ball beyond a line starts a stoppage that lasts until its restart:
                      the ball, lately still, moving off with the players getting going
                      (restart_signals), or a throw-in coming back over the touchline
                      (throw_ins); stoppages() is the plainer ending it falls back on
    dribbling         the ball unseen while the player who had it keeps running: at their
                      feet, not a stoppage, however slow everyone else is (bridge_dribbles
                      fills a lull inside one gap)

The values are fitted on Metrica's event-labelled sample games where marked (mtah.tune.restarts),
the others set from football and checked insensitive (scripts/sensitivity_report.py).
"""

import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d

from mtah.dead.features import ball_outside, speeds

# Params
# A stoppage from a ball beyond a line.
OUT_MIN_ROWS = 3  # beyond the line this many rows running, against single-row glitches
# Restart kicks and throws peak above 6 m/s about 90% of the time in Metrica's sample games
# (goal kicks and corners at 13-19 m/s median, throw-ins and free kicks at 8-13).
KICK_SPEED = 6.0
KICK_ROWS = 2
# A kick only restarts play after the ball has been slow (or unseen) this long, and no sooner
# than this after the stoppage began, so the ball's own flight out is not a restart.
SETTLE_S = 1.0
SLOW_BALL = 3.0  # m/s
# Without a seen kick, the stoppage ends once the players are back at this pace this long.
EXIT_SPEED = 2.0
EXIT_S = 2.0
WHISTLE_MAX_S = 45.0  # fitted: mtah.tune.restarts (Metrica)
# A free kick is taken from a stationary ball: the ball almost stopped for a while (within
# STILL_M of where it lies over STILL_S), then gone a little way (MOVE_M within MOVE_S): over
# stretches, not frame to frame, so the tracking's jitter and a kick's first frames of
# speeding up do not hide it. The restart is where the ball has moved OFF_M.
STILL_S = 1.0
STILL_M = 0.5
MOVE_S = 0.5
MOVE_M = 0.75  # fitted: mtah.tune.restarts (Metrica)
OFF_M = 0.2
# The ball needs to have lain still within this long before the kick, not up to it: it is
# often set down, moved into place, and kicked a moment later.
RECENT_S = 2.0
# ...and only if it sets the players going: their median pace over GOING_S after it averages
# GOING_SPEED. Play inside the stoppage (a ball stopped and passed on its way back to the
# spot) leaves them waiting; a restart does not, even when the play after it is slow.
GOING_S = 3.0
GOING_SPEED = 1.2  # m/s
# ...and they speed up: GOING_RISE over their pace in the GOING_S before. During a goal
# kick's wait they jog back into position above GOING_SPEED already.
GOING_RISE = 0.2  # m/s
# The ball keeps going: in the AFTER_S after it moves off, it does not come to rest again
# (within REST_M for REST_S). A nudge into position moves it a metre and it rests; a kick, or
# a free kick tapped to a teammate, does not.
AFTER_S = 1.0
REST_S = 0.5
REST_M = 0.3
# A throw-in is the ball back on the pitch over the touchline: seen IN_M inside it, last seen
# on it (within LINE_M) or beyond, at most THROW_GAP_S before (it may be unseen in the throw).
IN_M = 1.0
LINE_M = 0.375  # fitted: mtah.tune.restarts (Metrica)
THROW_GAP_S = 3.0
# Dribbling: the nearest player within this of the ball just before it vanished, and moving at
# least this fast (averaged over the window) while it is unseen.
CONTROL_RADIUS = 1.5
DRIBBLE_SPEED = 2.0
DRIBBLE_WINDOW_S = 0.5


def stoppages(ball, speed, period, fps, pitch_size):
    """Per row: inside a stoppage that began with the ball beyond a line, before its restart.

    speed is the median player speed (median_speed). A period change ends any stoppage.
    """
    n = len(ball)
    settle = max(1, round(SETTLE_S * fps))
    out_start = _all_ahead(ball_outside(ball, pitch_size), OUT_MIN_ROWS)
    ball_speed = speeds(ball, fps)
    with np.errstate(invalid="ignore"):
        kick = _all_ahead(ball_speed >= KICK_SPEED, KICK_ROWS)
        settled = _all_behind(~(ball_speed >= SLOW_BALL), settle)
        back_in_play = _all_ahead(speed > EXIT_SPEED, max(1, round(EXIT_S * fps)))

    inside = np.zeros(n, dtype=bool)
    since = None  # row the current stoppage began on
    for i in range(n):
        if i > 0 and period[i] != period[i - 1]:
            since = None
        if since is None:
            if out_start[i] and period[i] > 0:
                since = i
        elif i - since >= settle and ((kick[i] and settled[i]) or back_in_play[i]):
            since = None
        inside[i] = since is not None
    return inside


def out_stoppages(ball, speed, period, fps, pitch_size):
    """Per row: inside a stoppage that began with the ball beyond a line, ended at its
    restart: the ball, lately still, moving off with the players getting going.

    stoppages() ends one at the first kick after the ball has settled, which a ball thrown
    or kicked back into the box for the goal kick is, the players still waiting; here only a
    restart ends it, and stoppages() is the fallback. Over a goal line (a goal kick
    or a corner, taken from inside the pitch) the still ball must lie inside the pitch, so a
    ball returned from behind the goal is not the restart; over a touchline (a throw-in, from
    outside) it need not, and the ball coming back onto the pitch over the touchline is a
    restart too (throw_ins: the throw, whatever the thrower's hold and the players' pace
    look like), whichever comes first. The search runs to WHISTLE_MAX_S or a new period,
    not to the players' return (they jog back during a goal kick's wait); with no restart
    found, the stoppage ends where stoppages() ends it.
    """
    n = len(ball)
    length, width = pitch_size
    old = stoppages(ball, speed, period, fps, pitch_size)
    with np.errstate(invalid="ignore"):
        on_pitch = (
            (ball[:, 0] >= 0)
            & (ball[:, 0] <= length)
            & (ball[:, 1] >= 0)
            & (ball[:, 1] <= width)
        )
    signals = {
        anywhere: np.logical_and.reduce(
            restart_signals(ball, speed, fps, None if anywhere else on_pitch)
        )
        for anywhere in (True, False)
    }
    thrown = throw_ins(ball, pitch_size, fps)
    out = _all_ahead(ball_outside(ball, pitch_size), OUT_MIN_ROWS) & (period > 0)
    starts = np.flatnonzero(out & ~np.concatenate([[False], out[:-1]]))
    move = max(1, round(MOVE_S * fps))
    settle, longest = (max(1, round(s * fps)) for s in (SETTLE_S, WHISTLE_MAX_S))
    inside = np.zeros(n, dtype=bool)
    for start in starts:
        last = min(n, start + longest)
        changed = np.flatnonzero(period[start:last] != period[start])
        if len(changed):
            last = start + changed[0]
        over_goal_line = not 0 <= ball[start, 0] <= length
        restart = signals[not over_goal_line]
        if not over_goal_line:
            restart = restart | thrown
        placed = np.flatnonzero(restart[start + settle : last])
        if len(placed):
            k = start + settle + placed[0]
            if thrown[k] and not over_goal_line:
                end = k  # the ball back on the pitch: the throw
            else:
                with np.errstate(invalid="ignore"):
                    off = np.hypot(*(ball[k : k + move + 1] - ball[k]).T) >= OFF_M
                end = k + int(np.argmax(off))
        else:
            ended = np.flatnonzero(~old[start:])
            end = start + ended[0] if len(ended) else n
        inside[start:end] = True
    return inside


def throw_ins(ball, pitch_size, fps):
    """Per row: a throw-in, the first row the ball is seen after it was last on a touchline
    (within LINE_M) or beyond, when it goes on to be seen IN_M inside the pitch within
    THROW_GAP_S of that: a throw comes in over several frames, not in one jump."""
    length, width = pitch_size
    y = ball[:, 1]
    seen = np.flatnonzero(np.isfinite(y))
    out = np.zeros(len(ball), dtype=bool)
    side = np.minimum(y[seen], width - y[seen])  # inside the nearer touchline, m
    on_length = (ball[seen, 0] >= 0) & (ball[seen, 0] <= length)
    on_line = side <= LINE_M
    deep = (side >= IN_M) & on_length
    gap = max(1, round(THROW_GAP_S * fps))
    line = None  # index in `seen` of the latest row on or beyond a touchline
    for k in range(len(seen)):
        if on_line[k]:
            line = k
        elif deep[k] and line is not None:
            if seen[k] - seen[line] <= gap:
                out[seen[line + 1]] = True
            line = None
    return out


def restart_signals(ball, speed, fps, where=None):
    """(lies, gone, going) per row, the three tests of a stoppage's restart.

    lies   the ball has lain almost still at some point in the last RECENT_S: within
           STILL_M of where it was over STILL_S (unseen rows count as still), and, with
           `where`, on a row where `where` holds
    gone   it is MOVE_M away within MOVE_S, and does not come to rest again in the AFTER_S
           after that (within REST_M for REST_S)
    going  the players' median pace (speed) over GOING_S from here averages GOING_SPEED
           and GOING_RISE over its average in the GOING_S before
    """
    n = len(ball)
    move, ahead, recent, after_move = (
        max(1, round(s * fps)) for s in (MOVE_S, GOING_S, RECENT_S, AFTER_S)
    )
    still = _lies_still(ball, max(1, round(STILL_S * fps)), STILL_M)
    if where is not None:
        still &= where
    # Still at some row of the last `recent`: a count of still rows over that window.
    tally = np.concatenate([[0], np.cumsum(still)])
    rows = np.arange(n)
    lies = tally[rows + 1] - tally[np.maximum(rows + 1 - recent, 0)] > 0
    gone = np.zeros(n, dtype=bool)
    if n > move:
        with np.errstate(invalid="ignore"):
            gone[:-move] = np.hypot(*(ball[move:] - ball[:-move]).T) >= MOVE_M
    # Rests: a REST_S-long stretch within REST_M ending on the row; none from `move` to
    # `move + after_move` rows on.
    rests = np.concatenate(
        [[0], np.cumsum(_lies_still(ball, max(1, round(REST_S * fps)), REST_M))]
    )
    first, last = np.minimum(rows + move, n), np.minimum(rows + move + after_move, n)
    gone &= rests[last] - rests[first] == 0
    seen = np.isfinite(speed)
    total = np.concatenate([[0.0], np.cumsum(np.where(seen, speed, 0.0))])
    count = np.concatenate([[0], np.cumsum(seen)])
    stop = np.minimum(rows + ahead, n)
    begin = np.maximum(rows - ahead, 0)
    after, before = count[stop] - count[rows], count[rows] - count[begin]
    with np.errstate(invalid="ignore", divide="ignore"):
        pace = (total[stop] - total[rows]) / after
        earlier = (total[rows] - total[begin]) / before
        going = (after > 0) & (pace >= GOING_SPEED)
        going &= (before == 0) | (pace - earlier >= GOING_RISE)
    return lies, gone, going


def _lies_still(ball, rows, radius):
    """True at i where ball[i] is seen and every seen position of the rows before it, rows
    of them, is within radius of it; unseen rows count as still."""
    out = np.zeros(len(ball), dtype=bool)
    if len(ball) <= rows:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(ball, rows + 1, axis=0)
    gap = np.hypot(
        windows[:, 0] - windows[:, 0, -1:], windows[:, 1] - windows[:, 1, -1:]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all unseen
        far = np.nanmax(gap, axis=1)
    out[rows:] = ~(far > radius) & ~np.isnan(ball[rows:]).any(axis=1)
    return out


def dribbling(players, ball, fps):
    """Per row: the ball unseen while the player nearest it before it vanished keeps running.

    Only gaps that begin with a player within CONTROL_RADIUS of the ball count.
    """
    missing = np.isnan(ball).any(axis=1)
    offset = players - ball[:, None, :]
    distance = np.hypot(offset[..., 0], offset[..., 1])
    distance[np.isnan(distance)] = np.inf
    slot = distance.argmin(axis=1)
    near = distance[np.arange(len(distance)), slot] <= CONTROL_RADIUS

    player_speed = speeds(players, fps)
    window = max(1, round(DRIBBLE_WINDOW_S * fps))
    seen = ~np.isnan(player_speed)
    total = uniform_filter1d(np.where(seen, player_speed, 0.0), window, axis=0)
    count = uniform_filter1d(seen.astype(float), window, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pace = np.where(count > 0, total / count, np.nan)

    out = np.zeros(len(ball), dtype=bool)
    for first, last in runs(missing):
        before = first - 1
        if before < 0 or not near[before]:
            continue
        with np.errstate(invalid="ignore"):
            out[first : last + 1] = (
                pace[first : last + 1, slot[before]] >= DRIBBLE_SPEED
            )
    return out


def bridge_dribbles(dribbled, missing):
    """dribbled, with each gap filled from its first dribbled row to its last.

    The ball never reappeared in between, so it stayed at the same player's feet while they
    slowed for a moment. Step one's minimum dead duration drops such a lull; a model that adds
    evidence up needs it filled.
    """
    out = dribbled.copy()
    for first, last in runs(missing):
        inside = np.flatnonzero(dribbled[first : last + 1])
        if len(inside):
            out[first + inside[0] : first + inside[-1] + 1] = True
    return out


def runs(mask):
    """Inclusive (first, last) of each run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def _all_ahead(mask, k):
    """True at i where mask[i : i + k] is all True (and i + k fits)."""
    total = np.concatenate([[0], np.cumsum(mask)])
    out = np.zeros(len(mask), dtype=bool)
    if len(mask) >= k:
        out[: len(mask) - k + 1] = total[k:] - total[:-k] == k
    return out


def _all_behind(mask, k):
    """True at i where mask[i - k : i] is all True (and i >= k)."""
    total = np.concatenate([[0], np.cumsum(mask)])
    out = np.zeros(len(mask), dtype=bool)
    if len(mask) >= k:
        out[k:] = (total[k:-1] - total[: len(mask) - k]) == k
    return out
