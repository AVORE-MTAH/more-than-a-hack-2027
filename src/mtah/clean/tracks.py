"""Clean the automatic tracking before any possession logic: stale tracks and goalkeepers.

Two faults of the tracker, fixed in this order:

    stale tracks   a lost player's last position is left in place, nearly frozen, while play
                   goes on. A track that stays within a few cm for several seconds is set to NaN
                   from the start of that stillness. A real keeper standing still sways more.
                   Or the tracker carries a lost player's last movement on: a slow drift in
                   a perfectly straight line. A track whose path over the same window is
                   longer than its straight-line move by under 2 cm, at under 1 m/s, is
                   coasting and goes too: a tracked person's detection noise alone lengthens
                   the path by more over 4 s. Left in a goal's seed zone, such a ghost can
                   take the keeper's identity and keep it, the real keeper labelled as found.
    keeper teams   the team label is unreliable for keepers, reliable for outfield players.
                   Keepers are identified per track over time and moved into two slots appended
                   after the raw ones, home GK (team 0) and away GK (team 1), as the team that
                   defends their goal. Outfield labels are never changed.

Goal 0 is at x = 0, goal 1 at x = pitch length. Which team defends which goal is read per period
from the outfield players' mean x. A period inside which the teams swap ends and stay swapped
(a file that marks no halftime) is split where they do (split_periods): each team's mean x,
which side of the other's it is on, smoothed over `side_window_s`, flipping for `half_min_s` or
longer. The cleaned tracking carries the split periods.

A keeper is seeded when exactly one track is in a goal's seed zone (the penalty area, near the
goal centre) for `seed_s`; the identity starts at the start of that run and follows the track
until it jumps faster than a person can run, is lost for longer than `max_gap_s`, stays far from
its goal line for `far_s`, or another track takes the keeper's position: it came within
`takeover_contact_m` of this one and then holds the seed zone alone for `takeover_s` while this
one is out of it and moving away (it covers more ground over that time than the new one). Two
tracks in the zone (an attacker next to the keeper at a corner) seed nothing
and take nothing over, and an attacker waiting in the box while the keeper comes out takes nothing
over either.

An identity also reaches back along its track from where it was seeded, up to `back_s`: the
track was the keeper before he stood alone near his goal too (a keeper with the ball after a
catch is often at the box's edge or among attackers). It stops where the track breaks (lost
for longer than `max_gap_s`, or a jump), lies more than `far_from_line_m` from the goal line,
holds another identity, or the period starts.
"""

import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from mtah.io.tracking import Tracking

# Params
GOALS = (0, 1)


@dataclass(frozen=True)
class CleanConfig:
    """Every threshold of the cleaning, in seconds and metres."""

    # Stale: the bounding-box diagonal of a track over `stale_window_s` stays below this.
    stale_window_s: float = 4.0
    stale_max_spread_m: float = 0.064  # fitted: mtah.tune.cleaning (DFL)
    # Coasting: over the same window, path minus straight-line move below this, the move
    # slower than coast_max_speed_mps. 0 turns it off.
    coast_max_excess_m: float = 0.029  # fitted: mtah.tune.cleaning (DFL)
    coast_max_speed_mps: float = 1.0
    # Penalty area: depth from the goal line, half width about the pitch's centre line.
    box_depth_m: float = 16.5
    box_half_width_m: float = 20.16
    # Seed zone: the penalty area, within this distance of the goal centre.
    seed_radius_m: float = 20.0  # fitted: mtah.tune.keepers (DFL)
    seed_s: float = 6.0  # fitted: mtah.tune.keepers (DFL)
    # Faster than this between two observations is an identity switch, not a person.
    max_speed_mps: float = 12.0
    # A keeper track missing for longer than this has ended.
    max_gap_s: float = 0.2
    far_from_line_m: float = 40.0  # fitted: mtah.tune.keepers (DFL)
    far_s: float = 5.0
    takeover_s: float = 1.0
    # A takeover needs the new track to have been this close to the keeper's during its current
    # stay in the seed zone: it took the keeper's position, as in an identity swap.
    takeover_contact_m: float = (
        2.0  # fitted: mtah.tune.keepers (DFL; 1-2 m tie, the nearest the old kept)
    )
    # An identity reaches back along its track from its seed by up to this; 0 turns it off.
    back_s: float = 240.0  # fitted: mtah.tune.keepers (DFL)
    # The teams swapping ends inside a period: smoothed over this, held for half_min_s.
    side_window_s: float = 120.0
    half_min_s: float = 600.0


DEFAULT_CONFIG = CleanConfig()


@dataclass(frozen=True)
class CleanResult:
    """The cleaned tracking and what was changed to get it.

    tracking      the cleaned Tracking: raw slots, then home GK and away GK slots
    keeper_track  (frames, 2)       raw slot identified as keeper of goal 0 / goal 1, or -1
    keeper_slots  (2,)              slot indexes of home GK and away GK in `tracking`
    stale         (frames, slots)   raw positions removed as stale (coasting included)
    coasting      (frames, slots)   those removed for coasting
    defends       {period: (team defending goal 0, team defending goal 1)}
    log           one row per changed segment: frames and slots, never positions
    """

    tracking: Tracking
    keeper_track: np.ndarray
    keeper_slots: tuple[int, int]
    stale: np.ndarray
    coasting: np.ndarray
    defends: dict
    log: pd.DataFrame


def clean_tracking(tracking, config=DEFAULT_CONFIG) -> CleanResult:
    """Remove stale tracks, then move identified keepers into the GK slots."""
    raw = tracking.players
    n_frames, n_slots = raw.shape[:2]
    coasting = coast_mask(raw, tracking.fps, config)
    stale = stale_mask(raw, tracking.fps, config) | coasting
    players = raw.copy()
    players[stale] = np.nan
    tracking = replace(
        tracking,
        period=split_periods(
            players, tracking.team, tracking.period, tracking.fps, config
        ),
    )

    defends = defending_goals(players, tracking.team, tracking.period)
    keeper_track, segments = identify_keepers(
        players, tracking.period, tracking.pitch_size, tracking.fps, config
    )
    # Periods whose sides are unknown keep their keepers where they are.
    segments = [s for s in segments if int(tracking.period[s["first"]]) in defends]
    unknown = ~np.isin(tracking.period, list(defends))
    keeper_track[unknown] = -1

    cleaned = np.concatenate([players, np.full((n_frames, 2, 2), np.nan)], axis=1)
    for goal in GOALS:
        for period, teams in defends.items():
            rows = np.flatnonzero(
                (tracking.period == period) & (keeper_track[:, goal] >= 0)
            )
            slots = keeper_track[rows, goal]
            cleaned[rows, n_slots + teams[goal]] = cleaned[rows, slots]
            cleaned[rows, slots] = np.nan

    team = np.concatenate([tracking.team, np.array([0, 1], dtype=tracking.team.dtype)])
    log = _log(stale, segments, tracking, defends, n_slots)
    return CleanResult(
        tracking=replace(tracking, players=cleaned, team=team),
        keeper_track=keeper_track,
        keeper_slots=(n_slots, n_slots + 1),
        stale=stale,
        coasting=coasting,
        defends=defends,
        log=log,
    )


def stale_mask(players, fps, config=DEFAULT_CONFIG):
    """(frames, slots) True where a track is part of a stale segment.

    A window of `stale_window_s` is still when the track is observed on all of it and its
    bounding-box diagonal is below `stale_max_spread_m`; the mask is the union of still windows.
    """
    n_frames, n_slots = players.shape[:2]
    window = _frames(config.stale_window_s, fps)
    if n_frames < window:
        return np.zeros((n_frames, n_slots), dtype=bool)

    still = np.zeros((n_frames, n_slots), dtype=bool)
    with np.errstate(invalid="ignore"):
        still[: n_frames - window + 1] = (
            window_spread(players, window) < config.stale_max_spread_m
        )

    # Frame i is stale if a still window starts in [i - w + 1, i].
    counts = np.concatenate([np.zeros((1, n_slots)), np.cumsum(still, axis=0)])
    first = np.maximum(np.arange(n_frames) - window + 1, 0)
    return counts[1:] - counts[first] > 0


def coast_mask(players, fps, config=DEFAULT_CONFIG):
    """(frames, slots) True where a track is part of a coasting segment.

    A window of `stale_window_s` coasts when the track is observed on all of it, moves less
    than `coast_max_speed_mps` from its first frame to its last, and its path is longer than
    that straight line by less than `coast_max_excess_m`; the mask is the union of such windows.
    """
    n_frames, n_slots = players.shape[:2]
    window = _frames(config.stale_window_s, fps)
    out = np.zeros((n_frames, n_slots), dtype=bool)
    if n_frames < window or window < 2 or config.coast_max_excess_m <= 0:
        return out

    missing = np.isnan(players).any(axis=2)
    step = np.zeros((n_frames, n_slots))
    with np.errstate(invalid="ignore"):
        lengths = np.hypot(*np.diff(players, axis=0).transpose(2, 0, 1))
    step[1:] = np.where(np.isnan(lengths), 0.0, lengths)
    walked = np.concatenate([np.zeros((1, n_slots)), np.cumsum(step, axis=0)])
    gaps = np.concatenate([np.zeros((1, n_slots)), np.cumsum(missing, axis=0)])

    starts = np.arange(n_frames - window + 1)
    ends = starts + window - 1
    path = walked[ends + 1] - walked[starts + 1]  # the steps into frames start+1..end
    move = np.hypot(*(players[ends] - players[starts]).transpose(2, 0, 1))
    complete = gaps[ends + 1] - gaps[starts] == 0
    slowest = config.coast_max_speed_mps * (window - 1) / fps
    with np.errstate(invalid="ignore"):
        coasting = (
            complete & (path - move < config.coast_max_excess_m) & (move <= slowest)
        )

    # Frame i coasts if a coasting window starts in [i - w + 1, i].
    counts = np.concatenate([np.zeros((1, n_slots)), np.cumsum(coasting, axis=0)])
    counts = np.concatenate([counts, np.repeat(counts[-1:], window - 1, axis=0)])
    first = np.maximum(np.arange(n_frames) - window + 1, 0)
    return counts[1 : n_frames + 1] - counts[first] > 0


def window_spread(players, window):
    """(frames - window + 1, slots): each track's bounding-box diagonal over the `window` frames
    starting at each row; NaN where the window misses an observation."""
    n_frames = len(players)
    missing = np.isnan(players).any(axis=2)
    filled = np.where(np.isnan(players), 0.0, players)

    # Centred filters of size w cover [i - w//2, i - w//2 + w); the forward window starting at
    # t is the centred one at t + w//2.
    starts = slice(window // 2, window // 2 + n_frames - window + 1)

    def forward(filter_, values):
        return filter_(values, window, axis=0)[starts]

    spread = np.hypot(
        *(
            forward(maximum_filter1d, filled[..., k])
            - forward(minimum_filter1d, filled[..., k])
            for k in (0, 1)
        )
    )
    spread[forward(maximum_filter1d, missing.astype(np.uint8)) > 0] = np.nan
    return spread


def seed_zone(players, pitch_size, goal, config=DEFAULT_CONFIG):
    """(frames, slots) True where a track is in goal `goal`'s seed zone: the penalty area,
    within `seed_radius_m` of the goal centre."""
    length, width = pitch_size
    goal_x = 0.0 if goal == 0 else length
    x, y = players[..., 0], players[..., 1]
    with np.errstate(invalid="ignore"):
        in_box = (np.abs(x - goal_x) <= config.box_depth_m) & (
            np.abs(y - width / 2) <= config.box_half_width_m
        )
        return in_box & (np.hypot(x - goal_x, y - width / 2) <= config.seed_radius_m)


def split_periods(players, team, period, fps, config=DEFAULT_CONFIG):
    """period, with a period split where the teams swap ends inside it and stay swapped.

    Per row, whether home's mean x is below away's, as a share over a centred
    `side_window_s` window of the rows with both teams seen; a period whose smoothed side
    holds one way for `half_min_s` and later the other way for as long is split at the start
    of the second stretch, the rows from there on getting a new period number.
    """
    period = np.asarray(period).copy()
    x = players[..., 0]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # rows with a team unseen
        gap = np.nanmean(x[:, team == 0], axis=1) - np.nanmean(x[:, team == 1], axis=1)
    known = ~np.isnan(gap)
    left = known & (gap < 0)
    window = _frames(config.side_window_s, fps)
    longest = _frames(config.half_min_s, fps)
    next_id = int(period.max()) + 1 if len(period) else 1
    for p in np.unique(period[period > 0]):
        rows = np.flatnonzero(period == p)
        share = _centred_share(left[rows], known[rows], window)
        side = np.where(np.isnan(share), -1, share > 0.5).astype(int)
        held = [
            (a, b, side[a])
            for a, b in _runs(side >= 0)
            for a, b in _side_runs(side, a, b)
            if b - a + 1 >= longest
        ]
        swap = next((a for a, _, s in held[1:] if s != held[0][2]), None)
        if swap is not None:
            period[rows[swap:]] = next_id
            next_id += 1
    return period


def _side_runs(side, first, last):
    """[(a, b)] runs of one side value inside first..last."""
    part = side[first : last + 1]
    change = np.flatnonzero(np.diff(part) != 0) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(part)] - 1
    return [(first + int(a), first + int(b)) for a, b in zip(starts, ends, strict=True)]


def _centred_share(hits, counted, window):
    """Per row: hits over counted rows in a centred window, NaN where none counted."""
    total = np.concatenate([[0], np.cumsum(hits)])
    count = np.concatenate([[0], np.cumsum(counted)])
    n = len(hits)
    lo = np.clip(np.arange(n) - window // 2, 0, n)
    hi = np.clip(np.arange(n) + window - window // 2, 0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            count[hi] > count[lo],
            (total[hi] - total[lo]) / (count[hi] - count[lo]),
            np.nan,
        )


def defending_goals(players, team, period):
    """{period: (team defending goal 0, team defending goal 1)} from mean x per team.

    The team whose players' mean x is lower (median over the period's frames) defends goal 0.
    A period where either team is never observed is left out.
    """
    defends = {}
    for p in np.unique(period[period > 0]):
        rows = players[period == p]
        means = []
        for t in (0, 1):
            x = rows[:, team == t, 0]
            count = (~np.isnan(x)).sum(axis=1)
            seen = count > 0
            if not seen.any():
                break
            means.append(np.median(np.nansum(x, axis=1)[seen] / count[seen]))
        if len(means) == 2:
            defends[int(p)] = (0, 1) if means[0] <= means[1] else (1, 0)
    return defends


def identify_keepers(players, period, pitch_size, fps, config=DEFAULT_CONFIG):
    """(keeper_track (frames, 2), segments): the raw slot keeping each goal, or -1.

    Each segment is a dict with goal, slot, first and last row (inclusive) and end_reason.
    Goal 0 is tracked first; goal 1 cannot take a slot that goal 0 holds on that frame.
    """
    n_frames = len(players)
    observed = ~np.isnan(players[..., 0])
    jump = _jumps(players, observed, fps, config)
    keeper_track = np.full((n_frames, 2), -1, dtype=np.int64)
    segments = []
    for goal in GOALS:
        taken = keeper_track[:, 0] if goal == 1 else None
        segments += _track_goal(
            goal,
            players,
            observed,
            jump,
            period,
            pitch_size,
            fps,
            config,
            keeper_track[:, goal],
            taken,
        )
    return keeper_track, segments


def _frames(seconds, fps):
    return max(1, round(seconds * fps))


def _jumps(players, observed, fps, config):
    """(frames, slots) True where a track moved faster than `max_speed_mps` since it was last seen."""
    n_frames = len(observed)
    index = np.where(observed, np.arange(n_frames)[:, None], -1)
    last_seen = np.maximum.accumulate(index, axis=0)
    previous = np.full_like(last_seen, -1)
    previous[1:] = last_seen[:-1]

    has_previous = observed & (previous >= 0)
    rows = np.where(has_previous, previous, 0)
    before = np.take_along_axis(players, rows[..., None], axis=0)
    distance = np.hypot(*(players - before).transpose(2, 0, 1))
    elapsed = (np.arange(n_frames)[:, None] - rows) / fps
    with np.errstate(invalid="ignore"):
        return has_previous & (distance > config.max_speed_mps * elapsed)


def _track_goal(
    goal, players, observed, jump, period, pitch_size, fps, config, keeper, taken
):
    """Fill `keeper` (frames,) for one goal in place; return its segments."""
    length, width = pitch_size
    goal_x = 0.0 if goal == 0 else length
    x, y = players[..., 0], players[..., 1]
    from_line = np.abs(x - goal_x)
    centre = np.hypot(x - goal_x, y - width / 2)
    in_seed = seed_zone(players, pitch_size, goal, config)
    with np.errstate(invalid="ignore"):
        far = from_line > config.far_from_line_m
    n_in = in_seed.sum(axis=1)
    sole = np.where(n_in == 1, in_seed.argmax(axis=1), -1)
    # First row of each slot's current stay in the seed zone.
    rows = np.arange(len(players))[:, None]
    zone_since = np.maximum.accumulate(np.where(in_seed, 0, rows + 1), axis=0)
    contact = _Contact(players, zone_since, config.takeover_contact_m)

    seed_frames = _frames(config.seed_s, fps)
    gap_frames = _frames(config.max_gap_s, fps)
    far_frames = _frames(config.far_s, fps)
    take_frames = _frames(config.takeover_s, fps)

    segments = []
    k = -1  # current keeper slot
    start = last_seen = 0
    run_slot = run = 0  # current run of the same sole slot
    far_run = take_run = take_slot = 0  # take_run: frames `take_slot` has held the zone
    boundary = 0  # first row a new keeper may start on

    def close(last, reason):
        nonlocal k, boundary
        keeper[last + 1 : f + 1] = -1
        if last >= start:
            segments.append(
                {
                    "goal": goal,
                    "slot": k,
                    "first": start,
                    "last": last,
                    "end_reason": reason,
                }
            )
        k, boundary = -1, last + 1

    def begin(slot, first):
        nonlocal k, start, last_seen, far_run, take_run
        k, start, last_seen, far_run, take_run = slot, first, f, 0, 0
        keeper[first : f + 1] = slot

    for f in range(len(players)):
        if f > 0 and period[f] != period[f - 1]:
            if k >= 0:
                close(f - 1, "period_end")
            run, boundary = 0, f
        if period[f] == 0:
            continue

        s = int(sole[f])
        free = s >= 0 and (taken is None or taken[f] != s)
        if free and s == run_slot and run > 0:
            run += 1
        else:
            run_slot, run = s, 1 if free else 0

        if k >= 0:
            if taken is not None and taken[f] == k:
                close(f - 1, "other_goal")
            elif not observed[f, k]:
                if f - last_seen > gap_frames:
                    close(last_seen, "track_end")
                else:
                    keeper[f] = k
            elif jump[f, k]:
                close(last_seen, "jump")
            else:
                last_seen = f
                far_run = far_run + 1 if far[f, k] else 0
                if free and s != k and centre[f, s] < centre[f, k]:
                    if take_run > 0 and s == take_slot:
                        take_run += 1
                    elif contact.met(s, k, f):
                        take_run, take_slot = 1, s
                    else:
                        take_run = 0
                else:
                    take_run = 0
                first = f - take_frames + 1
                if far_run >= far_frames:
                    close(f - far_frames, "far")
                elif take_run >= take_frames and _moved(players, k, first, f) > _moved(
                    players, s, first, f
                ):
                    close(first - 1, "takeover")
                    begin(s, first)
                else:
                    keeper[f] = k

        if k < 0 and run >= seed_frames:
            begin(run_slot, max(f - run + 1, boundary))

    if k >= 0:
        f = len(players) - 1
        close(f, "period_end")
    reach = round(config.back_s * fps)
    for segment in segments:
        if reach > 0 and segment["end_reason"] != "takeover":
            _reach_back(
                segment, keeper, observed, jump, period, far, gap_frames, reach, taken
            )
    return segments


def _reach_back(segment, keeper, observed, jump, period, far, gap_frames, reach, taken):
    """Extend a seeded segment back along its track, in place (keeper and segment)."""
    slot, first = segment["slot"], segment["first"]
    earliest, gap, row = first, 0, first
    while first - row < reach:
        if observed[row, slot] and jump[row, slot]:
            break  # it got here faster than a person runs: someone else before
        f = row - 1
        if f < 0 or period[f] != period[first] or keeper[f] >= 0:
            break
        if taken is not None and taken[f] == slot:
            break
        if observed[f, slot]:
            if far[f, slot]:
                break
            earliest, gap = f, 0
        else:
            gap += 1
            if gap > gap_frames:
                break
        row = f
    if earliest < first:
        keeper[earliest:first] = slot
        segment["first"] = earliest


def _moved(players, slot, first, last):
    """Straight-line distance a track covered from row `first` to row `last`; NaN if unseen."""
    return np.hypot(*(players[last, slot] - players[first, slot]))


class _Contact:
    """Whether slot c came within `radius` of slot k during c's current stay in the seed zone.

    Each (c, k, stay) is scanned incrementally, so repeated checks over a long stay stay linear.
    """

    def __init__(self, players, zone_since, radius):
        self.players, self.zone_since, self.radius = players, zone_since, radius
        self._checked = {}  # (c, k, stay start) -> (last row checked, found)

    def met(self, c, k, f):
        key = (c, k, int(self.zone_since[f, c]))
        last, found = self._checked.get(key, (key[2] - 1, False))
        if not found and last < f:
            offset = (
                self.players[last + 1 : f + 1, c] - self.players[last + 1 : f + 1, k]
            )
            with np.errstate(invalid="ignore"):
                found = bool((np.hypot(*offset.T) <= self.radius).any())
        self._checked[key] = (max(last, f), found)
        return found


def _log(stale, segments, tracking, defends, n_slots):
    """One row per stale segment and per keeper segment, in video frames."""
    video_frame, team = tracking.video_frame, tracking.team
    rows = []
    for slot in range(stale.shape[1]):
        for first, last in _runs(stale[:, slot]):
            rows.append(
                {
                    "kind": "stale",
                    "slot": slot,
                    "goal": -1,
                    "to_slot": -1,
                    "first": first,
                    "last": last,
                    "from_team": int(team[slot]),
                    "to_team": int(team[slot]),
                    "end_reason": "",
                }
            )
    for segment in segments:
        defender = defends[int(tracking.period[segment["first"]])][segment["goal"]]
        rows.append(
            {
                "kind": "keeper",
                **segment,
                "to_slot": n_slots + defender,
                "from_team": int(team[segment["slot"]]),
                "to_team": defender,
            }
        )
    columns = [
        "kind",
        "slot",
        "goal",
        "to_slot",
        "first_frame",
        "last_frame",
        "n_frames",
        "from_team",
        "to_team",
        "end_reason",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    log = pd.DataFrame(rows)
    log["first_frame"] = video_frame[log["first"].to_numpy()]
    log["last_frame"] = video_frame[log["last"].to_numpy()]
    log["n_frames"] = log["last"] - log["first"] + 1
    return log[columns]


def _runs(mask):
    """[(first, last)] inclusive of each run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))
