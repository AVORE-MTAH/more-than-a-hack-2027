"""Possession as a hidden semi-Markov model over the tracking's signals.

Hard rules (a speed threshold for dead ball, control within a radius, a minimum run of control)
make one decision per signal, so several weak signals that each fall short (a slow pitch, a
still ball, a contested touch) can never add up. Here the signals are evidence in a model:

    states        H, A, D. Each is a chain of sub-states sharing its emissions: entering a
                  state walks the chain, so it lasts at least min_live_s / min_dead_s, then a
                  self-looping last sub-state: a minimum plus a geometric tail (semi-Markov
                  durations at plain Viterbi cost)
    transitions   from a last sub-state only: switch (H to A, A to H), stop (live to D),
                  restart (D to H or A); per-frame rates from a labelling (calibrate)
    emissions     a weighted sum of log-likelihood terms (naive Bayes, tempered by the weights
                  because consecutive frames and correlated signals are not independent):
                    team      the nearest player's team when the ball is seen, graded by its
                              distance: P(nearest is own team | live, distance bucket) for H and
                              A, a coin for D. A soft control in place of a fixed radius
                    dead      ball seen, missing but dribbled (dead.score.dribbling, a lull
                              between two dribbled stretches of one gap included:
                              bridge_dribbles), or missing by median player speed bucket
                              (dead.features.median_speed): P(feature | D) against
                              P(feature | live). The pitch's speed counts only when the ball is
                              missing: a seen ball on a slow pitch is a goal kick in one window
                              and build-up play in the next
                    stoppage  whether a ball-out stoppage is under way (dead.score.
                              out_stoppages), weighed apart from the dead table
                    kick      each flight of the ball (mtah.predict.flight.in_flight: fast and
                              straight, from a kick to its reception) with a kicker
                              (mtah.predict.flight.kickers) once, as an event: w_kick times
                              log P(kicker's team keeps it) in all, spread over the flight's
                              frames, so a short pass weighs as much as a long one; a short
                              possession (a touch and a pass) has too few frames to pay for
                              its two switches without it
                    classifier  the hybrid's per-frame classifier (mtah.predict.classifier):
                              its scaled likelihoods, weighed by w_clf
    decoding      Viterbi per half; the dead ranges given are forced D, frames outside the
                  halves are X.

A calibration is a dict of likelihood tables and rates, made by calibrate() from a labelling. The
one the submission uses is fitted on external data only (scripts/calibrate_external.py) and
ships with the code (EXTERNAL_CALIBRATION).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mtah.dead.features import median_speed
from mtah.dead.score import bridge_dribbles, dribbling, out_stoppages, runs
from mtah.predict.classifier import frame_table, scaled_loglik
from mtah.predict.flight import in_flight, kickers
from mtah.predict.possession import nearest

# Params
DISTANCE_EDGES = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)  # m: 7 buckets
SPEED_EDGES = (1.0, 1.5, 2.0)  # m/s: 4 buckets, for a missing ball
# Fitted on external data by scripts/weights_report.py --skillcorner: SkillCorner's 20 matches at
# 25 fps, scored by spell F1 against their phases of play, controlled possession
# (docs/tunable-values.md).
MIN_LIVE_S = 3.0
MIN_DEAD_S = 3.0  # with W_DEAD and W_STOPPAGE: scripts/dead_weights_report.py (Metrica)
W_TEAM = 0.1
W_DEAD = 0.1  # Metrica (scripts/dead_weights_report.py)
# The ball-out stoppage term, weighed apart from the dead table; Metrica, as W_DEAD: the DFL
# ball freezes once play is dead, so its stoppages could not choose these.
W_STOPPAGE = 0.2
W_KICK = 4.0

# The calibration fitted on external data only (scripts/calibrate_external.py), shipped with the
# code so that the submission is reproduced from the repository alone.
EXTERNAL_CALIBRATION = Path(__file__).with_name("external_calibration.json")

LABELS = np.array(["H", "A", "D"])
H, A, D = 0, 1, 2
N_DISTANCE = len(DISTANCE_EDGES) + 1
N_SPEED = len(SPEED_EDGES) + 1
# Dead categories: a missing ball is MISSING + its speed bucket.
SEEN, DRIBBLED, MISSING = 0, 1, 2
N_DEAD = MISSING + N_SPEED


@dataclass(frozen=True)
class Features:
    """Per video frame; -1 where there is nothing to say (no row, ball missing, no speed).

    nearest   team code of the nearest player, ball seen
    distance  its distance bucket (DISTANCE_EDGES)
    dead      SEEN, DRIBBLED, or MISSING + speed bucket
    stoppage  inside a ball-out stoppage, before its restart
    flight    the ball in flight (mtah.predict.flight.in_flight), or None
    kicker    in flight, the kicking team's code (mtah.predict.flight.kickers), or None
    """

    nearest: np.ndarray
    distance: np.ndarray
    dead: np.ndarray
    stoppage: np.ndarray
    flight: np.ndarray | None = None
    kicker: np.ndarray | None = None


def features(tracking, n_frames, flight=None):
    """Features of every video frame 0..n_frames-1 from (cleaned) tracking; flight, a dict of
    in_flight's thresholds (speed, shortest_s, kink_deg), for other than its defaults."""
    players, ball, fps = tracking.players, tracking.ball, tracking.fps
    codes, distance, _ = nearest(players, tracking.team, ball)
    speed = median_speed(players, fps)
    missing = np.isnan(ball).any(axis=1)
    dribbled = bridge_dribbles(dribbling(players, ball, fps), missing)
    speed_bucket = np.searchsorted(SPEED_EDGES, speed, side="right")
    category = np.where(
        missing,
        np.where(dribbled, DRIBBLED, MISSING + speed_bucket),
        SEEN,
    )
    dead = np.where(np.isnan(speed), -1, category)
    stopped = out_stoppages(ball, speed, tracking.period, fps, tracking.pitch_size)
    seen = codes >= 0
    flying = in_flight(ball, fps, tracking.period, **(flight or {}))
    rows = {
        "nearest": np.where(seen, codes, -1),
        "distance": np.where(
            seen, np.searchsorted(DISTANCE_EDGES, distance, side="right"), -1
        ),
        "dead": dead,
        "stoppage": stopped,
        "flight": flying,
        "kicker": kickers(flying, codes, distance, fps),
    }

    frame = tracking.video_frame
    inside = (frame >= 0) & (frame < n_frames)
    out = {}
    for name, values in rows.items():
        flag = name in ("stoppage", "flight")
        out[name] = np.full(n_frames, False if flag else -1, bool if flag else np.int8)
        out[name][frame[inside]] = values[inside]
    return Features(**out)


def calibrate(feats, labels):
    """Likelihood tables and transition rates from a labelling (one label per video frame).

    Counts are add-one smoothed. Only frames labelled H, A or D are used. With flight, also how
    often the kicking team is the possessing one in flight.
    """
    team = np.where(labels == "H", H, np.where(labels == "A", A, -1))
    live, dead = team >= 0, labels == "D"

    agree, total = np.ones(N_DISTANCE), 2 * np.ones(N_DISTANCE)
    seen = live & (feats.nearest >= 0)
    np.add.at(total, feats.distance[seen], 1)
    np.add.at(agree, feats.distance[seen & (feats.nearest == team)], 1)

    def table(mask):
        counts = np.ones(N_DEAD)
        known = mask & (feats.dead >= 0)
        np.add.at(counts, feats.dead[known], 1)
        return (counts / counts.sum()).tolist()

    def share(mask):
        return float((np.sum(feats.stoppage & mask) + 1) / (np.sum(mask) + 2))

    graded = live | dead
    pairs = graded[1:] & graded[:-1]
    before, after = labels[:-1][pairs], labels[1:][pairs]
    live_before = np.isin(before, ("H", "A"))
    switches = np.sum(live_before & np.isin(after, ("H", "A")) & (before != after))
    stops = np.sum(live_before & (after == "D"))
    restarts = np.sum((before == "D") & np.isin(after, ("H", "A")))
    extra = {}
    if feats.kicker is not None:
        kicked = live & (feats.kicker >= 0)
        extra["kicker_agree"] = float(
            (np.sum(kicked & (feats.kicker == team)) + 1) / (np.sum(kicked) + 2)
        )
        extra["kicked_frames"] = int(np.sum(kicked))
    return {
        "team_agree": (agree / total).tolist(),
        "dead_given_dead": table(dead),
        "dead_given_live": table(live),
        "stoppage_given_dead": share(dead),
        "stoppage_given_live": share(live),
        "switch": float((switches + 1) / (live_before.sum() + 2)),
        "stop": float((stops + 1) / (live_before.sum() + 2)),
        "restart": float((restarts + 1) / ((before == "D").sum() + 2)),
        "counts": {
            "live_frames": int(live.sum()),
            "dead_frames": int(dead.sum()),
            "switches": int(switches),
            "stops": int(stops),
            "restarts": int(restarts),
        },
        "possession_frames": possession_frames(team),
        **extra,
    }


def possession_frames(team):
    """The length in frames of each possession: a maximal run of one team (codes, -1 for
    neither)."""
    team = np.asarray(team)
    if not len(team):
        return []
    change = np.flatnonzero(np.diff(team) != 0) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(team)]
    return [int(e - s) for s, e in zip(starts, ends) if team[s] >= 0]


def save_calibration(calibration, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2), encoding="utf-8", newline="\n")


def load_calibration(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evidence(
    feats,
    calibration,
    w_team=W_TEAM,
    w_dead=W_DEAD,
    w_kick=W_KICK,
    w_stoppage=W_STOPPAGE,
):
    """(n_frames, 3) log-likelihoods of H, A, D per frame, tempered by the weights.

    w_team weighs the team term, w_dead the dead table (the ball seen, dribbled, or missing by
    the pitch's pace), w_stoppage the ball-out stoppage, and w_kick each flight's kick once over
    the flight, with a calibration that has kicker_agree.
    """
    if len(calibration["dead_given_dead"]) != N_DEAD:
        raise ValueError(
            "the calibration has another dead-feature layout: run calibrate_external.py again"
        )
    n = len(feats.nearest)
    ll = np.zeros((n, 3))

    seen = feats.nearest >= 0
    p = np.asarray(calibration["team_agree"])[feats.distance[seen]]
    own_home = feats.nearest[seen] == H
    team = np.full(n, float(w_team))[seen]
    ll[seen, H] = team * np.log(np.where(own_home, p, 1 - p))
    ll[seen, A] = team * np.log(np.where(own_home, 1 - p, p))
    ll[seen, D] = team * np.log(0.5)

    if feats.kicker is not None and w_kick and "kicker_agree" in calibration:
        q = calibration["kicker_agree"]
        for first, last in runs(feats.kicker >= 0):
            span = slice(first, last + 1)
            home = feats.kicker[first] == H
            share = w_kick / (last - first + 1)
            ll[span, H] += share * np.log(q if home else 1 - q)
            ll[span, A] += share * np.log(1 - q if home else q)
            ll[span, D] += share * np.log(0.5)

    known = feats.dead >= 0
    given_live = np.log(np.asarray(calibration["dead_given_live"]))[feats.dead[known]]
    given_dead = np.log(np.asarray(calibration["dead_given_dead"]))[feats.dead[known]]
    ll[known, H] += w_dead * given_live
    ll[known, A] += w_dead * given_live
    ll[known, D] += w_dead * given_dead

    s_live, s_dead = (
        calibration["stoppage_given_live"],
        calibration["stoppage_given_dead"],
    )
    stop = feats.stoppage
    live_term = np.where(stop, np.log(s_live), np.log(1 - s_live))
    ll[:, H] += w_stoppage * live_term
    ll[:, A] += w_stoppage * live_term
    ll[:, D] += w_stoppage * np.where(stop, np.log(s_dead), np.log(1 - s_dead))
    return ll


@dataclass(frozen=True)
class Chains:
    """The state layout and transition log-probabilities of the model."""

    min_live: int
    min_dead: int
    log_switch: float
    log_stop: float
    log_restart: float  # into one team: half the restart rate

    @classmethod
    def from_calibration(
        cls, calibration, fps, min_live_s=MIN_LIVE_S, min_dead_s=MIN_DEAD_S
    ):
        return cls(
            min_live=max(1, round(min_live_s * fps)),
            min_dead=max(1, round(min_dead_s * fps)),
            log_switch=float(np.log(calibration["switch"])),
            log_stop=float(np.log(calibration["stop"])),
            log_restart=float(np.log(calibration["restart"] / 2)),
        )


def viterbi(ll, chains):
    """Most likely label per frame (H, A, D codes) for (frames, 3) log-likelihoods.

    The path starts in the last sub-state of any label, so the first stretch has no minimum,
    and may end anywhere, so the last one may be cut short.
    """
    m = [chains.min_live, chains.min_live, chains.min_dead]
    start = np.cumsum([0] + m[:-1])
    last = start + np.array(m) - 1
    label = np.repeat([H, A, D], m)
    n_states = int(sum(m))

    n = len(ll)
    # Capped so that staying stays possible.
    stops = np.minimum(np.full(n, chains.log_stop), np.log(0.5))
    restarts = np.minimum(np.full(n, chains.log_restart), np.log(0.25))
    switches = np.minimum(np.full(n, chains.log_switch), np.log(0.25))
    stay_live = np.log1p(-(np.exp(switches) + np.exp(stops)))
    stay_dead = np.log1p(-2 * np.exp(restarts))
    # Within a chain each sub-state but the first comes from the one before it.
    pred = np.arange(n_states) - 1
    pred[start] = 0

    back = np.zeros((n, n_states), dtype=np.int16)
    score = np.full(n_states, -np.inf)
    score[last] = ll[0]
    for t in range(1, n):
        new = score[pred]
        back[t] = pred
        stay = (stay_live[t], stay_live[t], stay_dead[t])
        # Entries into each label's first sub-state: (from label, log-probability).
        entries = {
            H: ((A, switches[t]), (D, restarts[t])),
            A: ((H, switches[t]), (D, restarts[t])),
            D: ((H, stops[t]), (A, stops[t])),
        }
        for k in (H, A, D):
            first, end = start[k], last[k]
            best, arg = (new[end], pred[end]) if m[k] > 1 else (-np.inf, 0)
            if score[end] + stay[k] > best:
                best, arg = score[end] + stay[k], end
            if m[k] > 1:
                new[end], back[t, end] = best, arg
                best, arg = -np.inf, 0
            for source, log_p in entries[k]:
                value = score[last[source]] + log_p
                if value > best:
                    best, arg = value, last[source]
            new[first], back[t, first] = best, arg
        score = new + ll[t][label]

    states = np.empty(n, dtype=np.int64)
    states[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return label[states]


def hmm_labels(
    tracking,
    n_frames,
    halves,
    dead_ranges,
    calibration,
    w_team=W_TEAM,
    w_dead=W_DEAD,
    min_live_s=MIN_LIVE_S,
    min_dead_s=MIN_DEAD_S,
    w_kick=W_KICK,
    w_stoppage=W_STOPPAGE,
    classifier=None,
    w_clf=0.0,
):
    """One label per video frame, 0..n_frames-1 (H, A, D inside the halves, X outside them;
    D throughout dead_ranges), by the model.

    classifier: the hybrid's per-frame classifier (mtah.predict.classifier.load_model), its
    scaled likelihoods added to the evidence by w_clf.
    """
    for first, last in list(halves) + list(dead_ranges):
        if not 0 <= first <= last < n_frames:
            raise ValueError(f"range ({first}, {last}) is outside 0..{n_frames - 1}")
    feats = features(tracking, n_frames)
    chains = Chains.from_calibration(calibration, tracking.fps, min_live_s, min_dead_s)
    return labels_from_features(
        feats,
        halves,
        dead_ranges,
        calibration,
        chains,
        w_team,
        w_dead,
        w_kick,
        w_stoppage,
        extra_ll=(
            w_clf * scaled_loglik(classifier, frame_table(tracking, n_frames))
            if classifier is not None and w_clf
            else None
        ),
    )


def labels_from_features(
    feats,
    halves,
    dead_ranges,
    calibration,
    chains,
    w_team,
    w_dead,
    w_kick=W_KICK,
    w_stoppage=W_STOPPAGE,
    extra_ll=None,
):
    """hmm_labels from features already computed, so several weightings can share them.

    extra_ll, (frames, 3) log-likelihoods of H, A, D, is added to the evidence: the hybrid's
    classifier term (mtah.predict.classifier.scaled_loglik, weighed).
    """
    ll = evidence(feats, calibration, w_team, w_dead, w_kick, w_stoppage)
    if extra_ll is not None:
        ll = ll + extra_ll
    for first, last in dead_ranges:
        ll[first : last + 1, [H, A]] = -np.inf
    labels = np.full(len(ll), "X", dtype="<U1")
    for first, last in halves:
        span = slice(first, last + 1)
        labels[span] = LABELS[viterbi(ll[span], chains)]
    return labels
