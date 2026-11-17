"""The model's calibration (mtah.predict.hmm.calibrate) from external data only.

Each part is taken from the source that measures it properly (docs/tunable-values.md):

    possession  the team table, the transition rates, the kicker's agreement, the possession
              lengths: they need possession (H or A) on every frame, and a seen ball
              (POSSESSION_KEYS). From SkillCorner's phases of play when given (a team keeps
              the ball until the other one gains control, the challenge's controlled
              possession), else from DFL's possession (it changes on nearly every touch)
    Metrica   the dead table and the ball-out stoppage shares: they need a ball that goes
              missing as a tactical camera's does (39-41% of frames there, mostly while dead),
              and dead against live from the events (METRICA_KEYS)

SkillCorner's matches come at 10 fps; they are upsampled to 25 (mtah.io.skillcorner.upsampled)
before their features are computed, as the rates are per frame.

Each source's matches are cleaned as the submission's tracking is (mtah.clean.tracks), their
features computed (mtah.predict.hmm.features), and joined end to end with one X frame between two
matches, so that no transition is counted across them; then calibrate() runs once per source.
Metrica's live frames carry no team (its events say who passes, not who has the ball frame by
frame), so they are all labelled H there: harmless, as only its dead and stoppage parts are
kept.
"""

from dataclasses import fields

import numpy as np

from mtah.clean.tracks import clean_tracking
from mtah.predict.hmm import Features, calibrate, features

# Params
ATTRIBUTION = (
    "Fitted on external data only. DFL: Bassek et al., 'An integrated dataset of"
    " spatiotemporal and event data in elite soccer', Scientific Data 12, 195 (2025),"
    " doi:10.1038/s41597-025-04505-y, figshare doi:10.6084/m9.figshare.28196177, CC BY 4.0,"
    " data owner Deutsche Fussball Liga. Metrica Sports sample data,"
    " github.com/metrica-sports/sample-data."
)
SKILLCORNER_ATTRIBUTION = (
    " SkillCorner open data, github.com/SkillCorner/opendata, MIT licence."
)
POSSESSION_KEYS = (
    "team_agree",
    "switch",
    "stop",
    "restart",
    "counts",
    "possession_frames",
    "kicker_agree",
    "kicked_frames",
)
METRICA_KEYS = (
    "dead_given_dead",
    "dead_given_live",
    "stoppage_given_dead",
    "stoppage_given_live",
)


def match_features(tracking, labels, clean=True):
    """(Features, labels) of one match: labels per video frame 0..n-1, as features gives."""
    if clean:
        tracking = clean_tracking(tracking).tracking
    n = int(tracking.video_frame.max()) + 1 if len(tracking.video_frame) else 0
    labels = np.asarray(labels, dtype="<U1")
    if len(labels) != n:
        raise ValueError(f"{len(labels)} labels for {n} video frames")
    return features(tracking, n), labels


def joined(pairs):
    """One (Features, labels) of several, with an X frame between two."""
    parts, labels = {}, []
    names = [f.name for f in fields(Features)]
    for k, (feats, lab) in enumerate(pairs):
        if k:
            for name in names:
                values = getattr(feats, name)
                if values is not None:
                    parts.setdefault(name, []).append(_gap(values))
            labels.append(np.array(["X"]))
        for name in names:
            values = getattr(feats, name)
            if values is not None:
                parts.setdefault(name, []).append(values)
        labels.append(lab)
    return (
        Features(**{name: np.concatenate(v) for name, v in parts.items()}),
        np.concatenate(labels) if labels else np.zeros(0, "<U1"),
    )


def _gap(values):
    return (
        np.zeros(1, dtype=bool)
        if values.dtype == bool
        else np.full(1, -1, values.dtype)
    )


def merged(possession_calibration, metrica_calibration, possession="dfl"):
    """The calibration: POSSESSION_KEYS from the first (from `possession`: "dfl" or
    "skillcorner"), METRICA_KEYS from the second."""
    out = {
        key: possession_calibration[key]
        for key in POSSESSION_KEYS
        if key in possession_calibration
    }
    out.update({key: metrica_calibration[key] for key in METRICA_KEYS})
    attribution = ATTRIBUTION
    if possession == "skillcorner":
        attribution += SKILLCORNER_ATTRIBUTION
    out["sources"] = {
        "attribution": attribution,
        possession: list(POSSESSION_KEYS),
        "metrica": list(METRICA_KEYS),
        f"{possession}_frames": possession_calibration["counts"],
        "metrica_frames": {
            "live_frames": metrica_calibration["counts"]["live_frames"],
            "dead_frames": metrica_calibration["counts"]["dead_frames"],
        },
    }
    return out


def metrica_labels(tracking, dead, unknown):
    """Per video frame of a Metrica game: D in dead_intervals' stretches, X in its unknown
    ones and where the tracking has no row or no period, H elsewhere (live, team unknown)."""
    n = int(tracking.video_frame.max()) + 1
    labels = np.full(n, "X", dtype="<U1")
    in_play = np.zeros(n, dtype=bool)
    in_play[tracking.video_frame[tracking.period > 0]] = True
    labels[in_play] = "H"
    for first, last in dead:
        labels[first : last + 1] = np.where(in_play[first : last + 1], "D", "X")
    for first, last in unknown:
        labels[first : last + 1] = "X"
    return labels


def calibrate_sources(dfl_pairs, metrica_pairs, skillcorner_pairs=None):
    """(merged calibration, the possession source's, Metrica's) from lists of (Features,
    labels): the possession tables from SkillCorner's pairs if given, else from DFL's."""
    metrica_cal = calibrate(*joined(metrica_pairs))
    if skillcorner_pairs:
        own = calibrate(*joined(skillcorner_pairs))
        return merged(own, metrica_cal, "skillcorner"), own, metrica_cal
    own = calibrate(*joined(dfl_pairs))
    return merged(own, metrica_cal), own, metrica_cal
