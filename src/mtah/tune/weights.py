"""Choosing the model's weights on external data: the DFL matches, their ball removed as Metrica's.

The evaluation set needs possession on every frame and a ball that goes missing as a tactical
camera's does; no external dataset has both. So each DFL match (exact H, A and D) has its ball
removed by mtah.tune.missing's rule, fitted on Metrica's sample games (about 7% of live frames and
91% of dead ones), with a fixed seed per match. The matches are then cleaned, their features
computed (once per flight setting tried), as the submission's would be.

Possession weights are scored on SkillCorner's matches instead (prepare_skillcorner): their phases of play
change team only when the other team gains control, the challenge's controlled possession,
where DFL's possession changes on nearly every touch. Their ball is a broadcast's, missing where
the camera loses it, and is used as it is.

A setting (Setting) is every value the decoding takes: the weights, the shortest possession and
stoppage, the flight thresholds. score_match() labels one match with it and scores it with the
organisers' metrics (evaluation/possession_metrics.py: spell F1, the ranking metric, and macro
F1).
"""

import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from mtah.clean.tracks import clean_tracking
from mtah.io import skillcorner
from mtah.predict.classifier import W_CLF
from mtah.predict.hmm import (
    MIN_DEAD_S,
    MIN_LIVE_S,
    W_DEAD,
    W_KICK,
    W_STOPPAGE,
    W_TEAM,
    Chains,
    features,
    labels_from_features,
)
from mtah.tune.missing import degrade

EVALUATION = Path(__file__).resolve().parents[3] / "evaluation"
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))
import possession_metrics as pm

# Params
FLIGHTS = {  # the flight settings features are computed for: (speed m/s, shortest s)
    "8/0.3": {"speed": 8.0, "shortest_s": 0.3},
    "6/0.3": {"speed": 6.0, "shortest_s": 0.3},
    "6/0.5": {"speed": 6.0, "shortest_s": 0.5},
    "4/0.5": {"speed": 4.0, "shortest_s": 0.5},
    "10/0.3": {"speed": 10.0, "shortest_s": 0.3},
    "4/0.8": {"speed": 4.0, "shortest_s": 0.8},
    "3/0.5": {"speed": 3.0, "shortest_s": 0.5},
    "2/0.5": {"speed": 2.0, "shortest_s": 0.5},
}


@dataclass(frozen=True)
class Setting:
    w_team: float = W_TEAM
    w_dead: float = W_DEAD
    w_stoppage: float = W_STOPPAGE
    w_clf: float = W_CLF  # the hybrid's classifier term, with prepared["clf_ll"]
    w_kick: float = W_KICK
    min_live_s: float = MIN_LIVE_S
    min_dead_s: float = MIN_DEAD_S
    flight: str = "4/0.5"  # the module defaults (FLIGHT_SPEED, FLIGHT_S)

    def key(self):
        return tuple(sorted(asdict(self).items()))


def halves_of(period):
    """[(first, last)] rows of each period."""
    out = []
    for p in np.unique(period[period > 0]):
        rows = np.flatnonzero(period == p)
        out.append((int(rows[0]), int(rows[-1])))
    return out


def prepare(match, missing_fit, seed, flights=FLIGHTS):
    """What score_match needs of one DflMatch: features per flight setting, labels, halves,
    fps."""
    t = match.tracking
    rng = np.random.default_rng(seed)
    ball = degrade(t.ball, match.labels, t.fps, missing_fit, rng)
    cleaned = clean_tracking(replace(t, ball=ball)).tracking
    n = len(match.labels)
    halves = halves_of(cleaned.period)
    return {
        "features": {
            name: features(cleaned, n, flight=params)
            for name, params in flights.items()
        },
        "labels": np.asarray(match.labels),
        "halves": halves,
        "fps": cleaned.fps,
    }


def prepare_skillcorner(match, flights=FLIGHTS):
    """What score_match needs of one SkillCornerMatch, upsampled to 25 fps: the labels are its
    phases of play (controlled possession), the ball its own (a broadcast's: seen where
    detected), so nothing is removed."""
    match = skillcorner.upsampled(match)
    cleaned = clean_tracking(match.tracking).tracking
    n = len(match.labels)
    halves = halves_of(cleaned.period)
    return {
        "features": {
            name: features(cleaned, n, flight=params)
            for name, params in flights.items()
        },
        "labels": np.asarray(match.labels),
        "halves": halves,
        "fps": cleaned.fps,
    }


def labels_for(prepared, calibration, setting):
    chains = Chains.from_calibration(
        calibration, prepared["fps"], setting.min_live_s, setting.min_dead_s
    )
    return labels_from_features(
        prepared["features"][setting.flight],
        prepared["halves"],
        [],
        calibration,
        chains,
        setting.w_team,
        setting.w_dead,
        w_kick=setting.w_kick,
        w_stoppage=setting.w_stoppage,
        extra_ll=(
            setting.w_clf * prepared["clf_ll"]
            if setting.w_clf and prepared.get("clf_ll") is not None
            else None
        ),
    )


def scores(truth, predicted):
    """{spell_f1, macro_f1, n_gt, n_pred} by the organisers' metrics."""
    gt = pm.label_codes(np.asarray(truth, dtype=object))
    pred = pm.label_codes(np.asarray(predicted, dtype=object))
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    runs = pm.valid_runs(gt)
    spells = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)
    return {
        "spell_f1": float(spells["mean_f1"]),
        "macro_f1": float(pm.macro_f1(cm3)),
        "n_gt": int(spells["n_gt"]),
        "n_pred": int(spells["n_pred"]),
    }


def score_match(prepared, calibration, setting):
    return scores(prepared["labels"], labels_for(prepared, calibration, setting))


def leave_one_out(table):
    """For each match, the setting best on the others (mean spell F1) and its score on that
    match. table: {setting key: [per-match scores]}. Returns [(match index, key, spell_f1)]."""
    keys = list(table)
    n = len(next(iter(table.values())))
    out = []
    for held in range(n):
        best = max(
            keys,
            key=lambda k: np.mean(
                [table[k][i]["spell_f1"] for i in range(n) if i != held]
            ),
        )
        out.append((held, best, table[best][held]["spell_f1"]))
    return out
