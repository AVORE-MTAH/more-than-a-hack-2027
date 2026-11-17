"""Choose the dead-ball weights on Metrica's sample games: the dead table, the stoppage, the shortest stoppage.

    uv run python scripts/dead_weights_report.py --metrica <metrica>/data

External data only. Metrica's tracking is the one external source with a ball that goes
missing as a tactical camera's does and is tracked through stoppages, and its events time the
stoppages (mtah.io.metrica_csv.dead_intervals). Its live frames carry no team, so these games
score dead against live only: D's F1 (the ranking here) and the alive/dead F1, both from the
organisers' metrics. Every combination of GRID is labelled with the other values at their
defaults; the check leaving one game out picks on one game and scores on the other.

The DFL data cannot choose these weights: its ball freezes once play is dead, so its ball-out
stoppages are rarely seen. scripts/weights_report.py chooses the rest with these fixed.
"""

import argparse
import itertools
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtah.clean.tracks import clean_tracking
from mtah.io.metrica_csv import dead_intervals, load_metrica_csv
from mtah.predict.hmm import EXTERNAL_CALIBRATION, features, load_calibration
from mtah.tune.calibration import metrica_labels
from mtah.tune.weights import Setting, labels_for, pm

# Params
GRID = {
    "w_dead": (0.02, 0.05, 0.1, 0.2, 0.3, 0.5),
    "w_stoppage": (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0),
    "min_dead_s": (2.0, 3.0, 4.0),
}
GAMES = (1, 2)
D_CODE = 2  # possession_metrics' code for D


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--metrica", type=Path, default=os.environ.get("MTAH_METRICA_DIR")
    )
    parser.add_argument("--calibration", type=Path, default=EXTERNAL_CALIBRATION)
    parser.add_argument(
        "--grid",
        type=json.loads,
        default=None,
        help="JSON: {Setting field: [values]} in place of GRID (e.g. with w_stoppage)",
    )
    args = parser.parse_args(argv)
    if args.metrica is None:
        parser.error("--metrica (or MTAH_METRICA_DIR) is needed")
    calibration = load_calibration(args.calibration)
    games = [prepare(args.metrica, g) for g in GAMES]
    base = Setting()
    results = []
    grid = GRID if args.grid is None else {k: tuple(v) for k, v in args.grid.items()}
    names = list(grid)
    for values in itertools.product(*(grid[n] for n in names)):
        setting = replace(base, **dict(zip(names, values, strict=True)))
        results.append((setting, [score(g, calibration, setting) for g in games]))
    print("\n".join(report(results, base, names)))


def prepare(root, game):
    stem = root / f"Sample_Game_{game}" / f"Sample_Game_{game}_"
    t = load_metrica_csv(
        f"{stem}RawTrackingData_Home_Team.csv", f"{stem}RawTrackingData_Away_Team.csv"
    )
    dead, unknown = dead_intervals(f"{stem}RawEventsData.csv")
    labels = metrica_labels(t, dead, unknown)
    cleaned = clean_tracking(t).tracking
    n = len(labels)
    halves = []
    for p in np.unique(cleaned.period[cleaned.period > 0]):
        frames = cleaned.video_frame[cleaned.period == p]
        halves.append((int(frames.min()), int(frames.max())))
    return {
        "features": {Setting().flight: features(cleaned, n)},
        "labels": labels,
        "halves": halves,
        "fps": cleaned.fps,
    }


def score(game, calibration, setting):
    """{d_f1, alive_dead_f1} of one game."""
    truth = pm.label_codes(np.asarray(game["labels"], dtype=object))
    predicted = pm.label_codes(
        np.asarray(labels_for(game, calibration, setting), dtype=object)
    )
    cm3 = pm.scored_confusion(pm.confusion(truth, predicted))
    _, recall, f1 = pm.per_class_prf(cm3, D_CODE)
    return {
        "d_f1": float(f1),
        "d_recall": float(recall),
        "alive_dead_f1": float(pm.alive_dead_f1(cm3)),
    }


def report(results, base, names):
    mean = [np.mean([s["d_f1"] for s in scores]) for _, scores in results]
    order = np.argsort(mean)[::-1]
    lines = [
        "D's F1 per game, then alive/dead F1 per game (the best ten, and the defaults):"
    ]

    def row(k):
        setting, scores = results[k]
        values = ", ".join(f"{n} {getattr(setting, n):g}" for n in names)
        d = " ".join(f"{s['d_f1']:.3f}" for s in scores)
        ad = " ".join(f"{s['alive_dead_f1']:.3f}" for s in scores)
        rec = " ".join(f"{s['d_recall']:.2f}" for s in scores)
        return (
            f"  {values}: D F1 {d} (mean {mean[k]:.3f}; recall {rec}), alive/dead {ad}"
        )

    lines += [row(k) for k in order[:10]]
    defaults = [k for k, (s, _) in enumerate(results) if s == base]
    if defaults:
        lines += ["", "the defaults:", row(defaults[0])]
    n_games = len(results[0][1])
    lines += ["", "leaving one game out (chosen on the other, scored on it):"]
    for held in range(n_games):
        others = [g for g in range(n_games) if g != held]
        best = max(
            range(len(results)),
            key=lambda k: np.mean([results[k][1][g]["d_f1"] for g in others]),
        )
        setting, scores = results[best]
        values = ", ".join(f"{n} {getattr(setting, n):g}" for n in names)
        lines.append(f"  game {held + 1}: {values} -> D F1 {scores[held]['d_f1']:.3f}")
    return lines


if __name__ == "__main__":
    main()
