"""Score the hybrid on every external source, each held out of the classifier's fit.

    uv run python scripts/hybrid_report.py --dfl <dfl> \
        --skillcorner <skillcorner>/data/matches \
        --metrica <metrica>/data

The classifier is fitted as scripts/train_classifier.py fits it (SkillCorner's 20 matches and
Metrica's games 1 and 2, those weighed METRICA_WEIGHT), but never on what it is scored on:

    SkillCorner   out of fold: 4 folds by match, each fitted on the other 15 (and Metrica)
    Metrica       one game out: fitted on SkillCorner and the other game; D's F1
    DFL           not in any fit: fitted on all of the above (DFL's ball removed as Metrica's)

Each row is the model's evidence with the classifier's scaled likelihoods added at a weight
(--weights; 0 is the model without it). The weight shipped (mtah.predict.classifier.W_CLF) is the best one
that loses on no source. External data only.
"""

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dead_weights_report import prepare as prepare_metrica
from dead_weights_report import score as score_metrica
from train_classifier import METRICA_WEIGHT
from weights_report import SEED, missing_rule

from mtah.clean.tracks import clean_tracking
from mtah.io import dfl, skillcorner
from mtah.io.metrica_csv import load_metrica_csv
from mtah.predict import classifier as clf
from mtah.predict.hmm import EXTERNAL_CALIBRATION, features, load_calibration
from mtah.tune.missing import degrade
from mtah.tune.weights import FLIGHTS, Setting, halves_of, prepare, score_match

# Params
FOLDS = 4


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl", type=Path, default=os.environ.get("MTAH_DFL_DIR"))
    parser.add_argument(
        "--skillcorner", type=Path, default=os.environ.get("MTAH_SKILLCORNER_DIR")
    )
    parser.add_argument(
        "--metrica", type=Path, default=os.environ.get("MTAH_METRICA_DIR")
    )
    parser.add_argument("--calibration", type=Path, default=EXTERNAL_CALIBRATION)
    parser.add_argument(
        "--weights", type=float, nargs="*", default=[0.0, 0.05, 0.1, 0.2]
    )
    args = parser.parse_args(argv)
    if None in (args.dfl, args.skillcorner, args.metrica):
        parser.error("--dfl, --skillcorner and --metrica are needed")
    cal = load_calibration(args.calibration)
    base = Setting(w_clf=0.0)

    sc = []
    for mid in skillcorner.match_ids(args.skillcorner):
        m = skillcorner.upsampled(skillcorner.load_match(args.skillcorner, mid))
        cleaned = clean_tracking(m.tracking).tracking
        n = len(m.labels)
        sc.append(
            (
                {
                    "features": {base.flight: features(cleaned, n)},
                    "labels": np.asarray(m.labels),
                    "halves": halves_of(cleaned.period),
                    "fps": cleaned.fps,
                },
                clf.frame_table(cleaned, n),
            )
        )
    games, met_tables, met_labels = [], [], []
    for g in (1, 2):
        p = prepare_metrica(args.metrica, g)
        stem = args.metrica / f"Sample_Game_{g}" / f"Sample_Game_{g}_"
        t = load_metrica_csv(
            f"{stem}RawTrackingData_Home_Team.csv",
            f"{stem}RawTrackingData_Away_Team.csv",
        )
        games.append(p)
        met_tables.append(clf.frame_table(clean_tracking(t).tracking, len(p["labels"])))
        met_labels.append(np.where(p["labels"] == "H", "L", p["labels"]))
    rule = missing_rule(args.metrica)
    flights = {base.flight: FLIGHTS[base.flight]}
    dfl_prepared = []
    for k, mid in enumerate(dfl.match_files(args.dfl)):
        m = dfl.load_match(args.dfl, mid, cache_dir=args.dfl / "cache")
        p = prepare(m, rule, SEED + k, flights)
        ball = degrade(
            m.tracking.ball,
            m.labels,
            m.tracking.fps,
            rule,
            np.random.default_rng(SEED + k),
        )
        cleaned = clean_tracking(replace(m.tracking, ball=ball)).tracking
        dfl_prepared.append((p, clf.frame_table(cleaned, len(m.labels))))

    sc_tables = [t for _, t in sc]
    sc_labels = [p["labels"] for p, _ in sc]

    def fitted(idx, metrica):
        return clf.fit(
            [sc_tables[i] for i in idx] + [met_tables[g] for g in metrica],
            [sc_labels[i] for i in idx] + [met_labels[g] for g in metrica],
            weights=[1.0] * len(idx) + [METRICA_WEIGHT] * len(metrica),
        )

    every = list(range(len(sc)))
    for k in range(FOLDS):
        model = fitted([i for i in every if i % FOLDS != k], [0, 1])
        for i, (p, t) in enumerate(sc):
            if i % FOLDS == k:
                p["clf_ll"] = clf.scaled_loglik(model, t)
    for held in (0, 1):
        model = fitted(every, [1 - held])
        games[held]["clf_ll"] = clf.scaled_loglik(model, met_tables[held])
    model = fitted(every, [0, 1])
    for p, t in dfl_prepared:
        p["clf_ll"] = clf.scaled_loglik(model, t)

    lines = [
        (
            "classifier weight: SkillCorner spell / macro F1 (out of fold), DFL spell / macro"
            " F1, Metrica D's F1 per game (one game out)"
        )
    ]
    for w in args.weights:
        s = replace(base, w_clf=w)
        a = [score_match(p, cal, s) for p, _ in sc]
        b = [score_match(p, cal, s) for p, _ in dfl_prepared]
        c = [score_metrica(p, cal, s)["d_f1"] for p in games]
        lines.append(
            f"  {w:<5g} SkillCorner {np.mean([x['spell_f1'] for x in a]):.4f} /"
            f" {np.mean([x['macro_f1'] for x in a]):.4f}"
            f"   DFL {np.mean([x['spell_f1'] for x in b]):.4f} /"
            f" {np.mean([x['macro_f1'] for x in b]):.4f}"
            f"   Metrica {c[0]:.3f} {c[1]:.3f} (mean {np.mean(c):.3f})"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
