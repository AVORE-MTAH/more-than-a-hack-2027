"""Fit the hybrid's per-frame classifier on external data and write it as JSON.

    uv run python scripts/train_classifier.py \
        --skillcorner <skillcorner>/data/matches \
        --metrica <metrica>/data

SkillCorner's 20 matches (upsampled to 25 fps, labelled by their phases of play: H, A, D) and
Metrica's sample games 1 and 2 (dead against live from their events; their live frames carry
no team, so they count as "H or A": mtah.predict.classifier.fit), the Metrica frames weighed
METRICA_WEIGHT times: they are the one tactical feed whose ball goes missing as the provided
match's format does, and without them the classifier learns a broadcast's stoppages. Every
match is cleaned as the submission's tracking is. The provided match is never read.

The fit is deterministic (the same frames, L-BFGS from zeros), so the file is reproduced by
running this again. By default it is written to src/mtah/predict/frame_classifier.json, which
make_submission.py reads.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtah.clean.tracks import clean_tracking
from mtah.io import skillcorner
from mtah.io.metrica_csv import dead_intervals, load_metrica_csv
from mtah.predict import classifier as clf
from mtah.tune.calibration import ATTRIBUTION, SKILLCORNER_ATTRIBUTION, metrica_labels

# Params
METRICA_WEIGHT = 5.0
METRICA_GAMES = (1, 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skillcorner", type=Path, default=os.environ.get("MTAH_SKILLCORNER_DIR")
    )
    parser.add_argument(
        "--metrica", type=Path, default=os.environ.get("MTAH_METRICA_DIR")
    )
    parser.add_argument("--out", type=Path, default=clf.CLASSIFIER)
    args = parser.parse_args(argv)
    if args.skillcorner is None or args.metrica is None:
        parser.error(
            "--skillcorner and --metrica (or MTAH_SKILLCORNER_DIR, MTAH_METRICA_DIR) are needed"
        )
    tables, labels, weights = [], [], []
    ids = skillcorner.match_ids(args.skillcorner)
    for mid in ids:
        m = skillcorner.upsampled(skillcorner.load_match(args.skillcorner, mid))
        cleaned = clean_tracking(m.tracking).tracking
        tables.append(clf.frame_table(cleaned, len(m.labels)))
        labels.append(np.asarray(m.labels))
        weights.append(1.0)
    for g in METRICA_GAMES:
        stem = args.metrica / f"Sample_Game_{g}" / f"Sample_Game_{g}_"
        t = load_metrica_csv(
            f"{stem}RawTrackingData_Home_Team.csv",
            f"{stem}RawTrackingData_Away_Team.csv",
        )
        dead, unknown = dead_intervals(f"{stem}RawEventsData.csv")
        lab = metrica_labels(t, dead, unknown)
        tables.append(clf.frame_table(clean_tracking(t).tracking, len(lab)))
        labels.append(np.where(lab == "H", "L", lab))  # live, the team unknown
        weights.append(METRICA_WEIGHT)
    model = clf.fit(tables, labels, weights=weights)
    clf.save_model(model, args.out, attribution=ATTRIBUTION + SKILLCORNER_ATTRIBUTION)
    print(
        f"{len(ids)} SkillCorner matches, {len(METRICA_GAMES)} Metrica games (x{METRICA_WEIGHT:g}):"
        f" {model['train_frames']} training frames, converged {model['converged']};"
        f" written to {args.out}"
    )


if __name__ == "__main__":
    main()
