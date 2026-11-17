"""Build the submission: one possession label per video frame of the provided match.

    uv run python scripts/make_submission.py --out <team_name>.csv
    python evaluation/validate_submission.py <team_name>.csv

The match's automatic tracking (ATD, from games.py and MTAH_DATA_DIR) is cleaned
(mtah.clean.tracks: stale tracks, keepers' teams) and labelled by the semi-Markov model
(mtah.predict.hmm) with the hybrid's per-frame classifier (mtah.predict.classifier). Every
value the labelling uses ships in the package and was fitted on external data only, never on
the provided match (docs/technical-document.md, docs/tunable-values.md):

    src/mtah/predict/external_calibration.json   scripts/calibrate_external.py (SkillCorner's
                                                 open data and Metrica's sample games)
    src/mtah/predict/frame_classifier.json       scripts/train_classifier.py (the same)
    the weights and durations, module defaults   scripts/weights_report.py,
                                                 scripts/dead_weights_report.py,
                                                 scripts/hybrid_report.py (SkillCorner, DFL,
                                                 Metrica)

Nothing is fitted here and nothing is random, so the same data gives the same file. Frame
count, halves, the dead stretch before the second-half kickoff and the pitch size come from
games.py. The file is read back and checked with the organisers' validator before exiting. It is
never written into evaluation/results/ unless --allow-results is given.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm

from games import (
    DATA_DIR,
    GAME,
    GAMES,
    anchor_video_frame,
    half_video_blocks,
    require,
)
from mtah.clean.tracks import clean_tracking
from mtah.io.tracking import load_tracking
from mtah.predict.classifier import CLASSIFIER, W_CLF, load_model
from mtah.predict.hmm import EXTERNAL_CALIBRATION, hmm_labels, load_calibration
from mtah.predict.possession import possession_lengths

RESULTS_DIR = ROOT / "evaluation" / "results"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--data-dir", help=f"match data folder (default {DATA_DIR})")
    parser.add_argument(
        "--meta", type=Path, help="ATD metadata XML (default from games.py)"
    )
    parser.add_argument("--raw", type=Path, help="ATD raw text (default from games.py)")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="where the parsed tracking is cached (default cache/)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="always load through kloppy"
    )
    parser.add_argument(
        "--allow-results",
        action="store_true",
        help="allow writing into evaluation/results/",
    )
    args = parser.parse_args(argv)

    out = args.out.resolve()
    if out.is_relative_to(RESULTS_DIR.resolve()) and not args.allow_results:
        parser.error(
            f"{args.out} is in evaluation/results/; pass --allow-results to write there"
        )

    game = GAMES[GAME]
    n_frames = game["n_video_frames"]
    halves = [(first, last) for _, first, last in half_video_blocks(GAME)]
    # Graded but not yet in play: from the start of the second-half block to its kickoff.
    dead_ranges = [(halves[1][0], anchor_video_frame(game["kickoff_2nd"]))]

    if args.meta and args.raw:
        meta, raw = args.meta, args.raw
    else:
        meta, raw = require(["atd_meta", "atd_raw"], GAME, args.data_dir)
    tracking = load_tracking(
        meta, raw, game["pitch"], cache_dir=None if args.no_cache else args.cache_dir
    )
    tracking = clean_tracking(tracking).tracking
    labels = hmm_labels(
        tracking,
        n_frames,
        halves,
        dead_ranges,
        load_calibration(EXTERNAL_CALIBRATION),
        classifier=load_model(CLASSIFIER),
        w_clf=W_CLF,
    )
    write_checked(out, labels, n_frames)
    print(possession_summary(labels, halves, tracking.fps))


def write_checked(out, labels, n_frames):
    """Write the frame,label CSV and check it with the organisers' validator."""
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"frame": np.arange(n_frames), "label": labels})
    table.to_csv(out, index=False, lineterminator="\n")
    reason = pm.validate_submission(pm.read_label_csv(out), n_frames)
    if reason:
        sys.exit(f"{out}: INVALID -- {reason}")
    counts = ", ".join(f"{label} {int(np.sum(labels == label))}" for label in pm.LABELS)
    print(f"{out}: valid, {n_frames} frames ({counts})")


def possession_summary(labels, halves, fps):
    """`N possessions, P per minute, mean S s, D share` over the halves: counts only."""
    parts = [labels[first : last + 1] for first, last in halves]
    lengths = np.concatenate([possession_lengths(part) for part in parts])
    frames = sum(len(part) for part in parts)
    minutes = frames / fps / 60
    mean = lengths.mean() / fps if len(lengths) else 0.0
    dead = sum(int(np.sum(part == "D")) for part in parts) / frames
    return (
        f"{len(lengths)} possessions in the halves, {len(lengths) / minutes:.1f} per minute,"
        f" mean {mean:.1f} s, {dead:.0%} dead"
    )


if __name__ == "__main__":
    main()
