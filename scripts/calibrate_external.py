"""Calibrate the model (mtah.predict.hmm) on external data only, and print the calibration.

    uv run python scripts/calibrate_external.py \
        --skillcorner <skillcorner>/data/matches \
        --metrica <metrica>/data

The team table, the transition and flight rates and the possession lengths come from
SkillCorner's phases of play with --skillcorner (controlled possession, each match
upsampled to 25 fps), else from the DFL data with --dfl (its possession changes on
nearly every touch); the dead table and the ball-out stoppage shares from
Metrica's sample games 1 and 2, whose ball goes missing as a tactical camera's does, dead against
live from their events (mtah.tune.calibration). Every match is cleaned first, as the submission's
tracking is. The provided match is never read.

By default it is written to src/mtah/predict/external_calibration.json, which is committed and
which make_submission.py uses: it holds only numbers derived from the external data (credited in the
file), so the submission is reproduced from the repository alone.
"""

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtah.io import dfl, skillcorner
from mtah.io.metrica_csv import dead_intervals, load_metrica_csv
from mtah.predict.hmm import (
    DISTANCE_EDGES,
    EXTERNAL_CALIBRATION,
    MIN_LIVE_S,
    SPEED_EDGES,
    save_calibration,
)
from mtah.tune.calibration import calibrate_sources, match_features, metrica_labels

# Params
METRICA_GAMES = (1, 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl", type=Path, default=os.environ.get("MTAH_DFL_DIR"))
    parser.add_argument("--dfl-cache", type=Path)
    parser.add_argument(
        "--skillcorner",
        type=Path,
        default=os.environ.get("MTAH_SKILLCORNER_DIR"),
        help="SkillCorner's data/matches: the possession tables from its phases of play",
    )
    parser.add_argument(
        "--metrica", type=Path, default=os.environ.get("MTAH_METRICA_DIR")
    )
    parser.add_argument("--out", type=Path, default=EXTERNAL_CALIBRATION)
    args = parser.parse_args(argv)
    if (args.dfl is None and args.skillcorner is None) or args.metrica is None:
        parser.error(
            "--metrica and --skillcorner or --dfl (or MTAH_METRICA_DIR,"
            " MTAH_SKILLCORNER_DIR, MTAH_DFL_DIR) are needed"
        )

    dfl_pairs, skillcorner_pairs = [], []
    if args.skillcorner is not None:
        for mid in skillcorner.match_ids(args.skillcorner):
            m = skillcorner.upsampled(skillcorner.load_match(args.skillcorner, mid))
            skillcorner_pairs.append(match_features(m.tracking, m.labels))
    else:
        cache = args.dfl_cache or args.dfl / "cache"
        for mid in dfl.match_files(args.dfl):
            m = dfl.load_match(args.dfl, mid, cache_dir=cache)
            dfl_pairs.append(match_features(m.tracking, m.labels))
    metrica_pairs = []
    for g in METRICA_GAMES:
        stem = args.metrica / f"Sample_Game_{g}" / f"Sample_Game_{g}_"
        t = load_metrica_csv(
            f"{stem}RawTrackingData_Home_Team.csv",
            f"{stem}RawTrackingData_Away_Team.csv",
        )
        dead, unknown = dead_intervals(f"{stem}RawEventsData.csv")
        metrica_pairs.append(match_features(t, metrica_labels(t, dead, unknown)))

    calibration, _, _ = calibrate_sources(dfl_pairs, metrica_pairs, skillcorner_pairs)
    save_calibration(calibration, args.out)
    print("\n".join(summary_lines(calibration)))
    source = (
        f"{len(skillcorner_pairs)} SkillCorner matches"
        if skillcorner_pairs
        else f"{len(dfl_pairs)} DFL matches"
    )
    print(f"\n{source}, {len(metrica_pairs)} Metrica games; written to {args.out}")


def summary_lines(calibration):
    return summary(calibration)


def summary(calibration, fps=25.0):
    """The calibration's lines: shares, likelihood ratios and rates, no frames."""
    counts = calibration["counts"]
    lines = [
        (
            f"labelled: {counts['live_frames']} live frames, {counts['dead_frames']} dead;"
            f" {counts['switches']} switches, {counts['stops']} stops,"
            f" {counts['restarts']} restarts"
        ),
        *possession_lines(calibration, fps),
        "",
        "nearest player is the possessing team, by distance",
    ]
    edges = (0, *DISTANCE_EDGES, np.inf)
    for (low, high), share in zip(itertools.pairwise(edges), calibration["team_agree"]):
        name = f">={low:g} m" if np.isinf(high) else f"{low:g}-{high:g} m"
        lines.append(f"  {name:<10}{share:>6.0%}")

    lines += ["", "dead features: P(. | D) / P(. | live)"]
    speed_edges = (0, *SPEED_EDGES, np.inf)
    names = ["seen", "dribbled"] + [
        f"missing, >={a:g} m/s" if np.isinf(b) else f"missing, {a:g}-{b:g} m/s"
        for a, b in itertools.pairwise(speed_edges)
    ]
    for name, given_dead, given_live in zip(
        names, calibration["dead_given_dead"], calibration["dead_given_live"]
    ):
        lines.append(f"  {name:<22}{given_dead / given_live:>8.2f}")
    lines += [
        "",
        (
            f"ball-out stoppage under way: {calibration['stoppage_given_dead']:.0%} of dead"
            f" frames, {calibration['stoppage_given_live']:.1%} of live"
        ),
        (
            f"per frame: switch {calibration['switch']:.4f}, stop {calibration['stop']:.4f},"
            f" restart {calibration['restart']:.4f}"
        ),
    ]
    if "kicker_agree" in calibration:
        lines.append(
            f"possessing team in flight: the kicker's {calibration['kicker_agree']:.0%}"
            f" ({calibration['kicked_frames']} live frames with a kicker)"
        )
    return lines


def possession_lines(calibration, fps):
    """The labelling's possessions by length, against the model's minimum."""
    if "possession_frames" not in calibration:
        return []
    seconds = np.asarray(calibration["possession_frames"]) / fps
    edges = (*sorted({0, MIN_LIVE_S, 1, 2, 5, 10}), np.inf)
    cells = []
    for low, high in itertools.pairwise(edges):
        name = f">={low:g} s" if np.isinf(high) else f"{low:g}-{high:g} s"
        cells.append(f"{name}: {int(np.sum((seconds >= low) & (seconds < high)))}")
    return [
        f"possessions by length (the model's shortest is {MIN_LIVE_S:g} s): "
        + ", ".join(cells)
    ]


if __name__ == "__main__":
    main()
