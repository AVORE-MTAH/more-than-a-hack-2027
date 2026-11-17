"""Choose the model's weights on external data: SkillCorner's matches, or DFL's.

    uv run python scripts/weights_report.py \
        --skillcorner <skillcorner>/data/matches
    uv run python scripts/weights_report.py --dfl <dfl> \
        --metrica <metrica>/data

With --skillcorner: SkillCorner's 20 matches, upsampled to 25 fps, scored against
their phases of play, whose team changes only when the other team gains control (the
challenge's controlled possession); their ball is a broadcast's, used as it is. The search runs
over the possession values only (SKILLCORNER_GRID): the dead-ball ones are Metrica's
(scripts/dead_weights_report.py). The calibration should be the one calibrate_external.py
writes with --skillcorner (the shipped one, the default). A last check leaves each match out of the calibration too: its
tables from the other 19, the chosen setting, scored on it.

With --dfl, external data only (mtah.tune.weights): the seven DFL matches, each with its ball removed by the
rule fitted on Metrica's sample games (mtah.tune.missing), labelled with the calibration from
scripts/calibrate_external.py, and scored with the organisers' metrics against the DFL data's own
possession and status.

The search changes one value at a time (GRID), keeping a change only if the mean spell F1 over
the matches rises, for --sweeps passes. Every setting tried is kept with its per-match scores, so
the check leaving one match out comes from the same runs: for each match, the setting best on the
others, scored on it. Then the chosen setting with the ball removed less and more often
(--share-scales), to show it does not hinge on one rate.

The log of every setting goes to --log (external/, not committed). Nothing of the provided match
is read.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtah.io import dfl, skillcorner
from mtah.io.metrica_csv import dead_intervals, load_metrica_csv
from mtah.predict.hmm import EXTERNAL_CALIBRATION, calibrate, load_calibration
from mtah.tune.calibration import POSSESSION_KEYS, joined, metrica_labels
from mtah.tune.missing import fit
from mtah.tune.weights import (
    FLIGHTS,
    Setting,
    leave_one_out,
    prepare,
    prepare_skillcorner,
    score_match,
)

# Params
GRID = {
    "w_team": (0.05, 0.1, 0.15, 0.25, 0.4),
    "w_dead": (0.1, 0.2, 0.3, 0.5, 0.8),
    "w_kick": (0.0, 1.0, 2.0, 4.0),
    "min_live_s": (0.2, 0.4, 0.6, 1.0),
    "min_dead_s": (1.0, 2.0, 3.0),
    "flight": tuple(FLIGHTS),
}
SKILLCORNER_GRID = {  # the possession values; the dead-ball ones are Metrica's
    "w_team": (0.05, 0.1, 0.2, 0.3, 0.5, 0.8),
    "w_kick": (0.0, 0.5, 1.0, 2.0, 4.0),
    "min_live_s": (0.1, 0.5, 1.0, 2.0, 3.0),
    "flight": tuple(FLIGHTS),
}
START = Setting()  # the defaults
SEED = 2026


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl", type=Path, default=os.environ.get("MTAH_DFL_DIR"))
    parser.add_argument("--dfl-cache", type=Path)
    parser.add_argument(
        "--skillcorner",
        type=Path,
        default=os.environ.get("MTAH_SKILLCORNER_DIR"),
        help="SkillCorner's data/matches: score on its matches instead of DFL's",
    )
    parser.add_argument(
        "--metrica", type=Path, default=os.environ.get("MTAH_METRICA_DIR")
    )
    parser.add_argument("--calibration", type=Path, default=EXTERNAL_CALIBRATION)
    parser.add_argument("--log", type=Path, default=Path("external/weights_log.jsonl"))
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument(
        "--start",
        type=json.loads,
        default={},
        help="JSON: the start's values (Setting fields)",
    )
    parser.add_argument(
        "--grid",
        type=json.loads,
        default=None,
        help="JSON: {field: [values]} in place of GRID",
    )
    parser.add_argument(
        "--share-scales", type=float, nargs="*", default=[0.5, 1.5], metavar="X"
    )
    args = parser.parse_args(argv)
    on_skillcorner = args.skillcorner is not None
    if not on_skillcorner and (args.dfl is None or args.metrica is None):
        parser.error(
            "--skillcorner, or --dfl and --metrica (or MTAH_SKILLCORNER_DIR, MTAH_DFL_DIR,"
            " MTAH_METRICA_DIR), are needed"
        )

    say = progress()
    calibration = load_calibration(args.calibration)
    if on_skillcorner:
        ids = skillcorner.match_ids(args.skillcorner)
        say(f"preparing {len(ids)} SkillCorner matches at 25 fps")
        prepared = [
            prepare_skillcorner(skillcorner.load_match(args.skillcorner, mid))
            for mid in ids
        ]
    else:
        missing = missing_rule(args.metrica)
        cache = args.dfl_cache or args.dfl / "cache"
        ids = list(dfl.match_files(args.dfl))
        matches = [dfl.load_match(args.dfl, mid, cache_dir=cache) for mid in ids]
        say(f"preparing {len(matches)} DFL matches, ball removed as Metrica's")
        prepared = [prepare(m, missing, SEED + k) for k, m in enumerate(matches)]

    table = {}
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("w", encoding="utf-8")

    def evaluate(setting):
        key = setting.key()
        if key not in table:
            table[key] = [score_match(p, calibration, setting) for p in prepared]
            log.write(
                json.dumps({"setting": asdict(setting), "scores": table[key]}) + "\n"
            )
            log.flush()
        return float(np.mean([s["spell_f1"] for s in table[key]]))

    default_grid = SKILLCORNER_GRID if on_skillcorner else GRID
    grid = (
        default_grid
        if args.grid is None
        else {k: tuple(v) for k, v in args.grid.items()}
    )
    start = replace(START, **args.start)
    best, best_score = start, evaluate(start)
    say(f"start {best_score:.4f}")
    lines = [
        "search (mean spell F1 over the matches):",
        f"  start {fmt(best)}: {best_score:.4f}",
    ]
    for sweep in range(args.sweeps):
        improved = False
        for name, values in grid.items():
            for value in values:
                trial = replace(best, **{name: value})
                score = evaluate(trial)
                if score > best_score + 1e-6:
                    best, best_score, improved = trial, score, True
                    lines.append(f"  sweep {sweep + 1}, {name} = {value}: {score:.4f}")
                    say(f"{name} = {value}: {score:.4f}")
        if not improved:
            break
    log.close()

    final = table[best.key()]
    lines += [
        "",
        f"chosen: {fmt(best)}",
        "per match (spell F1, macro F1, possessions true/ours):",
    ]
    for mid, s in zip(ids, final, strict=True):
        lines.append(
            f"  {mid}: {s['spell_f1']:.3f}, {s['macro_f1']:.3f}, {s['n_gt']}/{s['n_pred']}"
        )
    held = leave_one_out(table)
    lines += [
        "",
        "leaving one match out (the setting best on the others, scored on it):",
        (
            f"  mean spell F1 {np.mean([h[2] for h in held]):.4f}"
            f" (in-sample {best_score:.4f}); settings chosen differ from the final"
            f" in {sum(h[1] != best.key() for h in held)} of {len(held)} folds"
        ),
    ]
    if on_skillcorner:
        held_out = held_out_calibration(prepared, calibration, best)
        lines += [
            "",
            "leaving each match out of the calibration too (tables from the others):",
            (
                f"  mean spell F1 {np.mean([s['spell_f1'] for s in held_out]):.4f},"
                f" macro F1 {np.mean([s['macro_f1'] for s in held_out]):.4f}"
            ),
        ]
    elif args.share_scales:
        lines += ["", "the chosen setting with the ball removed less or more often:"]
        for scale in args.share_scales:
            scaled = {
                state: {**v, "share": min(1.0, v["share"] * scale)}
                for state, v in missing.items()
            }
            again = [prepare(m, scaled, SEED + k) for k, m in enumerate(matches)]
            mean = np.mean(
                [score_match(p, calibration, best)["spell_f1"] for p in again]
            )
            lines.append(f"  missing share x{scale:g}: mean spell F1 {mean:.4f}")
    print("\n".join(lines))
    print(f"\n{len(table)} settings tried, logged to {args.log}")


def held_out_calibration(prepared, calibration, setting):
    """Per match: its scores with the possession tables calibrated on the other matches (the
    dead-ball ones kept: they are Metrica's)."""
    pairs = [(p["features"][setting.flight], p["labels"]) for p in prepared]
    out = []
    for k, p in enumerate(prepared):
        own = calibrate(*joined(pairs[:k] + pairs[k + 1 :]))
        cal = dict(calibration)
        cal.update({key: own[key] for key in POSSESSION_KEYS if key in own})
        out.append(score_match(p, cal, setting))
    return out


def missing_rule(root):
    pairs = []
    for g in (1, 2):
        stem = root / f"Sample_Game_{g}" / f"Sample_Game_{g}_"
        t = load_metrica_csv(
            f"{stem}RawTrackingData_Home_Team.csv",
            f"{stem}RawTrackingData_Away_Team.csv",
        )
        dead, unknown = dead_intervals(f"{stem}RawEventsData.csv")
        pairs.append((t.ball, metrica_labels(t, dead, unknown)[t.video_frame]))
    return fit(pairs, 25.0)


def fmt(setting):
    return ", ".join(f"{k} {v}" for k, v in asdict(setting).items())


def progress():
    start = time.perf_counter()

    def say(message):
        elapsed = time.perf_counter() - start
        print(
            f"[{elapsed // 60:3.0f}:{elapsed % 60:02.0f}] {message}",
            file=sys.stderr,
            flush=True,
        )

    return say


if __name__ == "__main__":
    main()
