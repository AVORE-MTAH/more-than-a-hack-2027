"""How much each value not fitted on external data matters: each at half and at double, scored.

    uv run python scripts/sensitivity_report.py --group keepers --dfl <dfl>
    uv run python scripts/sensitivity_report.py --group dead --metrica <metrica>/data
    uv run python scripts/sensitivity_report.py --group flight --dfl <dfl> \
        --metrica <metrica>/data

External data only. Each value of a group is set to half and to double its default for a run,
the others left at theirs, and scored by the fit it belongs to:

    keepers   the cleaning's keeper values, by the DFL keepers' coverage and team-changing
              errors (mtah.tune.keepers)
    dead      the dribbling and ball-out values, by D's F1 on Metrica's sample games, with the
              dead-ball weights at their defaults (scripts/dead_weights_report.py)
    flight    the flight helpers' values, by the mean spell F1 over the DFL matches with the
              ball removed as Metrica's, the weights at their defaults (mtah.tune.weights)

A value whose score barely moves (under FLAT) is recorded as set from physics and checked
insensitive; one that moves is to be fitted.
"""

import argparse
import contextlib
import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mtah.dead.features as dead_features
import mtah.dead.score as dead_score
import mtah.predict.flight as flight_module
from mtah.clean.tracks import DEFAULT_CONFIG
from mtah.io import dfl
from mtah.predict.flight import FLIGHT_S, FLIGHT_SPEED, KINK_DEG
from mtah.predict.hmm import EXTERNAL_CALIBRATION, load_calibration
from mtah.tune import keepers as keeper_fit
from mtah.tune.weights import Setting, prepare, score_match

# Params
SCALES = (0.5, 2.0)
FLAT = {"keepers": 0.01, "dead": 0.005, "flight": 0.003}
KEEPER_VALUES = ("max_gap_s", "far_s", "takeover_s", "takeover_contact_m", "back_s")
DEAD_VALUES = {
    "CONTROL_RADIUS": dead_score,
    "DRIBBLE_SPEED": dead_score,
    "DRIBBLE_WINDOW_S": dead_score,
    "OUTSIDE_MARGIN_M": dead_features,  # a default argument of ball_outside
    "KICK_SPEED": dead_score,
    "SETTLE_S": dead_score,
    "SLOW_BALL": dead_score,
    "EXIT_SPEED": dead_score,
    "EXIT_S": dead_score,
}
FLIGHT_VALUES = (
    "GAP_MAX_S",
    "GAP_SPEED",
    "KICKER_M",
    "KICK_S",
    "TURN_SPEED",
    "KINK_DEG",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", choices=("keepers", "dead", "flight"), required=True)
    parser.add_argument("--dfl", type=Path, default=os.environ.get("MTAH_DFL_DIR"))
    parser.add_argument(
        "--metrica", type=Path, default=os.environ.get("MTAH_METRICA_DIR")
    )
    parser.add_argument("--calibration", type=Path, default=EXTERNAL_CALIBRATION)
    args = parser.parse_args(argv)
    rows = {"keepers": keepers, "dead": dead, "flight": flight}[args.group](args)
    print("\n".join(report(args.group, rows)))


@contextlib.contextmanager
def patched(module, name, value):
    """Set a module constant, or ball_outside's margin, for the duration."""
    if name == "OUTSIDE_MARGIN_M":
        old = dead_features.ball_outside.__defaults__
        dead_features.ball_outside.__defaults__ = (value,)
        try:
            yield
        finally:
            dead_features.ball_outside.__defaults__ = old
        return
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


def keepers(args):
    matches = [
        dfl.load_match(args.dfl, mid, cache_dir=args.dfl / "cache")
        for mid in dfl.match_files(args.dfl)
    ]
    rows = []
    base = keeper_fit.score(matches, DEFAULT_CONFIG)
    for name in KEEPER_VALUES:
        default = getattr(DEFAULT_CONFIG, name)
        scores = [
            keeper_fit.score(matches, replace(DEFAULT_CONFIG, **{name: default * s}))
            for s in SCALES
        ]
        rows.append((name, default, base, scores))
    return rows


def dead(args):
    script = _script("dead_weights_report")
    calibration = load_calibration(args.calibration)

    def run():
        games = [script.prepare(args.metrica, g) for g in script.GAMES]
        return float(
            np.mean([script.score(g, calibration, Setting())["d_f1"] for g in games])
        )

    base = run()
    rows = []
    for name, module in DEAD_VALUES.items():
        default = (
            dead_features.ball_outside.__defaults__[0]
            if name == "OUTSIDE_MARGIN_M"
            else getattr(module, name)
        )
        scores = []
        for s in SCALES:
            value = type(default)(default * s)
            with patched(module, name, value):
                scores.append(run())
        rows.append((name, default, base, scores))
    return rows


def flight(args):
    script = _script("weights_report")
    missing = script.missing_rule(args.metrica)
    calibration = load_calibration(args.calibration)
    matches = [
        dfl.load_match(args.dfl, mid, cache_dir=args.dfl / "cache")
        for mid in dfl.match_files(args.dfl)
    ]
    setting = Setting()

    def run(kink=KINK_DEG):
        flights = {
            setting.flight: {
                "speed": FLIGHT_SPEED,
                "shortest_s": FLIGHT_S,
                "kink_deg": kink,
            }
        }
        prepared = [
            prepare(m, missing, script.SEED + k, flights) for k, m in enumerate(matches)
        ]
        return float(
            np.mean(
                [score_match(p, calibration, setting)["spell_f1"] for p in prepared]
            )
        )

    base = run()
    rows = []
    for name in FLIGHT_VALUES:
        default = getattr(flight_module, name)
        scores = []
        for s in SCALES:
            value = default * s
            if name == "KINK_DEG":
                scores.append(run(kink=value))
            else:
                with patched(flight_module, name, value):
                    scores.append(run())
        rows.append((name, default, base, scores))
    return rows


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(group, rows):
    flat = FLAT[group]
    lines = [
        f"{group}: each value at x{SCALES[0]:g} and x{SCALES[1]:g} (flat if under {flat:g})"
    ]
    for name, default, base, scores in rows:
        if group == "keepers":
            (bc, bf), parts = base, scores
            moves = max(abs(c - bc) for c, _ in parts)
            cells = ", ".join(f"coverage {c:.1%} flips {f:.3%}" for c, f in parts)
            head = f"coverage {bc:.1%} flips {bf:.3%}"
        else:
            moves = max(abs(s - base) for s in scores)
            cells = ", ".join(f"{s:.4f}" for s in scores)
            head = f"{base:.4f}"
        verdict = "flat: set from physics" if moves < flat else "moves: to fit"
        lines.append(
            f"  {name} = {default:g}: {head}; halved, doubled: {cells} -> {verdict}"
        )
    return lines


if __name__ == "__main__":
    main()
