"""Fit every value the external data can fit, and print each beside the module's default.

    uv run python scripts/tune_external.py --dfl <dfl> \
        --metrica <metrica>/data

External data only; never the provided match. Each fit is one module of
mtah.tune; the defaults in the code are to equal what this prints (docs/tunable-values.md):

    cleaning       mtah.tune.cleaning     DFL: real players' 4 s spread and path excess
    keepers        mtah.tune.keepers      DFL: its named keepers
    restarts       mtah.tune.restarts     Metrica sample games 1-2: a change kept only if it
                                          helps every game

Leave a source out to skip its fits. --dfl-cache caches the parsed DFL matches (default
<dfl>/cache).
"""

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mtah.dead.score as dead_score
from mtah.clean.tracks import DEFAULT_CONFIG
from mtah.io import dfl
from mtah.io.metrica_csv import dead_intervals, load_metrica_csv
from mtah.tune import cleaning, keepers, restarts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl", type=Path, default=os.environ.get("MTAH_DFL_DIR"))
    parser.add_argument("--dfl-cache", type=Path)
    parser.add_argument(
        "--metrica", type=Path, default=os.environ.get("MTAH_METRICA_DIR")
    )
    args = parser.parse_args(argv)
    lines = []
    if args.dfl:
        cache = args.dfl_cache or args.dfl / "cache"
        matches = [
            dfl.load_match(args.dfl, mid, cache_dir=cache)
            for mid in dfl.match_files(args.dfl)
        ]
        fitted = cleaning.fit([m.tracking for m in matches])
        lines += section(
            "cleaning (DFL)",
            [
                (name, getattr(DEFAULT_CONFIG, name), round(fitted[name], 3))
                for name in ("stale_max_spread_m", "coast_max_excess_m")
            ],
        )
        config = replace(
            DEFAULT_CONFIG,
            stale_max_spread_m=fitted["stale_max_spread_m"],
            coast_max_excess_m=fitted["coast_max_excess_m"],
        )
        best, _ = keepers.fit(matches, config)
        lines += section(
            "keepers (DFL)",
            [
                (name, getattr(DEFAULT_CONFIG, name), getattr(best, name))
                for name in keepers.GRID
            ],
        )
    if args.metrica:
        games = []
        for g in (1, 2):
            stem = args.metrica / f"Sample_Game_{g}" / f"Sample_Game_{g}_"
            t = load_metrica_csv(
                f"{stem}RawTrackingData_Home_Team.csv",
                f"{stem}RawTrackingData_Away_Team.csv",
            )
            dead, _ = dead_intervals(f"{stem}RawEventsData.csv")
            games.append([restarts.prepared_from(t, restarts.metrica_dead(t, dead))])
        chosen, maes, _ = restarts.fit_consistent(games)
        lines += section(
            "restarts (Metrica 1-2)",
            [
                (name, getattr(dead_score, name), value)
                for name, value in chosen.items()
                if value != getattr(dead_score, name)
            ]
            + [("(mean abs error per game, s)", "", tuple(round(m, 2) for m in maes))],
        )
    print("\n".join(lines) if lines else "no source given: nothing fitted")


def section(title, rows):
    lines = ["", title]
    for name, default, fitted in rows:
        mark = "" if default == "" or default == fitted else "   <- differs"
        lines.append(f"  {name:<28} default {default!s:<32} fitted {fitted!s}{mark}")
    return lines


if __name__ == "__main__":
    main()
