"""Metrica's public sample games (CSV format): tracking, and dead-ball stretches from their events.

Sample Games 1 and 2 (https://github.com/metrica-sports/sample-data) come as two tracking CSVs,
home and away, and an events CSV. Being public, they can calibrate what the match data cannot:
their events say when play stopped and restarted, about a hundred times a match, independently
of the tracking.

Tracking CSV: two header rows (team, shirt number), then `Period,Frame,Time [s],Player11,,
Player1,,...,Ball,` and one row per frame, each player and the ball as an x, y pair in 0-1 with
the origin at the top left, the orientation of mtah.io.tracking. Frames count from 1, so
video_frame = Frame - 1 as for the EPTS files.

Dead stretches: play stops at a trigger (the ball going out, a foul, a goal) and restarts at a set
piece or a goal kick. A stretch runs from the latest trigger since the previous restart to the
frame before the restart. A restart with no trigger in the LOOKBACK_S before it (a stoppage the
events do not explain: an injury, a dropped ball, a foul not logged) makes the stretch since the
previous event unknown, to be left out of every count.
"""

import numpy as np
import pandas as pd

from mtah.io.tracking import Tracking

# Params
SAMPLE_PITCH = (105.0, 68.0)
FPS = 25.0
LOOKBACK_S = 60
# Event rules, on upper-cased Type and Subtype.
RESTART_TYPES = ("SET PIECE",)
RESTART_SUBTYPE = "GOAL KICK"  # PASS / GOAL KICK, BALL LOST / GOAL KICK-INTERCEPTION
OUT_TYPES = ("BALL OUT",)


def load_metrica_csv(home_csv, away_csv, pitch_size=SAMPLE_PITCH):
    """Both teams' tracking CSVs as one Tracking, positions in metres."""
    home_period, home_frame, home = _read_team(home_csv)
    _, away_frame, away = _read_team(away_csv)
    if not np.array_equal(home_frame, away_frame):
        raise ValueError(f"{home_csv} and {away_csv} do not cover the same frames")

    scale = np.array(pitch_size, dtype=float)
    home_players = [xy for name, xy in home.items() if name != "Ball"]
    away_players = [xy for name, xy in away.items() if name != "Ball"]
    players = np.stack(home_players + away_players, axis=1) * scale
    team = np.array([0] * len(home_players) + [1] * len(away_players), dtype=np.int8)
    return Tracking(
        players=players,
        ball=home["Ball"] * scale,
        team=team,
        period=home_period.astype(np.int8),
        video_frame=home_frame.astype(np.int64) - 1,
        fps=FPS,
        pitch_size=(float(pitch_size[0]), float(pitch_size[1])),
    )


def dead_intervals(events_csv, fps=FPS, lookback_s=LOOKBACK_S):
    """(dead, unknown): inclusive (first, last) video frames, from the event data's stoppages."""
    events = pd.read_csv(events_csv)
    kind = events["Type"].fillna("").str.strip().str.upper()
    subtype = events["Subtype"].fillna("").str.strip().str.upper()
    start = events["Start Frame"].to_numpy(dtype=np.int64) - 1
    end = events["End Frame"].to_numpy(dtype=np.int64) - 1

    shot = kind == "SHOT"
    restart = kind.isin(RESTART_TYPES) | subtype.str.contains(RESTART_SUBTYPE)
    stops_at_end = kind.isin(OUT_TYPES) | (shot & subtype.str.endswith("-OUT"))
    stops_at_end |= shot & subtype.str.contains("GOAL")
    stops_at_start = (kind == "FAULT RECEIVED") | (
        (kind == "CHALLENGE") & subtype.str.contains("FAULT")
    )

    lookback = round(lookback_s * fps)
    dead, unknown = [], []
    trigger = last_event = None
    for i in np.argsort(start, kind="stable"):
        if restart.iloc[i]:
            frame = int(start[i])
            if trigger is not None and 0 < frame - trigger <= lookback:
                dead.append((trigger, frame - 1))
            elif last_event is not None and last_event < frame:
                unknown.append((max(last_event, frame - lookback), frame - 1))
            trigger = None
        elif stops_at_end.iloc[i]:
            trigger = int(end[i])
        elif stops_at_start.iloc[i]:
            trigger = int(start[i])
        last_event = int(max(start[i], end[i]))
    return dead, unknown


def _read_team(path):
    """(period, frame, {name: (frames, 2) x, y in 0-1}) for one team's CSV, the ball included."""
    table = pd.read_csv(path, skiprows=2)
    columns = list(table.columns)
    positions = {}
    for k, name in enumerate(columns[3:], start=3):
        if not str(name).startswith("Unnamed"):
            positions[name] = table.iloc[:, [k, k + 1]].to_numpy(dtype=float)
    if "Ball" not in positions:
        raise ValueError(f"{path}: no Ball column")
    return table["Period"].to_numpy(), table["Frame"].to_numpy(), positions
