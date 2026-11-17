"""A tiny synthetic Metrica sample game (CSV tracking pair and events), so tests need no download."""

from pathlib import Path

# Params
N_FRAMES = 60
HOME_PLAYERS = ("Player1", "Player2")
AWAY_PLAYERS = ("Player15", "Player16")
MISSING = (10, "Player2")  # (frame, player) left NaN
# (Type, Subtype, Start Frame, End Frame): frames count from 1 as in the real files.
EVENTS = [
    ("SET PIECE", "KICK OFF", 1, 0),
    ("PASS", "", 2, 4),
    ("BALL OUT", "", 5, 6),  # out at frame 6 -> dead 5..9 (video frames)
    ("SET PIECE", "THROW IN", 11, 11),
    ("FAULT RECEIVED", "", 15, 15),  # foul at frame 15 -> dead 14..17
    ("SET PIECE", "FREE KICK", 19, 19),
    ("SHOT", "ON TARGET-GOAL", 22, 24),  # goal at frame 24 -> dead 23..29
    ("SET PIECE", "KICK OFF", 31, 0),
    ("SHOT", "OFF TARGET-OUT", 34, 36),  # out at frame 36 -> dead 35..39
    ("PASS", "GOAL KICK", 41, 43),
    ("PASS", "", 44, 46),
    ("SET PIECE", "FREE KICK", 50, 50),  # no trigger -> unknown 45..48
]


def _team_csv(team, players):
    # As in the real files: every player and the ball take an x column and an unnamed y one.
    header = [",,," + "".join(f"{team},," for _ in players) + ","]
    header.append(",,," + "".join(f"{p[6:]},," for p in players) + ",")
    header.append(
        "Period,Frame,Time [s]," + "".join(f"{p},," for p in players) + "Ball,"
    )
    rows = []
    for frame in range(1, N_FRAMES + 1):
        period = 1 if frame <= 30 else 2
        cells = []
        for k, player in enumerate(players):
            if (frame, player) == MISSING:
                cells.append("NaN,NaN")
            else:
                x = 0.1 + 0.2 * k + (0.5 if team == "Away" else 0.0)
                cells.append(f"{x:.3f},0.500")
        cells.append(f"{0.01 * frame:.3f},0.250")
        rows.append(f"{period},{frame},{frame * 0.04:.2f}," + ",".join(cells))
    return "\n".join(header + rows) + "\n"


def write_metrica(directory):
    """Write home.csv, away.csv and events.csv; return their paths."""
    directory = Path(directory)
    home, away, events = (directory / n for n in ("home.csv", "away.csv", "events.csv"))
    home.write_text(_team_csv("Home", HOME_PLAYERS), encoding="utf-8")
    away.write_text(_team_csv("Away", AWAY_PLAYERS), encoding="utf-8")
    lines = [
        (
            "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],"
            "From,To,Start X,Start Y,End X,End Y"
        )
    ]
    for kind, subtype, start, end in EVENTS:
        lines.append(
            f"Home,{kind},{subtype},1,{start},{start * 0.04:.2f},{end},{end * 0.04:.2f},"
            "Player1,,NaN,NaN,NaN,NaN"
        )
    events.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return home, away, events
