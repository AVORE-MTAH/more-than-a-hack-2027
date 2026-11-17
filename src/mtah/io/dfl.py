"""Read one match of the DFL dataset into arrays (external data, CC BY 4.0).

"An integrated dataset of spatiotemporal and event data in elite soccer" (Bassek et al.,
Scientific Data 12, 195, 2025; figshare 10.6084/m9.figshare.28196177). Per match, three XML files
named after the match id:

    DFL_02_01_matchinformation_*_{id}.xml        teams (home and guest), players, keepers (TW)
    DFL_03_02_events_raw_*_{id}.xml              events, each timed (EventTime)
    DFL_04_03_positions_raw_observed_*_{id}.xml  25 fps positions: one FrameSet per person and
                                                 half; the ball's carries its height (Z), the
                                                 possessing team (BallPossession: 1 home, 2
                                                 guest) and its status (BallStatus: 1 alive)

Rows are the frames of each half in turn. DFL's coordinates are metres from the pitch centre,
y up; here they are moved to ours (mtah.io.tracking): metres from the top-left corner, y down.

Labels per row: D where the ball is dead, else H or A by its possession. Events become rows
too, by their time (and, for restarts, the time they were awarded: DecisionTimestamp).

A positions file is 350-420 MB, parsed in about a minute; load_match caches the arrays in
cache_dir, and later loads read that instead.
"""

import xml.etree.ElementTree as ET
from array import array
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from mtah.io.tracking import Tracking

# Params
FPS = 25.0
SECTIONS = ("firstHalf", "secondHalf")
CACHE_VERSION = 1


@dataclass(frozen=True)
class DflMatch:
    """One match.

    tracking         players and ball (x, y), period 1 or 2 by half, 25 fps
    ball_z           (rows,) the ball's height in metres
    labels           (rows,) "H", "A" or "D"
    times            (rows,) seconds since the epoch of each row (the ball's timestamps)
    keepers          slots of the players who play in goal (TW)
    person_ids       (slots,) DFL person ids, in slot order
    events           list of dicts: row, decision_row (or -1), kind (the event's element),
                     detail (the elements below it, joined by "/"), team (0 home, 1 guest,
                     -1 none), player, recipient (person ids or "")
    """

    tracking: Tracking
    ball_z: np.ndarray
    labels: np.ndarray
    times: np.ndarray
    keepers: tuple
    person_ids: np.ndarray
    events: list


def match_files(root):
    """{match id: (information, events, positions)} paths under root."""
    root = Path(root)
    out = {}
    for path in root.glob("DFL_02_01_matchinformation_*.xml"):
        mid = path.stem.split("_")[-1]
        events = next(root.glob(f"DFL_03_02_events_raw_*_{mid}.xml"), None)
        positions = next(
            root.glob(f"DFL_04_03_positions_raw_observed_*_{mid}.xml"), None
        )
        if events and positions:
            out[mid] = (path, events, positions)
    return dict(sorted(out.items()))


def _seconds(stamp):
    return datetime.fromisoformat(stamp).timestamp()


def read_information(path):
    """(home team id, guest team id, [(person id, team 0/1, keeper)], pitch (x, y))."""
    root = ET.parse(path).getroot()
    info = root.find(".//MatchInformation")
    general = info.find("General")
    env = info.find("Environment")
    home, guest = general.get("HomeTeamId"), general.get("GuestTeamId")
    people = []
    for team in info.iter("Team"):
        side = 0 if team.get("TeamId") == home else 1
        for player in team.iter("Player"):
            people.append(
                (player.get("PersonId"), side, player.get("PlayingPosition") == "TW")
            )
    pitch = (float(env.get("PitchX", 105.0)), float(env.get("PitchY", 68.0)))
    return home, guest, people, pitch


def read_positions(path):
    """{(section, person or "BALL"): columns} for the halves in SECTIONS, referees left out.

    Columns are compact arrays: N, X, Y for everyone; for the ball also Z, possession (P: 1
    home, 2 guest, 0 none), status (S: 1 alive) and the timestamp in seconds (T).
    """
    sets = {}
    current = is_ball = None
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            section = elem.get("GameSection")
            is_ball = elem.get("TeamId") == "BALL"
            person = "BALL" if is_ball else elem.get("PersonId")
            current = None
            if elem.get("TeamId") != "referee" and section in SECTIONS:
                fields = "NXYZPST" if is_ball else "NXY"
                current = sets.setdefault(
                    (section, person),
                    {f: array("q" if f in "NPS" else "d") for f in fields},
                )
        elif event == "end" and elem.tag == "Frame":
            if current is not None:
                a = elem.attrib
                current["N"].append(int(a["N"]))
                current["X"].append(float(a["X"]))
                current["Y"].append(float(a["Y"]))
                if is_ball:
                    current["Z"].append(float(a.get("Z", "nan")))
                    current["P"].append(int(a.get("BallPossession", 0)))
                    current["S"].append(int(a.get("BallStatus", 0)))
                    current["T"].append(_seconds(a["T"]))
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            current = None
            elem.clear()
    return {
        key: {f: np.asarray(v) for f, v in cols.items()} for key, cols in sets.items()
    }


def build(information, positions):
    """Arrays for one match from its parsed information and positions."""
    _, _, people, (length, width) = information
    ids = np.array([p for p, _, _ in people])
    slot_of = {p: k for k, p in enumerate(ids)}
    team = np.array([t for _, t, _ in people], dtype=np.int8)
    keepers = tuple(k for k, (_, _, gk) in enumerate(people) if gk)

    # Rows: the ball's frames, half after half; a frame number's row within its half.
    offsets, starts, n = {}, {}, 0
    for section in SECTIONS:
        frames = positions.get((section, "BALL"), {}).get("N", np.zeros(0, int))
        if len(frames):
            starts[section], offsets[section] = int(frames.min()), n
            n += int(frames.max() - frames.min() + 1)
    xy = np.full((n, len(ids), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    z = np.full(n, np.nan)
    labels = np.full(n, "D", dtype="<U1")
    times = np.full(n, np.nan)
    period = np.zeros(n, dtype=np.int8)

    for (section, person), cols in positions.items():
        if section not in offsets:
            continue
        rows = offsets[section] + cols["N"] - starts[section]
        keep = (rows >= 0) & (rows < n)
        rows = rows[keep]
        x = cols["X"][keep] + length / 2
        y = width / 2 - cols["Y"][keep]
        if person == "BALL":
            ball[rows] = np.column_stack([x, y])
            z[rows] = cols["Z"][keep]
            times[rows] = cols["T"][keep]
            period[rows] = SECTIONS.index(section) + 1
            alive = cols["S"][keep] == 1
            side = np.where(
                cols["P"][keep] == 1, "H", np.where(cols["P"][keep] == 2, "A", "D")
            )
            labels[rows] = np.where(alive, side, "D")
        elif person in slot_of:
            xy[rows, slot_of[person]] = np.column_stack([x, y])
    tracking = Tracking(
        players=xy,
        ball=ball,
        team=team,
        period=period,
        video_frame=np.arange(n),
        fps=FPS,
        pitch_size=(length, width),
    )
    return tracking, z, labels, times, keepers, ids


def read_events(path, times, home, guest):
    """The events as dicts, each placed on the row nearest its time."""
    order = np.argsort(times)
    sorted_times = times[order]

    def row(stamp):
        if not stamp:
            return -1
        t = _seconds(stamp)
        k = int(np.clip(np.searchsorted(sorted_times, t), 1, len(order) - 1))
        best = k if abs(sorted_times[k] - t) < abs(sorted_times[k - 1] - t) else k - 1
        return int(order[best])

    team_code = {home: 0, guest: 1}
    out = []
    for event in ET.parse(path).getroot().iter("Event"):
        children = list(event)
        if not children:
            continue
        kind = children[0]
        detail = "/".join(e.tag for e in kind.iter() if e is not kind)
        attributes = {}
        for e in kind.iter():
            for key in ("Team", "Player", "Recipient", "DecisionTimestamp"):
                if key in e.attrib and key not in attributes:
                    attributes[key] = e.attrib[key]
        out.append(
            {
                "row": row(event.get("EventTime")),
                "decision_row": row(attributes.get("DecisionTimestamp")),
                "kind": kind.tag,
                "detail": detail,
                "team": team_code.get(attributes.get("Team"), -1),
                "player": attributes.get("Player", ""),
                "recipient": attributes.get("Recipient", ""),
            }
        )
    return out


def load_match(root, match_id, cache_dir=None):
    """One DflMatch; the arrays cached in cache_dir if given."""
    information_path, events_path, positions_path = match_files(root)[match_id]
    information = read_information(information_path)
    home, guest = information[0], information[1]
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"dfl_{match_id}_v{CACHE_VERSION}.npz"
    if cache is not None and cache.exists():
        with np.load(cache) as saved:
            arrays = {k: saved[k] for k in saved.files}
        tracking = Tracking(
            players=arrays["players"],
            ball=arrays["ball"],
            team=arrays["team"],
            period=arrays["period"],
            video_frame=np.arange(len(arrays["ball"])),
            fps=FPS,
            pitch_size=tuple(float(v) for v in arrays["pitch"]),
        )
        z, labels, times = arrays["z"], arrays["labels"], arrays["times"]
        keepers, ids = tuple(int(k) for k in arrays["keepers"]), arrays["ids"]
    else:
        tracking, z, labels, times, keepers, ids = build(
            information, read_positions(positions_path)
        )
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache,
                players=tracking.players,
                ball=tracking.ball,
                team=tracking.team,
                period=tracking.period,
                pitch=np.array(tracking.pitch_size),
                z=z,
                labels=labels,
                times=times,
                keepers=np.array(keepers, dtype=np.int64),
                ids=ids,
            )
    events = read_events(events_path, times, home, guest)
    return DflMatch(tracking, z, labels, times, keepers, ids, events)
