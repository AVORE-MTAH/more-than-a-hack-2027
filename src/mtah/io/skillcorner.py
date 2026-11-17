"""Read one match of SkillCorner's open data (github.com/SkillCorner/opendata, MIT) into arrays.

External data, for fitting values (docs/technical-document.md). A match folder holds:

    {id}_match.json                 teams, players (with their role), pitch size
    {id}_tracking_extrapolated.jsonl   one frame per line at 10 fps: the ball (x, y, z,
                                    is_detected), each player (x, y, is_detected: seen or
                                    extrapolated), the camera's view on the pitch
    {id}_phases_of_play.csv         phases of play: first and last frame, team in possession

SkillCorner's coordinates are metres from the pitch centre, y up; here they are moved to ours
(mtah.io.tracking): metres from the top-left corner, y down.

Labels are per frame: H or A for the team in possession in a phase of play, D for a frame in a
period but in no phase (the phases cover the ball in play only), X outside the periods.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mtah.io.tracking import Tracking

# Params
FPS = 10.0
TARGET_FPS = 25.0  # the provided match's frame rate (upsampled)
CORNERS = ("top_left", "bottom_left", "bottom_right", "top_right")


@dataclass(frozen=True)
class SkillCornerMatch:
    """One match. Frames are SkillCorner's frame numbers, 0 onwards.

    tracking         players (detected or extrapolated) and the ball where detected, NaN else
    ball_estimate    (frames, 2) the ball where SkillCorner has a position, detected or not
    ball_z           (frames,) its height in metres, NaN without a position
    ball_detected    (frames,) bool
    player_detected  (frames, slots) bool: seen on the frame, not extrapolated
    view             (frames, 4, 2) the camera's view on the pitch, corners in CORNERS order
    labels           (frames,) "H", "A", "D" or "X"
    keepers          slots of the players whose role is goalkeeper
    player_ids       (slots,) SkillCorner player ids, in slot order
    """

    tracking: Tracking
    ball_estimate: np.ndarray
    ball_z: np.ndarray
    ball_detected: np.ndarray
    player_detected: np.ndarray
    view: np.ndarray
    labels: np.ndarray
    keepers: tuple
    player_ids: np.ndarray


def match_ids(root):
    """The match ids with a tracking file under root (the repository's data/matches)."""
    root = Path(root)
    return sorted(
        int(p.parent.name) for p in root.glob("*/*_tracking_extrapolated.jsonl")
    )


def load_match(root, match_id):
    """One match from root/{id}/: the folder layout of SkillCorner's data/matches."""
    folder = Path(root) / str(match_id)
    meta = json.loads((folder / f"{match_id}_match.json").read_text(encoding="utf-8"))
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home = meta["home_team"]["id"]
    players = meta["players"]
    ids = np.array([p["id"] for p in players], dtype=np.int64)
    slot_of = {int(pid): k for k, pid in enumerate(ids)}
    team = np.array([0 if p["team_id"] == home else 1 for p in players], dtype=np.int8)
    keepers = tuple(
        k
        for k, p in enumerate(players)
        if (p.get("player_role") or {}).get("acronym") == "GK"
    )

    lines = (folder / f"{match_id}_tracking_extrapolated.jsonl").read_text(
        encoding="utf-8"
    )
    frames = [json.loads(line) for line in lines.splitlines() if line.strip()]
    n = max(f["frame"] for f in frames) + 1 if frames else 0
    xy = np.full((n, len(ids), 2), np.nan)
    seen = np.zeros((n, len(ids)), dtype=bool)
    ball = np.full((n, 2), np.nan)
    z = np.full(n, np.nan)
    detected = np.zeros(n, dtype=bool)
    view = np.full((n, 4, 2), np.nan)
    period = np.zeros(n, dtype=np.int8)

    def ours(x, y):
        return x + length / 2, width / 2 - y

    for f in frames:
        i = f["frame"]
        period[i] = f["period"] or 0
        b = f["ball_data"] or {}
        if b.get("x") is not None:
            ball[i] = ours(b["x"], b["y"])
            z[i] = np.nan if b.get("z") is None else b["z"]
            detected[i] = bool(b.get("is_detected"))
        corners = f.get("image_corners_projection") or {}
        for c, name in enumerate(CORNERS):
            if corners.get(f"x_{name}") is not None:
                view[i, c] = ours(corners[f"x_{name}"], corners[f"y_{name}"])
        for p in f["player_data"] or []:
            k = slot_of.get(int(p["player_id"]))
            if k is not None and p.get("x") is not None:
                xy[i, k] = ours(p["x"], p["y"])
                seen[i, k] = bool(p.get("is_detected"))

    labels = _labels(folder / f"{match_id}_phases_of_play.csv", period, home)
    tracking = Tracking(
        players=xy,
        ball=np.where(detected[:, None], ball, np.nan),
        team=team,
        period=period,
        video_frame=np.arange(n),
        fps=FPS,
        pitch_size=(length, width),
    )
    return SkillCornerMatch(
        tracking=tracking,
        ball_estimate=ball,
        ball_z=z,
        ball_detected=detected,
        player_detected=seen,
        view=view,
        labels=labels,
        keepers=keepers,
        player_ids=ids,
    )


def upsampled(match, fps=TARGET_FPS):
    """The match at `fps` (the provided match's 25): what the model sees is then on the same
    time grid as the submission's, so per-frame rates and per-frame evidence carry over.

    Positions (players, ball, its estimate and height, the view) are linear between the two
    nearest original frames, NaN where either lacks one or they lie in different periods;
    labels, periods and detection flags are the nearest original frame's.
    """
    t = match.tracking
    n = len(t.period)
    if n == 0:
        return match
    m = int(np.floor((n - 1) * fps / t.fps + 1e-9)) + 1
    src = np.arange(m) * (t.fps / fps)
    lo = np.minimum(np.floor(src + 1e-9).astype(np.int64), n - 1)
    hi = np.minimum(lo + 1, n - 1)
    w = src - lo
    near = np.where(w > 0.5, hi, lo)
    split = t.period[lo] != t.period[hi]

    def linear(a):
        shape = (-1,) + (1,) * (a.ndim - 1)
        weight = w.reshape(shape)
        out = np.where(weight == 0, a[lo], a[lo] * (1 - weight) + a[hi] * weight)
        out[split & (w > 0)] = np.nan
        return out

    tracking = Tracking(
        players=linear(t.players),
        ball=linear(t.ball),
        team=t.team,
        period=t.period[near],
        video_frame=np.arange(m),
        fps=float(fps),
        pitch_size=t.pitch_size,
    )
    return SkillCornerMatch(
        tracking=tracking,
        ball_estimate=linear(match.ball_estimate),
        ball_z=linear(match.ball_z),
        ball_detected=match.ball_detected[near],
        player_detected=match.player_detected[near],
        view=linear(match.view),
        labels=match.labels[near],
        keepers=match.keepers,
        player_ids=match.player_ids,
    )


def _labels(path, period, home):
    """H/A inside the phases of play, D elsewhere in a period, X outside the periods."""
    labels = np.where(period > 0, "D", "X").astype("<U1")
    with open(path, encoding="utf-8", newline="") as handle:
        for phase in csv.DictReader(handle):
            first, last = int(phase["frame_start"]), int(phase["frame_end"])
            side = "H" if int(phase["team_in_possession_id"]) == home else "A"
            span = slice(max(first, 0), min(last, len(labels) - 1) + 1)
            labels[span] = np.where(period[span] > 0, side, labels[span])
    return labels
