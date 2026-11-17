"""Load Metrica's FIFA/EPTS tracking export into plain numpy arrays in metres.

kloppy parses the metadata XML and raw text pair; this module flattens its frame objects into
arrays so the rest of the code never touches kloppy. Positions keep the Metrica orientation
(origin at the top-left corner, y growing downwards, the same convention as the homography),
scaled from 0-1 to metres.

A full match through kloppy takes minutes, so a full load can be cached: the 0-1 arrays go to an
.npz in `cache_dir` and later loads of the same, unchanged files read that instead.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from kloppy import metrica
from kloppy.domain import Ground

# Metres, measured from the corrected tracking (games.py). The ATD metadata says 105 x 68.
PITCH_SIZE = (105.3, 68.0)
# Bump when the cached arrays change meaning, so old cache files are ignored.
CACHE_VERSION = 1
PER_FRAME = ("players", "ball", "period", "video_frame")


@dataclass(frozen=True)
class Tracking:
    """One tracking dataset as arrays. Missing observations are NaN.

    players      (frames, slots, 2)  x, y in metres
    ball         (frames, 2)         x, y in metres
    team         (slots,)            0 = home, 1 = away
    period       (frames,)           1, 2, ...; 0 outside any period
    video_frame  (frames,)           video frame index, frame_id - 1
    """

    players: np.ndarray
    ball: np.ndarray
    team: np.ndarray
    period: np.ndarray
    video_frame: np.ndarray
    fps: float
    pitch_size: tuple[float, float]


def load_tracking(
    meta_path, raw_path, pitch_size=PITCH_SIZE, limit=None, cache_dir=None
) -> Tracking:
    """Load an EPTS metadata/raw pair. `limit` keeps only the first `limit` records.

    With `cache_dir`, a cache file for these files is read if it exists; otherwise a full load
    (no `limit`) writes one.
    """
    meta_path, raw_path = Path(meta_path), Path(raw_path)
    cached = (
        cache_path(cache_dir, meta_path, raw_path) if cache_dir is not None else None
    )

    if cached is not None and cached.exists():
        arrays = _read_cache(cached, limit)
    else:
        arrays = _load_kloppy(meta_path, raw_path, limit)
        if cached is not None and limit is None:
            _write_cache(cached, arrays)

    length, width = pitch_size
    # In place: the arrays are fresh from kloppy or the cache, and a full match's players array
    # is ~150 MB, so a scaled copy would briefly double it.
    scale = np.array([length, width])
    arrays["players"] *= scale
    arrays["ball"] *= scale
    return Tracking(
        players=arrays["players"],
        ball=arrays["ball"],
        team=arrays["team"],
        period=arrays["period"],
        video_frame=arrays["video_frame"],
        fps=float(arrays["fps"]),
        pitch_size=(float(length), float(width)),
    )


def cache_path(cache_dir, meta_path, raw_path):
    """Cache file for this pair, keyed on both files' size and modification time."""
    meta_path, raw_path = Path(meta_path), Path(raw_path)
    stats = [path.stat() for path in (meta_path, raw_path)]
    key = "|".join(
        [str(CACHE_VERSION)] + [f"{s.st_size}:{s.st_mtime_ns}" for s in stats]
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return Path(cache_dir) / f"{raw_path.stem}-{digest}.npz"


def _load_kloppy(meta_path, raw_path, limit):
    """The 0-1 arrays, straight from kloppy."""
    dataset = metrica.load_tracking_epts(
        meta_data=str(meta_path), raw_data=str(raw_path), limit=limit
    )
    slots = [player for team in dataset.metadata.teams for player in team.players]
    n_frames, n_slots = len(dataset.records), len(slots)

    players = np.full((n_frames, n_slots, 2), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    period = np.zeros(n_frames, dtype=np.int8)
    video_frame = np.empty(n_frames, dtype=np.int64)

    for i, frame in enumerate(dataset.records):
        for j, player in enumerate(slots):
            data = frame.players_data.get(player)
            if data is not None and data.coordinates is not None:
                players[i, j] = data.coordinates.x, data.coordinates.y
        if frame.ball_coordinates is not None:
            ball[i] = frame.ball_coordinates.x, frame.ball_coordinates.y
        if frame.period is not None:
            period[i] = frame.period.id
        video_frame[i] = frame.frame_id - 1

    team = np.array(
        [0 if p.team.ground == Ground.HOME else 1 for p in slots], dtype=np.int8
    )
    return {
        "players": players,
        "ball": ball,
        "team": team,
        "period": period,
        "video_frame": video_frame,
        "fps": np.float64(dataset.metadata.frame_rate),
    }


def _read_cache(path, limit):
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    if limit is not None:
        for name in PER_FRAME:
            arrays[name] = arrays[name][:limit]
    return arrays


def _write_cache(path, arrays):
    """Write through a temporary file, so an interrupted write never leaves a broken cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    with open(partial, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(partial, path)
