"""Tracking cache and label loading, on synthetic data."""

import os

import numpy as np

from mtah.io import tracking as tracking_module
from mtah.io.tracking import cache_path, load_tracking


def _no_kloppy(*args, **kwargs):
    raise AssertionError("kloppy was called although a cache exists")


def test_cache_written_then_used(epts, tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    fresh = load_tracking(*epts, cache_dir=cache_dir)
    assert [p.name for p in cache_dir.iterdir()] == [cache_path(cache_dir, *epts).name]

    monkeypatch.setattr(tracking_module.metrica, "load_tracking_epts", _no_kloppy)
    cached = load_tracking(*epts, cache_dir=cache_dir)
    for name in ("players", "ball", "team", "period", "video_frame"):
        np.testing.assert_array_equal(getattr(cached, name), getattr(fresh, name))
    assert cached.fps == fresh.fps

    # The cache holds 0-1 positions, so another pitch size scales from the same file.
    other = load_tracking(*epts, pitch_size=(100.0, 50.0), cache_dir=cache_dir)
    np.testing.assert_allclose(other.ball[0], [50.0, 25.0])

    limited = load_tracking(*epts, limit=3, cache_dir=cache_dir)
    assert len(limited.video_frame) == 3 and limited.players.shape[0] == 3


def test_partial_load_writes_no_cache(epts, tmp_path):
    cache_dir = tmp_path / "cache"
    load_tracking(*epts, limit=3, cache_dir=cache_dir)
    assert not cache_dir.exists()


def test_changed_file_invalidates_cache(epts, tmp_path):
    cache_dir = tmp_path / "cache"
    load_tracking(*epts, cache_dir=cache_dir)
    before = cache_path(cache_dir, *epts)

    raw = epts[1]
    stat = raw.stat()
    os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert cache_path(cache_dir, *epts) != before
