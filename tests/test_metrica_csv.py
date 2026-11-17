"""Metrica sample-game CSV loader and dead stretches from events, on synthetic files."""

import numpy as np
import pytest
from synthetic_metrica import N_FRAMES, write_metrica

from mtah.io.metrica_csv import SAMPLE_PITCH, dead_intervals, load_metrica_csv


@pytest.fixture
def files(tmp_path):
    return write_metrica(tmp_path)


def test_load_tracking(files):
    tracking = load_metrica_csv(files[0], files[1])
    assert tracking.players.shape == (N_FRAMES, 4, 2)
    assert tracking.team.tolist() == [0, 0, 1, 1]
    assert tracking.video_frame[0] == 0 and tracking.video_frame[-1] == N_FRAMES - 1
    assert set(tracking.period) == {1, 2}
    length, width = SAMPLE_PITCH
    np.testing.assert_allclose(tracking.players[0, 0], [0.1 * length, 0.5 * width])
    np.testing.assert_allclose(tracking.players[0, 2], [0.6 * length, 0.5 * width])
    np.testing.assert_allclose(tracking.ball[9], [0.1 * length, 0.25 * width])
    # Player2 is NaN on frame 10 (video frame 9), observed elsewhere.
    assert np.isnan(tracking.players[9, 1]).all()
    assert not np.isnan(tracking.players[8, 1]).any()


def test_dead_intervals(files):
    dead, unknown = dead_intervals(files[2])
    # Ball out -> throw-in, foul -> free kick, goal -> kickoff, shot out -> goal kick.
    assert dead == [(5, 9), (14, 17), (23, 29), (35, 39)]
    # A free kick no event explains: unknown since the event before it.
    assert unknown == [(45, 48)]


def test_dead_intervals_lookback(files):
    dead, unknown = dead_intervals(files[2], lookback_s=0.1)  # 2 frames: none qualify
    assert dead == []
    assert (5, 9) not in unknown and len(unknown) == 5
