"""The SkillCorner reader (mtah.io.skillcorner) on a small synthetic match folder."""

import json

import numpy as np

from mtah.io.skillcorner import CORNERS, load_match, match_ids, upsampled


def write_match(root, match_id=7):
    folder = root / str(match_id)
    folder.mkdir(parents=True)
    meta = {
        "pitch_length": 105,
        "pitch_width": 68,
        "home_team": {"id": 1},
        "away_team": {"id": 2},
        "players": [
            {"id": 10, "team_id": 1, "player_role": {"acronym": "GK"}},
            {"id": 11, "team_id": 1, "player_role": {"acronym": "CF"}},
            {"id": 20, "team_id": 2, "player_role": {"acronym": "GK"}},
        ],
    }
    (folder / f"{match_id}_match.json").write_text(json.dumps(meta), encoding="utf-8")

    def frame(i, period, ball, players):
        corners = {f"{a}_{c}": 1.0 * k for k, c in enumerate(CORNERS) for a in "xy"}
        return {
            "frame": i,
            "period": period,
            "ball_data": ball,
            "possession": {"player_id": None, "group": None},
            "image_corners_projection": corners if period else {},
            "player_data": players,
        }

    nothing = {"x": None, "y": None, "z": None, "is_detected": None}
    rows = [
        frame(0, None, nothing, []),  # before the match
        frame(
            1,
            1,
            {"x": 0.0, "y": 0.0, "z": 0.1, "is_detected": True},
            [{"player_id": 11, "x": -10.0, "y": 5.0, "is_detected": True}],
        ),
        frame(
            2,
            1,
            {"x": 1.0, "y": -2.0, "z": 3.0, "is_detected": False},  # extrapolated
            [{"player_id": 20, "x": 50.0, "y": 0.0, "is_detected": False}],
        ),
        frame(3, 1, nothing, []),
    ]
    (folder / f"{match_id}_tracking_extrapolated.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (folder / f"{match_id}_phases_of_play.csv").write_text(
        "frame_start,frame_end,team_in_possession_id\n1,2,2\n", encoding="utf-8"
    )


def test_load_match(tmp_path):
    write_match(tmp_path)
    assert match_ids(tmp_path) == [7]
    m = load_match(tmp_path, 7)
    t = m.tracking
    assert t.fps == 10.0 and t.pitch_size == (105.0, 68.0)
    assert t.team.tolist() == [0, 0, 1] and m.keepers == (0, 2)
    assert t.period.tolist() == [0, 1, 1, 1]
    # Corner origin, y down: the centre spot is (52.5, 34).
    np.testing.assert_allclose(t.ball[1], [52.5, 34.0])
    assert np.isnan(t.ball[2]).all() and np.isnan(t.ball[3]).all()  # not detected
    np.testing.assert_allclose(m.ball_estimate[2], [53.5, 36.0])
    assert m.ball_detected.tolist() == [False, True, False, False]
    np.testing.assert_allclose(m.ball_z[1:3], [0.1, 3.0])
    np.testing.assert_allclose(t.players[1, 1], [42.5, 29.0])
    np.testing.assert_allclose(t.players[2, 2], [102.5, 34.0])
    assert m.player_detected[1, 1] and not m.player_detected[2, 2]
    assert m.labels.tolist() == ["X", "A", "A", "D"]  # away's phase on frames 1-2
    assert np.isfinite(m.view[1]).all() and np.isnan(m.view[0]).all()


def test_upsampled_to_25_fps(tmp_path):
    write_match(tmp_path)
    m = upsampled(load_match(tmp_path, 7))
    t = m.tracking
    # Frames 0-3 at 10 fps span 0.3 s: 8 frames at 25 fps, original frame 0.4 k.
    assert t.fps == 25.0 and len(t.period) == len(m.labels) == 8
    assert t.video_frame.tolist() == list(range(8))
    # The nearest original frame: 0 0 1 1 2 2 2 3.
    assert t.period.tolist() == [0, 0, 1, 1, 1, 1, 1, 1]
    assert m.labels.tolist() == ["X", "X", "A", "A", "A", "A", "A", "D"]
    assert m.ball_detected.tolist() == [False, False, True, True] + [False] * 4
    # Between frames of different periods (0 and 1): no position.
    assert np.isnan(t.ball[1:3]).all() and np.isnan(m.ball_estimate[1:3]).all()
    # 1.2: 0.2 of the way from frame 1 to 2; the ball is not detected on 2, so NaN,
    # while the estimate is there on both, so linear.
    assert np.isnan(t.ball[3]).all()
    np.testing.assert_allclose(m.ball_estimate[3], [52.7, 34.4])
    # 2.0 is frame 2 itself, though frame 3 has no estimate.
    np.testing.assert_allclose(m.ball_estimate[5], [53.5, 36.0])
    assert np.isnan(m.ball_estimate[6]).all()
    np.testing.assert_allclose(m.ball_z[5], 3.0)
    assert np.isnan(t.players[1:5, 1]).all()  # player 11 is only on frame 1


def test_load_real_layout_if_present():
    """On the real open data when it is on this machine (skipped otherwise)."""
    import os

    import pytest

    root = os.environ.get("MTAH_SKILLCORNER_DIR")
    if not root:
        pytest.skip("MTAH_SKILLCORNER_DIR not set")
    ids = match_ids(root)
    m = load_match(root, ids[0])
    assert set(np.unique(m.labels)) <= {"H", "A", "D", "X"}
