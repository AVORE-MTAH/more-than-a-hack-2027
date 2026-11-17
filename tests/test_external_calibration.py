"""The external calibration (mtah.tune.calibration, scripts/calibrate_external.py)."""

import json
from pathlib import Path

import numpy as np
from synthetic_metrica import write_metrica
from test_dfl import write_match

from mtah.io import dfl
from mtah.io.metrica_csv import dead_intervals, load_metrica_csv
from mtah.predict.hmm import Features
from mtah.tune.calibration import (
    METRICA_KEYS,
    POSSESSION_KEYS,
    joined,
    match_features,
    metrica_labels,
)


def feats(n, value):
    return Features(
        nearest=np.full(n, value, np.int8),
        distance=np.full(n, value, np.int8),
        dead=np.full(n, value, np.int8),
        stoppage=np.ones(n, bool),
    )


def test_joined_puts_an_x_between_matches():
    both, labels = joined(
        [(feats(3, 0), np.array(["H"] * 3)), (feats(2, 1), np.array(["D"] * 2))]
    )
    assert labels.tolist() == ["H", "H", "H", "X", "D", "D"]
    assert both.nearest.tolist() == [0, 0, 0, -1, 1, 1]
    assert both.stoppage.tolist() == [True, True, True, False, True, True]
    assert both.flight is None


def test_metrica_labels(tmp_path):
    home, away, events = write_metrica(tmp_path)
    t = load_metrica_csv(home, away)
    dead, unknown = dead_intervals(events)
    labels = metrica_labels(t, dead, unknown)
    assert len(labels) == t.video_frame.max() + 1
    for first, last in dead:
        assert set(labels[first : last + 1]) <= {"D", "X"}
    assert "H" in labels and set(labels) <= {"H", "D", "X"}


def test_match_features_checks_the_length(tmp_path):
    import pytest

    write_match(tmp_path)
    m = dfl.load_match(tmp_path, "DFL-MAT-TEST01")
    f, lab = match_features(m.tracking, m.labels, clean=False)
    assert len(f.nearest) == len(lab) == len(m.labels)
    with pytest.raises(ValueError, match="labels for"):
        match_features(m.tracking, m.labels[:-1], clean=False)


def test_calibrate_external_script(tmp_path, run_script, capsys):
    (tmp_path / "dfl").mkdir()
    write_match(tmp_path / "dfl")
    games = tmp_path / "metrica"
    for g in (1, 2):
        folder = games / f"Sample_Game_{g}"
        folder.mkdir(parents=True)
        written = write_metrica(folder)
        names = (
            "RawTrackingData_Home_Team",
            "RawTrackingData_Away_Team",
            "RawEventsData",
        )
        for path, name in zip(written, names, strict=True):
            Path(path).rename(folder / f"Sample_Game_{g}_{name}.csv")
    out = tmp_path / "calibration.json"
    run_script(
        "calibrate_external",
        "--dfl",
        tmp_path / "dfl",
        "--metrica",
        games,
        "--out",
        out,
    )
    cal = json.loads(out.read_text(encoding="utf-8"))
    assert set(METRICA_KEYS) <= set(cal) and {"team_agree", "switch"} <= set(cal)
    assert set(cal) - {"sources"} <= set(POSSESSION_KEYS) | set(METRICA_KEYS)
    assert "1 DFL matches, 2 Metrica games" in capsys.readouterr().out


def test_dead_weights_report(tmp_path, run_script, capsys):
    from test_hmm import calibration_for_tests

    from mtah.predict.hmm import save_calibration

    games = tmp_path / "metrica"
    for g in (1, 2):
        folder = games / f"Sample_Game_{g}"
        folder.mkdir(parents=True)
        written = write_metrica(folder)
        names = (
            "RawTrackingData_Home_Team",
            "RawTrackingData_Away_Team",
            "RawEventsData",
        )
        for path, name in zip(written, names, strict=True):
            Path(path).rename(folder / f"Sample_Game_{g}_{name}.csv")
    calibration = tmp_path / "calibration.json"
    save_calibration(calibration_for_tests(), calibration)
    run_script("dead_weights_report", "--metrica", games, "--calibration", calibration)
    out = capsys.readouterr().out
    assert "D's F1 per game" in out and "leaving one game out" in out
    assert out.count("  game ") == 2


def test_calibrate_sources_takes_possession_from_skillcorner():
    from mtah.tune.calibration import calibrate_sources

    def pair(labels):
        return feats(len(labels), 0), np.array(list(labels))

    dfl_pairs = [pair("HHHHAAAA")]  # a switch every 4 frames
    skillcorner_pairs = [pair("HHHHHHHHHHHHAAAAAAAAAAAA")]
    metrica_pairs = [pair("HHHHDDDD")]
    cal, own, _ = calibrate_sources(dfl_pairs, metrica_pairs, skillcorner_pairs)
    assert (
        cal["switch"]
        == own["switch"]
        < calibrate_sources(dfl_pairs, metrica_pairs)[0]["switch"]
    )
    assert cal["sources"]["skillcorner"] == list(POSSESSION_KEYS)
    assert "dfl" not in cal["sources"] and "skillcorner_frames" in cal["sources"]
    assert "SkillCorner" in cal["sources"]["attribution"]


def test_calibrate_external_script_with_skillcorner(tmp_path, run_script, capsys):
    from test_skillcorner import write_match as write_skillcorner

    write_skillcorner(tmp_path / "sc", 7)
    write_skillcorner(tmp_path / "sc", 8)
    games = tmp_path / "metrica"
    for g in (1, 2):
        folder = games / f"Sample_Game_{g}"
        folder.mkdir(parents=True)
        written = write_metrica(folder)
        names = (
            "RawTrackingData_Home_Team",
            "RawTrackingData_Away_Team",
            "RawEventsData",
        )
        for path, name in zip(written, names, strict=True):
            Path(path).rename(folder / f"Sample_Game_{g}_{name}.csv")
    out = tmp_path / "calibration.json"
    run_script(
        "calibrate_external",
        "--skillcorner",
        tmp_path / "sc",
        "--metrica",
        games,
        "--out",
        out,
    )
    cal = json.loads(out.read_text(encoding="utf-8"))
    assert set(METRICA_KEYS) <= set(cal) and "skillcorner" in cal["sources"]
    assert "2 SkillCorner matches, 2 Metrica games" in capsys.readouterr().out


def test_weights_report_on_skillcorner(tmp_path, run_script, capsys):
    from test_hmm import calibration_for_tests
    from test_skillcorner import write_match as write_skillcorner

    from mtah.predict.hmm import save_calibration

    write_skillcorner(tmp_path / "sc", 7)
    write_skillcorner(tmp_path / "sc", 8)
    calibration = tmp_path / "calibration.json"
    save_calibration(calibration_for_tests(), calibration)
    run_script(
        "weights_report",
        "--skillcorner",
        tmp_path / "sc",
        "--calibration",
        calibration,
        "--log",
        tmp_path / "log.jsonl",
        "--sweeps",
        "1",
        "--grid",
        '{"w_team": [0.1, 0.3], "min_live_s": [0.1, 1.0]}',
    )
    out = capsys.readouterr().out
    assert "chosen:" in out and out.count("  7: ") + out.count("  8: ") == 2
    assert "leaving each match out of the calibration too" in out
    assert len((tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()) >= 2
