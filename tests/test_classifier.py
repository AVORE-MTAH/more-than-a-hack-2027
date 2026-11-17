"""The hybrid's per-frame classifier (mtah.predict.classifier) on synthetic data."""

from dataclasses import replace

import numpy as np
from synthetic_tracks import scene

from mtah.predict import classifier as clf


def test_frame_table_from_tracking():
    tracking, t = scene((), seconds=8, periods=2)
    ball = np.full((len(t), 2), np.nan)
    ball[:100] = tracking.players[:100, 0]  # on home's first player
    tracking = replace(tracking, ball=ball)
    table = clf.frame_table(tracking, len(t))
    assert np.allclose(table["home_m"][:100], 0.0)
    assert (table["away_m"][:100] > 20).all()
    assert table["missing"][:100].sum() == 0 and table["missing"][100:].all()
    # Home defends x = 0 in period 1, x = 105 in period 2 (synthetic_tracks).
    assert np.allclose(table["along"][:100], ball[:100, 0])
    x, n_columns = clf.design(table)
    assert x.shape == (len(t), n_columns)
    assert (
        np.asarray(x.sum(axis=1)).ravel() == x[0].sum()
    ).all()  # one bin per feature


def table_for(home_m, away_m, fps=25.0):
    n = len(home_m)
    nan = np.full(n, np.nan)
    none = np.full(n, -1, np.int8)
    zero = np.zeros(n, np.int8)
    return {
        "home_m": np.asarray(home_m, float),
        "away_m": np.asarray(away_m, float),
        "missing": np.isnan(home_m).astype(np.int8),
        "dribbled": zero,
        "still": zero,
        "ball_speed": nan,
        "flight": zero,
        "kicker": none,
        "receiver": none,
        "pace": np.full(n, 1.5),
        "stoppage": zero,
        "offline": none,
        "along": np.full(n, 52.5),
        "edge": np.full(n, 30.0),
        "fps": fps,
        "length": 105.0,
    }


def test_fit_learns_the_nearer_team_and_is_symmetric():
    rng = np.random.default_rng(1)
    n = 3000
    labels = rng.choice(np.array(["H", "A", "D"]), n)
    near, far = rng.uniform(0, 1, n), rng.uniform(3, 8, n)
    home = np.where(labels == "H", near, far)
    away = np.where(labels == "A", near, far)
    home[labels == "D"] = np.nan
    away[labels == "D"] = np.nan
    table = table_for(home, away)
    model = clf.fit([table], [labels], every=1)
    assert model["converged"]
    assert np.allclose(
        model["priors"][0], model["priors"][1]
    )  # mirrored: H and A alike
    got = np.argmax(clf.log_proba(model, table), axis=1)
    assert np.mean(got == clf.labels_codes(labels)) > 0.9
    # Symmetric: the mirrored table's posterior is the plain one with H and A swapped.
    lp = clf.log_proba(model, table)
    lm = clf.log_proba(model, clf.mirrored(table))
    np.testing.assert_allclose(lp, lm[:, [1, 0, 2]], atol=1e-9)
    ll = clf.scaled_loglik(model, table)
    assert ll.shape == (n, 3) and np.isfinite(ll).all()


def test_save_and_load(tmp_path):
    model = {"coef": [[0.0, 0.0, 0.0]], "priors": [0.4, 0.4, 0.2], "n_columns": 1}
    path = tmp_path / "m.json"
    clf.save_model(model, path, attribution="test")
    back = clf.load_model(path)
    assert back["attribution"] == "test" and back["coef"] == model["coef"]


def test_hmm_labels_with_the_shipped_classifier():
    from test_hmm import calibration_for_tests

    from mtah.predict.hmm import hmm_labels

    tracking, t = scene((), seconds=8)
    ball = tracking.players[:, 0].copy()  # on home's first player throughout
    tracking = replace(tracking, ball=ball)
    n = len(t)
    model = clf.load_model()
    assert model["converged"] and "SkillCorner" in model["attribution"]
    labels = hmm_labels(
        tracking,
        n,
        [(0, n - 1)],
        [],
        calibration_for_tests(),
        classifier=model,
        w_clf=0.1,
    )
    assert len(labels) == n and set(labels) <= {"H", "A", "D"}
    assert np.mean(labels == "H") > 0.5


def test_train_classifier_script(tmp_path, run_script, capsys):
    from pathlib import Path

    from synthetic_metrica import write_metrica
    from test_skillcorner import write_match

    write_match(tmp_path / "sc", 7)
    games = tmp_path / "metrica"
    for g in (1, 2):
        folder = games / f"Sample_Game_{g}"
        folder.mkdir(parents=True)
        names = (
            "RawTrackingData_Home_Team",
            "RawTrackingData_Away_Team",
            "RawEventsData",
        )
        for path, name in zip(write_metrica(folder), names, strict=True):
            Path(path).rename(folder / f"Sample_Game_{g}_{name}.csv")
    out = tmp_path / "clf.json"
    run_script(
        "train_classifier",
        "--skillcorner",
        tmp_path / "sc",
        "--metrica",
        games,
        "--out",
        out,
    )
    model = clf.load_model(out)
    assert (
        len(model["coef"]) == model["n_columns"] and "Metrica" in model["attribution"]
    )
    assert "1 SkillCorner matches, 2 Metrica games" in capsys.readouterr().out


def test_hybrid_report_script(tmp_path, run_script, capsys):
    from pathlib import Path

    from synthetic_metrica import write_metrica
    from test_dfl import write_match as write_dfl
    from test_hmm import calibration_for_tests
    from test_skillcorner import write_match as write_skillcorner

    from mtah.predict.hmm import save_calibration

    write_skillcorner(tmp_path / "sc", 7)
    (tmp_path / "dfl").mkdir()
    write_dfl(tmp_path / "dfl")
    games = tmp_path / "metrica"
    for g in (1, 2):
        folder = games / f"Sample_Game_{g}"
        folder.mkdir(parents=True)
        names = (
            "RawTrackingData_Home_Team",
            "RawTrackingData_Away_Team",
            "RawEventsData",
        )
        for path, name in zip(write_metrica(folder), names, strict=True):
            Path(path).rename(folder / f"Sample_Game_{g}_{name}.csv")
    calibration = tmp_path / "calibration.json"
    save_calibration(calibration_for_tests(), calibration)
    run_script(
        "hybrid_report",
        "--dfl",
        tmp_path / "dfl",
        "--skillcorner",
        tmp_path / "sc",
        "--metrica",
        games,
        "--calibration",
        calibration,
        "--weights",
        "0",
        "0.1",
    )
    out = capsys.readouterr().out
    assert out.count("   DFL ") == 2 and "Metrica " in out
