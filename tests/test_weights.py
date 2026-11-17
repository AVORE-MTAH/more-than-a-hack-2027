"""The missing-ball rule (mtah.tune.missing) and the weight search's pieces (mtah.tune.weights)."""

import numpy as np

from mtah.tune.missing import degrade, fit
from mtah.tune.weights import Setting, halves_of, leave_one_out, scores


def test_missing_fit_and_degrade_follow_the_states():
    rng = np.random.default_rng(0)
    # 50 s live then 50 s dead, repeated: the ball unseen 10% of live, 90% of dead frames.
    labels = np.array((["H"] * 1250 + ["D"] * 1250) * 20)
    ball = np.ones((len(labels), 2))
    for first in range(0, len(labels), 2500):
        for k in range(0, 1250, 250):
            ball[first + k : first + k + 25] = np.nan  # 1 s in every 10 s live
        ball[first + 1250 : first + 2375] = np.nan  # 45 s of the 50 s dead
    fitted = fit([(ball, labels)], 25.0)
    assert abs(fitted["live"]["share"] - 0.10) < 1e-9
    assert abs(fitted["dead"]["share"] - 0.90) < 1e-9
    out = degrade(np.ones_like(ball), labels, 25.0, fitted, rng)
    missing = np.isnan(out[:, 0])
    live = labels == "H"
    assert 0.03 < missing[live].mean() < 0.2 and missing[~live].mean() > 0.7


def test_degrade_leaves_x_frames():
    labels = np.array(["X"] * 100)
    fitted = {
        "live": {"share": 0.5, "seen_s": [1.0], "missing_s": [1.0]},
        "dead": {"share": 0.5, "seen_s": [1.0], "missing_s": [1.0]},
    }
    out = degrade(np.ones((100, 2)), labels, 25.0, fitted, np.random.default_rng(1))
    assert not np.isnan(out).any()


def test_scores_and_leave_one_out():
    truth = np.array(["H"] * 50 + ["D"] * 20 + ["A"] * 50)
    perfect = scores(truth, truth)
    assert perfect["spell_f1"] == 1.0 and perfect["macro_f1"] == 1.0
    assert perfect["n_gt"] == perfect["n_pred"] == 2
    wrong = scores(truth, np.array(["A"] * 120))
    assert wrong["spell_f1"] < 1.0
    a, b = Setting(w_team=0.1).key(), Setting(w_team=0.2).key()
    table = {
        a: [{"spell_f1": 0.9}, {"spell_f1": 0.1}, {"spell_f1": 0.9}],
        b: [{"spell_f1": 0.5}, {"spell_f1": 0.8}, {"spell_f1": 0.5}],
    }
    held = leave_one_out(table)
    # Held out match 1, the others prefer a (0.9 + 0.9), which scores 0.1 on it.
    assert held[1] == (1, a, 0.1)


def test_halves_of():
    period = np.array([0, 1, 1, 1, 0, 2, 2])
    assert halves_of(period) == [(1, 3), (5, 6)]
