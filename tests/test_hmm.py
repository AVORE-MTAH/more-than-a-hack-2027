"""The semi-Markov possession model: Viterbi, evidence, calibration, labels."""

import itertools
import time

import numpy as np
import pytest
from synthetic_tracks import FPS, N_OUTFIELD, scene

from mtah.predict.hmm import (
    DRIBBLED,
    MISSING,
    N_DEAD,
    N_DISTANCE,
    SEEN,
    Chains,
    Features,
    calibrate,
    evidence,
    features,
    hmm_labels,
    load_calibration,
    save_calibration,
    viterbi,
)

H, A, D = 0, 1, 2


# The shortest possession these mechanics tests are written for (short possessions of a few
# frames); the submission's is MIN_LIVE_S, fitted on external data.
MECHANICS_MIN_LIVE_S = 0.1


def chains(min_live, min_dead, switch=0.05, stop=0.03, restart=0.1):
    return Chains(min_live, min_dead, np.log(switch), np.log(stop), np.log(restart / 2))


def path_score(states, ll, c, stops=None, restarts=None, switches=None):
    """Score of a label path under the model, or -inf if it breaks a minimum duration.

    stops, restarts and switches: per-frame log rates of the transition into each frame, as
    viterbi takes them; the chains' constants without.
    """
    n = len(states)
    stops = np.full(n, c.log_stop) if stops is None else stops
    restarts = np.full(n, c.log_restart) if restarts is None else restarts
    switches = np.full(n, c.log_switch) if switches is None else switches
    m = {H: c.min_live, A: c.min_live, D: c.min_dead}
    total = ll[0, states[0]]
    run_start, first_run = 0, True
    for t in range(1, n):
        live = np.log1p(-(np.exp(switches[t]) + np.exp(stops[t])))
        stay = {H: live, A: live, D: np.log1p(-2 * np.exp(restarts[t]))}
        move = {
            (H, A): switches[t],
            (A, H): switches[t],
            (H, D): stops[t],
            (A, D): stops[t],
            (D, H): restarts[t],
            (D, A): restarts[t],
        }
        prev, cur = states[t - 1], states[t]
        if cur == prev:
            # Inside a run's first m frames the chain advances for free; after, it stays.
            in_chain = not first_run and t - run_start < m[cur]
            total += 0.0 if in_chain else stay[cur]
        else:
            if not first_run and t - run_start < m[prev]:
                return -np.inf
            total += move[(prev, cur)]
            run_start, first_run = t, False
        total += ll[t, cur]
    return total


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(60):
        n = int(rng.integers(1, 8))
        c = chains(int(rng.integers(1, 3)), int(rng.integers(1, 3)))
        ll = rng.normal(0, 2, (n, 3))
        best = max(
            path_score(np.array(p), ll, c)
            for p in itertools.product((H, A, D), repeat=n)
        )
        got = viterbi(ll, c)
        np.testing.assert_allclose(path_score(got, ll, c), best)


def test_a_short_burst_does_not_switch_a_long_one_does():
    c = chains(10, 20, switch=0.002, stop=0.001, restart=0.01)
    ll = np.zeros((200, 3))
    ll[:, H], ll[:, D] = 1.0, -5.0
    ll[100:104, A] = 4.0  # 4 frames of strong away evidence: a blip
    assert (viterbi(ll, c)[:200] == H).all()
    ll[100:140, A] = 4.0  # 40 frames: a possession change
    got = viterbi(ll, c)
    assert (got[:100] == H).all() and (got[100:140] == A).all()


def test_minimum_dead_duration():
    c = chains(5, 30, switch=0.01, stop=0.01, restart=0.05)
    ll = np.zeros((200, 3))
    ll[:, H], ll[:, D] = 1.0, -3.0
    ll[100:110, D] = 6.0  # 10 frames leaning dead: under the 30-frame minimum
    assert not (viterbi(ll, c) == D).any()
    ll[100:140, D] = 6.0  # 40 frames: a stoppage, exactly where the evidence is
    got = viterbi(ll, c)
    assert (
        (got[100:140] == D).all() and (got[:100] == H).all() and (got[140:] == H).all()
    )


def test_viterbi_is_fast():
    rng = np.random.default_rng(1)
    ll = rng.normal(0, 1, (80_000, 3))
    c = chains(15, 50, switch=0.002, stop=0.002, restart=0.01)
    start = time.perf_counter()
    viterbi(ll, c)
    assert time.perf_counter() - start < 15


def calibration_for_tests():
    agree = np.linspace(0.95, 0.6, N_DISTANCE)
    dead_given_dead = np.full(N_DEAD, 1.0)
    dead_given_live = np.full(N_DEAD, 1.0)
    dead_given_dead[MISSING] = 20.0  # missing ball, slow pitch: dead
    return {
        "team_agree": agree.tolist(),
        "dead_given_dead": (dead_given_dead / dead_given_dead.sum()).tolist(),
        "dead_given_live": (dead_given_live / dead_given_live.sum()).tolist(),
        "stoppage_given_dead": 0.4,
        "stoppage_given_live": 0.01,
        "switch": 0.003,
        "stop": 0.002,
        "restart": 0.02,
    }


def test_evidence_terms():
    feats = Features(
        nearest=np.array([H, A, -1, -1, -1], dtype=np.int8),
        distance=np.array([0, 0, -1, -1, -1], dtype=np.int8),
        dead=np.array([-1, -1, MISSING, DRIBBLED, -1], dtype=np.int8),
        stoppage=np.array([False, False, False, False, True]),
    )
    ll = evidence(
        feats, calibration_for_tests(), w_team=1.0, w_dead=1.0, w_stoppage=1.0
    )
    assert ll[0, H] > ll[0, A] and ll[1, A] > ll[1, H]  # a near player's team
    assert ll[2, D] > ll[2, H]  # missing ball, slow pitch
    assert ll[3, D] - ll[3, H] < 0 < ll[2, D] - ll[2, H]  # dribbled: leans live instead
    assert ll[4, D] > ll[4, H]  # a ball-out stoppage
    # The dead term says nothing of teams.
    np.testing.assert_allclose(ll[2, H], ll[2, A])
    half = evidence(
        feats, calibration_for_tests(), w_team=0.5, w_dead=0.5, w_stoppage=0.5
    )
    np.testing.assert_allclose(half, ll * 0.5)


def test_evidence_refuses_an_old_calibration():
    old = calibration_for_tests()
    old["dead_given_dead"] = old["dead_given_live"] = [1 / 12] * 12
    feats = Features(
        *(np.array([-1], dtype=np.int8) for _ in range(3)), np.zeros(1, bool)
    )
    with pytest.raises(ValueError, match="calibrate_external.py again"):
        evidence(feats, old)


def test_features_calibrate_and_round_trip(tmp_path):
    # Home slot 0 holds the ball (0.5 m) for 200 frames, then away slot 4 for 200.
    tracking, _ = scene((), seconds=16)
    ball = tracking.players[:, 0] + [0.5, 0.0]
    ball[200:] = tracking.players[200:, 4] + [0.5, 0.0]
    tracking = tracking.__class__(**{**tracking.__dict__, "ball": ball})
    feats = features(tracking, 400)
    assert (feats.nearest[:200] == H).all() and (feats.nearest[200:] == A).all()
    assert (feats.distance == 1).all()  # 0.5 m: the 0.5-1 m bucket
    assert (feats.dead == SEEN).all()  # a seen ball, whatever the players' speed

    labels = np.array(["H"] * 200 + ["A"] * 200, dtype=object)
    cal = calibrate(feats, labels)
    assert cal["team_agree"][1] == (400 + 1) / (400 + 2)
    assert cal["counts"]["switches"] == 1 and cal["counts"]["dead_frames"] == 0
    save_calibration(cal, tmp_path / "cal.json")
    assert load_calibration(tmp_path / "cal.json") == cal


def test_hmm_labels_recover_a_possession_change():
    tracking, _ = scene((), seconds=40)
    n = len(tracking.video_frame)
    ball = tracking.players[:, 0] + [0.5, 0.0]
    ball[500:] = tracking.players[500:, N_OUTFIELD - 1] + [0.5, 0.0]
    ball[300:310] = tracking.players[300:310, 5] + [0.5, 0.0]  # a 10-frame away touch
    tracking = tracking.__class__(**{**tracking.__dict__, "ball": ball})
    labels = hmm_labels(
        tracking, n + 10, [(0, n - 1)], [(0, 24)], calibration_for_tests(), 0.3, 0.3
    )
    assert "".join(labels[:25]) == "D" * 25  # the forced dead range
    assert (labels[25:500] == "H").all() and (labels[500:n] == "A").all()
    assert (labels[n:] == "X").all()
    assert FPS == 25.0


def pass_over_an_opponent():
    """Features of 300 frames of home possession; on 100-139 the ball, in flight, passes
    nearest an away player."""
    n = 300
    nearest = np.full(n, H, dtype=np.int8)
    nearest[100:140] = A
    flight = np.zeros(n, dtype=bool)
    flight[95:145] = True
    return Features(
        nearest=nearest,
        distance=np.zeros(n, dtype=np.int8),
        dead=np.full(n, SEEN, dtype=np.int8),
        stoppage=np.zeros(n, dtype=bool),
        flight=flight,
    )


def test_a_kick_counts_once_and_finds_a_short_possession():
    from mtah.predict.hmm import labels_from_features, possession_frames

    assert possession_frames([0, 0, -1, 1, 1, 1, 0, -1, -1]) == [2, 3, 1]
    assert possession_frames([]) == []

    # Home all along, but away touches it on 200-204 and passes it, in flight on 205-229 with
    # nobody near, back to home: a short away possession of 30 frames.
    n = 400
    nearest = np.full(n, H, dtype=np.int8)
    nearest[200:205] = A
    flight = np.zeros(n, dtype=bool)
    flight[205:230] = True
    nearest[flight] = -1
    feats = Features(
        nearest=nearest,
        distance=np.zeros(n, dtype=np.int8),
        dead=np.full(n, SEEN, dtype=np.int8),
        stoppage=np.zeros(n, dtype=bool),
        flight=flight,
        kicker=np.where(flight, A, -1).astype(np.int8),
    )
    cal = {**calibration_for_tests(), "kicker_agree": 0.9}
    # The kick weighs w_kick x log(0.9 / 0.1) in all, whatever the flight's length.
    plain = evidence(feats, cal, 0.15, 0.3, w_kick=0.0)
    kicked = evidence(feats, cal, 0.15, 0.3, w_kick=2.0) - plain
    lean = kicked[:, A] - kicked[:, H]
    np.testing.assert_allclose(lean.sum(), 2.0 * np.log(9))
    assert (lean[~flight] == 0).all()
    c = Chains.from_calibration(cal, FPS, min_live_s=MECHANICS_MIN_LIVE_S)

    def label(w_kick):
        return labels_from_features(
            feats,
            [(0, n - 1)],
            [],
            cal,
            c,
            0.15,
            0.3,
            w_kick=w_kick,
        )

    assert (
        label(0.0) == "H"
    ).all()  # five frames of a touch cannot pay for two switches
    found = label(8.0)
    assert (found[205:225] == "A").all() and (found[:195] == "H").all()
