"""Tracking cleaning on synthetic scenes: stale tracks and goalkeeper identification."""

from dataclasses import replace

import numpy as np
from synthetic_tracks import FPS, LENGTH, N_OUTFIELD, WIDTH, keeper_path, scene

from mtah.clean.tracks import (
    CleanConfig,
    clean_tracking,
    stale_mask,
)

K, X = N_OUTFIELD, N_OUTFIELD + 1  # the two extra slots of each scene
# The scenes are timed for these values: the rules' behaviour is tested, not the values fitted
# on external data (mtah.tune), which are the defaults.
MECHANICS = CleanConfig(
    stale_max_spread_m=0.10,
    coast_max_excess_m=0.02,
    seed_radius_m=12.0,
    seed_s=1.0,
    far_from_line_m=35.0,
)


def keeper_rows(result):
    return result.log[result.log["kind"] == "keeper"].reset_index(drop=True)


def path(t, times, points, jostle=0.3):
    """Positions interpolated through `points` at `times`, plus a small jostle."""
    points = np.asarray(points, dtype=float)
    xy = np.column_stack([np.interp(t, times, points[:, k]) for k in (0, 1)])
    return xy + jostle * np.column_stack([np.sin(2 * t), np.cos(2 * t)])


def test_frozen_ghost_next_to_moving_keeper():
    tracking, t = scene((0, 0))
    players = tracking.players
    players[:, K] = keeper_path(t, 0)
    # The ghost runs in at 5 m/s and is left behind, frozen with 1 cm of jitter, 2 m from the
    # keeper.
    frozen_from = 70
    start, spot = np.array([20.0, 36.0]), np.array([6.0, 36.0])
    approach = np.linspace(0, 1, frozen_from + 1)[:, None]
    players[: frozen_from + 1, X] = start + approach * (spot - start)
    rng = np.random.default_rng(0)
    players[frozen_from:, X] = spot + rng.uniform(
        -0.01, 0.01, (len(t) - frozen_from, 2)
    )

    result = clean_tracking(tracking, MECHANICS)
    np.testing.assert_array_equal(result.stale[:, X], np.arange(len(t)) >= frozen_from)
    assert not result.stale[:, K].any()
    assert np.isnan(result.tracking.players[frozen_from:, X]).all()
    assert (result.keeper_track[:, 0] == K).all()
    np.testing.assert_array_equal(
        result.tracking.players[:, result.keeper_slots[0]], players[:, K]
    )

    stale = result.log[result.log["kind"] == "stale"]
    assert stale[["slot", "first_frame", "last_frame"]].values.tolist() == [
        [X, frozen_from, len(t) - 1]
    ]
    keeper = keeper_rows(result)
    assert keeper[["slot", "goal", "from_team", "to_team"]].values.tolist() == [
        [K, 0, 0, 0]
    ]

    # Thresholds come from the config: at 1 mm the ghost's jitter is movement.
    loose = clean_tracking(tracking, replace(MECHANICS, stale_max_spread_m=0.001))
    assert not loose.stale.any()


def test_keepers_with_the_wrong_team():
    # K is the home keeper labelled away, X the away keeper labelled home. Sides swap at half
    # time.
    tracking, t = scene((1, 0), periods=2)
    players, first = tracking.players, (tracking.period == 1)[:, None]
    players[:, K] = np.where(first, keeper_path(t, 0), keeper_path(t, 1))
    players[:, X] = np.where(first, keeper_path(t, 1), keeper_path(t, 0))

    result = clean_tracking(tracking, MECHANICS)
    assert result.defends == {1: (0, 1), 2: (1, 0)}
    home_gk, away_gk = result.keeper_slots
    cleaned = result.tracking
    np.testing.assert_array_equal(cleaned.players[:, home_gk], players[:, K])
    np.testing.assert_array_equal(cleaned.players[:, away_gk], players[:, X])
    assert np.isnan(cleaned.players[:, [K, X]]).all()
    assert cleaned.team.tolist() == tracking.team.tolist() + [0, 1]

    first = first[:, 0]
    assert (result.keeper_track[first] == [K, X]).all()
    assert (result.keeper_track[~first] == [X, K]).all()
    keepers = keeper_rows(result)
    assert len(keepers) == 4
    assert (keepers["from_team"] != keepers["to_team"]).all()
    assert set(keepers["end_reason"]) == {"period_end"}

    # Five tracks a team either way, but now the right five.


def test_corner_attacker_next_to_keeper_keeps_its_label():
    tracking, t = scene((0, 1))
    players = tracking.players
    players[:, K] = keeper_path(t, 0)
    # The attacker runs in, stands 1.4 m from the keeper for 5 s, and runs out again.
    stand = (6.0, WIDTH / 2 + 1.0)
    players[:, X] = path(t, [0, 5, 10, 15], [(40, 30), stand, stand, (40, 30)])

    result = clean_tracking(tracking, MECHANICS)
    assert (result.keeper_track[:, 0] == K).all()
    assert not (result.keeper_track == X).any()
    np.testing.assert_array_equal(result.tracking.players[:, X], players[:, X])
    assert result.tracking.team[X] == 1
    assert keeper_rows(result)["slot"].tolist() == [K]


def test_track_jumps_from_keeper_to_another_player():
    # The keeper's track (labelled away) switches to an away player 20 m out; the keeper is
    # picked up again by a new track.
    tracking, t = scene((1, 0))
    players, jump = tracking.players, 150
    players[:jump, K] = keeper_path(t[:jump], 0)
    players[jump:, K] = path(t[jump:], [0, 20], [(25, 30), (35, 20)])
    players[jump:, X] = keeper_path(t[jump:], 0)

    result = clean_tracking(tracking, MECHANICS)
    np.testing.assert_array_equal(result.keeper_track[:jump, 0], K)
    np.testing.assert_array_equal(result.keeper_track[jump:, 0], X)
    keepers = keeper_rows(result)
    assert keepers[
        ["slot", "first_frame", "last_frame", "end_reason"]
    ].values.tolist() == [
        [K, 0, jump - 1, "jump"],
        [X, jump, len(t) - 1, "period_end"],
    ]
    cleaned = result.tracking
    assert np.isnan(cleaned.players[:jump, K]).all()
    np.testing.assert_array_equal(cleaned.players[jump:, K], players[jump:, K])
    assert cleaned.team[K] == 1
    np.testing.assert_array_equal(
        cleaned.players[:, result.keeper_slots[0]],
        np.where((np.arange(len(t)) < jump)[:, None], players[:, K], players[:, X]),
    )


def test_keeper_far_from_goal_ends():
    tracking, t = scene((0,), seconds=25)
    players = tracking.players
    players[:, K] = path(t, [0, 4, 14, 25], [(5, 34), (5, 34), (45, 34), (45, 34)])

    result = clean_tracking(tracking, MECHANICS)
    first_far = int(np.argmax(players[:, K, 0] > 35))
    np.testing.assert_array_equal(result.keeper_track[:first_far, 0], K)
    np.testing.assert_array_equal(result.keeper_track[first_far:, 0], -1)
    assert keeper_rows(result)["end_reason"].tolist() == ["far"]


def test_another_track_takes_over():
    # The keeper's track walks out to 25 m while a new track holds the keeper's spot.
    tracking, t = scene((0, 0))
    players = tracking.players
    players[:, K] = path(t, [0, 4, 14, 20], [(5, 34), (5, 34), (25, 34), (25, 34)])
    players[100:, X] = keeper_path(t[100:], 0)

    result = clean_tracking(tracking, MECHANICS)
    out = int(np.argmax(np.hypot(players[:, K, 0], players[:, K, 1] - WIDTH / 2) > 12))
    np.testing.assert_array_equal(result.keeper_track[:out, 0], K)
    np.testing.assert_array_equal(result.keeper_track[out:, 0], X)
    keepers = keeper_rows(result)
    assert keepers[["slot", "last_frame", "end_reason"]].values.tolist()[0] == [
        K,
        out - 1,
        "takeover",
    ]


def test_attacker_alone_in_the_box_while_the_keeper_comes_out():
    # The keeper comes out 17 m for 4 s; an attacker waits alone near goal, never within 3 m.
    tracking, t = scene((0, 1))
    players = tracking.players
    out = (14.0, WIDTH / 2 + 10.0)
    players[:, K] = path(t, [0, 5, 7, 11, 13], [(5, 34), (5, 34), out, out, (5, 34)])
    players[:, X] = path(t, [0, 3, 20], [(40, 20), (6, 26), (6, 27)])

    result = clean_tracking(tracking, MECHANICS)
    assert (result.keeper_track[:, 0] == K).all()
    assert not (result.keeper_track == X).any()


def test_attacker_beside_keeper_at_the_post_does_not_take_over():
    # The keeper stands by the post, just outside the seed radius; an attacker comes within
    # 2.5 m, just inside it, and runs out again. The keeper did not move away: no takeover.
    tracking, t = scene((0, 1))
    players = tracking.players
    post, beside = (1.3, 21.0), (1.3, 23.5)
    players[:, K] = path(t, [0, 3, 6, 20], [(5, 34), (5, 34), post, post])
    players[:, X] = path(t, [0, 8, 10, 14], [(30, 10), beside, beside, (30, 26)])

    result = clean_tracking(tracking, MECHANICS)
    assert (result.keeper_track[:, 0] == K).all()
    assert not (result.keeper_track == X).any()


def test_keeper_survives_a_short_gap_and_ends_with_its_track():
    tracking, t = scene((0,))
    players, end = tracking.players, 200
    players[:end, K] = keeper_path(t[:end], 0)
    players[50:53, K] = np.nan  # 3 frames, within max_gap_s

    result = clean_tracking(tracking, MECHANICS)
    np.testing.assert_array_equal(result.keeper_track[:end, 0], K)
    np.testing.assert_array_equal(result.keeper_track[end:, 0], -1)
    keepers = keeper_rows(result)
    assert keepers[["last_frame", "end_reason"]].values.tolist() == [
        [end - 1, "track_end"]
    ]


def test_goal_one_keeper_and_no_keeper_without_a_seed():
    tracking, t = scene((1,))
    tracking.players[:, K] = keeper_path(t, 1)
    result = clean_tracking(tracking, MECHANICS)
    assert (result.keeper_track[:, 0] == -1).all()
    assert (result.keeper_track[:, 1] == K).all()
    assert keeper_rows(result)[["from_team", "to_team"]].values.tolist() == [[1, 1]]


def test_stale_mask_matches_brute_force():
    config = CleanConfig(stale_window_s=0.4, stale_max_spread_m=0.1)
    window = round(config.stale_window_s * FPS)
    rng = np.random.default_rng(1)
    n_frames, n_slots = 400, 4
    steps = rng.normal(0, 0.05, (n_frames, n_slots, 2))
    # Long near-still stretches, so some windows are still and some are not.
    steps[rng.uniform(size=(n_frames, n_slots)) < 0.7] *= 0.05
    players = np.cumsum(steps, axis=0) + LENGTH / 2
    players[rng.uniform(size=(n_frames, n_slots)) < 0.01] = np.nan

    expected = np.zeros((n_frames, n_slots), dtype=bool)
    for slot in range(n_slots):
        for start in range(n_frames - window + 1):
            xy = players[start : start + window, slot]
            if not np.isnan(xy).any() and np.hypot(*np.ptp(xy, axis=0)) < 0.1:
                expected[start : start + window, slot] = True
    assert expected.any() and not expected.all()
    np.testing.assert_array_equal(stale_mask(players, FPS, config), expected)


def test_frozen_track_split_by_a_gap_is_not_stale():
    players = np.full((200, 1, 2), 10.0)
    players[60] = np.nan  # 2.4 s, a gap, 5.6 s: only the second part spans a 4 s window
    stale = stale_mask(players, FPS)[:, 0]
    assert not stale[:61].any() and stale[61:].all()
    assert not stale_mask(players[:99], FPS).any()


def test_coasting_ghost_does_not_take_the_keeper():
    from mtah.clean.tracks import coast_mask

    # The keeper is lost for his first 2 s; a ghost the tracker carries on drifts straight
    # up at 0.25 m/s, 1 m from him, with no jitter at all.
    tracking, t = scene((1, 0))
    players = tracking.players
    players[:, K] = keeper_path(t, 0)
    players[:50, K] = np.nan
    players[:, X] = np.column_stack([np.full(len(t), 6.0), 30.0 + 0.25 * t])

    off = clean_tracking(tracking, replace(MECHANICS, coast_max_excess_m=0))
    assert (off.keeper_track[:, 0] == X).all()  # alone at first: it keeps the identity
    assert not off.coasting.any()

    result = clean_tracking(tracking, MECHANICS)
    assert result.coasting[:, X].all() and not result.coasting[:, K].any()
    assert result.stale[:, X].all() and np.isnan(result.tracking.players[:, X]).all()
    # The real keeper, labelled away, is found once alone, and moved to home's GK slot.
    seeded = result.keeper_track[:, 0] == K
    assert seeded[50:].all() and not seeded[:50].any()
    np.testing.assert_array_equal(
        result.tracking.players[50:, result.keeper_slots[0]], players[50:, K]
    )

    # A real player walking straight as slowly, with 1 cm of tracking noise, is no ghost.
    rng = np.random.default_rng(1)
    walk = np.column_stack([30.0 + 0.8 * t, np.full(len(t), 20.0)])
    walk = (walk + rng.uniform(-0.01, 0.01, walk.shape))[:, None]
    assert not coast_mask(walk, FPS, MECHANICS).any()
    # Faster than 1 m/s, a straight line is a run, whatever its smoothness.
    run = np.column_stack([30.0 + 2.0 * t, np.full(len(t), 20.0)])[:, None]
    assert not coast_mask(run, FPS, MECHANICS).any()
    assert coast_mask(run / 4, FPS, MECHANICS).all()  # 0.5 m/s


def test_coast_mask_covers_its_windows_only():
    from mtah.clean.tracks import coast_mask

    # Real movement, then 6 s of a straight drift, then real movement again.
    rng = np.random.default_rng(2)
    xy = rng.normal(0, 0.2, (400, 1, 2)).cumsum(axis=0)  # a random walk
    xy[150:300, 0] = xy[149, 0] + np.outer(np.arange(1, 151), [0.01, 0.0])
    xy[300:, 0] += xy[299, 0] - xy[300, 0] + [0.0, 0.5]  # a step aside ends it
    got = coast_mask(xy, FPS, MECHANICS)[:, 0]
    # Windows of 100 frames inside 149..299 (149 is the drift's start point).
    np.testing.assert_array_equal(
        got, (np.arange(400) >= 149) & (np.arange(400) <= 299)
    )


def test_the_identity_reaches_back_to_the_keeper_with_the_ball():
    # The keeper, labelled away, holds the ball at the box's edge, 15 m out, for 6 s, then
    # walks back and stands near the goal: alone there, he is identified from about 8 s.
    tracking, t = scene((1,))
    out, home = np.array([15.0, WIDTH / 2]), np.array([5.0, WIDTH / 2])
    tracking.players[:, K] = path(t, [0, 6, 8, 20], [out, out, home, home])
    result = clean_tracking(tracking, MECHANICS)
    assert (result.keeper_track[:, 0] == K).all()  # from the first frame: the catch too
    assert keeper_rows(result)["first_frame"].tolist() == [0]
    late = clean_tracking(tracking, replace(MECHANICS, back_s=0))
    assert late.keeper_track[:150, 0].tolist() == [-1] * 150  # only once he is near

    # Before a jump, the track was someone else: the identity stops there.
    jumped = tracking.players.copy()
    jumped[:100, K] = path(t[:100], [0, 4], [[60.0, 20.0], [60.0, 20.0]])  # moving
    tracking = tracking.__class__(**{**tracking.__dict__, "players": jumped})
    first = clean_tracking(tracking, MECHANICS).keeper_track[:, 0]
    assert (first[:100] == -1).all() and (first[100:] == K).all()


def test_a_period_is_split_where_the_teams_swap_ends():
    from mtah.clean.tracks import split_periods

    # Two halves, sides swapped, but the file marks one period throughout.
    tracking, t = scene((1, 0), periods=2)
    players = tracking.players
    first = (tracking.period == 1)[:, None]
    players[:, K] = np.where(first, keeper_path(t, 0), keeper_path(t, 1))
    players[:, X] = np.where(first, keeper_path(t, 1), keeper_path(t, 0))
    marked = tracking.period.copy()
    one = tracking.__class__(**{**tracking.__dict__, "period": np.ones_like(marked)})
    config = CleanConfig(side_window_s=2.0, half_min_s=10.0)  # 20 s halves here

    np.testing.assert_array_equal(
        split_periods(one.players, one.team, one.period, FPS, config), marked
    )
    # Cleaned as one period, the keepers come out as with the halves marked.
    result = clean_tracking(one, config)
    assert result.defends == {1: (0, 1), 2: (1, 0)}
    np.testing.assert_array_equal(result.tracking.period, marked)
    assert (result.keeper_track[first[:, 0]] == [K, X]).all()
    assert (result.keeper_track[~first[:, 0]] == [X, K]).all()
    # Marked halves are left alone; so is a swap shorter than half_min_s.
    np.testing.assert_array_equal(
        split_periods(players, tracking.team, marked, FPS, config), marked
    )
    short = CleanConfig(side_window_s=2.0, half_min_s=30.0)
    assert (split_periods(one.players, one.team, one.period, FPS, short) == 1).all()
