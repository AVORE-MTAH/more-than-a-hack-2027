"""The external fits (mtah.tune) on synthetic data: they run and restore what they patch."""

import numpy as np
from synthetic_tracks import FPS, scene

from mtah.tune import cleaning


def test_cleaning_fit_reads_real_players():
    tracking, _ = scene(())
    fitted = cleaning.fit([tracking])
    # The scene's players sway: their stillest windows still spread well over a centimetre.
    assert fitted["windows"] > 0
    assert 0.01 < fitted["stale_max_spread_m"] < 10
    assert fitted["coast_max_excess_m"] >= 0


def test_window_values_skip_windows_with_gaps():
    players = np.cumsum(np.ones((300, 1, 2)) * 0.01, axis=0)
    players[100:110] = np.nan
    spread, excess, move = cleaning.window_values(players, FPS, 4.0)
    assert len(spread) == len(excess) == len(move) > 0
    np.testing.assert_allclose(excess, 0, atol=1e-9)  # a straight line has no excess


def test_tune_external_without_sources(run_script, capsys, monkeypatch):
    for name in ("MTAH_DFL_DIR", "MTAH_SKILLCORNER_DIR", "MTAH_METRICA_DIR"):
        monkeypatch.delenv(name, raising=False)
    run_script("tune_external")
    assert "nothing fitted" in capsys.readouterr().out
