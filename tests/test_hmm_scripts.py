"""The submission script on synthetic tracking: make_submission.py end to end."""

import pandas as pd
import pytest


def test_make_submission(epts, tmp_path, run_script, capsys):
    out = tmp_path / "team.csv"
    meta, raw = epts
    run_script(
        "make_submission", "--meta", meta, "--raw", raw, "--no-cache", "--out", out
    )
    printed = capsys.readouterr().out
    assert f"{out}: valid, 199319 frames" in printed
    assert "possessions in the halves" in printed
    table = pd.read_csv(out)
    assert list(table.columns) == ["frame", "label"] and len(table) == 199319
    # the same input, the same file
    again = tmp_path / "again.csv"
    run_script(
        "make_submission", "--meta", meta, "--raw", raw, "--no-cache", "--out", again
    )
    assert again.read_bytes() == out.read_bytes()


def test_make_submission_keeps_out_of_results(epts, run_script):
    from pathlib import Path

    results = Path(__file__).resolve().parents[1] / "evaluation" / "results" / "x.csv"
    meta, raw = epts
    with pytest.raises(SystemExit):
        run_script("make_submission", "--meta", meta, "--raw", raw, "--out", results)
