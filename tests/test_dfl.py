"""The DFL reader (mtah.io.dfl) on small synthetic files in the dataset's layout."""

import numpy as np

from mtah.io.dfl import load_match, match_files

MID = "DFL-MAT-TEST01"


def write_match(root):
    (root / f"DFL_02_01_matchinformation_DFL-COM-1_{MID}.xml").write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation>
 <General MatchId="{MID}" HomeTeamId="T-H" GuestTeamId="T-G"/>
 <Environment PitchX="105.00" PitchY="68.00"/>
 <Teams>
  <Team TeamId="T-G" Role="guest"><Players>
   <Player PersonId="G1" PlayingPosition="TW"/>
  </Players></Team>
  <Team TeamId="T-H" Role="home"><Players>
   <Player PersonId="H1" PlayingPosition="STZ"/>
  </Players></Team>
 </Teams>
</MatchInformation></PutDataRequest>""",
        encoding="utf-8",
    )
    (root / f"DFL_03_02_events_raw_DFL-COM-1_{MID}.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest>
<Event EventTime="2023-05-27T15:30:00.000+02:00"><KickOff GameSection="firstHalf">
 <Play Team="T-H" Player="H1" Recipient="G1"><Pass/></Play></KickOff></Event>
<Event EventTime="2023-05-27T15:30:00.040+02:00"><ShotAtGoal Team="T-G" Player="G1">
 <SuccessfulShot/></ShotAtGoal></Event>
<Event EventTime="2023-05-27T16:30:00.040+02:00">
 <ThrowIn Team="T-G" DecisionTimestamp="2023-05-27T16:30:00.000+02:00"/></Event>
</PutDataRequest>""",
        encoding="utf-8",
    )

    def frames(start, n, t0, extra=""):
        return "".join(
            f'<Frame N="{start + k}" T="{t0}.{k * 40:03d}+00:00" X="{k:.2f}" Y="{-k:.2f}"'
            f"{extra.format(k=k)}/>"
            for k in range(n)
        )

    ball = ' Z="{k}.0" BallPossession="{k}" BallStatus="1"'
    (root / f"DFL_04_03_positions_raw_observed_DFL-COM-1_{MID}.xml").write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><Positions>
<FrameSet GameSection="firstHalf" TeamId="referee" PersonId="R1">{frames(10000, 3, "2023-05-27T13:30:00")}</FrameSet>
<FrameSet GameSection="firstHalf" TeamId="T-H" PersonId="H1">{frames(10000, 3, "2023-05-27T13:30:00")}</FrameSet>
<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="B">{frames(10000, 3, "2023-05-27T13:30:00", ball)}</FrameSet>
<FrameSet GameSection="secondHalf" TeamId="BALL" PersonId="B">{frames(100000, 2, "2023-05-27T14:30:00", ' Z="0" BallPossession="1" BallStatus="0"')}</FrameSet>
</Positions></PutDataRequest>""",
        encoding="utf-8",
    )


def test_load_match(tmp_path):
    write_match(tmp_path)
    assert list(match_files(tmp_path)) == [MID]
    m = load_match(tmp_path, MID, cache_dir=tmp_path / "cache")
    t = m.tracking
    assert t.fps == 25.0 and t.pitch_size == (105.0, 68.0)
    assert t.team.tolist() == [1, 0] and m.keepers == (
        0,
    )  # guest's keeper, home striker
    assert t.period.tolist() == [1, 1, 1, 2, 2]
    # Centre origin, y up, to corner origin, y down.
    np.testing.assert_allclose(t.ball[1], [53.5, 35.0])
    np.testing.assert_allclose(t.players[2, 1], [54.5, 36.0])
    assert np.isnan(t.players[:, 0]).all()  # the keeper is never in the positions
    np.testing.assert_allclose(m.ball_z[:3], [0, 1, 2])
    # Possession 0 none, 1 home, 2 guest; the second half dead throughout.
    assert m.labels.tolist() == ["D", "H", "A", "D", "D"]

    kickoff, _goal, throw_in = m.events
    assert (kickoff["kind"], kickoff["team"], kickoff["row"]) == ("KickOff", 0, 0)
    assert kickoff["player"] == "H1" and kickoff["recipient"] == "G1"
    assert throw_in["row"] == 4 and throw_in["decision_row"] == 3

    cached = load_match(tmp_path, MID, cache_dir=tmp_path / "cache")
    np.testing.assert_array_equal(cached.labels, m.labels)
    np.testing.assert_array_equal(cached.tracking.players, t.players)
