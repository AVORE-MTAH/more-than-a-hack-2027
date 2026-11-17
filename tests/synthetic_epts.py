"""A minimal synthetic Metrica EPTS pair, so tests never need match data."""

from pathlib import Path

# Params
# Two home and two away slots, ten records at 25 fps. Periods cover frame_ids 1-5 and 7-10, so
# frame_id 6 is between the halves. Slot 1 is missing on frame_id 3, the ball on frame_id 4.
SLOTS = [("T1", "A"), ("T2", "A"), ("T3", "B"), ("T4", "B")]
N_RECORDS = 10
PERIODS = {
    "first_half_start": 1,
    "first_half_end": 5,
    "second_half_start": 7,
    "second_half_end": 10,
}
MISSING_PLAYER = (3, 1)
MISSING_BALL = 4


def _params(values):
    return "".join(
        f"<ProviderParameter><Name>{k}</Name><Value>{v}</Value></ProviderParameter>"
        for k, v in values.items()
    )


def write_epts(directory):
    """Write a minimal Metrica EPTS metadata XML and raw text pair; return their paths."""
    players = "".join(
        f'<Player id="{p}" teamId="{t}"><Name>Track {i + 1}</Name>'
        f"<ShirtNumber>{i + 1}</ShirtNumber></Player>"
        for i, (p, t) in enumerate(SLOTS)
    )
    channels = "".join(
        f'<PlayerChannel id="{p}_x" playerId="{p}" channelId="x"/>'
        f'<PlayerChannel id="{p}_y" playerId="{p}" channelId="y"/>'
        for p, _ in SLOTS
    )
    refs = "".join(
        f'<SplitRegister separator=","><PlayerChannelRef playerChannelId="{p}_x"/>'
        f'<PlayerChannelRef playerChannelId="{p}_y"/></SplitRegister>'
        for p, _ in SLOTS
    )
    direction = _params({"attack_direction_first_half": "left_to_right"})
    meta = f"""<?xml version="1.0" encoding="utf-8"?>
<main>
<Metadata>
<GlobalConfig><FrameRate>25</FrameRate><ProviderName>Metrica Sports</ProviderName>
<ProviderGlobalParameters>{_params(PERIODS)}</ProviderGlobalParameters></GlobalConfig>
<Sessions><Session id="1"><FieldSize><Width>105</Width><Height>68</Height></FieldSize>
</Session></Sessions>
<Teams>
<Team id="A"><Name>Home</Name><ProviderTeamsParameters>{direction}</ProviderTeamsParameters></Team>
<Team id="B"><Name>Away</Name></Team>
</Teams>
<Players>{players}</Players>
<Devices><Device id="d"><Name>d</Name><Sensors><Sensor id="position"><Name>Position</Name>
<Channels><Channel id="x"><Name>x</Name><Unit>normalized</Unit></Channel>
<Channel id="y"><Name>y</Name><Unit>normalized</Unit></Channel></Channels>
</Sensor></Sensors></Device></Devices>
<PlayerChannels>{channels}</PlayerChannels>
</Metadata>
<DataFormatSpecifications>
<DataFormatSpecification separator=":" startFrame="1" endFrame="{N_RECORDS}">
<StringRegister name="frameCount"/>
<SplitRegister separator=";">{refs}</SplitRegister>
<SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>
</SplitRegister>
</DataFormatSpecification>
</DataFormatSpecifications>
</main>
"""
    lines = []
    for frame_id in range(1, N_RECORDS + 1):
        players_txt = ";".join(
            "NaN,NaN"
            if (frame_id, i) == MISSING_PLAYER
            else f"{0.1 * (i + 1):.3f},0.250"
            for i in range(len(SLOTS))
        )
        ball_txt = "NaN,NaN" if frame_id == MISSING_BALL else "0.500,0.500"
        lines.append(f"{frame_id}:{players_txt}:{ball_txt}")

    meta_path, raw_path = Path(directory) / "meta.xml", Path(directory) / "raw.txt"
    meta_path.write_text(meta, encoding="utf-8")
    raw_path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return meta_path, raw_path
