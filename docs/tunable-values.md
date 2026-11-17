# Tunable values: where each one comes from

Every number the submission depends on, and the source it must have. The rule we follow: a
value that needs tuning is tuned on external data only. None is set from the provided match:
not from its possession ground truth, not from its video or audio, and not by checking the
output against either.

Each value falls in one of five groups:

| group | meaning | needs data? |
|---|---|---|
| **Laws** | fixed by the Laws of the Game or the pitch's markings | no |
| **Physics** | what a person, a ball or the sensor can physically do | no |
| **Resolution** | a window, a bucket edge or a batch size: how finely something is measured, not where a line is drawn | no |
| **Fit** | a threshold or a weight with no value given by the game, fitted on external data | yes, external |
| **None yet** | needs tuning, but no external data supports it so far | — |

Values in **Fit** are produced from external data only, by scripts that can be rerun:
`scripts/tune_external.py` (cleaning, keepers, restarts),
`scripts/calibrate_external.py` (the calibration, shipped as
`src/mtah/predict/external_calibration.json`), `scripts/train_classifier.py` (the per-frame
classifier, shipped as `src/mtah/predict/frame_classifier.json`), `scripts/dead_weights_report.py`
(the dead-ball weights), `scripts/weights_report.py` (the other weights and the flight
thresholds) and `scripts/hybrid_report.py` (the classifier's weight). The
module defaults are set to what they output. A value that is not fitted was set to half and
double its value and scored by the fit it belongs to (`scripts/sensitivity_report.py`); the ones
that barely moved are recorded as set from physics and checked insensitive. The **Status**
column says where each value stands; none is left to fit. The scores behind each chosen value
are in "The last external run behind each chosen value", below.

## External data

The sources, their licences and what each one fits are in `technical-document.md`, section 3.

**The DFL dataset**: Bassek et al., "An integrated dataset of spatiotemporal and
event data in elite soccer", *Scientific Data* 12, 195 (2025), doi:10.1038/s41597-025-04505-y;
figshare doi:10.6084/m9.figshare.28196177; CC BY 4.0, data owner Deutsche Fussball Liga. Seven
matches of 2022/23 (2 Bundesliga, 5 2. Bundesliga), each with:

- **match information:** the teams, the line-ups, and each keeper's position (`TW`);
- **raw events:** kick-offs, throw-ins, free kicks, goal kicks, corners and penalties, with the
  time each restart was awarded (`DecisionTimestamp`); fouls; goals (`SuccessfulShot`); and
  passes with their recipient;
- **raw positions** at 25 fps: every player, and the ball with its height (`Z`), its possessing
  team (`BallPossession`: 1 home, 2 away) and its status (`BallStatus`: 1 alive, 0 dead) on every
  frame.

It is read from outside the repository and never committed.

Labels come from the provider's own frame data: **D** where `BallStatus` is 0, otherwise **H**
or **A** by `BallPossession`. The events give the rest:

- **Flights:** from each pass to its recipient, and the ball's height directly.
- **Restarts:** each set piece's time, and the time it was awarded.

**The Metrica Sports sample data** (github.com/metrica-sports/sample-data), games 1 and 2: tracking whose ball goes
missing as a tactical camera's does, and events. Dead ball is labelled from each ball-out, foul
or goal to the next set piece (`mtah.io.metrica_csv`); the team in possession is not used.

**SkillCorner's open data** (github.com/SkillCorner/opendata, MIT), 20 A-League matches of
broadcast tracking with phases of play (the team in controlled possession), upsampled from 10 to
25 fps.

**What the external data does not have:**

- **A missing ball, except Metrica's.** The DFL ball has a position on every frame, and
  SkillCorner's goes missing mostly in replays. Values that act on an unseen ball are fitted on
  Metrica, or on DFL with its ball removed by a rule fitted on Metrica (see The missing ball).
- **Tracking faults.** Keepers are labelled correctly and there are no leftover tracks, so the
  cleaning values are fitted against how real players and keepers move, not against the faults.
- **Audio or a broadcast with labels.** See Dropped.

The DFL possession and status are a provider's mechanical definitions, like the challenge's,
and may differ from them in detail.

What each **Fit** value is fitted against is given in its row.

## Tracking cleaning (`mtah.clean.tracks.CleanConfig`)

| value | default | what it is | group | fitted against (external) | status |
|---|---|---|---|---|---|
| `stale_window_s` | 4.0 | window a still track is measured over | Resolution | — | fixed |
| `stale_max_spread_m` | 0.064 | a track that stays within this is stale | Fit | tracked players' 4 s spread: well below the lowest real one | fitted (DFL, 1st percentile of real players) |
| `coast_max_excess_m` | 0.029 | path minus straight move under this is coasting | Fit | tracked players' path excess over 4 s: below the lowest real one | fitted (DFL, 1st percentile of real players) |
| `coast_max_speed_mps` | 1.0 | coasting only below this speed | Physics | — (a lost track drifts slowly; a tracked person walking covers more) | fixed |
| `box_depth_m`, `box_half_width_m` | 16.5, 20.16 | the penalty area | Laws | — | fixed |
| `seed_radius_m` | 20.0 | keeper seed zone around the goal centre | Fit | keepers' distance from their goal centre | fitted (DFL keepers) |
| `seed_s` | 6.0 | alone in the zone this long seeds a keeper | Fit | keeper identification against the known keepers | fitted (DFL keepers) |
| `max_speed_mps` | 12.0 | faster than a person can run | Physics | — | fixed |
| `max_gap_s` | 0.2 | a keeper track lost longer has ended | Fit | track gap lengths of the known keepers | set from physics; checked insensitive on DFL (half and double, scripts/sensitivity_report.py); DFL has no keeper faults |
| `far_from_line_m`, `far_s` | 40.0, 5.0 | a keeper this far from his line this long is not one | Fit | keepers' distance from their goal line | far_from_line_m fitted (DFL keepers); far_s set from physics; checked insensitive on DFL (half and double, scripts/sensitivity_report.py) |
| `takeover_s`, `takeover_contact_m` | 1.0, 2.0 | another track taking the keeper's place | Fit | keeper identification against the known keepers | takeover_contact_m fitted (DFL keepers: team errors 0.047% -> 0.013%); takeover_s set from physics; checked insensitive on DFL (half and double, scripts/sensitivity_report.py) |
| `back_s` | 240.0 | how far an identity reaches back along its track | Fit | keeper identification against the known keepers | fitted (DFL keepers: coverage 83.4% -> 84.6%); DFL has no keeper faults, so a long reach-back is untested on faulty tracking |
| `side_window_s` | 120.0 | smoothing of which team is on which side | Resolution | — | fixed |
| `half_min_s` | 600.0 | a swap of ends must hold this long to split a period | Laws | — (a half is 45 min) | fixed |

## Dead-ball features (`mtah.dead.features`)

| value | default | what it is | group | fitted against (external) | status |
|---|---|---|---|---|---|
| `SPEED_WINDOW_S` | 2.0 | smoothing of the median player speed | Resolution | — | fixed |
| `OUTSIDE_MARGIN_M` | 0.75 | ball this far over a line is out | Fit | ball position at the events' ball-outs | fitted (Metrica: D's F1 0.911 -> 0.921, both games) |
| `STILL_WINDOW_S`, `STILL_MAX_SPEED` | 1.5, 0.5 | a still ball | Physics | — | fixed: a binary input of the per-frame classifier, whose weights are fitted |

## Dead ball and restarts (`mtah.dead.score`)

| value | default | used by | group | fitted against (external) | status |
|---|---|---|---|---|---|
| `OUT_MIN_ROWS` | 3 | `stoppages`, `out_stoppages` | Resolution | — (single-row glitches) | fixed |
| `KICK_SPEED`, `KICK_ROWS`, `SETTLE_S`, `SLOW_BALL`, `EXIT_SPEED`, `EXIT_S` | 6.0, 2, 1.0, 3.0, 2.0, 2.0 | `stoppages` (ball-out restarts) | Fit | restart frames of the set pieces (Metrica's events) | set from physics; checked insensitive on Metrica (half and double); EXIT_SPEED flat once the margin was fitted |
| `WHISTLE_MAX_S` | 45.0 | `out_stoppages`: the longest stoppage | Fit | stoppage lengths (Metrica's ball-outs) | fitted (Metrica) |
| `STILL_S`, `STILL_M`, `MOVE_S`, `MOVE_M`, `OFF_M`, `RECENT_S` | 1.0, 0.5, 0.5, 0.75, 0.2, 2.0 | `restart_signals` (S and M) | Fit | restart frames of the set pieces (Metrica's events) | MOVE_M fitted (Metrica); the rest checked on Metrica: no change helps both games |
| `GOING_S`, `GOING_SPEED`, `GOING_RISE` | 3.0, 1.2, 0.2 | `restart_signals` (P) | Fit | restart frames of the set pieces (Metrica's events) | checked (Metrica): no change helps both games |
| `AFTER_S`, `REST_S`, `REST_M` | 1.0, 0.5, 0.3 | `restart_signals` (no rest after the move) | Fit | restart frames of the set pieces (Metrica's events) | checked (Metrica): no change helps both games |
| `IN_M`, `LINE_M`, `THROW_GAP_S` | 1.0, 0.375, 3.0 | `throw_ins` | Fit | throw-in frames (Metrica's events) | LINE_M fitted (Metrica); the rest checked on Metrica: no change helps both games |
| `CONTROL_RADIUS`, `DRIBBLE_SPEED`, `DRIBBLE_WINDOW_S` | 1.5, 2.0, 0.5 | `dribbling` | Fit | D's F1 on Metrica, ball unseen | set from physics; checked insensitive on Metrica (half and double) |

## Ball in flight (`mtah.predict.flight`)

| value | default | what it is | group | fitted against (external) | status |
|---|---|---|---|---|---|
| `FLIGHT_SPEED`, `SPAN_S`, `FLIGHT_S` | 4.0, 0.2, 0.5 | fast enough, measured over, for long enough | Fit | spell F1 on SkillCorner's phases | fitted (scripts/weights_report.py --skillcorner) |
| `KINK_DEG`, `TURN_SPEED` | 30.0, 3.0 | a turn this sharp at speed is a touch | Fit | spell F1 on DFL, its ball removed as Metrica's | fitted (DFL): TURN_SPEED; KINK_DEG unchanged (15, 20, 40 and 60 worse) |
| `GAP_MAX_S`, `GAP_SPEED` | 1.5, 5.0 | a ball lost in flight | Physics | — | set from physics; checked insensitive on DFL, its ball removed as Metrica's (half and double, scripts/sensitivity_report.py) |
| `KICKER_M`, `KICK_S` | 3.0, 0.12 | the kicker or receiver: nearest player this close, this soon | Fit | spell F1 on DFL, its ball removed as Metrica's | fitted (DFL: 0.678 -> 0.686, past the grid's edges checked) |

## The model (`mtah.predict.hmm`)

| value | default | what it is | group | fitted against (external) | status |
|---|---|---|---|---|---|
| `DISTANCE_EDGES`, `SPEED_EDGES` | 0.5–5 m, 1–2 m/s | buckets of the evidence tables | Resolution | — | fixed; SPEED_EDGES checked insensitive on Metrica (five edge sets, D's F1 0.921 each) |
| `MIN_LIVE_S`, `MIN_DEAD_S` | 3.0, 3.0 | shortest possession and stoppage | Fit | possession and stoppage lengths | MIN_DEAD_S with the dead-ball weights (Metrica); MIN_LIVE_S fitted (scripts/weights_report.py --skillcorner: SkillCorner's phases) |
| **the calibration** | — | every likelihood table and transition rate | Fit | `scripts/calibrate_external.py --skillcorner`: SkillCorner's phases (upsampled to 25 fps) for the team table, transition and kicker rates; Metrica for the dead table and the ball-out stoppage shares | fitted |
| `W_TEAM`, `W_DEAD` | 0.1, 0.1 | tempering of the team and dead terms | Fit | spell F1 on SkillCorner's phases (W_TEAM); D's F1 on Metrica (W_DEAD) | W_DEAD fitted on Metrica (scripts/dead_weights_report.py), kept where SkillCorner disagrees; W_TEAM fitted (scripts/weights_report.py --skillcorner) |
| `W_KICK` | 4.0 | the kick term | Fit | spell F1 on SkillCorner's phases | fitted (scripts/weights_report.py --skillcorner) |
| `W_STOPPAGE` | 0.2 | the ball-out stoppage term, weighed apart from the dead table | Fit | Metrica: D's F1 against its event-timed stoppages | fitted (scripts/dead_weights_report.py) |
| **the per-frame classifier** (`frame_classifier.json`) | — | the hybrid's coefficients | Fit | `scripts/train_classifier.py`: SkillCorner's phases (H, A, D) and Metrica's games (dead against live), Metrica x5 | fitted |
| `W_CLF` | 0.05 | the classifier's scaled likelihoods in the evidence | Fit | held-out spell and macro F1 on SkillCorner and DFL, D's F1 on Metrica (`scripts/hybrid_report.py`) | fitted: the largest that loses on no source and no game |
| `METRICA_WEIGHT` (`scripts/train_classifier.py`) | 5.0 | Metrica's frames' weight in the classifier's training | Fit | x1 against x5, held out on Metrica (one game out) and DFL | chosen: x5 better on Metrica's dead ball (D's F1 0.910 against 0.906 at a classifier weight of 0.1), DFL about equal (spell F1 0.538 against 0.539); no other value tried |
| `classifier.L2` | 1.0 | the regression's penalty (x 1e-4 per training frame) | None yet | — | set once, not searched: a light penalty on about 120 one-hot columns and 1.3 million frames |
| the classifier's bins (`DIST_EDGES`, `DIFF_EDGES`, `BALL_SPEED_EDGES`, `PACE_EDGES`, `ALONG_EDGES`, `EDGE_EDGES`, `SHARE_EDGES`) | 0.5–10 m, ... | how finely each input is binned; `ALONG_EDGES` are the pitch's markings (the boxes, the halfway line) | Resolution / Laws | — | fixed: the bins' weights are fitted |
| `classifier.CONTEXT_S` | 1, 4 s | the centred windows of the context inputs | Resolution | — | fixed |
| `restart.OFFLINE_M` | 0.5 | a player this far beyond a touchline (a classifier input) | Physics | — (beyond the tracking's noise) | fixed |

**How the possession weights were chosen.** On SkillCorner's 20 matches, upsampled to 25
fps, scored by the organisers' spell F1 against their phases of play: a team keeps possession
until the other one gains control, the challenge's controlled possession. Their ball is a
broadcast's, used as it is. The setting best on
19 matches, scored on the 20th, with the possession tables calibrated on the 19 too, scores as
the chosen one (0.515).

**The check on DFL.** No external dataset has both possession on every frame and a
ball that goes missing as a tactical camera's does, so the seven DFL matches (exact possession and
status) had their ball removed by a rule fitted on Metrica's sample games (`mtah.tune.missing`:
about 7% of live frames and 91% of dead ones, with Metrica's run lengths). Each setting was scored
by the organisers' spell F1 against the DFL labels (`scripts/weights_report.py --dfl`): the
flight helpers were fitted there, and every later choice was checked on it held out. Terms that
were tried and set to 0 by these searches (the team term's weakening in flight, the kicking
team carried through a flight, kicks weighed by their receiver, a seen-ball term, a
restarting-team state) are not in the code.

## The last external run behind each chosen value

The scores below come from the last run that chose or checked each value, on external data
only, as the scripts printed them; their logs are kept outside the repository with the external
data. In each row one value moves and the others stay at their shipped values. The runs were
made with the development code of the time. The terms dropped since were set to 0 in all of
them, and the shipped model with the classifier off gives the same starting scores: SkillCorner
spell F1 0.5152, DFL 0.5124, Metrica D's F1 0.910 (the W_CLF table, weight 0).

The values were chosen in stages, and an earlier stage was not searched again after a later one:

1. the cleaning, the keepers and the flight helpers on DFL;
2. the possession weights on SkillCorner, with the classifier off;
3. the dead-ball weights on Metrica, checked on SkillCorner;
4. the classifier's weight last.

**Possession weights.** Mean spell F1 over SkillCorner's 20 matches
(`scripts/weights_report.py --skillcorner`). This run used the calibration of the time, where
the shipped setting scores 0.5156; with the shipped calibration it scores 0.5152. With one match
left out, the setting best on the other 19 is the shipped one in 19 of 20 folds (held-out mean
0.5144).

| value | shipped (0.5156) | tried, spell F1 |
|---|---|---|
| `W_TEAM` | 0.1 | 0.03: 0.4933 · 0.05: 0.5030 · 0.075: 0.5082 · 0.15: 0.5106 · 0.2: 0.5027 · 0.3: 0.4880 · 0.5: 0.4619 · 0.8: 0.4426 |
| `W_KICK` | 4 | 0: 0.4722 · 0.5: 0.4813 · 1: 0.4878 · 2: 0.5011 · 3: 0.5082 · 6: 0.5092 · 8: 0.5019 · 12: 0.4798 |
| `MIN_LIVE_S` | 3 s | 0.1: 0.5041 · 0.5: 0.5053 · 1: 0.5051 · 2: 0.5091 · 2.5: 0.5122 · 3.5: 0.5142 · 4: 0.5126 · 5: 0.5077 · 6: 0.4990 · 8: 0.4790 |
| `FLIGHT_SPEED` / `FLIGHT_S` | 4 m/s / 0.5 s | 2/0.5: 0.5028 · 3/0.5: 0.5130 · 6/0.5: 0.5139 · 4/0.8: 0.5137 · 6/0.3: 0.5146 · 8/0.3: 0.5061 · 10/0.3: 0.5044 |

**Dead-ball weights.** D's F1 on Metrica's games 1 and 2, with the shipped calibration
(`scripts/dead_weights_report.py`):

| `W_DEAD`, `W_STOPPAGE`, `MIN_DEAD_S` | D's F1, game 1 / game 2 (mean) | SkillCorner spell F1 |
|---|---|---|
| **0.1, 0.2, 3 s (shipped)** | 0.913 / 0.907 (0.910) | **0.5152** |
| 0.5, 0.5, 3 s (best on Metrica; with one game left out, the other game picks 0.5, 0.5, with `MIN_DEAD_S` 3 s or 4 s) | 0.916 / 0.912 (0.914) | 0.4988, with more possessions than the true ones in all 20 matches |

Metrica's best setting loses 0.016 of spell F1 on SkillCorner, so it fails the every-source
rule, and the shipped weights stay. `MIN_DEAD_S` hardly matters on Metrica: at the best setting,
3 and 4 s tie at 0.914. An earlier run, made with the calibration of the time at the shipped
weights, gave 0.922, 0.922 and 0.921 for 2, 3 and 4 s.

**The classifier's weight `W_CLF`.** Held out (`scripts/hybrid_report.py`): out of fold on
SkillCorner (4 folds by match), the classifier trained on SkillCorner and Metrica and scored on
DFL, and one game out on Metrica.

| `W_CLF` | SkillCorner spell / macro F1 | DFL spell / macro F1 | Metrica D's F1, game 1 / game 2 |
|---|---|---|---|
| 0 | 0.5152 / 0.6957 | 0.5124 / 0.8795 | 0.913 / 0.907 |
| **0.05 (shipped)** | **0.5585 / 0.8403** | **0.5259 / 0.8806** | **0.914 / 0.911** |
| 0.1 | 0.5615 / 0.8443 | 0.5377 / 0.8817 | 0.904 / 0.915 |
| 0.2 | 0.5668 / 0.8515 | 0.5461 / 0.8857 | 0.894 / 0.912 |

The higher weights score better on SkillCorner and DFL, but they lose on Metrica's game 1.
Metrica is the only source whose ball really goes missing as a tactical camera's does. 0.05 is
the largest weight in the grid that loses on no source's mean and on neither Metrica game.

**Flight helpers.** Mean spell F1 over DFL's 7 matches, ball removed as Metrica's. This was
fitted at stage 1, with the possession tables then counted on DFL, so the scores are on a
different scale from the tables above. The search that chose them was removed with the unused
code. `scripts/sensitivity_report.py` still rechecks these values at half and double.

| value | shipped (0.6858) | tried, spell F1 |
|---|---|---|
| `KICK_S` | 0.12 s | 0.08: 0.6832 · 0.2: 0.6816 |
| `KICKER_M` | 3 m | 1.5: 0.6784 · 2: 0.6832 · 4: 0.6830 · 5: 0.6832 |
| `TURN_SPEED` | 3 m/s | 2: 0.6797 · 2.5: 0.6821 · 4: 0.6846 · 5: 0.6614 |
| `KINK_DEG` | 30° | 15: 0.6673 · 20: 0.6795 · 40: 0.6817 |

**Ball over the line, `OUTSIDE_MARGIN_M`.** D's F1 on Metrica, game 1 / game 2, at
`EXIT_SPEED` 2:

| margin | 0.25 m | 0.5 m | **0.75 m (shipped)** | 1.0 m (the earlier default) |
|---|---|---|---|---|
| D's F1 | 0.9215 / 0.9169 | 0.9221 / 0.9189 | **0.9220 / 0.9205** | 0.9116 / 0.9109 |

0.75 m is the best value on game 2, and better than the earlier default on both games. 0.5 m
beats it on game 1 by 0.0001 and loses on game 2 by 0.0016. `EXIT_SPEED` 3 and 4 score the
same as 2 within 0.0003.

**Keepers.** On DFL's named keepers (`scripts/tune_external.py`, `mtah.tune.keepers`). Coverage
is the share of keeper frames identified; flips are frames given the wrong team.

| `back_s` | 60 s | 120 s | **240 s (shipped)** |
|---|---|---|---|
| coverage, flips (`takeover_contact_m` 1 m) | 83.4%, 0.013% | 84.4%, 0.013% | **84.6%, 0.013%** |

At `takeover_contact_m` 1, 1.5 and 2 m, flips are the same (0.013%); at 3 m they rise to 0.047%.
2 m is shipped, the value nearest the old default of 3 m among the tied best.

**Values whose run left no saved log:** the cleaning thresholds (`stale_max_spread_m`,
`coast_max_excess_m`), `seed_radius_m`, `seed_s`, `far_from_line_m`, `MOVE_M`, `LINE_M` and
`WHISTLE_MAX_S`. Their fits are in the repository: `scripts/tune_external.py` prints each
fitted value beside the code's default, and `scripts/calibrate_external.py` does the same for
the calibration. `METRICA_WEIGHT`'s comparison is in its row above. The values that were only
checked at half and double are reproduced by `scripts/sensitivity_report.py`.

## Provided metadata, not tuned

The halves and the second-half kick-off frame come from `games.py`, the organisers' published
metadata. They place the halves; they are not tuned.

## The missing ball (`mtah.tune.missing`)

The external tracking either never loses the ball (DFL) or loses it mostly while play is dead,
because a broadcast camera cuts to replays (SkillCorner). **Metrica's sample games lose it the
way a tactical camera does**: on 39–41% of their frames and 92–95% of their dead frames (their
events time the stoppages). So:

- the calibration's missing-ball terms (the dead table, the stoppage shares) are counted on
  Metrica directly (`scripts/calibrate_external.py`);
- to score settings on DFL, its ball is removed by a rule fitted on Metrica (`mtah.tune.missing`:
  how often the ball goes missing in live and in dead play, about 7% and 91%, and its gaps'
  lengths, by their quantiles, `QUANTILES` 2.5%–97.5%, Resolution), with a fixed seed per match;
  the chosen setting scores about the same with the ball removed half or one and a half times as
  often.

The provided match's own tracking is not used to set any of it.

## Dropped

- **The referee's whistle from the broadcast audio.** A detector's threshold could be set on
  synthetic mixtures of licensed whistle recordings, but the rest of the term cannot: how much
  likelier a stop is around a whistle, and how much the stoppage after one weighs, need a match
  with audio, whistles and possession labels together, and none is available to us (SoccerTrack
  v2 has no audio; SoccerNet is not used). Setting them without data would be tuning them by
  hand, so the submission does not use the audio, and its code is not in this repository.
- **The broadcast's cut-aways** (replays and close-ups), for the same reason at the time: no
  external match with a broadcast and labels was in hand. SkillCorner's broadcast tracking could
  now fit them (its replays show as frames with no player detected); not done in time.
