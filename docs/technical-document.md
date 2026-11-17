# Ball possession from automatic tracking: a semi-Markov model with a hybrid classifier

More Than a Hack 2027, Phase 1. This document describes the system that produces our submission,
the data it was built on, and how to reproduce it. All code is in this repository; every number
the submission depends on is fitted on external data, listed in `docs/tunable-values.md`.

## 1. Approach

For every frame of the tactical video we label H (home in possession), A (away), or D (dead ball),
and X outside the two halves. The ranking metric, spell F1, rewards whole possessions, so the
labelling is a **segmentation** problem rather than a per-frame classification: a single wrong
frame inside a possession splits it in two. We therefore decode the whole match with a
**hidden semi-Markov model** whose states are H, A and D, each with a minimum duration, and feed
it per-frame evidence from the tracking.

The pipeline (`scripts/make_submission.py`):

1. **Clean the automatic tracking** (`mtah.clean.tracks`). Positions are converted to metres
   (105.3 x 68). Stale tracks (a lost player left frozen, or drifting in a perfectly straight
   line) are removed; goalkeepers are identified over time from their position near their own
   goal and given the team that defends it, because the provider's team label is unreliable for
   keepers; which team attacks which side is read per half from the players' mean position.
2. **Per-frame signals** (`mtah.predict.hmm.features`), all from the tracking:
   - the team of the player nearest the ball and its distance (team evidence);
   - the ball seen, missing, or *dribbled* (missing while the player who had it keeps running),
     crossed with the players' median speed (dead-ball evidence: a missing ball on a slow pitch
     is the strongest sign of a stoppage);
   - a **ball-out stoppage** (`mtah.dead.score.out_stoppages`): the ball beyond a line starts a
     stoppage that lasts until its restart, detected as a ball lying still and then moved off
     with the players setting off, or a throw-in coming back over the touchline;
   - the **ball in flight** (`mtah.predict.flight`): fast and straight between a kick and its
     reception, with the kicking and receiving teams; each kick counts once as evidence that the
     kicking team keeps the ball.
3. **Evidence**: each signal has a likelihood table P(signal | H, A, D), fitted on external data
   (section 3) and combined as a tempered naive Bayes, each term with its own weight because the
   signals are correlated (team 0.1, dead table 0.1, ball-out stoppage 0.2, kick 4).
4. **A per-frame classifier** (`mtah.predict.classifier`), the "hybrid" of speech recognition:
   a multinomial logistic regression over about 120 one-hot bins of the same signals plus the
   ball's position on the pitch, its distance to the touchlines, and 1 s and 4 s centred means.
   Its posteriors divided by the class priors are scaled likelihoods, added to the evidence with
   weight 0.05. **Team symmetry is tied**: every training frame is used twice, as it is and with
   the teams swapped, and at decoding both posteriors are averaged, so the model cannot prefer a
   team. Its coefficients ship as `src/mtah/predict/frame_classifier.json`.
5. **Decoding**: Viterbi over the semi-Markov chains per half, a possession and a stoppage each
   lasting at least 3 s, with switch, stop and restart rates fitted on external data. The frames
   between the second-half block's start and its kick-off (from `games.py`) are forced D.

Why this rather than something else: a model that only classifies frames scores well per frame
and badly on spell F1, because it flickers; hard rules (thresholds on speed, a control radius)
make one decision per signal, so several weak signals can never add up. The semi-Markov model
lets weak signals accumulate over time and pays explicitly for every change of possession. The
classifier then weighs the signals jointly where the naive-Bayes terms cannot.

## 2. Assets used, and those tried and dropped

**Used:** the automatic tactical tracking (ATD) of the provided match, and the organisers'
metadata in `games.py` (frame count, halves, second-half kick-off, pitch size). Nothing else of
the provided match is read by the submission.

**Not used:** the tactical and broadcast videos, the broadcast tracking, the homography, and
Smart Tagging.

**Tried and dropped** (judged on external data where it could be, section 4; otherwise not pursued for lack of it):

| idea | why it was dropped |
|---|---|
| the referee's whistle from the broadcast audio | a detector could be built (e.g. tuned on licensed whistle recordings), but its weight and timing in the model need a match with audio and possession labels; none was available to us, so it could not be fitted |
| the broadcast's cut-aways (replays, close-ups) as dead-ball evidence | as above at the time; we later found SkillCorner's broadcast tracking could fit it (section 6) |
| self-training on the match's own tracking (re-estimating the dead tables on the decoded labels) | no gain on the DFL matches |
| a label-free transform of distances and speeds onto the training data's | no change on any source |
| a team-of-restart state split and a seen-ball term | failed on at least one source |
| free-kick features (opponents' distance to a still ball, the wall, the pitch walking, the time to the next restart) | gains of a few thousandths, not on every source |

## 3. Training data

No data of the provided match was used to train, fit, calibrate or select anything: not its
ground truth (full or public window), not its videos, not its tracking. All of the following are
other matches:

| dataset | licence | what we used | labels: by whom | how much |
|---|---|---|---|---|
| **SkillCorner open data** (github.com/SkillCorner/opendata) | MIT | broadcast tracking, 10 fps upsampled to 25 fps; phases of play as controlled possession (H, A; D outside any phase) | SkillCorner (their phases of play) | 20 A-League 2024/25 matches |
| **Metrica Sports sample data** (github.com/metrica-sports/sample-data) | no named licence; credited | tracking whose ball goes missing as a tactical camera's does; dead ball from each ball-out, foul or goal to the next set piece, from their events (the team is not used) | Metrica Sports (their event data) | sample games 1 and 2 |
| **DFL / IDSSE dataset** (Bassek et al., *Scientific Data* 12, 195, 2025, doi:10.1038/s41597-025-04505-y; figshare doi:10.6084/m9.figshare.28196177) | CC BY 4.0, data owner Deutsche Fussball Liga | tracking at 25 fps with the possessing team and ball status on every frame, events, named keepers; its ball removed with a rule fitted on Metrica so that it goes missing as a tactical camera's does | the DFL's data provider | 7 matches of 2022/23: 2 Bundesliga, 5 2. Bundesliga |

What each one fits:

- **SkillCorner**: the team table, the switch, stop and restart rates, the flight and kick
  tables (`scripts/calibrate_external.py`, shipped as `external_calibration.json`); the
  possession weights and shortest possession (`scripts/weights_report.py --skillcorner`); the
  classifier, with Metrica (`scripts/train_classifier.py`).
- **Metrica**: the dead-ball table and the ball-out stoppage shares, the dead-ball weights
  (`scripts/dead_weights_report.py`), the restart detector's values
  (`scripts/tune_external.py`); in the classifier's training, weighed 5 times, its live frames
  counting as "H or A".
- **DFL**: the tracking cleaning and keeper identification (`scripts/tune_external.py`), the
  flight helpers, and a held-out source for every choice; it never enters the classifier's fit
  when it is scored.

The classifier is trained on 1.29 million frames (every 5th labelled frame, each also mirrored);
the calibration counts 1.58 million live and 1.37 million dead frames.

## 4. How values were chosen

Every candidate change was scored **held out** on three sources, with the organisers' metrics
(`evaluation/possession_metrics.py`): SkillCorner out of fold (4 folds by match), Metrica one
game out (D's F1, the only label it has), and DFL, which fits only the tracking cleaning and the
flight helpers, never a likelihood table, a weight or the classifier. A change was kept only if it
helped **every source and every game**. Values that could not be fitted were set from the Laws
of the Game or physics and checked insensitive at half and double their value
(`scripts/sensitivity_report.py`).

| configuration | SkillCorner spell / macro F1 | DFL spell / macro F1 | Metrica D's F1, game 1 / 2 |
|---|---|---|---|
| the semi-Markov model alone | 0.515 / 0.696 | 0.512 / 0.880 | 0.913 / 0.907 |
| **with the classifier at 0.05 (submitted)** | **0.559 / 0.840** | **0.526 / 0.881** | **0.914 / 0.911** |
| with the classifier at 0.1 | 0.562 / 0.844 | 0.538 / 0.882 | 0.904 / 0.915 |

A larger classifier weight gains more where the team is labelled but costs a Metrica game, so
0.05 is kept. No value of the submission was fitted or tuned on the public 5-minute window, and
the configuration was chosen on the external results above; the window was used to check that
the file is well formed and aligned.

## 5. The possession rules implemented

- **Possession** is controlled possession: a team keeps the ball until the other team gains
  control. Being nearest the ball for a moment does not change it; the model needs at least
  3 s of evidence for a new possession, and a ball in flight changes team only where it is
  received.
- **Loose and contested balls** stay with the team that had the ball (the challenge's
  definition), by the same minimum duration and the flight rule.
- **Dead ball** runs from the ball leaving the pitch, or a stoppage seen in the tracking (the
  ball missing while the players slow down), to the restart: the ball lying still and moved
  off with the players setting off, or the throw-in. A stoppage lasts at least 3 s.
- **Restarts** go to the team the evidence after the stoppage favours; the second half's
  graded frames before its kick-off are D; frames outside the halves are X.

## 6. Limitations

- **Undetected stoppages, mostly free kicks.** A foul in midfield with the ball rolling on and
  the players slowing gradually is often labelled live. In a tactical feed the ball is missing on
  most dead frames, so shape-around-the-ball signals cannot see these stoppages either (section
  2). On external data this is the largest remaining loss.
- **One tactical source with dead ball, none with the team.** Metrica's two games are the only
  tactical tracking in training and carry no possession team; the team evidence is learnt on
  SkillCorner's broadcast tracking.
- **Domain differences.** A-League broadcast tracking and German league tracking are not LaLiga
  tactical tracking; the held-out scores are evidence, not a promise.
- **With more time:** fit a cut-away term on SkillCorner (in an exploratory count, not part of
  this repository, its replays show as frames with no player detected on about 73% of dead
  frames against 6% of live ones) and use the provided broadcast as input; a
  larger tactical dataset with possession labels (e.g. PFF's World Cup 2022 data) for the
  classifier.

## 7. Reproduction

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/) (dependencies pinned in
`uv.lock`). The match data is not in the repository.

```bash
uv sync --locked
# the organisers' bundle in ./data, or set MTAH_DATA_DIR (PowerShell: $env:MTAH_DATA_DIR = "...")
uv run python scripts/make_submission.py --out <team_name>.csv
uv run python evaluation/validate_submission.py <team_name>.csv
```

Nothing is fitted and nothing is random at this step, so the same data gives the same file.

The shipped calibration, classifier and weights can be rebuilt from the external data (paths to
local copies of each dataset):

```bash
uv run python scripts/calibrate_external.py --skillcorner <skillcorner>/data/matches --metrica <metrica>/data
uv run python scripts/train_classifier.py --skillcorner <skillcorner>/data/matches --metrica <metrica>/data
uv run python scripts/weights_report.py --skillcorner <skillcorner>/data/matches --metrica <metrica>/data
uv run python scripts/dead_weights_report.py --metrica <metrica>/data
uv run python scripts/hybrid_report.py --dfl <dfl> --skillcorner <skillcorner>/data/matches --metrica <metrica>/data
uv run python scripts/tune_external.py --dfl <dfl> --metrica <metrica>/data
```

`uv run pytest` runs the tests, on synthetic data only.
