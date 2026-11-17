"""A per-frame classifier for the hybrid model: P(H, A, D | the frame's features).

The model's own evidence is a tempered naive Bayes: each signal's table multiplied in
with a weight, because the signals are not independent. Here one discriminative model weighs
them together: a multinomial logistic regression over binned features of the frame and its
surroundings, fitted on external data (SkillCorner's phases and Metrica's dead ball, scripts/train_classifier.py). Its
posteriors, divided by the class priors, are scaled likelihoods (the hybrid NN/HMM of speech
recognition) that the same semi-Markov decoding smooths over time (mtah.predict.hmm).

Features, from the home team's side (frame_table): the nearest home and away players' distances
to the ball and their difference; the ball missing, dribbled, still, in flight, its speed; the
kicking and receiving teams in flight; the players' median pace; a ball-out stoppage; a team
beyond a touchline; where the ball is along the pitch from home's goal and from the touchlines;
and centred means over CONTEXT_S of who is nearer, the ball missing, and the pace. Each is
binned and one-hot coded (design), so the regression is a weighted vote of bins.

Team symmetry is tied: every training frame is also used mirrored (home and away swapped, the
label too), and at decoding the two posteriors are averaged, so the model cannot prefer a team.
Coefficients and bins are plain numbers: the model ships as JSON, like the calibration.
"""

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize

from mtah.dead.features import ball_still, median_speed, speeds
from mtah.dead.score import bridge_dribbles, dribbling, out_stoppages
from mtah.predict.flight import in_flight, kickers, receivers
from mtah.predict.possession import nearest
from mtah.predict.restart import offline_team

# Params
DIST_EDGES = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)  # m, each team's nearest player
DIFF_EDGES = (-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0)  # m, away's minus home's
BALL_SPEED_EDGES = (0.5, 2.0, 5.0, 10.0, 20.0)  # m/s
PACE_EDGES = (0.5, 1.0, 1.5, 2.0, 3.0)  # m/s
ALONG_EDGES = (5.5, 16.5, 35.0, 52.5, 70.0, 88.5, 99.5)  # m from home's goal line
EDGE_EDGES = (1.0, 5.0, 15.0)  # m from the nearer touchline
SHARE_EDGES = (0.1, 0.3, 0.5, 0.7, 0.9)  # the context shares
CONTEXT_S = (1.0, 4.0)  # centred windows, seconds
L2 = 1.0  # the regression's penalty, per training frame x 1e-4
# The hybrid's weight on the classifier's scaled likelihoods, added to the model's evidence:
# the largest that loses on no external source and no game (docs/technical-document.md, section 4).
W_CLF = 0.05
CLASSES = ("H", "A", "D")
CLASSIFIER = Path(__file__).with_name("frame_classifier.json")


def frame_table(tracking, n_frames, flight=None):
    """Per video frame 0..n_frames-1, from (cleaned) tracking, home's side: a dict of arrays
    (float NaN, or int -1, where unknown)."""
    players, ball, fps = tracking.players, tracking.ball, tracking.fps
    team = np.asarray(tracking.team)
    codes, best, opponent = nearest(players, team, ball)
    home = np.where(codes == 0, best, opponent)
    away = np.where(codes == 1, best, opponent)
    seen = codes >= 0
    home[~seen] = np.nan
    away[~seen] = np.nan
    missing = np.isnan(ball).any(axis=1)
    pace = median_speed(players, fps)
    flying = in_flight(ball, fps, tracking.period, **(flight or {}))
    length, width = tracking.pitch_size
    defends_left = _home_defends_left(players, team, tracking.period)
    along = np.where(defends_left, ball[:, 0], length - ball[:, 0])
    edge = np.minimum(ball[:, 1], width - ball[:, 1])
    rows = {
        "home_m": home,
        "away_m": away,
        "missing": missing.astype(np.int8),
        "dribbled": bridge_dribbles(dribbling(players, ball, fps), missing).astype(
            np.int8
        ),
        "still": ball_still(ball, fps).astype(np.int8),
        "ball_speed": speeds(ball, fps),
        "flight": flying.astype(np.int8),
        "kicker": kickers(flying, codes, best, fps),
        "receiver": receivers(flying, codes, best, fps),
        "pace": pace,
        "stoppage": out_stoppages(
            ball, pace, tracking.period, fps, tracking.pitch_size
        ).astype(np.int8),
        "offline": offline_team(players, team, tracking.pitch_size),
        "along": along,
        "edge": edge,
    }
    frame = tracking.video_frame
    inside = (frame >= 0) & (frame < n_frames)
    out = {}
    for name, values in rows.items():
        values = np.asarray(values)
        fill = np.nan if values.dtype.kind == "f" else -1
        out[name] = np.full(n_frames, fill, values.dtype)
        out[name][frame[inside]] = values[inside]
    out["fps"] = fps
    out["length"] = float(length)
    return out


def _home_defends_left(players, team, period):
    """Per row: home's goal at x = 0, from the teams' mean x over each period."""
    out = np.ones(len(period), bool)
    for p in np.unique(period[period > 0]):
        rows = period == p
        home_x, away_x = (_mean(players[rows][:, team == t, 0]) for t in (0, 1))
        out[rows] = not home_x > away_x
    return out


def _mean(values):
    """The mean of the finite values, NaN without any."""
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def mirrored(table):
    """The table with home and away swapped."""
    out = dict(table)
    out["home_m"], out["away_m"] = table["away_m"], table["home_m"]
    for name in ("kicker", "receiver", "offline"):
        v = table[name]
        out[name] = np.where(v >= 0, 1 - v, v).astype(v.dtype)
    out["along"] = table["length"] - table["along"]
    return out


def _onehot(values, edges, block):
    """(column per row, or -1) for values binned by edges, NaN in an extra last bin."""
    values = np.asarray(values, float)
    col = np.searchsorted(edges, values, side="right")
    col = np.where(np.isnan(values), len(edges) + 1, col)
    return block + col, len(edges) + 2


def _codes(values, n, block):
    """Columns for integer codes 0..n-1, -1 in an extra last bin."""
    v = np.asarray(values)
    return block + np.where(v < 0, n, v), n + 1


def design(table):
    """(sparse one-hot design matrix, n columns) of a frame table."""
    fps = table["fps"]
    n = len(table["home_m"])
    home, away = table["home_m"], table["away_m"]
    nearer = np.where(np.isnan(home), np.nan, np.where(home <= away, 1.0, 0.0))
    parts = [
        (home, DIST_EDGES),
        (away, DIST_EDGES),
        (away - home, DIFF_EDGES),
        (table["ball_speed"], BALL_SPEED_EDGES),
        (table["pace"], PACE_EDGES),
        (table["along"], ALONG_EDGES),
        (table["edge"], EDGE_EDGES),
    ]
    for seconds in CONTEXT_S:
        window = max(1, round(seconds * fps))
        parts += [
            (_window_mean(nearer, window), SHARE_EDGES),
            (_window_mean(table["missing"].astype(float), window), SHARE_EDGES),
            (_window_mean(table["pace"], window), PACE_EDGES),
        ]
    columns, block = [], 0
    for values, edges in parts:
        col, width = _onehot(values, edges, block)
        columns.append(col)
        block += width
    for name, k in (
        ("missing", 2),
        ("dribbled", 2),
        ("still", 2),
        ("flight", 2),
        ("kicker", 2),
        ("receiver", 2),
        ("stoppage", 2),
        ("offline", 2),
    ):
        col, width = _codes(table[name], k, block)
        columns.append(col)
        block += width
    columns.append(np.full(n, block))  # the intercept
    cols = np.column_stack(columns)
    rows = np.repeat(np.arange(n), cols.shape[1])
    data = np.ones(cols.size, np.float32)
    x = sparse.csr_matrix((data, (rows, cols.ravel())), shape=(n, block + 1))
    return x, block + 1


def _window_mean(values, window):
    """Centred mean of the non-NaN values; NaN where the window has none."""
    values = np.asarray(values, float)
    seen = ~np.isnan(values)
    total = uniform_filter1d(np.where(seen, values, 0.0), window, mode="constant")
    count = uniform_filter1d(seen.astype(float), window, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 1e-9, total / count, np.nan)


def labels_codes(labels):
    """H, A, D as 0, 1, 2; L (live, team unknown: Metrica's games) as 3; -1 for anything else."""
    labels = np.asarray(labels)
    return np.select([labels == c for c in (*CLASSES, "L")], [0, 1, 2, 3], -1)


# The classes each code allows: H, A, D, and L (H or A).
ALLOWED = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], float)
SWAP = np.array([1, 0, 2, 3])


def fit(tables, labels, l2=L2, every=5, max_iter=300, weights=None):
    """{coef, priors, n_columns}: the regression fitted on frame tables and their labels (H, A,
    D, or L for live with the team unknown, scored as H or A), every `every`-th labelled frame,
    each also mirrored; `weights`, one per table, weighs its frames (default 1)."""
    weights = [1.0] * len(tables) if weights is None else list(weights)
    xs, ys, ws = [], [], []
    for table, lab, weight in zip(tables, labels, weights, strict=True):
        y = labels_codes(lab)
        keep = np.flatnonzero(y >= 0)[::every]
        for t, yy in ((table, y[keep]), (mirrored(table), SWAP[y[keep]])):
            x, n_columns = design(t)
            xs.append(x[keep])
            ys.append(yy)
            ws.append(np.full(len(keep), float(weight)))
    x = sparse.vstack(xs).tocsr()
    y = np.concatenate(ys)
    sample = np.concatenate(ws)
    sample /= sample.mean()
    m = len(y)
    allowed = ALLOWED[y]
    penalty = l2 * 1e-4

    def loss(w):
        w = w.reshape(n_columns, 3)
        z = np.asarray(x @ w)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        mass = np.maximum((p * allowed).sum(axis=1), 1e-300)
        value = -np.sum(sample * np.log(mass)) / m + 0.5 * penalty * np.sum(w[:-1] ** 2)
        # d(-log mass)/dz = p - p * allowed / mass
        g = (p - p * allowed / mass[:, None]) * sample[:, None]
        grad = np.asarray(x.T @ g) / m
        grad[:-1] += penalty * w[:-1]
        return value, grad.ravel()

    result = minimize(
        loss,
        np.zeros(n_columns * 3),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    dead = float(np.sum(sample * (y == 2)) / np.sum(sample))
    return {
        "coef": result.x.reshape(n_columns, 3).tolist(),
        "priors": [(1 - dead) / 2, (1 - dead) / 2, dead],
        "n_columns": n_columns,
        "train_frames": int(m),
        "converged": bool(result.success),
    }


def log_proba(model, table):
    """(n, 3) log P(H, A, D | frame), the plain and the mirrored posteriors averaged."""
    w = np.asarray(model["coef"])

    def one(t):
        x, n_columns = design(t)
        if n_columns != len(w):
            raise ValueError("the classifier was fitted on another feature layout")
        z = np.asarray(x @ w)
        z -= z.max(axis=1, keepdims=True)
        return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

    plain = one(table)
    swapped = one(mirrored(table))[:, [1, 0, 2]]
    return np.log(np.clip(0.5 * (plain + swapped), 1e-12, 1.0))


def scaled_loglik(model, table):
    """(n, 3) log P(frame | H, A, D) up to a constant: the posteriors over the priors, the
    priors made symmetric in H and A as the model is."""
    priors = np.asarray(model["priors"], float)
    priors[:2] = priors[:2].mean()
    return log_proba(model, table) - np.log(priors)


def save_model(model, path=CLASSIFIER, attribution=""):
    payload = {"attribution": attribution, **model}
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load_model(path=CLASSIFIER):
    return json.loads(Path(path).read_text(encoding="utf-8"))
