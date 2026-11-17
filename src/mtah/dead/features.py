"""Per-frame signals of a dead ball, from the tracking alone. Positions in metres.

    median_speed          the players' median speed, averaged over a short centred window: the
                          whole pitch slows down at a stoppage, ball visible or not
    ball_outside          the ball seen beyond a touchline or goal line, past a noise margin
    ball_still            the ball seen and barely moving over a window: a placed set piece

Each returns one value per tracking row.
"""

import warnings

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

# Params
SPEED_WINDOW_S = 2.0
OUTSIDE_MARGIN_M = 0.75  # fitted on Metrica's sample games (D's F1, both games)
STILL_WINDOW_S = 1.5
STILL_MAX_SPEED = 0.5  # m/s


def speeds(xy, fps):
    """Speed in m/s from each row to the previous one; NaN on the first row and where unseen.

    xy is (rows, 2) or (rows, slots, 2).
    """
    out = np.full(xy.shape[:-1], np.nan)
    step = np.diff(xy, axis=0)
    out[1:] = np.hypot(step[..., 0], step[..., 1]) * fps
    return out


def median_speed(players, fps, window_s=SPEED_WINDOW_S):
    """Median speed over the observed players, then a centred mean over `window_s`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # rows with nobody observed
        median = np.nanmedian(speeds(players, fps), axis=1)
    return _centred_mean(median, max(1, round(window_s * fps)))


def ball_outside(ball, pitch_size, margin=OUTSIDE_MARGIN_M):
    """The ball seen more than `margin` beyond a touchline or goal line."""
    length, width = pitch_size
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (
            (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)
        )


def ball_still(ball, fps, window_s=STILL_WINDOW_S, max_speed=STILL_MAX_SPEED):
    """The ball seen on every row of a centred `window_s` window, never faster than max_speed."""
    speed = speeds(ball, fps)
    speed[np.isnan(speed)] = np.inf
    window = max(1, round(window_s * fps))
    return maximum_filter1d(speed, size=window, mode="nearest") < max_speed


def _centred_mean(values, window):
    """Mean of the non-NaN values in a centred window; NaN where the window has none."""
    seen = ~np.isnan(values)
    # Ratio of two window means is the ratio of the sums; zeros pad beyond the ends.
    total = uniform_filter1d(np.where(seen, values, 0.0), window, mode="constant")
    count = uniform_filter1d(seen.astype(float), window, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)
