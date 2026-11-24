"""Interval-valued envelopes of a speech signal.

Each window of w samples becomes an interval [lower, upper] around the
window median. Three ways to choose the bounds:

    peak      : the window max and min. Covers every sample, but one click
                moves it.
    quantile  : the q and 1-q quantiles. A simple robust baseline.
    granular  : the principle of justifiable granularity. For the upper bound
                b over candidate values (the samples above the median):
                    coverage(b)    = #{x : med <= x <= b} / w
                    specificity(b) = max(0, 1 - (b - med) / R)
                and b maximizes coverage * specificity. The lower bound is
                chosen the same way below the median. R is the signal's
                range; a robust value (from the 99.9 % quantile of |x|) is
                used so that a few clicks do not change it.

Why the granular bound ignores a click: covering one extra sample adds only
1/w to the coverage, while stretching b up to the click costs
(click - previous bound) / R in specificity. For a click well above the
speech, the loss is far larger than the gain, so the bound stays put.
"""

import numpy as np


def _windows(x, w):
    n = len(x) // w
    return x[: n * w].reshape(n, w)


def robust_range(x, q=0.999):
    return 2.0 * np.quantile(np.abs(x - np.median(x)), q) + 1e-8


def peak_envelope(x, w=320):
    X = _windows(x, w)
    return X.min(1), X.max(1)


def quantile_envelope(x, w=320, q=0.05):
    X = _windows(x, w)
    return np.quantile(X, q, axis=1), np.quantile(X, 1 - q, axis=1)


def _justified_bound(X, med, R):
    """Upper bound for each row of X (values are taken relative to med)."""
    w = X.shape[1]
    S = np.sort(X, axis=1)
    above = S >= med[:, None]  # sorted, so this is a suffix of each row
    start = w - above.sum(1)  # index of the first sample >= med
    k = np.arange(w)[None, :]
    count = np.where(above, k - start[:, None] + 1, 0)
    cov = count / w
    spec = np.clip(1.0 - (S - med[:, None]) / R, 0.0, None)
    score = np.where(above, cov * spec, -1.0)
    best = score.argmax(1)
    return S[np.arange(len(S)), best], score.max(1)


def granular_envelope(x, w=320, R=None, return_coverage=False):
    R = robust_range(x) if R is None else R
    X = _windows(x, w)
    med = np.median(X, axis=1)
    upper, _ = _justified_bound(X, med, R)
    lower_neg, _ = _justified_bound(-X, -med, R)
    lower = -lower_neg
    if return_coverage:
        inside = ((X >= lower[:, None]) & (X <= upper[:, None])).mean(1)
        return lower, upper, inside
    return lower, upper


def add_clicks(x, n_clicks=20, amplitude=0.9, rng=None):
    rng = np.random.default_rng(rng)
    y = x.copy()
    pos = rng.choice(len(x), n_clicks, replace=False)
    y[pos] = amplitude * np.abs(x).max() * rng.choice([-1, 1], n_clicks)
    return y, pos
