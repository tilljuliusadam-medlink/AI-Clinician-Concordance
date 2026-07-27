"""Inter-rater agreement coefficients and the case-level bootstrap.

Krippendorff's alpha (ordinal metric) and Gwet's AC2 (ordinal weights) are
implemented directly so the package needs no dependency beyond numpy, pandas,
scipy and statsmodels.

Both statistics are functions of a SUM of per-case contribution vectors, so one
bootstrap resample is a single matrix product and one function serves both.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

import config as cfg

Q = len(cfg.RATING_LEVELS)


def ordinal_weights(q=Q):
    """Gwet's ordinal weights."""
    k = np.arange(q)
    steps = np.abs(k[:, None] - k[None, :]) + 1
    raw = steps * (steps - 1) / 2.0
    return 1.0 - raw / raw.max()


def ratio(totals):
    totals = np.asarray(totals, float)
    numerator, denominator = totals[..., 0], totals[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def case_contrib(frame, value_col, case_col="case_id"):
    """Per-case [sum, count]: the contribution form of any mean."""
    grouped = frame.groupby(case_col)[value_col].agg(["sum", "count"])
    return grouped.to_numpy(float)


def pair_matrix(frame, col, case_col="case_id", rater_col="reviewer_id"):
    """Reviewer-pair values for cases rated exactly twice."""
    counts = frame.groupby(case_col)[col].transform("count")
    two = frame[counts == 2].sort_values([case_col, rater_col])
    if two.empty:
        return np.zeros((0, 2), dtype=int)
    values = two.groupby(case_col)[col].apply(list)
    return np.array([v for v in values if len(v) == 2], dtype=int)


def alpha_contrib(pairs, q=Q):
    """Per-unit coincidence-matrix contributions, flattened."""
    out = np.zeros((len(pairs), q * q))
    for i, (a, b) in enumerate(pairs - 1):
        cell = np.zeros((q, q))
        cell[a, b] += 1.0
        cell[b, a] += 1.0
        out[i] = cell.ravel()
    return out


def alpha_from_coincidence(totals, q=Q):
    """Krippendorff's alpha with the ordinal difference metric."""
    totals = np.asarray(totals, float)
    obs = totals.reshape(totals.shape[:-1] + (q, q))
    n_c = obs.sum(-1)
    n_total = n_c.sum(-1)
    cumulative = np.cumsum(n_c, axis=-1)
    index = np.arange(q)
    low = np.minimum(index[:, None], index[None, :])
    high = np.maximum(index[:, None], index[None, :])
    cum_b = np.broadcast_to(cumulative[..., None, :],
                            cumulative.shape[:-1] + (q, q))
    counts_b = np.broadcast_to(n_c[..., None, :], n_c.shape[:-1] + (q, q))
    upper = np.take_along_axis(cum_b, np.broadcast_to(high, cum_b.shape), -1)
    lower = np.take_along_axis(
        cum_b, np.broadcast_to(np.clip(low - 1, 0, q - 1), cum_b.shape), -1)
    lower = np.where(np.broadcast_to(low, lower.shape) == 0, 0.0, lower)
    n_i = np.take_along_axis(counts_b, np.broadcast_to(low, counts_b.shape), -1)
    n_j = np.take_along_axis(counts_b, np.broadcast_to(high, counts_b.shape), -1)
    distance = ((upper - lower) - (n_i + n_j) / 2.0) ** 2
    observed = (obs * distance).sum((-2, -1))
    expected = ((n_c[..., :, None] * n_c[..., None, :]) * distance).sum((-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            expected > 0,
            1.0 - (n_total - 1.0) * observed / np.where(expected > 0,
                                                        expected, 1.0),
            np.nan)


def ac2_contrib(frame, col, case_col="case_id", q=Q):
    """Per-case contributions for Gwet's AC2."""
    weights = ordinal_weights(q)
    rows = []
    for _, sub in frame.groupby(case_col):
        values = sub[col].to_numpy(int)
        n_raters = len(values)
        counts = np.bincount(values - 1, minlength=q).astype(float)
        if n_raters >= 2:
            weighted = counts @ weights
            agreement = float((counts * (weighted - 1)).sum()) \
                / (n_raters * (n_raters - 1))
            denominator = 1.0
        else:
            agreement, denominator = 0.0, 0.0
        rows.append(np.concatenate([[agreement, denominator],
                                    counts / n_raters, [1.0]]))
    return np.array(rows, float)


def ac2_from_contrib(totals, q=Q):
    totals = np.asarray(totals, float)
    weight_sum = ordinal_weights(q).sum()
    observed, denominator = totals[..., 0], totals[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        proportions = totals[..., 2:2 + q] / totals[..., 2 + q][..., None]
        agreement = np.where(denominator > 0, observed / denominator, np.nan)
        expected = (weight_sum * (proportions * (1 - proportions)).sum(-1)
                    / (q * (q - 1)))
        return np.where(expected < 1, (agreement - expected) / (1 - expected),
                        np.nan)


def bootstrap_ci(contrib, statistic, n_boot=cfg.BOOTSTRAP_N, seed=cfg.SEED,
                 alpha=0.05):
    """Bias-corrected and accelerated CI, resampling cases with replacement.

    Falls back to a percentile interval when the acceleration term cannot be
    computed, and reports which method was used rather than hiding it.
    """
    contrib = np.asarray(contrib, float)
    n_cases = len(contrib)
    total = contrib.sum(0)
    point = float(statistic(total))
    if n_cases < 3 or not np.isfinite(point):
        return point, np.nan, np.nan, "undefined"

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_cases, np.full(n_cases, 1.0 / n_cases),
                             size=n_boot).astype(float)
    boots = np.asarray(statistic(counts @ contrib), float)
    boots = boots[np.isfinite(boots)]
    if len(boots) < 100:
        return point, np.nan, np.nan, "undefined"

    below = float((boots < point).mean())
    jackknife = np.asarray(statistic(total - contrib), float)
    if below <= 0 or below >= 1 or not np.isfinite(jackknife).all():
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
        return point, float(lo), float(hi), "percentile"

    z0 = float(norm.ppf(below))
    mean_jack = float(jackknife.mean())
    denominator = 6.0 * float(((mean_jack - jackknife) ** 2).sum()) ** 1.5
    accel = (float(((mean_jack - jackknife) ** 3).sum()) / denominator
             if denominator > 0 else 0.0)
    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    q1 = float(norm.cdf(z0 + (z0 + z_lo) / (1 - accel * (z0 + z_lo))))
    q2 = float(norm.cdf(z0 + (z0 + z_hi) / (1 - accel * (z0 + z_hi))))
    method = "bca"
    if not (np.isfinite(q1) and np.isfinite(q2)) or q1 >= q2:
        q1, q2, method = alpha / 2, 1 - alpha / 2, "percentile"
    lo, hi = np.quantile(boots, [q1, q2])
    return point, float(lo), float(hi), method
