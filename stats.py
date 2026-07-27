"""Shared statistical primitives.

Every table in the package routes its p-values, effect sizes and E-values
through this module, so each rule is written once.
"""

import numpy as np
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

import config as cfg


def fmt_p(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def phi_ci(table):
    """Phi coefficient with 95% CI and chi-square p for a 2 x k table.

    Phi is the Pearson correlation of the two indicator variables, so the
    confidence interval uses the Fisher z transformation.
    """
    table = np.asarray(table, float)
    if table.shape[0] < 2 or table.shape[1] < 2 or table.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return np.nan, np.nan, np.nan, np.nan
    chi2, p, _, _ = sps.chi2_contingency(table, correction=False)
    n = table.sum()
    phi = float(np.sqrt(chi2 / n))
    if table.shape == (2, 2):
        a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
        phi *= np.sign(a * d - b * c)
    if n <= 3:
        return phi, np.nan, np.nan, float(p)
    z = np.arctanh(np.clip(phi, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return phi, float(lo), float(hi), float(p)


def smd_ci(x, y):
    """Standardized mean difference with 95% CI and a Welch t-test p-value."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan, np.nan
    pooled = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1))
                     / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan, np.nan, np.nan, np.nan
    d = (x.mean() - y.mean()) / pooled
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    p = sps.ttest_ind(x, y, equal_var=False).pvalue
    return float(d), float(d - 1.96 * se), float(d + 1.96 * se), float(p)


def welch_or_anova(groups):
    """Welch t-test for two groups, one-way ANOVA for more than two."""
    groups = [np.asarray(g, float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return np.nan
    if len(groups) == 2:
        return float(sps.ttest_ind(*groups, equal_var=False).pvalue)
    return float(sps.f_oneway(*groups).pvalue)


def evalue_from_rr(rr, ci_lo=None, ci_hi=None):
    """E-value (VanderWeele and Ding 2017) from a risk-ratio scale estimate.

    Returns (E_point, E_ci). The confidence bound nearest the null is used; if
    the interval crosses the null the bound E-value is 1.
    """
    if rr is None or not np.isfinite(rr) or rr <= 0:
        return np.nan, np.nan

    def _e(r):
        r = max(r, 1.0 / r)
        return r + np.sqrt(r * (r - 1.0))

    e_point = _e(rr)
    if ci_lo is None or ci_hi is None or not np.isfinite(ci_lo) \
            or not np.isfinite(ci_hi):
        return e_point, np.nan
    if rr >= 1.0:
        bound = ci_lo
        if bound <= 1.0:
            return e_point, 1.0
    else:
        bound = ci_hi
        if bound >= 1.0:
            return e_point, 1.0
    return e_point, _e(bound)


def evalue_from_beta(beta, lo, hi, sd_outcome):
    """E-value for a continuous outcome.

    The coefficient is standardized by the outcome's standard deviation and
    mapped to an approximate risk ratio using VanderWeele and Ding's
    RR = exp(0.91 * d) transformation, then handed to evalue_from_rr.
    """
    if not np.isfinite(sd_outcome) or sd_outcome <= 0:
        return np.nan, np.nan
    scale = 0.91 / sd_outcome
    return evalue_from_rr(np.exp(beta * scale),
                          np.exp(lo * scale), np.exp(hi * scale))


def bh_adjust(pvalues):
    """Benjamini-Hochberg FDR correction that tolerates missing p-values."""
    raw = np.asarray(pvalues, float)
    out = np.full(len(raw), np.nan)
    valid = np.isfinite(raw)
    if valid.any():
        out[valid] = multipletests(raw[valid], method="fdr_bh")[1]
    return out


def write_table(frame, filename, count_col="num_patients"):
    """Write one table to results/. Every row must report its patient count."""
    if count_col not in frame.columns:
        raise KeyError(f"table has no '{count_col}' column; every exported row "
                       f"must report how many patients it describes")
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.RESULTS_DIR / filename
    frame.to_csv(path, index=False)
    print(f"[write] {filename}  ({len(frame)} rows)")
    return path
