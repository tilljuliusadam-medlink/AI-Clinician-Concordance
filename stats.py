"""
Shared statistical primitives.
"""

import numpy as np
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

import config as cfg


def fmt_p(value):
    if value is None or not np.isfinite(value):
        return "NA"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def fmt_ci(point, lo, hi, digits=2):
    return f"{point:.{digits}f} ({lo:.{digits}f}, {hi:.{digits}f})"


def wald_p(estimate, se):
    """Two-sided Wald p for estimate = 0."""
    return float(2 * sps.norm.sf(abs(estimate) / se))


def phi_ci(table):
    """Phi coefficient with 95% CI (Fisher z) and chi-square p for a 2 x 2 table."""
    table = np.asarray(table, float)
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return np.nan, np.nan, np.nan, np.nan
    chi2, p, _, _ = sps.chi2_contingency(table, correction=False)
    n = table.sum()
    (a, b), (c, d) = table
    phi = float(np.sqrt(chi2 / n) * np.sign(a * d - b * c))
    z, se = np.arctanh(np.clip(phi, -0.999999, 0.999999)), 1.0 / np.sqrt(n - 3)
    return phi, float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)), float(p)


def cohens_d(x, y):
    """Difference of two means over their pooled SD."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    pooled = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    return float((x.mean() - y.mean()) / pooled)


def smd_ci(x, y):
    """Standardized mean difference with 95% CI and the Welch t-test p."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    d = cohens_d(x, y)
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    return d, d - 1.96 * se, d + 1.96 * se, float(sps.ttest_ind(x, y, equal_var=False).pvalue)


def welch_anova_p(groups):
    """Welch's one-way analysis of variance (unequal variances); with two groups it equals Welch's t-test."""
    groups = [np.asarray(g, float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return np.nan
    n = np.array([len(g) for g in groups], float)
    mean = np.array([g.mean() for g in groups])
    var = np.array([g.var(ddof=1) for g in groups])
    k, w = len(groups), n / var
    grand = (w * mean).sum() / w.sum()
    lam = (((1 - w / w.sum()) ** 2) / (n - 1)).sum()
    f_stat = ((w * (mean - grand) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) * lam / (k ** 2 - 1))
    return float(sps.f.sf(f_stat, k - 1, (k ** 2 - 1) / (3 * lam)))


def evalue_from_rr(rr, lo, hi):
    """E-value (VanderWeele and Ding 2017) of a risk ratio and of its CI bound nearest the null.

    The bound E-value is 1 when the interval crosses the null.
    """
    def e(r):
        r = max(r, 1.0 / r)
        return r + np.sqrt(r * (r - 1.0))

    bound = lo if rr >= 1 else hi
    crosses = (rr >= 1 and bound <= 1) or (rr < 1 and bound >= 1)
    return e(rr), 1.0 if crosses else e(bound)


def evalue_binary(odds_ratio, lo, hi, prevalence):
    """Conversion of OR to RR scale."""
    if prevalence >= 0.15:
        odds_ratio, lo, hi = np.sqrt(odds_ratio), np.sqrt(lo), np.sqrt(hi)
    return evalue_from_rr(odds_ratio, lo, hi)


def evalue_continuous(beta, lo, hi, sd_outcome):
    """beta / SD to RR scale."""
    scale = 0.91 / sd_outcome
    return evalue_from_rr(np.exp(beta * scale), np.exp(lo * scale), np.exp(hi * scale))


def fmt_e(pair):
    return f"{pair[0]:.2f} ({pair[1]:.2f})"


def bh_adjust(pvalues):
    """Benjamini-Hochberg correction."""
    return multipletests(np.asarray(pvalues, float), method="fdr_bh")[1]


def write_table(frame, filename, count_col="num_patients"):
    """Write one table to results/. Every row must report how many patients it describes."""
    if count_col not in frame.columns:
        raise KeyError(f"table has no '{count_col}' column; every exported row must report its patient count")
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cfg.RESULTS_DIR / filename, index=False, lineterminator="\n")
    print(f"[write] {filename}  ({len(frame)} rows)")
