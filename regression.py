"""Tables 3 and 4 and Supplementary Tables 1 to 10.

One routine runs all six analysis blocks. Each block fits the 13 binary
outcomes with logistic regression and the 7 continuous outcomes with linear
regression, adjusting for the six covariates, then applies Benjamini-Hochberg
correction across all 20 tests of the block at once.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as cfg
import derive
import stats as st

FAILED = "NOT ESTIMATED"
EXPOSURE = "concordance"


def _drop_constant(design):
    """Drop columns with no variation. The exposure may never be one of them.

    A column carrying a single value cannot be estimated, but dropping the
    exposure would leave a fitted model whose reported term is a covariate, so
    that case stops the run instead.
    """
    constant = [c for c in design.columns
                if c != "intercept" and design[c].nunique(dropna=False) <= 1]
    if EXPOSURE in constant:
        raise AssertionError(f"'{EXPOSURE}' takes a single value in this "
                             f"sample, so it cannot be the exposure of a "
                             f"regression")
    return design.drop(columns=constant)


def build_design(frame, metric):
    """Intercept, the concordance metric, and the six adjustment covariates."""
    parts = [pd.Series(1.0, index=frame.index, name="intercept"),
             frame[metric].rename(EXPOSURE)]
    for cov in cfg.COVARIATES:
        if cov in cfg.CATEGORICAL_COVARIATES:
            dummies = pd.get_dummies(frame[cov], prefix=cov, dtype=float)
            reference = f"{cov}_{cfg.COVARIATE_REFERENCE[cov]}"
            if reference not in dummies.columns:
                raise AssertionError(f"reference level '{reference}' is absent "
                                     f"from this sample, so '{cov}' would be "
                                     f"coded against a different baseline than "
                                     f"in every other row of the table")
            parts.append(dummies.drop(columns=reference))
        else:
            parts.append(pd.to_numeric(frame[cov], errors="coerce").rename(cov))
    return _drop_constant(pd.concat(parts, axis=1))


def block_sample(df, block):
    """Rows entering one block: subgroup filter, then a computable metric."""
    sample = df if block["subgroup"] is None \
        else df[df["subgroup"] == block["subgroup"]]
    return sample[sample[block["metric"]].notna()].copy()


def _fit_ready(sample, outcome, metric, weights):
    """Listwise deletion for one outcome. Returns (y, X, w) or None."""
    design = build_design(sample, metric)
    y = pd.to_numeric(sample[outcome], errors="coerce")
    keep = y.notna() & design.notna().all(axis=1)
    if keep.sum() < cfg.MIN_FIT_N:
        return None
    y, design = y[keep], design[keep]
    design = _drop_constant(design)
    w = None if weights is None else weights[keep].to_numpy(float)
    return y.to_numpy(float), design, w


def _summary(result):
    """Coefficient, 95% CI and p for the concordance term, selected by name."""
    ci = result.conf_int()
    return float(result.params[EXPOSURE]), float(ci.loc[EXPOSURE, 0]), \
        float(ci.loc[EXPOSURE, 1]), float(result.pvalues[EXPOSURE])


def _cov_type(weights):
    """Covariance estimator for one fit.

    Post-stratification weights are sampling weights, not variance weights, so
    a model-based interval would ignore the uncertainty the weighting design
    itself carries. Weighted blocks therefore use HC0, the linearization
    (survey) sandwich; unweighted blocks keep the model-based estimator.
    """
    return "nonrobust" if weights is None else "HC0"


def _mean_sd(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return "NA"
    return f"{values.mean():.3f} ({values.std(ddof=1):.3f})"


def _binary_evalue(odds_ratio, lo, hi, prevalence):
    """Odds ratio to approximate risk ratio, then E-value.

    For an outcome occurring in at least 15% of the sample the odds ratio
    overstates the risk ratio, so the square-root approximation is used.
    """
    if prevalence >= 0.15:
        return st.evalue_from_rr(np.sqrt(odds_ratio), np.sqrt(lo), np.sqrt(hi))
    return st.evalue_from_rr(odds_ratio, lo, hi)


def fit_binary(prepared):
    """Logistic regression. Weighted and unweighted are the same call."""
    y, design, w = prepared
    weights = np.ones(len(y)) if w is None else w
    result = sm.GLM(y, design, family=sm.families.Binomial(),
                    var_weights=weights).fit(cov_type=_cov_type(w))
    return _summary(result)


def fit_continuous(prepared):
    """Linear regression. Weighted and unweighted are the same call."""
    y, design, w = prepared
    weights = np.ones(len(y)) if w is None else w
    result = sm.WLS(y, design, weights=weights).fit(cov_type=_cov_type(w))
    coef, lo, hi, p = _summary(result)
    return coef, lo, hi, p, float(result.bse[EXPOSURE])


def _binary_row(block, label, sample, prepared, engine, result):
    y, design, _ = prepared
    values = sample.loc[design.index, block["metric"]]
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    row = {"Block": block["name"], "Tables": block["tables"],
           "Predictor": block["metric"], "Sample": block["subgroup"] or "full",
           "Outcome": label, "Model": engine, "num_patients": len(y),
           "num_patients (outcome=0)": n0,
           "% (outcome=0)": f"{100 * n0 / len(y):.1f}",
           "concordance (outcome=0)": _mean_sd(values[y == 0]),
           "num_patients (outcome=1)": n1,
           "% (outcome=1)": f"{100 * n1 / len(y):.1f}",
           "concordance (outcome=1)": _mean_sd(values[y == 1])}
    if result is None:
        row.update({"OR (95% CI)": FAILED, "_p": np.nan,
                    "E (point)": "NA", "E (CI)": "NA"})
        return row
    coef, lo, hi, p = result
    odds, lo_or, hi_or = np.exp(coef), np.exp(lo), np.exp(hi)
    e_point, e_ci = _binary_evalue(odds, lo_or, hi_or, n1 / len(y))
    row.update({"OR (95% CI)": f"{odds:.3f} ({lo_or:.3f} to {hi_or:.3f})",
                "_p": p,
                "E (point)": f"{e_point:.2f}" if np.isfinite(e_point) else "NA",
                "E (CI)": f"{e_ci:.2f}" if np.isfinite(e_ci) else "NA"})
    return row


def _continuous_row(block, label, engine, prepared, result):
    y, _, _ = prepared
    row = {"Block": block["name"], "Tables": block["tables"],
           "Predictor": block["metric"],
           "Sample": block["subgroup"] or "full", "Outcome": label,
           "Model": engine, "num_patients": len(y)}
    if result is None:
        row.update({"Beta (95% CI)": FAILED, "Beta SE": "NA", "_p": np.nan,
                    "E (point)": "NA", "E (CI)": "NA"})
        return row
    coef, lo, hi, p, se = result
    e_point, e_ci = st.evalue_from_beta(coef, lo, hi, float(np.std(y, ddof=1)))
    row.update({"Beta (95% CI)": f"{coef:.2f} ({lo:.2f} to {hi:.2f})",
                "Beta SE": f"{se:.2f}", "_p": p,
                "E (point)": f"{e_point:.2f}" if np.isfinite(e_point) else "NA",
                "E (CI)": f"{e_ci:.2f}" if np.isfinite(e_ci) else "NA"})
    return row


def run_all_blocks(df, weights):
    """Fit every block. Returns the binary and continuous result tables."""
    binary_rows, continuous_rows = [], []

    for block in cfg.BLOCKS:
        sample = block_sample(df, block)
        w = weights.loc[sample.index] if block["weighted"] else None
        print(f"[block] {block['name']:24s} n={len(sample):5d}  "
              f"{block['tables']}")

        for outcome, label, _, _, _ in cfg.BINARY_OUTCOMES:
            prepared = _fit_ready(sample, outcome, block["metric"], w)
            if prepared is None:
                print(f"    [skipped] {label}: fewer than {cfg.MIN_FIT_N} "
                      f"complete rows, so it leaves this block's BH family")
                continue
            engine = "Logistic (weighted, HC0)" if block["weighted"] \
                else "Logistic"
            binary_rows.append(_binary_row(
                block, label, sample, prepared, engine,
                _guard(fit_binary, prepared)))

        for outcome, label in cfg.CONTINUOUS_OUTCOMES:
            prepared = _fit_ready(sample, outcome, block["metric"], w)
            if prepared is None:
                print(f"    [skipped] {label}: fewer than {cfg.MIN_FIT_N} "
                      f"complete rows, so it leaves this block's BH family")
                continue
            engine = "Linear (weighted, HC0)" if block["weighted"] else "Linear"
            continuous_rows.append(_continuous_row(
                block, label, engine, prepared,
                _guard(fit_continuous, prepared)))

    return _finalize(binary_rows, continuous_rows)


def _guard(fn, *args):
    """Run a fit; on a numerical failure record it instead of hiding it."""
    try:
        return fn(*args)
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError,
            RuntimeError) as exc:
        print(f"    [fit failed] {fn.__name__}: {type(exc).__name__}: {exc}")
        return None


def _finalize(binary_rows, continuous_rows):
    """Benjamini-Hochberg within each block, across both outcome families.

    A block tests one predictor in one sample, so its binary and its
    continuous regressions form a single family and are corrected together.
    """
    rows = binary_rows + continuous_rows
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    keys = pd.DataFrame({"Block": [r["Block"] for r in rows],
                         "_p": [r["_p"] for r in rows]})
    adjusted = np.full(len(rows), np.nan)
    for _, index in keys.groupby("Block").groups.items():
        adjusted[index] = st.bh_adjust(keys.loc[index, "_p"])
    split = len(binary_rows)
    return (_format_p(pd.DataFrame(binary_rows), adjusted[:split]),
            _format_p(pd.DataFrame(continuous_rows), adjusted[split:]))


def _format_p(table, adjusted):
    """Swap the raw p column for the formatted corrected and raw columns."""
    if table.empty:
        return table
    raw = table.pop("_p")
    table["p (BH)"] = [st.fmt_p(v) for v in adjusted]
    table["p"] = [st.fmt_p(v) for v in raw]
    return table


def main():
    df = derive.load_analytic_table()
    weights = derive.compute_us_weights(df)
    binary, continuous = run_all_blocks(df, weights)
    st.write_table(binary, "table3_binary_regressions.csv")
    st.write_table(continuous, "table4_continuous_regressions.csv")


if __name__ == "__main__":
    main()
