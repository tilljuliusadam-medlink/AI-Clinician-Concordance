"""
Manuscript Tables 2-3 and Supplementary Tables 1-18 and 20-21.
"""

import gpboost as gpb
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

import config as cfg
import derive
import stats as st


def block_sample(df, block):
    """Patients of one block: its subset, with a computable metric copied into the exposure column."""
    sample = df if block.subset is None else df[df[block.subset] == 1]
    sample = sample[sample[block.metric].notna()].copy()
    sample[cfg.EXPOSURE] = sample[block.metric]
    return sample


def roster(block):
    return [m for m in cfg.BINARY_MODELS + cfg.CONTINUOUS_MODELS
            if block.roster == "all" or m.name in cfg.COLLAPSED]


def design(sample, model):
    """Outcome y and fixed-effect design X of one model; rows with a missing value drop (listwise deletion).

    A column constant in the sample is dropped (e.g. prior acute care inside the subgroup defined by it); the
    exposure may never be one of them.
    """
    data = sample.query(model.gate) if model.gate else sample
    y, X = patsy.dmatrices(cfg.expand(model.formula), data, return_type="dataframe", NA_action="drop")
    X = X.loc[:, (X.nunique() > 1) | (X.columns == "Intercept")]
    assert cfg.EXPOSURE in X, f"{model.name}: the exposure is constant in this sample"
    return y.iloc[:, 0], X


def fit_gpboost(y, X, clinicians, likelihood, weights=None):
    """Random intercept per clinician, GPBoost. Returns (coefficient, SE, intercept variance, residual SD)."""
    model = gpb.GPModel(group_data=clinicians, likelihood=likelihood, weights=weights, num_parallel_threads=1)
    model.fit(y=y.to_numpy(float), X=X.to_numpy(float), params={"maxit": cfg.MAX_ITERATIONS})
    if model._get_num_optim_iter() >= cfg.MAX_ITERATIONS:
        raise RuntimeError(f"GPBoost did not converge within {cfg.MAX_ITERATIONS} iterations")
    coef, variances = model.get_coef(std_err=True), model.get_cov_pars()
    j = X.columns.get_loc(cfg.EXPOSURE)
    resid_sd = float(np.sqrt(variances["Error_term"].iloc[0])) if "Error_term" in variances else np.nan
    return _checked(float(coef.iloc[0, j]), float(coef.iloc[1, j]), float(variances["Group_1"].iloc[0]), resid_sd,
                    max_abs=20 if likelihood == "bernoulli_logit" else np.inf)


def _checked(coef, se, var_u, resid_sd, max_abs=np.inf):
    """A degenerate fit (separation, singular Hessian) stops the run instead of printing a meaningless estimate.
    max_abs bounds a log-odds coefficient: |log OR| >= 20 is separation, never an estimate."""
    if not (np.isfinite(coef) and np.isfinite(se) and se > 0 and abs(coef) < max_abs and np.isfinite(var_u)):
        raise RuntimeError(f"degenerate fit: coefficient {coef}, SE {se}, clinician variance {var_u}")
    return coef, se, var_u, resid_sd


def fit_mixedlm(y, X, clinicians):
    """Random intercept per clinician, statsmodels MixedLM by REML. Same return as fit_gpboost."""
    model = sm.MixedLM(y, X, groups=clinicians)
    result = model.fit(reml=True)
    if not result.converged:
        result = model.fit(reml=True, method="powell", maxiter=500)
    if not result.converged:
        raise RuntimeError("MixedLM did not converge (lbfgs, powell)")
    return _checked(float(result.params[cfg.EXPOSURE]), float(result.bse[cfg.EXPOSURE]),
                    float(np.asarray(result.cov_re)[0, 0]), float(np.sqrt(result.scale)))


def fit_row(block, model, sample, weights):
    """One table row: counts, the concordance estimate with 95% CI, raw p, E-value, clinician variance."""
    y, X = design(sample, model)
    clinicians = sample.loc[X.index, "clinician_id"].to_numpy()
    w = None if weights is None else weights.loc[X.index].to_numpy(float)
    if w is not None:
        w = w * len(w) / w.sum()          # normalized: the weights of a fit sum to its number of patients
    n_clin = len(set(clinicians))
    assert n_clin >= 2, f"{block.name}/{model.name}: {n_clin} clinician(s), a random intercept needs at least 2"
    row = {"Block": block.name, "Tables": block.tables, "Metric": block.metric, "Domain": model.domain,
           "Outcome": model.label, "num_patients": len(y), "num_clinicians": n_clin}
    if model.kind == "binary":
        n1 = int(y.sum())
        n0 = len(y) - n1
        assert min(n0, n1) >= cfg.MIN_EVENTS, f"{block.name}/{model.name}: {n1} events, {n0} non-events"
        coef, se, var_u, _ = fit_gpboost(y, X, clinicians, "bernoulli_logit", w)
        lo, hi = coef - 1.96 * se, coef + 1.96 * se
        odds = np.exp([coef, lo, hi])
        row.update({"Without outcome, n (%)": f"{n0} ({100 * n0 / len(y):.1f}%)",
                    "With outcome, n (%)": f"{n1} ({100 * n1 / len(y):.1f}%)",
                    "OR (95% CI)": st.fmt_ci(*odds),
                    "E (CI)": st.fmt_e(st.evalue_binary(*odds, n1 / len(y)))})
    else:
        if block.weighted:
            coef, se, var_u, _ = fit_gpboost(y, X, clinicians, "gaussian", w)
            mean = np.average(y, weights=w)
            sd = float(np.sqrt(np.average((y - mean) ** 2, weights=w)))
        else:
            coef, se, var_u, _ = fit_mixedlm(y, X, clinicians)
            mean, sd = float(y.mean()), float(y.std())
        lo, hi = coef - 1.96 * se, coef + 1.96 * se
        row.update({"Beta (95% CI)": st.fmt_ci(coef, lo, hi),
                    "Standardized beta (95% CI)": st.fmt_ci(coef / sd, lo / sd, hi / sd),
                    "Mean (SD)": f"{mean:.2f} ({sd:.2f})", "SE": f"{se:.2f}",
                    "E (CI)": st.fmt_e(st.evalue_continuous(coef, lo, hi, sd))})
    row["Clinician intercept SD"] = f"{np.sqrt(var_u):.3f}"
    row["_p"] = st.wald_p(coef, se)
    return row


def run_all_blocks(df, weights):
    """Benjamini-Hochberg correction within each block."""
    tables = {"binary": [], "continuous": []}
    for block in cfg.BLOCKS:
        sample = block_sample(df, block)
        print(f"[block] {block.name:20s} n={len(sample):5d}  {block.tables}")
        rows = {"binary": [], "continuous": []}
        for model in roster(block):
            try:
                rows[model.kind].append(fit_row(block, model, sample, weights if block.weighted else None))
            except Exception as exc:
                raise RuntimeError(f"fit failed: block {block.name}, outcome {model.name}") from exc
        block_rows = rows["binary"] + rows["continuous"]
        for row, q in zip(block_rows, st.bh_adjust([r["_p"] for r in block_rows])):
            row["p"], row["p (BH)"] = st.fmt_p(row.pop("_p")), st.fmt_p(q)
        for kind in tables:
            tables[kind] += rows[kind]
    return pd.DataFrame(tables["binary"]), pd.DataFrame(tables["continuous"])


def main():
    df = derive.load_analytic_table()
    binary, continuous = run_all_blocks(df, derive.compute_us_weights(df))
    st.write_table(binary, "table2_binary_regressions.csv")
    st.write_table(continuous, "table3_continuous_regressions.csv")


if __name__ == "__main__":
    main()
