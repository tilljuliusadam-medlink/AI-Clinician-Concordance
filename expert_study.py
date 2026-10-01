"""
Expert-rated clinical appropriateness sub-study (manuscript Table 4).
"""

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import fisher_exact, ttest_ind

import config as cfg
import stats as st

NE = "NE"


def load_ratings(path=None):
    """Read the ratings and check the sampling design."""
    df = pd.read_csv(path or cfg.RATINGS_CSV)
    per_case = df.groupby("case_id")["reviewer_id"].nunique()
    cases = df.groupby("case_id")[["concordance_band", "negative_outcome"]].first()
    checks = {
        "cases": (df["case_id"].nunique() == cfg.EXPERT_CASES, df["case_id"].nunique()),
        "reviewers": (df["reviewer_id"].nunique() == cfg.EXPERT_REVIEWERS, df["reviewer_id"].nunique()),
        "two distinct reviewers per case": ((per_case == cfg.RATERS_PER_CASE).all()
                                            and len(df) == cfg.RATERS_PER_CASE * cfg.EXPERT_CASES, len(df)),
        "concordance bands": (cases["concordance_band"].value_counts().to_dict() == cfg.EXPERT_BANDS,
                              cases["concordance_band"].value_counts().to_dict()),
        "negative outcome 1:1 in every band": ((cases.groupby("concordance_band")["negative_outcome"].mean()
                                                == 0.5).all(), cases.groupby("concordance_band")["negative_outcome"]
                                               .sum().to_dict()),
    }
    for col, _ in cfg.EXPERT_DOMAINS:
        checks[f"{col} in 0-3"] = (df[col].dropna().isin([0, 1, 2, 3]).all(), sorted(df[col].dropna().unique()))
    failed = {name: found for name, (ok, found) in checks.items() if not ok}
    if failed:
        raise ValueError(f"the ratings file breaks the sampling design: {failed}")
    df["low"] = df["concordance_band"] == "low"
    df["med_high"] = ~df["low"]
    df["negative"] = df["negative_outcome"] == 1
    df["non_negative"] = ~df["negative"]
    return df


# ---------------------------------------------------------------------------
# Crossed random intercepts for case and reviewer
# ---------------------------------------------------------------------------

def _random_effect_design(frame):
    """One-hot columns per case and per reviewer, and the block sizes."""
    z_case = pd.get_dummies(frame["case_id"]).to_numpy(float)
    z_reviewer = pd.get_dummies(frame["reviewer_id"]).to_numpy(float)
    return np.hstack([z_case, z_reviewer]), [z_case.shape[1], z_reviewer.shape[1]]


def _fixed_design(frame):
    return np.column_stack([np.ones(len(frame)), frame["group"].to_numpy(float)])


def _reml_negll(theta, y, X, z, blocks):
    d = np.concatenate([np.full(nb, theta[k]) for k, nb in enumerate(blocks)])
    v = (z * d) @ z.T + theta[2] * np.eye(len(y))
    try:
        factor = cho_factor(v, lower=True)
    except np.linalg.LinAlgError:
        return 1e12
    xtvix = X.T @ cho_solve(factor, X)
    beta = np.linalg.solve(xtvix, X.T @ cho_solve(factor, y))
    resid = y - X @ beta
    return 0.5 * (2 * np.log(np.diag(factor[0])).sum() + np.linalg.slogdet(xtvix)[1]
                  + resid @ cho_solve(factor, resid))


def fit_crossed_reml(frame, y_col):
    """y ~ 1 + group + (1|case) + (1|reviewer), REML. Returns the group coefficient and its SE."""
    y, X = frame[y_col].to_numpy(float), _fixed_design(frame)
    z, blocks = _random_effect_design(frame)
    var_y = float(np.var(y, ddof=1))
    best = min((minimize(_reml_negll, np.array(start) * var_y, args=(y, X, z, blocks), method="L-BFGS-B",
                         bounds=[(0, None), (0, None), (1e-10, None)])
                for start in [(0.05, 0.02, 0.9), (0.3, 0.15, 0.6), (0.0, 0.0, 1.0)]), key=lambda r: r.fun)
    d = np.concatenate([np.full(nb, best.x[k]) for k, nb in enumerate(blocks)])
    factor = cho_factor((z * d) @ z.T + best.x[2] * np.eye(len(y)), lower=True)
    cov = np.linalg.inv(X.T @ cho_solve(factor, X))
    beta = cov @ X.T @ cho_solve(factor, y)
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def _inner_mode(eta0, dinv, y, z):
    """Mode of the random effects for a fixed linear predictor eta0 and fixed variances (Newton steps)."""
    u = np.zeros(len(dinv))
    for _ in range(60):
        p = expit(eta0 + z @ u)
        hessian = (z.T * (p * (1 - p))) @ z + np.diag(dinv)
        step = cho_solve(cho_factor(hessian, lower=True), z.T @ (y - p) - dinv * u)
        u += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = expit(eta0 + z @ u)
    return u, eta0 + z @ u, (z.T * (p * (1 - p))) @ z + np.diag(dinv)


def _laplace_negll(theta, y, X, z, blocks):
    d = np.concatenate([np.full(nb, np.exp(theta[2 + k])) for k, nb in enumerate(blocks)])
    u, eta, hessian = _inner_mode(X @ theta[:2], 1 / d, y, z)
    loglik = np.sum(y * eta - np.logaddexp(0, eta)) - 0.5 * u @ (u / d) - 0.5 * np.log(d).sum()
    return -(loglik - 0.5 * np.linalg.slogdet(hessian)[1])


def fit_crossed_logit(frame, y_col):
    """logit P(y = 1) ~ 1 + group + (1|case) + (1|reviewer), Laplace maximum likelihood.

    Returns the group log odds ratio and its SE from the joint information matrix of fixed and random effects.
    """
    y, X = frame[y_col].to_numpy(float), _fixed_design(frame)
    z, blocks = _random_effect_design(frame)
    p0 = y.mean()
    best = min((minimize(_laplace_negll, np.array([np.log(p0 / (1 - p0)), 0.0, v, v]), args=(y, X, z, blocks),
                         method="L-BFGS-B", bounds=[(None, None)] * 2 + [(-30, 5)] * 2)
                for v in (np.log(0.25), np.log(0.02))), key=lambda r: r.fun)
    d = np.concatenate([np.full(nb, np.exp(best.x[2 + k])) for k, nb in enumerate(blocks)])
    _, eta, _ = _inner_mode(X @ best.x[:2], 1 / d, y, z)
    w = expit(eta) * (1 - expit(eta))
    a, b = (X.T * w) @ X, (X.T * w) @ z
    c = (z.T * w) @ z + np.diag(1 / d)
    cov = np.linalg.inv(a - b @ cho_solve(cho_factor(c, lower=True), b.T))
    return float(best.x[1]), float(np.sqrt(cov[1, 1]))


# ---------------------------------------------------------------------------
# Table 4
# ---------------------------------------------------------------------------

def describe(frame, col):
    """Cases, mean (SD), Not appropriate ratings overall and by both reviewers of a case."""
    zero = frame[col] == cfg.NOT_APPROPRIATE
    both = int((frame.assign(zero=zero).groupby("case_id")["zero"].sum() == cfg.RATERS_PER_CASE).sum())
    cases = frame["case_id"].nunique()
    return {"Cases": cases, "Appropriateness rating (0-3), mean (SD)": f"{frame[col].mean():.2f} "
                                                                       f"({frame[col].std(ddof=1):.2f})",
            "Not appropriate, n (%)": f"{int(zero.sum())} ({100 * zero.mean():.1f}%)",
            "Not appropriate by both reviewers, n (%)": f"{both} ({100 * both / cases:.1f}%)"}


def compare(frame, col, first, second):
    """First group versus second on one rating domain."""
    sub = frame[frame[first] | frame[second]].reset_index(drop=True)
    sub["group"] = sub[first].astype(float)
    sub["zero"] = (sub[col] == cfg.NOT_APPROPRIATE).astype(float)
    a, b = sub.loc[sub["group"] == 1, col], sub.loc[sub["group"] == 0, col]
    beta, se = fit_crossed_reml(sub, col)
    out = {"beta (95% CI)": st.fmt_ci(beta, beta - 1.96 * se, beta + 1.96 * se),
           "p (linear mixed)": st.fmt_p(st.wald_p(beta, se)),
           "Cohen's d": f"{st.cohens_d(a, b):.2f}",
           "p (Welch)": st.fmt_p(ttest_ind(a, b, equal_var=False).pvalue)}
    zeros = sub.groupby("group")["zero"].agg(["sum", "count"])
    if (zeros["sum"] == 0).any() or (zeros["sum"] == zeros["count"]).any():
        out.update({"OR (95% CI)": NE, "p (logistic mixed)": NE})      # no finite odds ratio exists (too few data points)
    else:
        log_or, se_or = fit_crossed_logit(sub, "zero")
        out.update({"OR (95% CI)": st.fmt_ci(*np.exp([log_or, log_or - 1.96 * se_or, log_or + 1.96 * se_or])),
                    "p (logistic mixed)": st.fmt_p(st.wald_p(log_or, se_or))})
    table = [[zeros.loc[g, "sum"], zeros.loc[g, "count"] - zeros.loc[g, "sum"]] for g in (1.0, 0.0)]
    out["p (Fisher's exact)"] = st.fmt_p(fisher_exact(table)[1])
    return out


def table4(df):
    rows = []
    for col, domain in cfg.EXPERT_DOMAINS:
        rated = df[df[col].notna()]                  # no medication recommended = no sub-class rating
        rows.append({"Rating": domain, "Group": "Overall", **describe(rated, col)})
        for first, second in cfg.EXPERT_COMPARISONS:
            stats_row = compare(rated, col, first, second)
            rows.append({"Rating": domain, "Group": cfg.EXPERT_GROUPS[first],
                         **describe(rated[rated[first]], col), **stats_row})
            rows.append({"Rating": domain, "Group": cfg.EXPERT_GROUPS[second], **describe(rated[rated[second]], col)})
    return pd.DataFrame(rows).fillna("")


def main():
    st.write_table(table4(load_ratings()), "table4_expert_ratings.csv", count_col="Cases")


if __name__ == "__main__":
    main()
