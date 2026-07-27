"""Post-hoc expert-rated clinical appropriateness sub-study.

Ten psychiatrists rated 300 cases, each case independently by two reviewers.
The unit of analysis is the difference between the AI and clinician
appropriateness rating (range -3 to +3). The fixed-effect intercept of a linear
mixed model with two crossed random intercepts, one for case and one for
reviewer, estimates the mean difference.

Non-inferiority: lower bound of the two-sided 95% CI at or above -0.5.
Superiority:     lower bound of the two-sided 97.5% CI at or above 0.

Why the models are fitted here rather than with statsmodels MixedLM: on a
design of this shape MixedLM does not fit crossed random effects reliably. Its
default optimizer reports convergence failure and the Powell optimizer returns
a different answer with both variance components collapsed to zero and a
materially different standard error. The REML profile likelihood below is
evaluated exactly by Cholesky factorization, and the forced-choice model
integrates the random effects out by Laplace approximation.
"""

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

import agreement as agr
import config as cfg
import stats as st

Z_95 = float(norm.ppf(1 - (1 - cfg.CI_NONINFERIORITY) / 2))
Z_975 = float(norm.ppf(1 - (1 - cfg.CI_SUPERIORITY) / 2))


def load_ratings(path=None):
    df = pd.read_csv(path or cfg.RATINGS_CSV)
    for _, clinician_col, ai_col, _ in cfg.EXPERT_ENDPOINTS:
        for col in (clinician_col, ai_col):
            bad = int((~df[col].isin(cfg.RATING_LEVELS)).sum())
            if bad:
                raise ValueError(f"{col} has {bad} values outside "
                                 f"{cfg.RATING_LEVELS}")
    bad = int((~df["complexity"].isin(cfg.COMPLEXITY_ORDER)).sum())
    if bad:
        raise ValueError(f"complexity has {bad} unrecognised values")
    # Krippendorff's alpha is computed from reviewer PAIRS and silently ignores
    # any case not rated exactly twice, while Gwet's AC2 accepts any case with
    # two or more reviewers. The two coefficients would then be estimated on
    # different cases while reporting the same case count, so an off-design
    # case count stops the run here instead.
    per_case = df.groupby("case_id")["reviewer_id"].nunique()
    off_design = per_case[per_case != cfg.RATERS_PER_CASE]
    if len(off_design):
        raise ValueError(f"{len(off_design)} cases are not rated by exactly "
                         f"{cfg.RATERS_PER_CASE} reviewers; first offenders: "
                         f"{list(off_design.index[:5])}")
    return df


def endpoint_frame(df, endpoint):
    _, clinician_col, ai_col, choice_col = next(
        e for e in cfg.EXPERT_ENDPOINTS if e[0] == endpoint)
    out = pd.DataFrame({
        "case_id": df["case_id"].to_numpy(),
        "reviewer_id": df["reviewer_id"].to_numpy(),
        "arm": df["selection_arm"].to_numpy(),
        "clinician": df[clinician_col].to_numpy(float),
        "ai": df[ai_col].to_numpy(float),
        "choice": df[choice_col].to_numpy(),
        "complexity": df["complexity"].to_numpy(),
        "elapsed_seconds": df["elapsed_seconds"].to_numpy(float),
    })
    out["delta"] = out["ai"] - out["clinician"]
    out["prefers_ai"] = (out["choice"] == "ai").astype(float)
    out["complexity_num"] = out["complexity"].map(cfg.COMPLEXITY_ORDER)
    return out


# ---------------------------------------------------------------------------
# Crossed random intercepts, exact profile REML
# ---------------------------------------------------------------------------

def _reml_negll(theta, y, k_case, k_rater, design, n):
    v = theta[0] * k_case + theta[1] * k_rater + theta[2] * np.eye(n)
    try:
        factor = cho_factor(v, lower=True)
    except np.linalg.LinAlgError:
        return 1e12
    logdet_v = 2.0 * float(np.log(np.diag(factor[0])).sum())
    xtvix = design.T @ cho_solve(factor, design)
    sign, logdet_xtvix = np.linalg.slogdet(xtvix)
    if sign <= 0:
        return 1e12
    beta = np.linalg.solve(xtvix, design.T @ cho_solve(factor, y))
    residual = y - design @ beta
    return 0.5 * (logdet_v + float(logdet_xtvix)
                  + float(residual @ cho_solve(factor, residual)))


def _gls(theta, y, k_case, k_rater, design, n):
    v = theta[0] * k_case + theta[1] * k_rater + theta[2] * np.eye(n)
    factor = cho_factor(v, lower=True)
    xtvix = design.T @ cho_solve(factor, design)
    beta = np.linalg.solve(xtvix, design.T @ cho_solve(factor, y))
    return beta, np.linalg.inv(xtvix)


def fit_crossed_reml(frame, design_cols=None):
    """delta ~ 1 (+ fixed terms) + (1|case) + (1|reviewer), by REML.

    A variance component landing on the zero boundary is singular: it is
    dropped, the model refitted, and the refit asserts the fixed estimate did
    not move.
    """
    y = frame["delta"].to_numpy(float)
    n = len(y)
    if n < 3:
        return {"error": f"only {n} ratings", "n_ratings": n}
    z_case = pd.get_dummies(frame["case_id"]).to_numpy(float)
    z_rater = pd.get_dummies(frame["reviewer_id"]).to_numpy(float)
    k_case, k_rater = z_case @ z_case.T, z_rater @ z_rater.T
    if design_cols is None:
        design, names = np.ones((n, 1)), ["Intercept"]
    else:
        design = np.column_stack(
            [np.ones(n)] + [frame[c].to_numpy(float) for c in design_cols])
        names = ["Intercept"] + list(design_cols)

    var_y = float(np.var(y, ddof=1)) or 1.0
    starts = [(0.05 * var_y, 0.02 * var_y, 0.90 * var_y),
              (0.0, 0.0, var_y),
              (0.30 * var_y, 0.15 * var_y, 0.60 * var_y)]
    best = None
    for start in starts:
        candidate = minimize(_reml_negll, np.array(start, float),
                             method="L-BFGS-B",
                             bounds=[(0, None), (0, None), (1e-10, None)],
                             args=(y, k_case, k_rater, design, n))
        if best is None or candidate.fun < best.fun:
            best = candidate
    theta = np.maximum(best.x, 0.0)

    dropped, tol = [], 1e-8
    if theta[0] <= tol or theta[1] <= tol:
        fixed = [theta[0] <= tol, theta[1] <= tol]
        before, _ = _gls(theta, y, k_case, k_rater, design, n)
        refit = minimize(
            _reml_negll,
            np.array([0.0 if fixed[0] else theta[0],
                      0.0 if fixed[1] else theta[1], theta[2]]),
            method="L-BFGS-B",
            bounds=[(0, 0) if fixed[0] else (0, None),
                    (0, 0) if fixed[1] else (0, None), (1e-10, None)],
            args=(y, k_case, k_rater, design, n))
        theta = np.maximum(refit.x, 0.0)
        after, _ = _gls(theta, y, k_case, k_rater, design, n)
        if abs(float(before[0]) - float(after[0])) >= 1e-6:
            raise AssertionError("dropping a singular variance component "
                                 "changed the fixed estimate")
        best = refit
        dropped = [name for name, flag in zip(("case", "reviewer"), fixed)
                   if flag]

    beta, cov = _gls(theta, y, k_case, k_rater, design, n)
    se = np.sqrt(np.diag(cov))
    out = {"n_ratings": n, "n_cases": int(frame["case_id"].nunique()),
           "n_reviewers": int(frame["reviewer_id"].nunique()),
           "vc_case": float(theta[0]), "vc_reviewer": float(theta[1]),
           "residual_var": float(theta[2]),
           "singular_dropped": ",".join(dropped) if dropped else "none",
           "converged": bool(best.success)}
    for i, name in enumerate(names):
        out[f"est_{name}"] = float(beta[i])
        out[f"se_{name}"] = float(se[i])
    return out


# ---------------------------------------------------------------------------
# Crossed random intercepts, logistic, by Laplace approximation
# ---------------------------------------------------------------------------

def _inner_mode(mu, dinv, y, z, n_u, iterations=60):
    u = np.zeros(n_u)
    for _ in range(iterations):
        eta = mu + z @ u
        p = expit(eta)
        gradient = z.T @ (y - p) - dinv * u
        hessian = (z.T * (p * (1.0 - p))) @ z + np.diag(dinv)
        try:
            step = cho_solve(cho_factor(hessian, lower=True), gradient)
        except np.linalg.LinAlgError:
            return u, None, None
        u = u + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = mu + z @ u
    p = expit(eta)
    hessian = (z.T * (p * (1.0 - p))) @ z + np.diag(dinv)
    return u, eta, hessian


def _laplace_negll(theta, y, z, blocks):
    mu = theta[0]
    var = np.exp(np.clip(theta[1:], -30.0, 10.0))
    d = np.concatenate([np.full(nb, var[k]) for k, nb in enumerate(blocks)])
    u, eta, hessian = _inner_mode(mu, 1.0 / d, y, z, len(d))
    if eta is None:
        return 1e12
    loglik = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    loglik -= 0.5 * float(u @ (u / d))
    loglik -= 0.5 * float(np.log(d).sum())
    sign, logdet = np.linalg.slogdet(hessian)
    if sign <= 0:
        return 1e12
    return -(loglik - 0.5 * float(logdet))


def _laplace_se(mu, var, y, z, blocks):
    d = np.concatenate([np.full(nb, var[k]) for k, nb in enumerate(blocks)])
    _, eta, _ = _inner_mode(mu, 1.0 / d, y, z, len(d))
    weight = expit(eta) * (1.0 - expit(eta))
    a = float(weight.sum())
    b = (z.T * weight) @ np.ones(len(y))
    c = (z.T * weight) @ z + np.diag(1.0 / d)
    schur = a - float(b @ cho_solve(cho_factor(c, lower=True), b))
    return float(np.sqrt(1.0 / schur)) if schur > 0 else float("nan")


def fit_crossed_logit(frame, y_col="prefers_ai"):
    """logit P(y=1) ~ 1 + (1|case) + (1|reviewer), Laplace maximum likelihood."""
    y = frame[y_col].to_numpy(float)
    n = len(y)
    if n < 3 or y.min() == y.max():
        return {"error": f"n={n}, outcome not variable", "n_ratings": n}
    z_case = pd.get_dummies(frame["case_id"]).to_numpy(float)
    z_rater = pd.get_dummies(frame["reviewer_id"]).to_numpy(float)
    p0 = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
    mu0 = float(np.log(p0 / (1 - p0)))

    def run(use_case, use_reviewer):
        names = (["case"] if use_case else []) + (["reviewer"]
                                                  if use_reviewer else [])
        parts = ([z_case] if use_case else []) + ([z_rater] if use_reviewer
                                                  else [])
        if not parts:
            return mu0, {}, float(np.sqrt(1.0 / (n * p0 * (1 - p0)))), True
        z = np.hstack(parts)
        blocks = [part.shape[1] for part in parts]
        bounds = [(None, None)] + [(-30.0, 5.0)] * len(blocks)
        best = None
        for start_var in (np.log(0.25), np.log(0.02)):
            candidate = minimize(
                _laplace_negll,
                np.array([mu0] + [start_var] * len(blocks), float),
                method="L-BFGS-B", bounds=bounds, args=(y, z, blocks))
            if best is None or candidate.fun < best.fun:
                best = candidate
        mu = float(best.x[0])
        var = np.exp(np.clip(best.x[1:], -30.0, 10.0))
        return (mu, dict(zip(names, (float(v) for v in var))),
                _laplace_se(mu, var, y, z, blocks), bool(best.success))

    mu, var, se, converged = run(True, True)
    tol = 1e-6
    drop_case = var.get("case", 1.0) <= tol
    drop_reviewer = var.get("reviewer", 1.0) <= tol
    dropped = []
    if drop_case or drop_reviewer:
        before = mu
        mu, var, se, converged = run(not drop_case, not drop_reviewer)
        if abs(before - mu) >= 1e-3:
            raise AssertionError("dropping a singular variance component "
                                 "changed the fixed estimate")
        dropped = [name for name, flag in zip(("case", "reviewer"),
                                              (drop_case, drop_reviewer))
                   if flag]
    return {"n_ratings": n, "n_cases": int(frame["case_id"].nunique()),
            "n_reviewers": int(frame["reviewer_id"].nunique()),
            "est_logodds": mu, "se_logodds": se,
            "probability": float(expit(mu)),
            "vc_case": var.get("case", 0.0),
            "vc_reviewer": var.get("reviewer", 0.0),
            "singular_dropped": ",".join(dropped) if dropped else "none",
            "converged": converged}


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def _strata(frame):
    for arm in cfg.SELECTION_ARMS:
        yield arm, frame[frame["arm"] == arm].reset_index(drop=True)


def _decision_row(endpoint, stratum, fit, label="crossed REML"):
    estimate, se = fit["est_Intercept"], fit["se_Intercept"]
    lo95, hi95 = estimate - Z_95 * se, estimate + Z_95 * se
    lo975, hi975 = estimate - Z_975 * se, estimate + Z_975 * se
    return {"endpoint": endpoint, "stratum": stratum, "model": label,
            "num_patients": fit["n_cases"], "n_ratings": fit["n_ratings"],
            "n_reviewers": fit["n_reviewers"],
            "mean_difference": round(estimate, 4), "se": round(se, 4),
            "ci95": f"{lo95:.3f} to {hi95:.3f}",
            "ci97.5": f"{lo975:.3f} to {hi975:.3f}",
            "ni_margin": cfg.NI_MARGIN,
            "non_inferiority": bool(lo95 >= cfg.NI_MARGIN),
            "superiority": bool(lo975 >= 0.0),
            "vc_case": round(fit["vc_case"], 4),
            "vc_reviewer": round(fit["vc_reviewer"], 4),
            "residual_var": round(fit["residual_var"], 4),
            "singular_dropped": fit["singular_dropped"],
            "converged": fit["converged"]}


def sample_table(df):
    rows = []
    for arm in cfg.SELECTION_ARMS:
        sub = df[df["selection_arm"] == arm]
        rows.append({"stratum": arm, "num_patients": sub["case_id"].nunique(),
                     "n_ratings": len(sub),
                     "n_reviewers": sub["reviewer_id"].nunique(),
                     "ratings_per_case": round(
                         len(sub) / max(sub["case_id"].nunique(), 1), 2),
                     "rushed_ratings": int(
                         (sub["elapsed_seconds"] < cfg.RUSHED_SECONDS).sum()),
                     "median_seconds": float(sub["elapsed_seconds"].median())})
    return pd.DataFrame(rows)


def primary_table(frames):
    rows = []
    for endpoint, frame in frames.items():
        for stratum, sub in _strata(frame):
            rows.append(_decision_row(endpoint, stratum,
                                      fit_crossed_reml(sub)))
    return pd.DataFrame(rows)


def win_tie_loss_table(frames):
    rows = []
    for endpoint, frame in frames.items():
        for stratum, sub in _strata(frame):
            delta = sub["delta"].to_numpy(float)
            marked = sub.assign(
                _win=(delta > 0).astype(float) + 0.5 * (delta == 0))
            point, lo, hi, method = agr.bootstrap_ci(
                agr.case_contrib(marked, "_win"), agr.ratio)
            rows.append({"endpoint": endpoint, "stratum": stratum,
                         "num_patients": int(sub["case_id"].nunique()),
                         "n_ratings": len(sub),
                         "p_ai_higher": round(float((delta > 0).mean()), 4),
                         "p_tie": round(float((delta == 0).mean()), 4),
                         "p_clinician_higher": round(
                             float((delta < 0).mean()), 4),
                         "win_probability": round(point, 4),
                         "ci95": f"{lo:.3f} to {hi:.3f}",
                         "bootstrap_method": method})
    return pd.DataFrame(rows)


def preference_table(frames):
    rows = []
    for endpoint, frame in frames.items():
        for stratum, sub in _strata(frame):
            fit = fit_crossed_logit(sub)
            if "error" in fit:
                rows.append({"endpoint": endpoint, "stratum": stratum,
                             "num_patients": int(sub["case_id"].nunique()),
                             "model": "crossed logistic",
                             "probability_prefer_ai": "NOT ESTIMATED",
                             "note": fit["error"]})
                continue
            est, se = fit["est_logodds"], fit["se_logodds"]
            rows.append({
                "endpoint": endpoint, "stratum": stratum,
                "num_patients": fit["n_cases"], "model": "crossed logistic",
                "n_ratings": fit["n_ratings"],
                "n_reviewers": fit["n_reviewers"],
                "probability_prefer_ai": round(fit["probability"], 4),
                "log_odds": round(est, 4), "se": round(se, 4),
                "ci95": f"{expit(est - Z_95 * se):.3f} to "
                        f"{expit(est + Z_95 * se):.3f}",
                "vc_case": round(fit["vc_case"], 4),
                "vc_reviewer": round(fit["vc_reviewer"], 4),
                "singular_dropped": fit["singular_dropped"],
                "converged": fit["converged"], "note": ""})
    return pd.DataFrame(rows)


def agreement_table(frames):
    rows = []
    for endpoint, frame in frames.items():
        for who in ("clinician", "ai"):
            column = who
            pairs = agr.pair_matrix(frame, column)
            alpha = agr.bootstrap_ci(agr.alpha_contrib(pairs),
                                     agr.alpha_from_coincidence)
            ac2 = agr.bootstrap_ci(agr.ac2_contrib(frame, column),
                                   agr.ac2_from_contrib)
            for name, (point, lo, hi, method) in (("Krippendorff alpha", alpha),
                                                  ("Gwet AC2", ac2)):
                rows.append({"endpoint": endpoint, "rated_plan": who,
                             "coefficient": name,
                             "num_patients": int(frame["case_id"].nunique()),
                             "estimate": round(point, 4),
                             "ci95": f"{lo:.3f} to {hi:.3f}",
                             "bootstrap_method": method})
    return pd.DataFrame(rows)


def complexity_table(frames):
    """Sensitivity III: expert-rated complexity as a fixed effect."""
    rows = []
    for endpoint, frame in frames.items():
        for stratum, sub in _strata(frame):
            fit = fit_crossed_reml(sub, design_cols=["complexity_num"])
            rows.append({
                "endpoint": endpoint, "stratum": stratum,
                "num_patients": fit["n_cases"], "n_ratings": fit["n_ratings"],
                "mean_difference_at_complexity_0": round(
                    fit["est_Intercept"], 4),
                "se_intercept": round(fit["se_Intercept"], 4),
                "complexity_slope": round(fit["est_complexity_num"], 4),
                "se_slope": round(fit["se_complexity_num"], 4),
                "vc_case": round(fit["vc_case"], 4),
                "vc_reviewer": round(fit["vc_reviewer"], 4),
                "converged": fit["converged"]})
    return pd.DataFrame(rows)


def sensitivity_table(frames):
    """Sensitivity I (leave one reviewer out) and II (exclude rushed)."""
    rows = []
    for endpoint, frame in frames.items():
        for stratum, sub in _strata(frame):
            for reviewer in sorted(sub["reviewer_id"].unique()):
                kept = sub[sub["reviewer_id"] != reviewer]
                fit = fit_crossed_reml(kept)
                row = _decision_row(endpoint, stratum, fit)
                row["analysis"] = f"leave out {reviewer}"
                rows.append(row)
            kept = sub[sub["elapsed_seconds"] >= cfg.RUSHED_SECONDS]
            fit = fit_crossed_reml(kept)
            row = _decision_row(endpoint, stratum, fit)
            row["analysis"] = f"exclude ratings under {cfg.RUSHED_SECONDS}s"
            rows.append(row)
    table = pd.DataFrame(rows)
    return table[["analysis"] + [c for c in table.columns if c != "analysis"]]


def main():
    df = load_ratings()
    frames = {name: endpoint_frame(df, name)
              for name, _, _, _ in cfg.EXPERT_ENDPOINTS}
    st.write_table(sample_table(df), "expert_t1_sample.csv")
    st.write_table(primary_table(frames), "expert_t2_primary.csv")
    st.write_table(win_tie_loss_table(frames), "expert_t3_win_tie_loss.csv")
    st.write_table(preference_table(frames), "expert_t4_preferences.csv")
    st.write_table(agreement_table(frames), "expert_t5_agreement.csv")
    st.write_table(complexity_table(frames), "expert_t6_complexity.csv")
    st.write_table(sensitivity_table(frames), "expert_t7_sensitivity.csv")


if __name__ == "__main__":
    main()
