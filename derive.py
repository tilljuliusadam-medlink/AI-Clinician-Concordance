"""
Turn the three-row panel into the one-row-per-patient analytic table.
Every covariate, concordance metric, outcome and subgroup flag is derived here.
"""

import numpy as np
import pandas as pd

import config as cfg

PATIENT_LEVEL = ["clinician_id", "age_years", "sex", "race", "ethnicity", "education"]
VISIT_LEVEL = (["bmi"] + [c for c, _ in cfg.DIAGNOSES]
               + ["dx_smi", "n_psychiatric_diagnoses", "n_medical_diagnoses", "charlson_index",
                  "primary_dx_psychiatric", "primary_dx_medical", "ai_confidence_level", "ai_input_tokens",
                  "clinician_chose_stimulant", "clinician_chose_nonstimulant_adhd"]
               + [f"ai_recommends_{o}" for o in cfg.TREATMENT_OPTIONS]
               + [f"clinician_chose_{o}" for o in cfg.TREATMENT_OPTIONS])
SUICIDALITY = ["suicidal_thoughts", "suicidal_behavior"]
HISTORY = ["er_psych", "hosp_psych", "er_med", "hosp_med", "nonadherence", "noshow_count", "appts_psych",
           "appts_med", "hosp_days_psych", "hosp_days_med", "cost_psych", "cost_med"]


def load_panel(path=None):
    """Read and validate the simulated cohort panel."""
    panel = pd.read_csv(path or cfg.COHORT_CSV)
    counts = panel.groupby("patient_id")["timepoint"].agg(["count", "nunique"])
    bad = counts[(counts["count"] != len(cfg.TIMEPOINTS)) | (counts["nunique"] != len(cfg.TIMEPOINTS))]
    if len(bad):
        raise ValueError(f"{len(bad)} patients do not have exactly one row per timepoint; "
                         f"first offenders: {list(bad.index[:5])}")
    return panel


def build_patient_table(panel):
    """One row per patient: covariates, concordance metrics, outcomes, subgroup flags."""
    pre, visit, follow = (panel[panel["timepoint"] == t].set_index("patient_id") for t in cfg.TIMEPOINTS)
    df = visit[PATIENT_LEVEL + VISIT_LEVEL].copy()
    df["followup_months"] = follow["months_from_decision"]
    df["fu_death"] = follow["death"]
    for col in SUICIDALITY:
        df[f"pre_{col}"], df[f"dp_{col}"], df[f"fu_{col}"] = pre[col], visit[col], follow[col]
    for col in HISTORY:
        df[f"pre_{col}"], df[f"fu_{col}"] = pre[col], follow[col]
    _add_outcomes(df)
    _add_concordance(df)
    _add_treatment_families(df)
    _add_subgroups(df)
    return df.reset_index()


def _any(*cols):
    return np.maximum.reduce([c.to_numpy() for c in cols]).astype(int)


def _add_outcomes(df):
    """Combined events, totals, remission and the two negative-outcome composites."""
    for w in ("pre", "fu"):
        df[f"{w}_acute_psych"] = _any(df[f"{w}_er_psych"], df[f"{w}_hosp_psych"])
        df[f"{w}_acute_med"] = _any(df[f"{w}_er_med"], df[f"{w}_hosp_med"])
        df[f"{w}_er_any"] = _any(df[f"{w}_er_psych"], df[f"{w}_er_med"])
        df[f"{w}_hosp_any"] = _any(df[f"{w}_hosp_psych"], df[f"{w}_hosp_med"])
        df[f"{w}_acute_any"] = _any(df[f"{w}_er_any"], df[f"{w}_hosp_any"])
        df[f"{w}_noshow_any"] = (df[f"{w}_noshow_count"] > 0).astype(int)
        for kind in ("cost", "hosp_days", "appts"):
            df[f"{w}_{kind}_total"] = df[f"{w}_{kind}_psych"] + df[f"{w}_{kind}_med"]
    for w in ("pre", "dp", "fu"):
        df[f"{w}_suicidal_any"] = _any(df[f"{w}_suicidal_thoughts"], df[f"{w}_suicidal_behavior"])
    # remission 
    for kind in ("suicidal_any", "suicidal_thoughts", "suicidal_behavior"):
        df[f"fu_{kind}_remission"] = 1 - df[f"fu_{kind}"]
    # composites count outcomes
    four = ["suicidal_any", "er_any", "hosp_any"]
    six = four + ["noshow_any", "nonadherence"]
    for name, types in (("composite4", four), ("composite6", six)):
        df[f"pre_{name}_types"] = sum(df[f"pre_{t}"] for t in types)
        df[f"fu_{name}_types"] = sum(df[f"fu_{t}"] for t in types) + df["fu_death"]
        df[f"fu_{name}_any"] = (df[f"fu_{name}_types"] > 0).astype(int)


def _add_concordance(df):
    """One two-by-two table per patient over all 20 options, and every metric derived from it."""
    ai = np.column_stack([df[f"ai_recommends_{o}"] for o in cfg.TREATMENT_OPTIONS]).astype(float)
    clin = np.column_stack([df[f"clinician_chose_{o}"] for o in cfg.TREATMENT_OPTIONS]).astype(float)
    tp = ((ai == 1) & (clin == 1)).sum(axis=1)
    fp = ((ai == 1) & (clin == 0)).sum(axis=1)
    fn = ((ai == 0) & (clin == 1)).sum(axis=1)
    tn = ((ai == 0) & (clin == 0)).sum(axis=1)
    n = tp + fp + fn + tn
    assert (n == len(cfg.TREATMENT_OPTIONS)).all(), "two-by-two counts do not sum to the number of options"

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        specificity = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
        # F1-score definition
        f1 = np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        # MCC definition
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(denom > 0, (tp * tn - fp * fn) / denom, np.nan)
    # balanced accuracy definition
    balanced = np.where(np.isnan(recall), specificity,
                        np.where(np.isnan(specificity), recall, (recall + specificity) / 2))
    df["recall"], df["precision"], df["specificity"] = recall, precision, specificity
    df["f1"], df["balanced_accuracy"], df["pabak"], df["mcc"] = f1, balanced, 2 * (tp + tn) / n - 1, mcc
    df["n_treatments_ai"], df["n_treatments_clinician"] = tp + fp, tp + fn

    assert (np.isnan(f1) == (tp == 0)).all(), "F1 must be missing exactly when TP = 0"
    for col, lo in (("f1", 0), ("balanced_accuracy", 0), ("pabak", -1), ("mcc", -1)):
        values = df[col].dropna()
        assert values.between(lo, 1).all(), f"{col} outside [{lo}, 1]"
    median_f1 = np.nanmedian(f1)
    # ties go to the lower half
    df["f1_half"] = np.where(np.isnan(f1), "not analyzable", np.where(f1 > median_f1, "upper", "lower"))


def _add_treatment_families(df):
    """Table 1 drug families: the union of their options."""
    for family, options in cfg.TREATMENT_FAMILIES.items():
        df[f"clinician_chose_family_{family}"] = _any(*(df[f"clinician_chose_{o}"] for o in options))
    split = _any(df["clinician_chose_stimulant"], df["clinician_chose_nonstimulant_adhd"])
    assert (split == df["clinician_chose_stimulant_adhd"]).all(), \
        "stimulant / non-stimulant flags disagree with the stimulant_adhd option"


def _add_subgroups(df):
    """Severe mental illness and its three narrower subgroups; each narrower one lies inside the first."""
    smi = df["dx_smi"] == 1
    df["smi"] = smi.astype(int)
    df["smi_plus1"] = (smi & (df["n_psychiatric_diagnoses"] >= 2)).astype(int)   # + >=1 psychiatric comorbidity
    df["smi_plus2"] = (smi & (df["n_psychiatric_diagnoses"] >= 3)).astype(int)   # + >=2 psychiatric comorbidities
    df["smi_acute"] = (smi & (df["pre_acute_any"] == 1)).astype(int)             # + ER visit/hospitalization
    assert (df["smi_plus2"] <= df["smi_plus1"]).all() and (df["smi_plus1"] <= df["smi"]).all() \
        and (df["smi_acute"] <= df["smi"]).all(), "severe mental illness subgroups are not nested"


def compute_us_weights(df):
    """
    US-population weighting for age, sex and race. The weights are normalized to sum to the number of patients in the input table.
    """
    targets = cfg.US_POP_TARGETS
    age_group = pd.cut(df["age_years"], [17, 29, 44, 64, 200], labels=list(targets["age_group"]))
    cells = pd.DataFrame({"age_group": age_group, "sex": df["sex"], "race": df["race"]})
    matchable = cells["race"].isin(list(targets["race"])) & cells["age_group"].notna()
    share = cells[matchable].value_counts(normalize=True)
    weights = pd.Series(1.0, index=df.index)
    weights[matchable] = [targets["age_group"][a] * targets["sex"][s] * targets["race"][r] / share[(a, s, r)]
                          for a, s, r in cells[matchable].itertuples(index=False)]
    effective = weights.sum() ** 2 / (weights ** 2).sum()
    print(f"[weights] matched={int(matchable.sum())}/{len(df)} range=[{weights.min():.3f}, "
          f"{weights.max():.3f}] effective n={effective:.0f}")
    return weights


def load_analytic_table(path=None):
    """Convenience entry point used by every table script."""
    return build_patient_table(load_panel(path))
