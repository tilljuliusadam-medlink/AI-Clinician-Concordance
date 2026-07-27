"""Turn the three-row panel into the one-row-per-patient analytic table.

Every covariate, concordance metric and outcome is derived exactly once, here.
No other module recomputes any of them.
"""

import numpy as np
import pandas as pd

import config as cfg

PATIENT_LEVEL = ["age_years", "sex", "race", "ethnicity", "education"]
VISIT_LEVEL = (["bmi"] + cfg.DIAGNOSIS_FLAGS
               + ["n_psychiatric_diagnoses", "medical_comorbidity",
                  "primary_diagnosis_category", "ai_confidence_level"]
               + [f"ai_recommends_{o}" for o in cfg.TREATMENT_OPTIONS]
               + [f"clinician_chose_{o}" for o in cfg.TREATMENT_OPTIONS])
FOLLOWUP_LEVEL = ["cost_psychiatric_usd", "cost_medical_usd",
                  "hospital_days_psychiatric", "hospital_days_medical",
                  "appointments_psychiatric", "appointments_medical"]


def load_panel(path=None):
    """Read and validate the simulated cohort panel."""
    panel = pd.read_csv(path or cfg.COHORT_CSV)
    counts = panel.groupby("patient_id")["timepoint"].agg(["count", "nunique"])
    bad = counts[(counts["count"] != len(cfg.TIMEPOINTS))
                 | (counts["nunique"] != len(cfg.TIMEPOINTS))]
    if len(bad):
        raise ValueError(f"{len(bad)} patients do not have exactly one row per "
                         f"timepoint; first offenders: {list(bad.index[:5])}")
    missing = set(cfg.TIMEPOINTS) - set(panel["timepoint"])
    if missing:
        raise ValueError(f"panel is missing timepoints {sorted(missing)}")
    return panel


def build_patient_table(panel):
    """One row per patient: covariates, concordance metrics, all outcomes."""
    by_time = {t: panel[panel["timepoint"] == t].set_index("patient_id")
               for t in cfg.TIMEPOINTS}
    pre, visit, follow = (by_time["pre_visit"], by_time["decision_point"],
                          by_time["followup"])

    df = visit[PATIENT_LEVEL + VISIT_LEVEL].copy()
    df["months_to_followup"] = follow["months_from_decision"]
    df[cfg.ACUTE_CARE_3MO] = pre[cfg.ACUTE_CARE_3MO]
    for col in FOLLOWUP_LEVEL:
        df[col] = follow[col]
    # Death is the follow-up status, so it is named like every other one. The
    # outcome column plain 'death' is written once, by _add_binary_outcomes.
    df["fu_death"] = follow["death"].astype(int)
    for col in cfg.STATUS_COLUMNS:
        df[f"pre_{col}"] = pre[col]
        df[f"dp_{col}"] = visit[col]
        df[f"fu_{col}"] = follow[col]

    _add_composites(df)
    _add_treatment_families(df)
    _add_concordance(df)
    _add_binary_outcomes(df)

    # Total cost is defined as the sum, so it cannot disagree with its parts.
    # What it can be is missing, which would drop the patient from every cost
    # regression without a word.
    incomplete = df[["cost_psychiatric_usd",
                     "cost_medical_usd"]].isna().any(axis=1)
    if incomplete.any():
        raise AssertionError(f"{int(incomplete.sum())} patients are missing a "
                             f"psychiatric or a medical cost, so no total cost "
                             f"can be formed for them")
    df["cost_total_usd"] = df["cost_psychiatric_usd"] + df["cost_medical_usd"]

    df["subgroup"] = df["primary_diagnosis_category"]
    return df.reset_index()


def _add_composites(df):
    """Definitions over stored columns, so none of them can drift."""
    df["severe_mental_illness"] = (
        (df["dx_depressive"] == 1) | (df["dx_bipolar"] == 1)
        | (df["dx_psychotic"] == 1)).astype(int)
    df["two_or_more_psychiatric_diagnoses"] = (
        df["n_psychiatric_diagnoses"] >= 2).astype(int)

    # Combined acute care over the 12-month prior window and over follow-up.
    for stage in ("pre", "fu"):
        for kind in ("psychiatric", "medical"):
            df[f"{stage}_acute_care_{kind}"] = (
                (df[f"{stage}_er_visit_{kind}"] == 1)
                | (df[f"{stage}_hospitalization_{kind}"] == 1)).astype(int)
    df["fu_composite_adverse"] = (
        (df["fu_death"] == 1)
        | (df["fu_suicidal_thoughts"] == 1)
        | (df["fu_suicidal_behavior"] == 1)
        | (df["fu_acute_care_psychiatric"] == 1)
        | (df["fu_acute_care_medical"] == 1)).astype(int)

    # The 3-month window is nested inside the 12-month one.
    nested = df[cfg.ACUTE_CARE_3MO] <= (
        (df["pre_acute_care_psychiatric"] == 1)
        | (df["pre_acute_care_medical"] == 1)).astype(int)
    if not nested.all():
        raise AssertionError(
            f"{int((~nested).sum())} patients report acute care in the last 3 "
            f"months without any in the last 12 months")


def _add_treatment_families(df):
    """Table 1 also reports drug families; each is the union of its options."""
    for family, options in cfg.TREATMENT_FAMILIES.items():
        selected = np.zeros(len(df), dtype=bool)
        for option in options:
            selected |= df[f"clinician_chose_{option}"] == 1
        df[f"clinician_chose_family_{family}"] = selected.astype(int)


def _add_concordance(df):
    """Concordance metrics, derived from the raw AI and clinician decisions.

    Every treatment option contributes ONE cell to ONE two-by-two table per
    patient. The options are pooled flat: there is no per-family score and no
    averaging across families, which would be a different quantity.
    """
    ai = np.column_stack([df[f"ai_recommends_{o}"].to_numpy(float)
                          for o in cfg.TREATMENT_OPTIONS])
    clin = np.column_stack([df[f"clinician_chose_{o}"].to_numpy(float)
                            for o in cfg.TREATMENT_OPTIONS])
    tp = ((ai == 1) & (clin == 1)).sum(axis=1)
    fp = ((ai == 1) & (clin == 0)).sum(axis=1)
    fn = ((ai == 0) & (clin == 1)).sum(axis=1)
    tn = ((ai == 0) & (clin == 0)).sum(axis=1)
    total = tp + fp + fn + tn
    if not (total == len(cfg.TREATMENT_OPTIONS)).all():
        raise AssertionError("two-by-two counts do not sum to the number of "
                             "treatment options")

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        specificity = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
        # F1 is undefined without a true positive; those patients are excluded
        # from the F1 analyses and retained for balanced accuracy and PABAK.
        f1 = np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    accuracy = (tp + tn) / total
    df["concordance_recall"] = recall
    df["concordance_precision"] = precision
    df["concordance_specificity"] = specificity
    df["f1"] = f1
    df["balanced_accuracy"] = (recall + specificity) / 2.0
    df["pabak"] = 2.0 * accuracy - 1.0
    df["f1_analyzable"] = (tp > 0).astype(int)

    median_f1 = np.nanmedian(f1)
    df["f1_half"] = np.where(np.isnan(f1), "not analyzable",
                             np.where(f1 > median_f1, "upper", "lower"))


def _add_binary_outcomes(df):
    """Apply each outcome rule once. Ineligible patients get NaN, not zero."""
    for name, _, rule, fu_col, base_col in cfg.BINARY_OUTCOMES:
        fu = df[fu_col].to_numpy(float)
        if rule == "event":
            value, eligible = fu, np.ones(len(df), bool)
        else:
            base = df[base_col].to_numpy(float)
            if rule == "onset":
                value, eligible = fu, base == 0
            elif rule == "remission":
                value, eligible = 1.0 - fu, base == 1
            elif rule == "relapse":
                value, eligible = fu, base == 1
            else:
                raise ValueError(f"unknown outcome rule '{rule}'")
        df[name] = np.where(eligible, value, np.nan)


def compute_us_weights(df):
    """Post-stratification weights to the US adult population.

    Cells are age group by sex by race. Patients whose race is outside the four
    weightable categories pass through at weight 1.
    """
    age_group = pd.cut(df["age_years"], [17, 29, 44, 64, 200],
                       labels=["18-29", "30-44", "45-64", "65+"])
    matchable = df["race"].isin(cfg.WEIGHTABLE_RACES) & age_group.notna()
    weights = pd.Series(1.0, index=df.index)
    if not matchable.any():
        raise AssertionError("no patient matches a post-stratification cell")

    cells = pd.DataFrame({"age_group": age_group, "sex": df["sex"],
                          "race": df["race"]})[matchable]
    proportions = cells.value_counts(normalize=True)
    targets = {}
    for cell, share in proportions.items():
        age, sex, race = cell
        target = (cfg.US_POP_TARGETS["age_group"][age]
                  * cfg.US_POP_TARGETS["sex"][sex]
                  * cfg.US_POP_TARGETS["race"][race])
        targets[cell] = target / share
    keys = list(zip(cells["age_group"], cells["sex"], cells["race"]))
    weights.loc[matchable] = [targets[k] for k in keys]

    if weights.nunique() <= 1:
        raise AssertionError("post-stratification produced uniform weights, so "
                             "the weighted analysis would equal the unweighted "
                             "one; check the race, sex and age categories")
    effective = weights.sum() ** 2 / (weights ** 2).sum()
    print(f"[weights] matched={int(matchable.sum())}/{len(df)} "
          f"range=[{weights.min():.3f}, {weights.max():.3f}] "
          f"effective n={effective:.0f}")
    return weights


def load_analytic_table(path=None):
    """Convenience entry point used by every table script."""
    return build_patient_table(load_panel(path))
