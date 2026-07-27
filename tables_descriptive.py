"""Table 1 (baseline characteristics) and Table 2 (concordance metrics).

Row labels and subgroup definitions follow the manuscript exactly, including
its two different acute-care windows: Table 2 uses "Hospitalization/ER visit in
the last 3 months", while every Table 3 denominator uses 12 months.

Table 1 compares the upper and lower halves of the F1 median split: chi-square
with a phi effect size for categorical rows, Welch's t-test with a standardized
mean difference for continuous rows.
"""

import numpy as np
import pandas as pd

import config as cfg
import derive
import stats as st

RACE_LEVELS = ["African American", "Asian",
               "Native American or Pacific Islander", "White", "Other"]
EDUCATION_LEVELS = ["College/associate degree", "High school graduate or GED",
                    "Bachelor's degree", "Graduate/professional degree",
                    "Less than high school", "Other"]
DIAGNOSIS_LABELS = [("Anxiety disorder", "dx_anxiety"),
                    ("Depressive disorder", "dx_depressive"),
                    ("ADHD", "dx_adhd"),
                    ("Bipolar-spectrum disorder", "dx_bipolar"),
                    ("Psychotic disorder", "dx_psychotic"),
                    ("OCD", "dx_ocd")]

# (label, column, value). Families come from derive._add_treatment_families and
# are the union of their options, so a family can never contradict its members.
TREATMENT_ROWS = [
    ("Antidepressant", "clinician_chose_family_antidepressant"),
    ("SSRI", "clinician_chose_ssri"),
    ("Atypical Antidepressant", "clinician_chose_atypical_antidepressant"),
    ("SNRI", "clinician_chose_snri"),
    ("TCA", "clinician_chose_tca"),
    ("MAOI", "clinician_chose_maoi"),
    ("Antipsychotic", "clinician_chose_family_antipsychotic"),
    ("SGA", "clinician_chose_family_sga"),
    ("SGA excl. Clozapine", "clinician_chose_sga_excl_clozapine"),
    ("Clozapine", "clinician_chose_clozapine"),
    ("FGA", "clinician_chose_fga"),
    ("Mood Stabilizer", "clinician_chose_mood_stabilizer"),
    ("Anxiolytic/Hypnotic", "clinician_chose_family_anxiolytic_hypnotic"),
    ("Benzodiazepine", "clinician_chose_benzodiazepine"),
    ("Z Drug", "clinician_chose_z_drug"),
    ("Stimulant", "clinician_chose_stimulant"),
    ("Non-stimulant ADHD medication", "clinician_chose_nonstimulant_adhd"),
    ("Psychotherapy", "clinician_chose_family_psychotherapy"),
    ("CBT", "clinician_chose_cbt"),
    ("DBT", "clinician_chose_dbt"),
    ("ACT", "clinician_chose_act"),
    ("EMDR", "clinician_chose_emdr"),
    ("Other", "clinician_chose_other_psychotherapy"),
]

CATEGORICAL_ROWS = (
    [("Female sex", "sex", "Female")]
    + [(f"Race: {level}", "race", level) for level in RACE_LEVELS]
    + [("Ethnicity: Hispanic/Latino", "ethnicity", "Hispanic/Latino"),
       ("Ethnicity: Not Hispanic/Latino", "ethnicity", "Not Hispanic/Latino")]
    + [(f"Education: {level}", "education", level)
       for level in EDUCATION_LEVELS]
    + [(f"Psychiatric diagnosis: {label}", column, 1)
       for label, column in DIAGNOSIS_LABELS]
    + [("Severe mental illness", "severe_mental_illness", 1),
       ("At least 2 psychiatric diagnoses",
        "two_or_more_psychiatric_diagnoses", 1),
       ("Medical comorbidity", "medical_comorbidity", 1)]
    + [(f"Treatment: {label}", column, 1) for label, column in TREATMENT_ROWS]
)

CONTINUOUS_ROWS = [("Age", "age_years", "mean"),
                   ("Body Mass Index", "bmi", "mean"),
                   ("Number of psychiatric diagnoses",
                    "n_psychiatric_diagnoses", "median")]


def build_table1(df):
    analyzable = df[df["f1_analyzable"] == 1]
    upper = analyzable[analyzable["f1_half"] == "upper"]
    lower = analyzable[analyzable["f1_half"] == "lower"]
    groups = [("Overall", df), ("F1-analyzable", analyzable),
              ("F1 upper half", upper), ("F1 lower half", lower)]

    rows = []
    for label, column, value in CATEGORICAL_ROWS:
        row = {"Characteristic": label}
        cells = []
        for name, frame in groups:
            hits = int((frame[column] == value).sum())
            cells.append(hits)
            row[f"{name} (n={len(frame)})"] = (
                "NA" if len(frame) == 0
                else f"{hits} ({100 * hits / len(frame):.1f})")
        hit_u = int((upper[column] == value).sum())
        hit_l = int((lower[column] == value).sum())
        phi, lo, hi, p = st.phi_ci([[hit_u, len(upper) - hit_u],
                                    [hit_l, len(lower) - hit_l]])
        row["Effect size (95% CI)"] = (
            "NA" if not np.isfinite(phi)
            else f"phi {phi:.3f} ({lo:.3f} to {hi:.3f})")
        row["p"] = st.fmt_p(p)
        # The disclosure rule applies cell by cell, so the count that governs
        # the row is the smallest cell the row actually prints, not the total
        # of the analyzable column.
        row["num_patients"] = min(cells)
        rows.append(row)

    for label, column, kind in CONTINUOUS_ROWS:
        row = {"Characteristic": label
               + (", mean (SD)" if kind == "mean" else ", median (IQR)")}
        for name, frame in groups:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if kind == "mean":
                text = f"{values.mean():.2f} ({values.std(ddof=1):.2f})"
            else:
                q1, q3 = values.quantile([0.25, 0.75])
                text = f"{values.median():.0f} ({q1:.0f}-{q3:.0f})"
            row[f"{name} (n={len(frame)})"] = text
        d, lo, hi, p = st.smd_ci(
            pd.to_numeric(upper[column], errors="coerce"),
            pd.to_numeric(lower[column], errors="coerce"))
        row["Effect size (95% CI)"] = (
            "NA" if not np.isfinite(d)
            else f"SMD {d:.3f} ({lo:.3f} to {hi:.3f})")
        row["p"] = st.fmt_p(p)
        row["num_patients"] = len(analyzable)
        rows.append(row)

    return pd.DataFrame(rows)


def _subgroups(df):
    """Yield (subgroup name, [(level label, mask), ...]) exactly as Table 2."""
    yield "Overall", [("Overall", pd.Series(True, index=df.index))]
    yield "Sex", [(level, df["sex"] == level) for level in ["Male", "Female"]]
    yield "Race", [(level, df["race"] == level) for level in
                   ["White", "African American",
                    "Native American or Pacific Islander", "Asian", "Other"]]
    yield "Psychiatric diagnosis", [(label, df[column] == 1)
                                    for label, column in DIAGNOSIS_LABELS]
    median_dx = df["n_psychiatric_diagnoses"].median()
    yield "Number of psychiatric diagnoses", [
        (f"at or above median ({median_dx:.0f})",
         df["n_psychiatric_diagnoses"] >= median_dx),
        (f"below median ({median_dx:.0f})",
         df["n_psychiatric_diagnoses"] < median_dx)]
    yield "Severe mental illness", [
        ("Yes", df["severe_mental_illness"] == 1),
        ("No", df["severe_mental_illness"] == 0)]
    # The manuscript's Table 2 window is 3 months, not the 12 months used for
    # the Table 3 denominators.
    yield "Hospitalization/ER visit in the last 3 months", [
        ("Yes", df[cfg.ACUTE_CARE_3MO] == 1),
        ("No", df[cfg.ACUTE_CARE_3MO] == 0)]
    yield "Suicidal thoughts/behaviors", [
        ("Yes", (df["dp_suicidal_thoughts"] == 1)
         | (df["dp_suicidal_behavior"] == 1)),
        ("No", (df["dp_suicidal_thoughts"] == 0)
         & (df["dp_suicidal_behavior"] == 0))]


# Table 2 reports a p-value for F1, balanced accuracy and PABAK only.
METRIC_COLUMNS = [("Recall/Sensitivity", "concordance_recall", False),
                  ("Precision", "concordance_precision", False),
                  ("Specificity", "concordance_specificity", False),
                  ("F1", "f1", True),
                  ("Balanced Accuracy", "balanced_accuracy", True),
                  ("PABAK", "pabak", True)]


def build_table2(df):
    rows = []
    for name, levels in _subgroups(df):
        tests = {}
        if len(levels) > 1:
            for label, column, tested in METRIC_COLUMNS:
                if tested:
                    tests[label] = st.welch_or_anova(
                        [df.loc[mask, column].dropna().to_numpy()
                         for _, mask in levels])
        for level, mask in levels:
            frame = df[mask]
            row = {"Subgroup": name, "Level": level,
                   "num_patients": int(len(frame)),
                   "num_patients (F1 computable)":
                       int(frame["f1_analyzable"].sum())}
            for label, column, _ in METRIC_COLUMNS:
                values = frame[column].dropna()
                row[label] = ("NA" if values.empty
                              else f"{values.mean():.2f} "
                                   f"({values.std(ddof=1):.2f})")
            for label, _, tested in METRIC_COLUMNS:
                if tested:
                    row[f"p ({label})"] = (st.fmt_p(tests[label])
                                           if label in tests else "")
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = derive.load_analytic_table()
    st.write_table(build_table1(df), "table1_baseline_characteristics.csv")
    st.write_table(build_table2(df), "table2_concordance_metrics.csv")


if __name__ == "__main__":
    main()
