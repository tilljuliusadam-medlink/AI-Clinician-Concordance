"""
Table 1 (baseline characteristics) and Supplementary Table 23 (concordance metrics by subgroup).
"""

import numpy as np
import pandas as pd

import config as cfg
import derive
import stats as st

RACE_LEVELS = ["African American", "Asian", "Native American or Pacific Islander", "White", "Other"]
ETHNICITY_LEVELS = ["Hispanic or Latino", "Not Hispanic or Latino"]
EDUCATION_LEVELS = ["Less than high school", "High school graduate or GED", "College/associate degree",
                    "Bachelor's degree", "Graduate/professional degree"]
TREATMENT_ROWS = [
    ("Antidepressant", "family_antidepressant"), ("Selective serotonin reuptake inhibitor", "ssri"),
    ("Atypical antidepressant", "atypical_antidepressant"),
    ("Serotonin-norepinephrine reuptake inhibitor", "snri"), ("Tricyclic antidepressant", "tca"),
    ("Monoamine oxidase inhibitor", "maoi"), ("Antipsychotic", "family_antipsychotic"),
    ("Second-generation antipsychotic (with or without clozapine)", "family_sga"),
    ("Second-generation antipsychotic excl. clozapine", "sga_excl_clozapine"), ("Clozapine", "clozapine"),
    ("First-generation antipsychotic", "fga"), ("Mood stabilizer", "mood_stabilizer"),
    ("Anxiolytic/Hypnotic", "anxiolytic_sedative"),
    ("Stimulant/Attention-deficit/hyperactivity disorder medication", "stimulant_adhd"),
    ("Stimulant", "stimulant"), ("Non-stimulant attention-deficit/hyperactivity disorder medication",
                                 "nonstimulant_adhd"),
    ("Glutamatergic agent", "glutamatergic"), ("Cognitive enhancer", "cognitive_enhancer"),
    ("Addiction medication", "addiction_medication"), ("Psychotherapy", "family_psychotherapy"),
    ("Cognitive-behavioral therapy", "cbt"), ("Dialectical behavior therapy", "dbt"),
    ("Acceptance and commitment therapy", "act"), ("Eye movement desensitization and reprocessing", "emdr"),
    ("Other psychotherapy", "other_psychotherapy"), ("Neurostimulation (ECT / TMS / tDCS)", "neurostimulation"),
]

# (label, column, value): one categorical Table 1 row
CATEGORICAL_ROWS = (
    [("Female sex", "sex", "Female")]
    + [(f"Race: {v}", "race", v) for v in RACE_LEVELS]
    + [(f"Ethnicity: {v}", "ethnicity", v) for v in ETHNICITY_LEVELS]
    + [(f"Education: {v}", "education", v) for v in EDUCATION_LEVELS]
    + [(f"Psychiatric diagnosis: {label}", col, 1) for col, label in cfg.DIAGNOSES]
    + [("Severe mental illness", "smi", 1), (">=2 psychiatric diagnoses", "psy_dx_2plus", 1),
       ("Medical diagnosis", "medical_dx_any", 1), ("Charlson comorbidity index >=1", "charlson_1plus", 1)]
    + [(f"Treatment at the index visit: {label}", f"clinician_chose_{col}", 1) for label, col in TREATMENT_ROWS]
)
# (label, column, summaries): one continuous Table 1 row
CONTINUOUS_ROWS = [
    ("Age, mean (SD)", "age_years", ["mean"]),
    ("Number of psychiatric diagnoses, median (IQR), mean (SD)", "n_psychiatric_diagnoses", ["median", "mean"]),
    ("Number of medical diagnoses, median (IQR), mean (SD)", "n_medical_diagnoses", ["median", "mean"]),
    ("Charlson comorbidity index, median (IQR)", "charlson_index", ["median"]),
    ("BMI, mean (SD)", "bmi", ["mean"]),
    ("Number of treatments at the index visit, clinician, median (IQR)", "n_treatments_clinician", ["median"]),
    ("Number of treatments at the index visit, Comentra™, median (IQR)", "n_treatments_ai", ["median"]),
    ("Comentra™ input message length, tokens, mean (SD)", "ai_input_tokens", ["mean"]),
]


def _summary(values, kinds):
    parts = {"mean": lambda v: f"{v.mean():.1f} ({v.std(ddof=1):.1f})",
             "median": lambda v: f"{v.median():.0f} ({v.quantile(0.25):.0f}-{v.quantile(0.75):.0f})"}
    return ", ".join(parts[k](values) for k in kinds)


def build_table1(df):
    df = df.assign(psy_dx_2plus=(df["n_psychiatric_diagnoses"] >= 2).astype(int),
                   medical_dx_any=(df["n_medical_diagnoses"] >= 1).astype(int),
                   charlson_1plus=(df["charlson_index"] >= 1).astype(int))
    analyzable = df[df["f1"].notna()]
    upper, lower = analyzable[analyzable["f1_half"] == "upper"], analyzable[analyzable["f1_half"] == "lower"]
    groups = [("Overall", df), ("F1-analyzable", analyzable), ("F1 upper half", upper), ("F1 lower half", lower)]
    rows = []
    for label, col, value in CATEGORICAL_ROWS:
        row = {"Characteristic": label}
        for name, frame in groups:
            recorded = frame[col].notna()
            hits = int((frame[col] == value).sum())
            row[f"{name} (n={len(frame)})"] = f"{hits} ({100 * hits / recorded.sum():.1f}%)"
        u, lo = upper[upper[col].notna()], lower[lower[col].notna()]
        hu, hl = int((u[col] == value).sum()), int((lo[col] == value).sum())
        phi, ci_lo, ci_hi, p = st.phi_ci([[hu, len(u) - hu], [hl, len(lo) - hl]])
        row["Effect (95% CI)"] = st.fmt_ci(phi, ci_lo, ci_hi) if np.isfinite(phi) else "NA"
        row["p"], row["num_patients"] = st.fmt_p(p), int(analyzable[col].notna().sum())
        rows.append(row)
    for label, col, kinds in CONTINUOUS_ROWS:
        row = {"Characteristic": label}
        for name, frame in groups:
            row[f"{name} (n={len(frame)})"] = _summary(frame[col].dropna(), kinds)
        d, ci_lo, ci_hi, p = st.smd_ci(upper[col], lower[col])
        row["Effect (95% CI)"], row["p"] = st.fmt_ci(d, ci_lo, ci_hi), st.fmt_p(p)
        row["num_patients"] = int(analyzable[col].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _yes_no(heading, flag):
    return heading, [("Yes", flag == 1), ("No", flag == 0)]


def _median_split(heading, values):
    m = values.median()
    return heading, [(f">=median ({m:.0f})", values >= m), (f"<median ({m:.0f})", values < m)]


def subgroups(df):
    """(heading, [(level, mask)]) in the order of Supplementary Table 23."""
    out = [("Overall", [("Overall", pd.Series(True, index=df.index))]),
           _median_split("Age", df["age_years"]),
           ("Sex", [(v, df["sex"] == v) for v in ("Male", "Female")]),
           ("Race", [(v, df["race"] == v) for v in RACE_LEVELS]),
           ("Ethnicity", [(v, df["ethnicity"] == v) for v in ETHNICITY_LEVELS])]
    out += [_yes_no(f"Psychiatric diagnosis: {label}", df[col]) for col, label in cfg.DIAGNOSES]
    out += [_yes_no("Medical comorbidity", (df["n_medical_diagnoses"] >= 1).astype(int)),
            _median_split("Number of psychiatric diagnoses", df["n_psychiatric_diagnoses"]),
            _median_split("Number of medical diagnoses", df["n_medical_diagnoses"])]
    out += [_yes_no(f"Pre-index visit {label}", df[col]) for label, col in (
        ("psychiatric ER visit/hospitalization", "pre_acute_psych"),
        ("medical ER visit/hospitalization", "pre_acute_med"),
        ("suicidal thoughts/behaviors", "pre_suicidal_any"),
        ("medication nonadherence", "pre_nonadherence"),
        ("psychiatric appointment no-show", "pre_noshow_any"))]
    for model in cfg.BINARY_MODELS:          # every binary outcome, among the patients its model is fitted on
        eligible = df.eval(model.gate) if model.gate else pd.Series(True, index=df.index)
        outcome = df[model.formula.split("~")[0].strip()].where(eligible)
        out.append(_yes_no(f"Follow-up: {model.label}", outcome))
    out += [_yes_no(label, df[col]) for label, col in (
        ("Severe mental illness", "smi"), ("Severe mental illness + >=1 psychiatric comorbidity", "smi_plus1"),
        ("Severe mental illness + >=2 psychiatric comorbidities", "smi_plus2"),
        ("Severe mental illness with pre-index ER visit/hospitalization", "smi_acute"),
        ("Primary psychiatric diagnosis", "primary_dx_psychiatric"),
        ("Primary medical diagnosis", "primary_dx_medical"))]
    return out


METRIC_COLUMNS = [("Recall/Sensitivity", "recall", False), ("Precision", "precision", False),
                  ("Specificity", "specificity", False), ("F1", "f1", True),
                  ("Balanced Accuracy", "balanced_accuracy", True), ("PABAK", "pabak", True), ("MCC", "mcc", True)]


def build_concordance_table(df):
    rows = []
    for heading, levels in subgroups(df):
        tests = {col: st.welch_anova_p([df.loc[mask, col] for _, mask in levels])
                 for _, col, tested in METRIC_COLUMNS if tested and len(levels) > 1}
        for level, mask in levels:
            row = {"Subgroup": heading, "Level": level, "num_patients": int(mask.sum())}
            for label, col, tested in METRIC_COLUMNS:
                values = df.loc[mask, col].dropna()
                row[label] = f"{values.mean():.2f} ({values.std(ddof=1):.2f})"
                if tested:
                    row[f"p ({label})"] = st.fmt_p(tests[col]) if col in tests else ""
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = derive.load_analytic_table()
    st.write_table(build_table1(df), "table1_baseline_characteristics.csv")
    st.write_table(build_concordance_table(df), "supp_table23_concordance_metrics.csv")


if __name__ == "__main__":
    main()
