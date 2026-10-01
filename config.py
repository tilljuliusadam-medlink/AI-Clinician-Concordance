"""
Configuration file for the simulated analysis reproduction.
"""

import re
from collections import namedtuple
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
COHORT_CSV = DATA_DIR / "simulated_cohort.csv"
RATINGS_CSV = DATA_DIR / "simulated_expert_ratings.csv"
TIMEPOINTS = ["pre_visit", "decision_point", "followup"]

# ---------------------------------------------------------------------------
# Exposure: concordance over 20 treatment options
# ---------------------------------------------------------------------------
# Every option contributes one cell to ONE two-by-two table per patient (Comentra™ yes/no x clinician yes/no); every
# concordance metric follows from that table. 
TREATMENT_OPTIONS = [
    "ssri", "snri", "atypical_antidepressant", "tca", "maoi", "mood_stabilizer",
    "sga_excl_clozapine", "clozapine", "fga", "anxiolytic_sedative", "stimulant_adhd",
    "glutamatergic", "cognitive_enhancer", "addiction_medication",
    "cbt", "dbt", "act", "emdr", "other_psychotherapy", "neurostimulation",
]
TREATMENT_FAMILIES = {
    "antidepressant": ["ssri", "snri", "atypical_antidepressant", "tca", "maoi"],
    "antipsychotic": ["sga_excl_clozapine", "clozapine", "fga"],
    "sga": ["sga_excl_clozapine", "clozapine"],
    "psychotherapy": ["cbt", "dbt", "act", "emdr", "other_psychotherapy"],
}
CONCORDANCE_METRICS = ["f1", "balanced_accuracy", "pabak", "mcc"]
EXPOSURE = "concordance"
DIAGNOSES = [("dx_anxiety", "Anxiety disorders"), ("dx_adhd", "Attention-deficit/hyperactivity disorder"),
             ("dx_bipolar", "Bipolar-spectrum disorders"), ("dx_depressive", "Depressive disorders"),
             ("dx_psychotic", "Psychosis-spectrum disorders"), ("dx_ocd", "Obsessive-compulsive disorders"),
             ("dx_ptsd", "Posttraumatic stress disorder")]

# ---------------------------------------------------------------------------
# Covariates. Every model: sex, age, race, Comentra™ response confidence, and the number of options chosen by
# Comentra™ and by the clinician (UNIVERSAL); available follow-up months (FU; not for mortality); and outcome-specific
# severity terms: the psychiatric (PSY) and / or medical (MED) diagnosis count and the pre-index value of the
# outcome.
# ---------------------------------------------------------------------------
UNIVERSAL = ("C(sex, Treatment('Male')) + age_years + C(race, Treatment('White')) "
             "+ C(ai_confidence_level, Treatment('high')) + n_treatments_ai + n_treatments_clinician")
MACROS = {"UNIVERSAL": UNIVERSAL, "PSY": "n_psychiatric_diagnoses", "MED": "n_medical_diagnoses",
          "FU": "np.log1p(followup_months)"}


def expand(formula):
    """Replace the macros of a roster formula by the columns they stand for."""
    return re.sub(r"\b(" + "|".join(MACROS) + r")\b", lambda m: MACROS[m.group(1)], formula)


# ---------------------------------------------------------------------------
# Outcomes: one model per row of manuscript Tables 2 (binary, logistic) and 3 (continuous, linear), table order.
# ---------------------------------------------------------------------------
Model = namedtuple("Model", "name kind domain label formula gate")


def binary(name, domain, label, formula, gate=None):
    return Model(name, "binary", domain, label, formula, gate)


def continuous(name, domain, label, formula):
    return Model(name, "continuous", domain, label, formula, None)


BINARY_MODELS = [
    binary("acute_psych", "Acute care", "Psychiatric ER visit/hospitalization",
           "fu_acute_psych ~ concordance + pre_acute_psych + PSY + UNIVERSAL + FU"),
    binary("er_psych", "Acute care", "Psychiatric ER visit",
           "fu_er_psych ~ concordance + pre_er_psych + PSY + UNIVERSAL + FU"),
    binary("hosp_psych", "Acute care", "Psychiatric hospitalization",
           "fu_hosp_psych ~ concordance + pre_hosp_psych + PSY + UNIVERSAL + FU"),
    binary("acute_med", "Acute care", "Medical ER visit/hospitalization",
           "fu_acute_med ~ concordance + pre_acute_med + MED + UNIVERSAL + FU"),
    binary("er_med", "Acute care", "Medical ER visit",
           "fu_er_med ~ concordance + pre_er_med + MED + UNIVERSAL + FU"),
    binary("hosp_med", "Acute care", "Medical hospitalization",
           "fu_hosp_med ~ concordance + pre_hosp_med + MED + UNIVERSAL + FU"),
    binary("acute_any", "Acute care", "Any ER visit/hospitalization",
           "fu_acute_any ~ concordance + pre_acute_any + PSY + MED + UNIVERSAL + FU"),
    binary("er_any", "Acute care", "Any ER visit",
           "fu_er_any ~ concordance + pre_er_any + PSY + MED + UNIVERSAL + FU"),
    binary("hosp_any", "Acute care", "Any hospitalization",
           "fu_hosp_any ~ concordance + pre_hosp_any + PSY + MED + UNIVERSAL + FU"),
    binary("suicidal_any_onset", "Suicidality", "Onset of suicidal thoughts/behaviors",
           "fu_suicidal_any ~ concordance + pre_suicidal_any + PSY + UNIVERSAL + FU", "dp_suicidal_any == 0"),
    binary("suicidal_thoughts_onset", "Suicidality", "Onset of suicidal thoughts",
           "fu_suicidal_thoughts ~ concordance + pre_suicidal_thoughts + PSY + UNIVERSAL + FU",
           "dp_suicidal_thoughts == 0"),
    binary("suicidal_behavior_onset", "Suicidality", "Onset of suicidal behaviors",
           "fu_suicidal_behavior ~ concordance + pre_suicidal_behavior + PSY + UNIVERSAL + FU",
           "dp_suicidal_behavior == 0"),
    binary("suicidal_any_remission", "Suicidality", "Remission of suicidal thoughts/behaviors",
           "fu_suicidal_any_remission ~ concordance + pre_suicidal_any + PSY + UNIVERSAL + FU",
           "dp_suicidal_any == 1"),
    binary("suicidal_thoughts_remission", "Suicidality", "Remission of suicidal thoughts",
           "fu_suicidal_thoughts_remission ~ concordance + pre_suicidal_thoughts + PSY + UNIVERSAL + FU",
           "dp_suicidal_thoughts == 1"),
    binary("suicidal_behavior_remission", "Suicidality", "Remission of suicidal behaviors",
           "fu_suicidal_behavior_remission ~ concordance + pre_suicidal_behavior + PSY + UNIVERSAL + FU",
           "dp_suicidal_behavior == 1"),
    binary("nonadherence", "Engagement", "Psychotropic medication nonadherence",
           "fu_nonadherence ~ concordance + pre_nonadherence + PSY + UNIVERSAL + FU"),
    binary("noshow_any", "Engagement", "Psychiatric appointment no-show",
           "fu_noshow_any ~ concordance + pre_noshow_count + PSY + UNIVERSAL + FU"),
    binary("death", "Mortality", "All-cause mortality",
           "fu_death ~ concordance + pre_suicidal_behavior + PSY + MED + UNIVERSAL"),
    binary("composite4", "Composite", "Any of 4 negative outcomes",
           "fu_composite4_any ~ concordance + pre_composite4_types + PSY + MED + UNIVERSAL + FU"),
    binary("composite6", "Composite", "Any of 6 negative outcomes",
           "fu_composite6_any ~ concordance + pre_composite6_types + PSY + MED + UNIVERSAL + FU"),
]
CONTINUOUS_MODELS = [
    continuous("cost_psych", "Cumulative cost (USD)", "Cumulative psychiatric cost",
               "fu_cost_psych ~ concordance + pre_cost_psych + PSY + UNIVERSAL + FU"),
    continuous("cost_med", "Cumulative cost (USD)", "Cumulative medical cost",
               "fu_cost_med ~ concordance + pre_cost_med + MED + UNIVERSAL + FU"),
    continuous("cost_total", "Cumulative cost (USD)", "Cumulative total cost",
               "fu_cost_total ~ concordance + pre_cost_total + PSY + MED + UNIVERSAL + FU"),
    continuous("hosp_days_psych", "Utilization", "Psychiatric hospital days",
               "fu_hosp_days_psych ~ concordance + pre_hosp_days_psych + PSY + UNIVERSAL + FU"),
    continuous("hosp_days_med", "Utilization", "Medical hospital days",
               "fu_hosp_days_med ~ concordance + pre_hosp_days_med + MED + UNIVERSAL + FU"),
    continuous("hosp_days_total", "Utilization", "Total hospital days",
               "fu_hosp_days_total ~ concordance + pre_hosp_days_total + PSY + MED + UNIVERSAL + FU"),
    continuous("appts_psych", "Utilization", "Psychiatric appointments",
               "fu_appts_psych ~ concordance + pre_appts_psych + PSY + UNIVERSAL + FU"),
    continuous("appts_med", "Utilization", "Medical appointments",
               "fu_appts_med ~ concordance + pre_appts_med + MED + UNIVERSAL + FU"),
    continuous("appts_total", "Utilization", "Total appointments",
               "fu_appts_total ~ concordance + pre_appts_total + PSY + MED + UNIVERSAL + FU"),
    continuous("noshow_count", "Engagement", "Number of psychiatric appointment no-shows",
               "fu_noshow_count ~ concordance + pre_noshow_count + PSY + UNIVERSAL + FU"),
    continuous("composite4_count", "Composite", "Number of any of 4 negative outcomes",
               "fu_composite4_types ~ concordance + pre_composite4_types + PSY + MED + UNIVERSAL + FU"),
    continuous("composite6_count", "Composite", "Number of any of 6 negative outcomes",
               "fu_composite6_types ~ concordance + pre_composite6_types + PSY + MED + UNIVERSAL + FU"),
]
# The three narrower severe-mental-illness subgroups are fitted on the collapsed outcomes only.
COLLAPSED = {"acute_psych", "acute_med", "acute_any", "suicidal_any_onset", "suicidal_any_remission",
             "nonadherence", "death", "composite4", "composite6",
             "cost_psych", "hosp_days_psych", "composite4_count", "composite6_count"}

# ---------------------------------------------------------------------------
# Analysis blocks and BH-correction
# ---------------------------------------------------------------------------
Block = namedtuple("Block", "name metric subset roster weighted tables")
BLOCKS = [
    Block("f1_main", "f1", None, "all", False, "Tables 2-3"),
    Block("smi", "f1", "smi", "all", False, "Supplementary Tables 1-2"),
    Block("smi_plus1", "f1", "smi_plus1", "collapsed", False, "Supplementary Tables 3-4"),
    Block("smi_plus2", "f1", "smi_plus2", "collapsed", False, "Supplementary Tables 5-6"),
    Block("smi_acute", "f1", "smi_acute", "collapsed", False, "Supplementary Tables 7-8"),
    Block("primary_psychiatric", "f1", "primary_dx_psychiatric", "all", False, "Supplementary Tables 9-10"),
    Block("primary_medical", "f1", "primary_dx_medical", "all", False, "Supplementary Tables 11-12"),
    Block("balanced_accuracy", "balanced_accuracy", None, "all", False, "Supplementary Tables 13-14"),
    Block("pabak", "pabak", None, "all", False, "Supplementary Tables 15-16"),
    Block("mcc", "mcc", None, "all", False, "Supplementary Tables 17-18"),
    Block("us_weighted", "f1", None, "all", True, "Supplementary Tables 20-21"),
]
MIN_EVENTS = 10
MAX_ITERATIONS = 1000

# US-population weighting
US_POP_TARGETS = {
    "sex": {"Male": 0.491, "Female": 0.509},
    "age_group": {"18-29": 0.217, "30-44": 0.261, "45-64": 0.313, "65+": 0.209},
    "race": {"White": 0.746, "African American": 0.168, "Asian": 0.075,
             "Native American or Pacific Islander": 0.011},
}

# ---------------------------------------------------------------------------
# Expert-rated clinical appropriateness sub-study (manuscript Table 4)
# ---------------------------------------------------------------------------
EXPERT_CASES, EXPERT_REVIEWERS, RATERS_PER_CASE = 120, 8, 2
EXPERT_BANDS = {"low": 60, "medium": 30, "high": 30}
NOT_APPROPRIATE = 0
EXPERT_DOMAINS = [("rating_treatment_plan", "Treatment plan"),
                  ("rating_medication_subclass", "Medication sub-class")]

EXPERT_GROUPS = {"low": "Low concordance (F1-score<0.5)",
                 "med_high": "Medium/High concordance (F1-score>=0.5)",
                 "negative": "Negative outcome", "non_negative": "Non-negative outcome"}
EXPERT_COMPARISONS = [("low", "med_high"), ("negative", "non_negative")]
