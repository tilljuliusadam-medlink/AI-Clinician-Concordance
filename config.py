"""Constants for the replication package.

Every roster, covariate list and analysis block used by the paper lives here so
that a reader can see the whole analytical surface in one file. Labels and time
windows are quoted from the manuscript verbatim.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

COHORT_CSV = DATA_DIR / "simulated_cohort.csv"
RATINGS_CSV = DATA_DIR / "simulated_expert_ratings.csv"

# Smallest sample a single outcome regression may be fit on. Below this the
# outcome is skipped for that block rather than fitted on a handful of rows.
MIN_FIT_N = 10

TIMEPOINTS = ["pre_visit", "decision_point", "followup"]

# ---------------------------------------------------------------------------
# Exposure: the treatment options
# ---------------------------------------------------------------------------
# Concordance is computed FLAT across every treatment option at once. The AI
# recommendation and the clinician decision are recorded per option, the whole
# set forms ONE two-by-two table per patient, and every metric follows from it.
# There is no grouping into drug families first and no averaging of per-family
# scores: that would be a different quantity.
TREATMENT_OPTIONS = [
    "ssri", "atypical_antidepressant", "snri", "tca", "maoi",
    "sga_excl_clozapine", "clozapine", "fga",
    "mood_stabilizer",
    "benzodiazepine", "z_drug",
    "stimulant", "nonstimulant_adhd",
    "cbt", "dbt", "act", "emdr", "other_psychotherapy",
]

# Table 1 also reports the drug families. They are DERIVED from the options
# above (a family is present when any of its options is), never stored, so a
# family cannot disagree with its own members.
TREATMENT_FAMILIES = {
    "antidepressant": ["ssri", "atypical_antidepressant", "snri", "tca",
                       "maoi"],
    "antipsychotic": ["sga_excl_clozapine", "clozapine", "fga"],
    "sga": ["sga_excl_clozapine", "clozapine"],
    "anxiolytic_hypnotic": ["benzodiazepine", "z_drug"],
    "psychotherapy": ["cbt", "dbt", "act", "emdr", "other_psychotherapy"],
}

# ---------------------------------------------------------------------------
# Covariates (Methods: sex, age, race, number of psychiatric diagnoses, months
# to follow-up, AI response-confidence level)
# ---------------------------------------------------------------------------
COVARIATES = [
    "sex",
    "age_years",
    "race",
    "n_psychiatric_diagnoses",
    "months_to_followup",
    "ai_confidence_level",
]
CATEGORICAL_COVARIATES = ["sex", "race", "ai_confidence_level"]

# Reference level of each categorical covariate, pinned here rather than left
# to alphabetical order. A rare category can vanish from one listwise-deleted
# sample, which would silently move that row's reference level away from the
# one every other row uses.
COVARIATE_REFERENCE = {
    "sex": "Female",
    "race": "African American",
    "ai_confidence_level": "high",
}

DIAGNOSIS_FLAGS = [
    "dx_anxiety", "dx_depressive", "dx_adhd",
    "dx_bipolar", "dx_psychotic", "dx_ocd",
]

# Status variables recorded at every timepoint. At pre_visit they mean "at any
# point in the 12 months before the decision point", which is the window Table 3
# uses for every acute-care denominator.
STATUS_COLUMNS = [
    "suicidal_thoughts",
    "suicidal_behavior",
    "er_visit_psychiatric",
    "hospitalization_psychiatric",
    "er_visit_medical",
    "hospitalization_medical",
    "appointment_no_show",
    "treatment_nonadherence",
]

# The manuscript uses a THREE-month window in exactly two places, both distinct
# from the 12-month outcome denominators: identifying a decision-point month,
# and the Table 2 concordance subgroup "Hospitalization/ER visit in the last 3
# months". It is therefore its own column.
ACUTE_CARE_3MO = "acute_care_last_3_months"

# ---------------------------------------------------------------------------
# Outcomes. Labels are the manuscript's Table 3 and Table 4 wording verbatim.
# ---------------------------------------------------------------------------
# (name, label, rule, follow-up column, baseline column). Rules:
#   event     -- the follow-up status, across the whole analytic sample
#   onset     -- follow-up status among patients negative at baseline
#   remission -- absence at follow-up among patients positive at baseline
#   relapse   -- follow-up status among patients positive at baseline
BINARY_OUTCOMES = [
    ("death", "All-cause mortality",
     "event", "fu_death", None),
    ("suicidal_thoughts_onset",
     "Onset of suicidal thoughts in patients with no suicidal thoughts at "
     "baseline",
     "onset", "fu_suicidal_thoughts", "dp_suicidal_thoughts"),
    ("suicidal_behavior_onset",
     "Onset of suicidal behaviors in patients with no suicidal behaviors at "
     "baseline",
     "onset", "fu_suicidal_behavior", "dp_suicidal_behavior"),
    ("suicidal_thoughts_remission",
     "Remission of suicidal thoughts in patients with suicidal thoughts at "
     "baseline",
     "remission", "fu_suicidal_thoughts", "dp_suicidal_thoughts"),
    ("suicidal_behavior_remission",
     "Remission of suicidal behaviors in patients with suicidal behaviors at "
     "baseline",
     "remission", "fu_suicidal_behavior", "dp_suicidal_behavior"),
    # Denominator is the SAME event type, not any psychiatric acute care.
    ("er_visit_psychiatric_onset",
     "Psychiatric ER visit in patients with no recent psychiatric ER visit "
     "(12 months)",
     "onset", "fu_er_visit_psychiatric", "pre_er_visit_psychiatric"),
    ("hospitalization_psychiatric_onset",
     "Psychiatric hospitalization in patients with no recent psychiatric "
     "hospitalization (12 months)",
     "onset", "fu_hospitalization_psychiatric",
     "pre_hospitalization_psychiatric"),
    ("acute_care_medical_onset",
     "Medical hospitalization/ER visit in patients with no recent medical "
     "hospitalization/ER visit (12 months)",
     "onset", "fu_acute_care_medical", "pre_acute_care_medical"),
    ("acute_care_psychiatric_relapse",
     "Psychiatric re-hospitalization/repeated ER visit in patients with recent "
     "psychiatric hospitalization/ER visit (12 months)",
     "relapse", "fu_acute_care_psychiatric", "pre_acute_care_psychiatric"),
    ("acute_care_medical_relapse",
     "Medical re-hospitalization/repeated ER visit in patients with recent "
     "medical hospitalization/ER visit (12 months)",
     "relapse", "fu_acute_care_medical", "pre_acute_care_medical"),
    # "prior" with no window: any no-show or nonadherence before the decision
    # point, however long ago.
    ("appointment_no_show_onset",
     "Appointment no-shows in patients with no prior no-shows",
     "onset", "fu_appointment_no_show", "pre_appointment_no_show"),
    ("nonadherence_onset",
     "Treatment nonadherence in patients with no prior nonadherence",
     "onset", "fu_treatment_nonadherence", "pre_treatment_nonadherence"),
    ("composite_adverse",
     "Composite of adverse events (death, suicidality, hospitalization, "
     "ER visit)",
     "event", "fu_composite_adverse", None),
]

CONTINUOUS_OUTCOMES = [
    ("cost_psychiatric_usd",
     "Cumulative healthcare cost related to psychiatric diagnoses"),
    ("cost_medical_usd",
     "Cumulative healthcare cost related to medical diagnoses"),
    ("cost_total_usd", "Cumulative healthcare cost overall"),
    ("hospital_days_psychiatric",
     "Cumulative days of hospitalization related to psychiatric diagnoses"),
    ("hospital_days_medical",
     "Cumulative days of hospitalization related to medical diagnoses"),
    ("appointments_psychiatric",
     "Number of appointments for psychiatric diagnoses"),
    ("appointments_medical", "Number of appointments for medical diagnoses"),
]

# ---------------------------------------------------------------------------
# The six analysis blocks. One routine runs all of them.
# ---------------------------------------------------------------------------
BLOCKS = [
    dict(name="f1_full", metric="f1", subgroup=None, weighted=False,
         tables="Table 3 / Table 4"),
    dict(name="balacc_full", metric="balanced_accuracy", subgroup=None,
         weighted=False, tables="Supplementary Table 1 / 2"),
    dict(name="pabak_full", metric="pabak", subgroup=None, weighted=False,
         tables="Supplementary Table 3 / 4"),
    dict(name="f1_primary_psychiatric", metric="f1", subgroup="psychiatric",
         weighted=False, tables="Supplementary Table 5 / 6"),
    dict(name="f1_primary_medical", metric="f1", subgroup="medical",
         weighted=False, tables="Supplementary Table 7 / 8"),
    dict(name="f1_us_weighted", metric="f1", subgroup=None, weighted=True,
         tables="Supplementary Table 9 / 10"),
]

# ---------------------------------------------------------------------------
# Post-stratification targets: published 2022 American Community Survey
# marginals for the US adult population. Public data, quoted not derived.
# Race shares are renormalised across the four matchable categories.
# ---------------------------------------------------------------------------
US_POP_TARGETS = {
    "sex": {"Male": 0.491, "Female": 0.509},
    "age_group": {"18-29": 0.217, "30-44": 0.261, "45-64": 0.313, "65+": 0.209},
    "race": {"White": 0.746, "African American": 0.168,
             "Asian": 0.075, "Native American or Pacific Islander": 0.011},
}
WEIGHTABLE_RACES = list(US_POP_TARGETS["race"])

# ---------------------------------------------------------------------------
# Post-hoc expert-rated clinical appropriateness sub-study
# ---------------------------------------------------------------------------
RATING_LEVELS = (1, 2, 3, 4)
RATERS_PER_CASE = 2     # every case is rated by exactly two of the reviewers
COMPLEXITY_ORDER = {"low": 1, "middle": 2, "high": 3}
SELECTION_ARMS = ["discordant_bad_outcome", "high_concordance_no_bad_outcome"]

NI_MARGIN = -0.5        # points on the 1-4 appropriateness scale
CI_NONINFERIORITY = 0.95
CI_SUPERIORITY = 0.975
RUSHED_SECONDS = 120    # sensitivity analysis exclusion threshold
BOOTSTRAP_N = 2000
SEED = 20260727         # seeds the bootstrap resampling in agreement.py

EXPERT_ENDPOINTS = [
    ("treatment", "approp_treatment_clinician", "approp_treatment_ai",
     "choice_treatment"),
    ("diagnostic", "approp_diagnostic_clinician", "approp_diagnostic_ai",
     "choice_diagnostic"),
]
