# Replication package: AI-clinician treatment concordance and 12-month outcomes

## The data are simulated

Every value in `data/` comes from a seeded random number generator with invented parameters; nothing is derived
from patient data. The real electronic health record data are held inside the Mayo Clinic Platform and cannot be
redistributed. The AI clinical decision support system Comentra™ is proprietary and none of its code appears here.

The simulation plants a known effect to mimic main analyses effects. 
The expert ratings carry no planted effect: any group difference in Table 4 is sampling noise. 
The planted values are invented and say nothing about the study's findings, which are reported in the manuscript.

## Run it
```bash
pip install -r requirements.txt
python main.py
```

Verified on Python 3.11.9 with the package versions in `requirements.txt`. Runtime is about 75 seconds on a laptop:
about 40 seconds for the 295 mixed-effects regressions and about 30 seconds for the expert-study models. Every
estimator is deterministic, so a run reproduces the CSV files in `results/`.

## The analysis in brief

1. Concordance: per patient, each of the 20 treatment options is one cell of a single two-by-two table (Comentra™ yes/no x
   clinician yes/no). F1 is undefined without a true positive.
2. Models: one logistic (binary outcomes; GPBoost, Laplace maximum likelihood) or linear (continuous outcomes;
   statsmodels MixedLM, restricted maximum likelihood) mixed-effects model per outcome, with the concordance metric
   as independent variable and one random intercept per clinician. Covariates: sex, age, race, Comentra™ response
   confidence, the number of options chosen by Comentra™ and by the clinician, available follow-up months (not for
   mortality), and outcome-specific psychiatric / medical diagnosis counts and pre-index values of the outcome.
3. Blocks: the main F1 analysis; severe mental illness and three narrower subgroups; primary psychiatric and
   primary medical diagnosis; balanced accuracy, PABAK and MCC as the concordance metric; US-population weighting
   (GPBoost with post-stratification weights by age, sex and race). Benjamini-Hochberg correction within each
   block.
4. Expert sub-study: low versus medium/high concordance and negative versus non-negative outcome, compared with
   linear and logistic mixed models with crossed random intercepts for case and reviewer, Welch's t-test, Cohen's d
   and Fisher's exact test.