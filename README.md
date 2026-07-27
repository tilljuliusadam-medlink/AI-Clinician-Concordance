# Replication package: AI-clinician treatment concordance and 12-month outcomes

Simulated data and the analysis code behind the tables in "Higher clinician concordance with AI psychiatric treatment recommendations is associated with superior 12-month health outcomes."

## The data are simulated

Every value in `data/` came from a seeded random number generator. Outcomes were generated
independently of the treatment decisions, so the simulated association between concordance and every outcome is exactly zero. Any estimate in `results/` is sampling noise, never a study result. The published findings with aggregated statistics are in the manuscript. 

The AI clinical decision support system is proprietary and none of it appears here: no source code, prompts, knowledge-base content, retrieval configuration or base model. The real electronic health record data are held inside the Mayo Clinic Platform and cannot be redistributed.

## Run it

```bash
pip install -r requirements.txt
python main.py
```

Verified on Python 3.11.9 with the package versions in `requirements.txt` (numpy 2.4.4, pandas 2.3.3, scipy 1.16.3, statsmodels 0.14.6).

Runtime is 1 to 2 minutes on a laptop, nearly all of it in `expert_study.py`, whose crossed mixed models are fitted by numerical optimization. `tables_descriptive.py` and `regression.py` together finish in about 2 seconds. Progress prints as each table is written.

The run writes 11 CSV files to `results/`. The copies shipped in `results/` were produced by that same command: the data are a fixed committed file and every estimator is seeded, so a correct run reproduces all 11 files byte for byte.

## What is here

| File | Contents |
|---|---|
| `data/simulated_cohort.csv` | simulated data of 2000 patients, three rows each: `pre_visit`, `decision_point`, `followup` |
| `data/simulated_expert_ratings.csv` | 300 cases, each rated independently by 2 of 10 psychiatrists |
| `data/data_dictionary.csv` | all 82 columns of both files |

| Script | Writes to `results/` | Manuscript |
|---|---|---|
| `tables_descriptive.py` | `table1_baseline_characteristics.csv`, `table2_concordance_metrics.csv` | Tables 1 and 2 |
| `regression.py` | `table3_binary_regressions.csv`, `table4_continuous_regressions.csv` | Tables 3 and 4, Supplementary Tables 1 to 10 |
| `expert_study.py` | `expert_t1_sample.csv` to `expert_t7_sensitivity.csv` | Expert-rated appropriateness sub-study |

`config.py` holds every roster, covariate and analysis block. `derive.py` turns the three-row
panel into the one-row-per-patient analytic table, including the concordance metrics. `stats.py`
and `agreement.py` hold the shared estimators.

## License

MIT, see [LICENSE](LICENSE). It covers the code and the simulated data in this repository. It
does not extend to the AI clinical decision support system, which is proprietary and, as stated
above, appears nowhere here.
