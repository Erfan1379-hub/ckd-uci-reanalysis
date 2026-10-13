# UCI CKD benchmark: leakage-free re-analysis

Code and results for the manuscript *"Near-perfect accuracy on the UCI chronic kidney disease
benchmark reflects how its control group was assembled: a leakage-free re-analysis"*
(Salgi K, Ghapanvari E).

## Contents

| Path | Content |
|---|---|
| `data/kidney_disease.csv` | UCI CKD data set (400 records; source: UCI ML Repository, dataset 336) |
| `src/ckd_common.py` | Rule-based cleaning, leakage-free preprocessing pipeline, classifiers, envelope rule |
| `src/run_analysis.py` | All experiments (E0–E10) → `results/` |
| `src/make_figures.py` | Fig 1–5 and S1–S3 Fig (PNG + 300-dpi TIFF) → `figures/` |
| `src/make_tables.py` | Supplementary tables (LaTeX) → `results/S1_tables.tex` |
| `results/` | All result tables (CSV/JSON), `versions.json`, run log |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src
python run_analysis.py      # ~30 min on 2 CPU cores; each experiment is skipped if its result file exists
python make_figures.py
python make_tables.py
```

Delete the files in `results/` to recompute from scratch. Random seed 42 is used throughout;
other library versions may change late decimal places.

## Experiments

| ID | Experiment | Output |
|---|---|---|
| E0 | Data audit: missingness by class, control-group ranges, excursions per patient | `E0_*.csv/json` |
| E1 | Envelope rule, decision stump, depth-2 tree, missingness-only model (5×10 CV) | `E1_*` |
| E2 | Seven classifiers (5×10 CV) + corrected resampled t-tests vs logistic regression | `E2_*` |
| E3 | Nested CV of model selection (outer 5×5, inner 5-fold) | `E3_nested.csv` |
| E4 | Leakage: preprocessing/selection on all data; class-conditional imputation | `E4_*` |
| E5 | 200 random 80/20 splits | `E5_splits.csv` |
| E6 | Ablation of variable groups | `E6_*` |
| E7 | Imputation methods × missingness indicators × model | `E7_imputation.csv` |
| E8 | Label-permutation test | `E8_permutation_test.json` |
| E9 | Conventional hold-out protocol, permutation importance, mutual information | `E9_*` |
| E10 | Clustering (supplementary) | `E10_*` |

## Note on an earlier code version

An earlier version (`ckd_project.py`) winsorised all 24 variables. For binary and ordinal variables
whose first and third quartiles coincide, this clipped them to a constant (9 variables). The current
code winsorises only the 11 continuous variables.

## License

Code: MIT. Data: see the UCI Machine Learning Repository terms (CC BY 4.0).
