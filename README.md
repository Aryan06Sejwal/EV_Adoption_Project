# Electric Vehicle Adoption Intelligence

**Data-Driven Analysis and Growth Prediction Across Indian States**

Course project for *Fundamentals of Data Science*.
Authors: Dhruv Pandey (590028550) and Aryan Sejwal (590023034).

## Overview

This project analyses new electric vehicle (EV) registrations in India for **calendar year 2025**, using data from the Government of India's VAHAN dashboard. It compares states, vehicle categories and months, tests a few hypotheses, and builds simple machine learning models to see how well EV adoption can be predicted.

**Research questions**

1. Which states lead in EV adoption, by volume and by share of all registrations?
2. Which vehicle categories (2W, 3W, 4W, Bus, Other) are growing fastest?
3. How did EV registrations change over 2025?
4. What is associated with higher EV adoption across states?
5. Can EV registrations be predicted with simple machine learning models?

## How the code follows the course plan

The single script `ev_adoption_analysis.py` is split into sections in the same order as the course plan.

| Stage | What the code does |
|---|---|
| 1. Problem understanding (Unit I) | Problem statement, business objective and CRISP-DM mapping are in the script's opening docstring. Only 2025 data is used. |
| 2. Data understanding (Unit III) | Data types, data sources and a data quality assessment: duplicates, missing values, inconsistent state names, impossible values and outliers. |
| 3. Exploratory data analysis (Unit III) | Bar charts, line chart, histograms, scatter plot and correlation heatmap, plus descriptive statistics and hypothesis tests. |
| 4. Feature engineering (Unit IV) | Log and scaling transformations, feature selection (F-test, mutual information, correlation filter) and PCA. |
| 5. Machine learning models (Unit V) | Linear Regression, Decision Tree, Logistic Regression, KNN, K-Means and hierarchical clustering. |
| 6. Model evaluation (Unit V) | MAE, RMSE and R2 for regression; accuracy and confusion matrix for classification; silhouette score and inertia for clustering. |

## Project structure

```
EV_Adoption_Project/
├── ev_adoption_analysis.py     # the full analysis (stages 1-6)
├── convert_vahan.py            # builds the dataset from VAHAN exports
├── requirements.txt            # Python libraries
├── data/
│   └── ev_registrations.csv    # combined dataset used by the analysis
├── raw_data/                   # original VAHAN downloads (_ALL and _EV files)
├── outputs/
│   ├── tables/                 # result tables (CSV, JSON)
│   └── figures/                # charts (PNG)
├── EV_Adoption_Report.docx     # final report
└── EV_Adoption_Intelligence_PPT.pptx
```

## Setup and usage

You need Python 3.9 or newer.

```bash
pip install -r requirements.txt
python ev_adoption_analysis.py
```

The script reads `data/ev_registrations.csv` and writes everything to `outputs/tables/` and `outputs/figures/`. The same results are produced on every run because the random seed is fixed.

You can also choose other folders:

```bash
python ev_adoption_analysis.py --data-dir path/to/data --out-dir path/to/outputs
```

## Dataset

**Source:** VAHAN Public Report, Ministry of Road Transport and Highways (Parivahan Analytics), exported in November 2026.

**Period:** calendar year 2025 (12 months).

**Coverage:** 18 states and union territories, 5 vehicle categories, 1,080 rows after cleaning.

| Column | Meaning |
|---|---|
| `state` | State or union territory |
| `month` | Month, format `YYYY-MM` |
| `vehicle_category` | `2W`, `3W`, `4W`, `Bus` or `Other` |
| `ev_registrations` | New electric registrations (fuel: ELECTRIC(BOV) and PURE EV) |
| `total_registrations` | New registrations of all fuel types |

### Rebuilding the dataset from VAHAN

1. On the VAHAN Public Report page, set Year Type to Calendar Year, 2025 to 2025, Y-Axis to Vehicle Category and X-Axis to Month Wise.
2. For each state, download two CSV files. The first has no Fuel filter and is saved as `<STATE>_ALL`. The second has Fuel set to ELECTRIC(BOV) and PURE EV (and no "EV Type" filter) and is saved as `<STATE>_EV`.
3. Put all the files in one folder and run:

```bash
python convert_vahan.py raw_data data/ev_registrations.csv
```

The converter groups VAHAN's vehicle classes into the five categories, and it warns about any missing pair or any row where EV registrations exceed total registrations.

## Key results

Results from the run on the 18-state dataset:

- The 18 states registered **1,697,376 EVs** in 2025. Uttar Pradesh leads on volume (405,365), then Maharashtra (266,552). Assam (14.3%), Delhi (13.9%) and Kerala (12.0%) lead on EV share.
- Two-wheelers and three-wheelers make up **91.8%** of EVs. Among the large categories, four-wheelers grew fastest between the first and second half of the year (+33.9%).
- EV registrations were **16.7% higher** in the second half than in the first (Wilcoxon test, p = 0.00014).
- **Regression:** Linear Regression reaches R2 = 0.785 and the Decision Tree 0.728, but neither beats the naive "same as last month" baseline (R2 = 0.791).
- **Classification** of states into Low / Medium / High EV-share tiers scored only 29-34% accuracy, close to the 33% chance level.
- **Clustering:** K-Means with k = 6 has the best silhouette score (0.232), but all scores are low, so the cluster structure is weak.

## Limitations

- The sample has only **18 states**. Karnataka, Tamil Nadu and Telangana are not included, so rankings apply to the sampled states only.
- The data covers one year, so long-term trends and seasonality cannot be measured.
- Charging-station data was not available, so the effect of charging infrastructure is not tested.
- The models are simple and the test set is small, so results are indicative, not conclusive.
- The best classifier is chosen on the same cross-validation scores that are reported, so its accuracy is slightly optimistic.

## Output files

`outputs/tables/` includes `data_quality_report.csv`, `descriptive_stats.csv`, `stat_tests.csv`, `correlation_matrix.csv`, `correlation_pairs.csv`, `feature_scores.csv`, `pca_summary.csv`, `regression_metrics.csv`, `classification_metrics.csv`, `confusion_matrix.csv`, `kmeans_selection.csv`, `cluster_summary.csv`, `cluster_assignments.csv`, `state_features.csv` and `results_summary.json`.

`outputs/figures/` includes the state, category and monthly charts (01-03), the histograms (04), the market-size scatter plot (05), the correlation heatmap (06), the PCA and clustering plots (07-09), and the regression, decision tree and confusion matrix plots (10-12).
