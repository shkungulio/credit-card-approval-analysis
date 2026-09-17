[![Project Cover](data/credit_card_analysis.svg)]()

# Credit Card Approval & Spending Behavior Analysis

**An end-to-end machine learning analysis of 1,319 credit card applications — including the target-leakage trap that makes most public analyses of this dataset invalid.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-337AB7)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Executive summary

| | |
|:---|:--------|
| **Problem** | Which credit card applications get approved, and why? |
| **Data** | 1,319 historical applications, 12 features, 77.6% approval rate |
| **Headline finding** | Two features (`share`, `expenditure`) are **target leakage** — they inflate ROC-AUC from **0.85 to 0.99**. Every denied applicant has exactly zero card spending, because a denied applicant has no card to spend on. |
| **Selected model** | Regularized Logistic Regression (L2, C=5) — chosen for regulatory explainability, statistically tied with the tree ensembles |
| **Honest performance** | **Test ROC-AUC 0.823**, PR-AUC 0.939, denied-class recall **0.78** at a tuned threshold |
| **Top driver** | Derogatory credit reports — approval falls from **86.3%** (clean file) to **41.7%** (any derogatory mark) |

> **Diagnostic heuristic for this dataset:** any reported ROC-AUC near 0.99 is a leakage bug, not a good model.

---

## Business questions answered

### 1. What actually drives a credit card approval decision?

Three factors carry essentially all the signal, confirmed by two independent methods (logistic coefficients and model-agnostic permutation importance):

| Rank | Driver | Effect on approval odds | Permutation importance (drop in ROC-AUC) |
|:---|:----|:-------|:---|
| 1 | **Derogatory reports** | Odds ratio **0.057** — each additional derogatory report cuts approval odds by ~94% | **0.245** |
| 2 | **Income (log)** | Odds ratio **3.94** — higher income raises approval odds sharply | **0.116** |
| 3 | **Credit-seeking ratio** (`cards_per_active`) | Odds ratio **0.538** — holding cards relative to few active accounts reads as thin-file risk | **0.053** |
| 4 | Major cards held | Odds ratio 1.65 — an existing major card is a positive prior | 0.045 |
| 5 | Dependents | Odds ratio 0.684 — more dependents, lower approval odds | 0.016 |

Home ownership, age, self-employment, address tenure and active account count contribute little once the above are known (permutation importance ≤ 0.004, statistically indistinguishable from noise).

### 2. How much does a bad credit history cost an applicant?

Approval collapses non-linearly — this is a cliff, not a slope:

| Derogatory reports | 0 | 1 | 2 | 3 | 4 | 5+ |
|:---|:---|:---|:---|:---|:---|:---|
| **Approval rate** | 86.3% | 65.7% | 26.0% | 16.7% | 5.9% | **0.0%** |

**Business read:** the decisive break is between one and two reports — approval more than halves (65.7% → 26.0%). Five or more derogatory reports is an absolute decline in this portfolio, with zero exceptions across 1,319 applications.

### 3. Which applicant segments are underserved?

| Segment | Approval rate | Gap vs. peer |
|:----|:---|:-----|
| Homeowners | **84.5%** | +12.3 pts vs. renters (72.1%) |
| Self-employed | **69.2%** | −9.0 pts vs. conventionally employed (78.2%) |
| Lowest income quartile | **68.2%** | −13.5 pts vs. top quartile (81.7%) |
| Holds no major card | **68.0%** | −11.7 pts vs. cardholders (79.7%) |
| Fewest active accounts (0–2) | **69.3%** | −13.2 pts vs. most active (82.5%) |

**Business read:** income lifts approval only up to the third quartile (81.8%) and then flattens — beyond roughly the median-plus income band, more income buys no additional approval likelihood. Self-employed and thin-file applicants are the clearest growth segments if the portfolio wants volume without loosening credit-history standards.

### 4. Can approval be predicted from information available at application time?

Yes, but far less impressively than the leaked version suggests:

| Feature set | CV ROC-AUC | Verdict |
|:---|:---|:---|
| With `share` + `expenditure` | **0.988** | Invalid — circular, unusable in production |
| Leakage-free (application-time only) | **0.853** | Honest ceiling for this dataset |

The honest model separates approved from denied applicants well (test ROC-AUC 0.823) but cannot flag every decline: precision on the denied class is 0.475, so roughly half the applicants it flags for decline would in fact have been approved. Useful as a triage and reason-code engine, not as an autonomous decision system.

### 5. Where should the approve/decline cutoff sit?

The 0.50 default is a software convention, not an analytical conclusion. Sweeping 91 candidate thresholds shows the trade-off explicitly:

| Threshold | Denied recall | Denied precision | Approved recall | Overall accuracy |
|:----|:---|:---|:---|:---|
| 0.50 (default) | 0.644 | 0.475 | 0.795 | 76.1% |
| Tuned for denied-class F1 | **0.780** | 0.465 | 0.741 | 75.0% |

**Business read:** raising the cutoff catches 78% of eventual declines instead of 64% — a 21% improvement in risk capture — at the cost of turning away about 5% more good customers. The correct choice depends on the cost of a default relative to a lost profitable customer, and that ratio belongs to whoever owns the P&L, not to a library default.

---

## The leakage finding, in detail

This is the analytical core of the project and the reason the notebook exists.

`expenditure` is average monthly **credit card** spending; `share` is that spending divided by yearly income. Both are measured *after* the approval decision. The audit is unambiguous:

| Outcome | n | % with exactly zero expenditure | Mean expenditure |
|:----|:---|:---|:---|
| Denied | 296 | **100.0%** | $0 |
| Approved | 1,023 | 2.1% | $238.60 |

- If `expenditure > 0` → approval rate **100.00%**
- If `expenditure == 0` → approval rate **6.62%**

A denied applicant cannot spend on a card they were never issued. Using these fields to predict approval is circular: the model isn't learning who deserves credit, it's reading the answer. Both are excluded from all modeling — and the exclusion is encoded in code, not done by hand, so it survives a data refresh.

This is the single most common reason a model that performs superbly in development collapses in production, where post-decision fields do not exist at scoring time.

---

## Model benchmark

Six candidates spanning the interpretability–flexibility spectrum, all inside the same preprocessing pipeline, scored with identical 5-fold stratified cross-validation:

| Model | ROC-AUC | ± SD | PR-AUC | F1 | Recall | Balanced acc. |
|:-----|:---|:---|:---|:---|:---|:---|
| **Logistic Regression** | **0.853** | 0.031 | 0.928 | 0.856 | 0.807 | **0.768** |
| Random Forest | 0.839 | 0.019 | 0.931 | 0.905 | 0.929 | 0.751 |
| Gradient Boosting | 0.837 | 0.024 | 0.922 | 0.915 | 0.958 | 0.743 |
| XGBoost | 0.828 | 0.024 | 0.923 | 0.872 | 0.856 | 0.742 |
| K-Nearest Neighbors | 0.822 | 0.030 | 0.919 | 0.899 | 0.954 | 0.688 |
| Decision Tree | 0.785 | 0.039 | 0.900 | 0.845 | 0.821 | 0.708 |

**Why logistic regression wins.** At n≈1,300 the 0.014 ROC-AUC gap over Random Forest sits well inside the fold-to-fold standard deviation (0.019–0.031) — the models are statistically tied. The tie-breaker is regulatory: under ECOA and Regulation B, a lender declining an application must state the specific principal reasons. Logistic coefficients produce per-applicant reason codes as odds ratios, in language a compliance officer or applicant can follow. A marginally more accurate model that cannot justify its decisions is not deployable in regulated lending.

**Tuned configuration:** `C=5`, `penalty='l2'`, `solver='liblinear'` → CV ROC-AUC 0.8535 → **test ROC-AUC 0.8231**.

### Model recommendations for reuse

| Rank | Model | Use it when | Watch out for |
|:---|:-----|:------|:-------|
| 1 | Regularized Logistic Regression | Reason codes are required; sample is small | Assumes monotone/linear effects — add splines for nonlinear income/age |
| 2 | Gradient Boosting / XGBoost | Raw ranking power matters; interactions expected | Overfits at n≈1,300 — cap `max_depth` at 2–3, use early stopping, add SHAP |
| 3 | Random Forest | Strong low-effort baseline | Poorly calibrated — wrap in `CalibratedClassifierCV` before using scores as risk estimates |
| 4 | Shallow Decision Tree (depth 3–4) | You need a scorecard for a credit committee | Highest variance; a communication device, not a production model |
| 5 | KNN | Sanity-check baseline only | Degrades on mixed categorical/numeric data; no interpretability |

---

## Methodology

1. **Setup & load** — fixed `RANDOM_STATE = 42` throughout, so every split, fold and randomized model is reproducible
2. **Data quality audit** — dtypes, missingness, cardinality, duplicates (none found), distributional summary
3. **Domain-driven cleaning** — 7 applicants recorded as under 18 (ages 0.17–0.75 years, clearly data-entry errors); the single faulty field is invalidated to `NaN` and imputed in-pipeline, rather than deleting 7 otherwise-valid records from a modest dataset
4. **EDA** — distributions, target-split box plots, segment approval rates, correlation matrix (which surfaced the leakage anomaly)
5. **Leakage audit** — hypothesis tested and confirmed empirically, not assumed
6. **Feature engineering** — 5 business-meaningful features: `income_per_dependent`, `has_derogatory`, `cards_per_active`, `years_at_address`, `log_income` (correcting the income skew found in step 4)
7. **Leak-proof preprocessing** — median imputation + standardization for numerics, most-frequent imputation + one-hot with `handle_unknown='ignore'` for binaries, all inside a `ColumnTransformer` so nothing is fitted on the test fold
8. **Benchmarking** — 6 models × 6 metrics × 5 stratified folds, fold-level SD retained so marginal wins aren't oversold
9. **Tuning** — `GridSearchCV` on training CV only; test set stays sealed. Grids are pre-specified for all six candidates, so the notebook runs correctly whichever model wins after a data refresh
10. **Evaluation** — first and only look at held-out data: ROC curve, PR curve, confusion matrix, per-class report, threshold sweep
11. **Interpretability** — coefficients as odds ratios plus permutation importance (25 repeats, with error bars) as a model-agnostic cross-check

**Class imbalance:** at 77.6% approval, a model that approves everyone scores 78% accuracy while being a catastrophic lending policy. Accuracy is therefore never used as the scoreboard — ROC-AUC, PR-AUC and denied-class recall are. `class_weight='balanced'` handled the 78/22 ratio; SMOTE is rarely worth it at this ratio and can distort calibration.

---

## Repository structure

```
.
├── credit_card_approval_analysis.ipynb   # Full analysis, 52 cells with narrative markdown
├── data/
│   └── CreditCard.csv                    # 1,319 applications, 12 features + target
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/<your-username>/credit-card-approval-analysis.git
cd credit-card-approval-analysis

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter

jupyter notebook credit_card_approval_analysis.ipynb
```

Runs top-to-bottom in under two minutes on a laptop. No GPU required.

---

## Data dictionary

| Column | Meaning | Role |
|:---|:------|:----|
| `card` | Application accepted (yes/no) | **Target** |
| `reports` | Number of major derogatory reports | Feature — strongest predictor |
| `age` | Age in years | Feature — protected attribute under ECOA |
| `income` | Yearly income, USD 10,000s | Feature |
| `owner` | Owns a home | Feature |
| `selfemp` | Self-employed | Feature |
| `dependents` | Number of dependents | Feature |
| `months` | Months at current address | Feature |
| `majorcards` | Number of major credit cards held | Feature |
| `active` | Number of active credit accounts | Feature |
| `share` | Monthly card expenditure / yearly income | **Excluded — target leakage** |
| `expenditure` | Average monthly card expenditure | **Excluded — target leakage** |

---

## Limitations & next steps

**Known limitations**
- **Selection bias:** the data observes outcomes only for a filtered applicant population; it cannot describe applicants who never applied or were screened out earlier
- **Denied-class precision is modest** (0.475) — the model triages, it does not decide
- **Small sample** (n=1,319) caps how much complexity is justifiable and widens all confidence intervals
- **Fairness untested:** `age` is a protected attribute under ECOA. Disparate-impact testing across age bands is required before any deployment, and excluding `age` costs almost nothing (permutation importance 0.004)

**Next steps**
1. Nested CV for an unbiased estimate of the tuned model's performance
2. Spline or binned transforms of `income` and `age` in the logistic model, then re-benchmark
3. SHAP values for per-decision explanations if a boosted model is selected instead
4. Reject inference to model the selection process and score a broader applicant pool
5. Calibration check (reliability curve, isotonic/Platt scaling) if scores feed pricing or limit assignment
6. Persist the winning pipeline with `joblib` behind a scoring function with input validation

---

## Skills demonstrated

`Target-leakage detection` · `Domain-driven data cleaning` · `Class-imbalance handling` · `Leak-proof scikit-learn pipelines` · `Controlled model benchmarking` · `Hyperparameter tuning` · `Threshold optimization as a business decision` · `Model interpretability (odds ratios, permutation importance)` · `Regulatory awareness (ECOA / Regulation B)` · `Reproducible analysis` · `Technical writing for mixed audiences`

---

## Author

**Seif Kungulio** - M.S. Data Analytics


## Dataset
`CreditCard.csv`, a well-known credit card application dataset distributed with the R **AER** package ([Applied Econometrics with R](https://cran.r-project.org/package=AER)), originally from Greene, *Econometric Analysis*.
