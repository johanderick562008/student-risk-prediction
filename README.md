# Student Academic Risk Prediction System

A Python project that analyses student academic data, labels each student as **Low**, **Medium** or
**High** academic risk with a transparent rule, trains and compares machine-learning classifiers,
gives supportive recommendations, and exports a CSV report. A Streamlit dashboard presents everything.

> **This is a data-driven support indicator, not a judgement of any student.** The risk levels are
> *proxy labels* produced by a rule we chose. They are not observed outcomes such as failing a course.

---

## Contents

1. [Recruitment task requirements](#1-recruitment-task-requirements)
2. [Dataset](#2-dataset)
3. [Project structure](#3-project-structure)
4. [Technology stack](#4-technology-stack)
5. [Installation (Windows)](#5-installation-windows)
6. [How to run everything](#6-how-to-run-everything)
7. [Data exploration: what we found](#7-data-exploration-what-we-found)
8. [Data cleaning and preprocessing](#8-data-cleaning-and-preprocessing)
9. [How risk categories are defined](#9-how-risk-categories-are-defined)
10. [Feature selection and leakage prevention](#10-feature-selection-and-leakage-prevention)
11. [Models, training and evaluation](#11-models-training-and-evaluation)
12. [Prediction system and recommendations](#12-prediction-system-and-recommendations)
13. [Reports and dashboard](#13-reports-and-dashboard)
14. [Testing](#14-testing)
15. [Limitations and ethical considerations](#15-limitations-and-ethical-considerations)
16. [Future improvements](#16-future-improvements)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. Recruitment task requirements

| Requirement | Where it is done |
|---|---|
| Explore the data, find patterns, attendance vs marks, visualisations | `src/eda.py`, `reports/figures/` |
| Clean data: missing values, duplicates, invalid values, encoding | `src/preprocessing.py` |
| Define Low / Medium / High risk and explain the approach | `src/risk_labeling.py`, `src/config.py`, [section 9](#9-how-risk-categories-are-defined) |
| Train and compare Logistic Regression, Decision Tree, Random Forest | `src/train_model.py` |
| Evaluate with accuracy, precision, recall, F1, confusion matrix | `src/evaluate_model.py`, `reports/metrics.json` |
| Predict a student's risk and give recommendations | `src/predict.py`, `src/recommendations.py` |
| Bonus: automatic CSV report `Student \| Attendance \| Risk \| Recommendation` | `src/predict.py` -> `reports/risk_report.csv`, dashboard download |
| Optional dashboard | `app.py` (Streamlit) |

## 2. Dataset

- **Source:** <https://www.kaggle.com/datasets/ganeshkumarofficial/student-dataset>
- **File inside the download:** `dcs_student_data.csv` (about 1.2 MB)
- **Size:** 10,030 rows x 21 columns
- The Kaggle page says the data was *"modified for DCSXGDG recruitment"*.

| Column | Type in file | Notes |
|---|---|---|
| Student_ID, First_Name, Last_Name, Email | text | Personal data - **removed** during cleaning |
| Gender | text | Inconsistent case/spaces (`" FEMALE "`) - EDA only |
| Age | number | 201 missing, two impossible values (-3, 87) - EDA only |
| Department | text | 8 spellings of 4 departments (`CS`, `Computer Science`, `BUSINESS`, ...) |
| Attendance (%) | number | 199 missing, impossible values -12, 108, 135 |
| Midterm_Score, Final_Score | number | ~200 missing each, impossible values (-8, 145, -5, 132) |
| Assignments_Avg, Quizzes_Avg, Projects_Score | number (0-100) | ~200 missing each |
| Participation_Score | number (0-10) | ~200 missing |
| Total_Score | number | Range **50.02 - 99.99**, no missing values |
| Grade | text (A-F) | Does **not** agree with Total_Score (see EDA) |
| test_preparation_course | 0 / 1 | 39% completed |
| math_score | text | Stored as text because one cell is `"\t41"` |
| reading_score, writing_score, science_score | number | Complete |

The column names in the file match the task PDF exactly, so no renaming was needed. The loader
(`src/data_loader.py`) still maps common variants (e.g. `Attendance`, `Total Score`) to these names
and stops with a clear message if a required column is missing.

### Dataset setup

You do **not** need a Kaggle API key.

1. Open the Kaggle link above and click **Download** (you may need to sign in to Kaggle).
2. Unzip the downloaded `archive.zip`.
3. Put `dcs_student_data.csv` in the project's `data\` folder:

```
student-risk-prediction\data\dcs_student_data.csv
```

The file may also be named `student_dataset.csv`, or be the only `.csv` in `data\`.

The repository includes `data/students_anonymized.csv`: the same rows with Student_ID, names and
emails removed (created by `python src/anonymize_data.py`). It is used automatically when the raw
file is missing, e.g. for the online deployment, and gives identical results.

## 3. Project structure

```
student-risk-prediction/
├── data/
│   ├── dcs_student_data.csv        # downloaded from Kaggle (not committed)
│   └── students_anonymized.csv     # same data without personal columns (committed)
├── models/
│   ├── risk_model.pkl              # saved scikit-learn pipeline
│   └── model_metadata.json         # chosen model, features, thresholds, versions
├── reports/
│   ├── figures/                    # all charts (EDA + confusion matrices)
│   ├── metrics.json                # every metric from the training run
│   ├── model_comparison.csv        # comparison table (CV / train / test)
│   └── risk_report.csv             # Student | Attendance | Risk | Recommendation
├── src/
│   ├── config.py                   # paths, thresholds, feature lists (change things here)
│   ├── anonymize_data.py           # creates the anonymised copy of the dataset
│   ├── data_loader.py              # find + load CSV, map column names, validate
│   ├── eda.py                      # exploratory analysis and all chart functions
│   ├── preprocessing.py            # cleaning + scikit-learn preprocessing pipeline
│   ├── risk_labeling.py            # transparent Low/Medium/High rule
│   ├── train_model.py              # baselines, 3 models, selection, saving
│   ├── evaluate_model.py           # metrics, confusion matrices, re-evaluation script
│   ├── predict.py                  # single + bulk prediction, CSV report
│   └── recommendations.py          # rule-based supportive recommendations
├── tests/
│   └── test_pipeline.py            # pytest checks with small example records
├── .streamlit/config.toml          # light theme for the dashboard
├── app.py                          # Streamlit dashboard
├── requirements.txt
└── README.md
```

`config.py` and `eda.py` were added to the suggested structure: `config.py` keeps thresholds and
feature lists in one place, and `eda.py` lets the script and the dashboard share the same charts.

## 4. Technology stack

Python 3 · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Joblib · Streamlit · pytest.
The whole project is plain Python scripts.
No deep learning, databases, cloud services or APIs: this is a small tabular classification problem.

## 5. Installation (Windows)

Open **Command Prompt** in the `student-risk-prediction` folder.

Create and activate a virtual environment:

```bat
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bat
pip install -r requirements.txt
```

Then download the dataset as described in [Dataset setup](#dataset-setup).

## 6. How to run everything

Run these from the `student-risk-prediction` folder with the virtual environment active.

| Step | Command | Output |
|---|---|---|
| 1. Explore the data | `python src/eda.py` | overview + findings in the console, 13 charts in `reports/figures/` |
| 2. Train and save the model | `python src/train_model.py` | `models/risk_model.pkl`, `reports/metrics.json`, confusion matrices |
| 3. Evaluate the saved model | `python src/evaluate_model.py` | train vs test metrics, classification reports |
| 4. Predict everyone + report | `python src/predict.py` | `reports/risk_report.csv` + an example student |
| 5. Launch the dashboard | `streamlit run app.py` | opens <http://localhost:8501> |
| Run the tests | `python -m pytest -v` | 27 tests |

Training takes well under a minute. Everything uses `random_state = 42`, so results are reproducible.

## 7. Data exploration: what we found

All numbers below come from the actual file (`python src/eda.py` reproduces them).

**Data quality**
- 30 exact duplicate rows; ~2% missing values in Age, Attendance and every assessment column.
- Impossible values: Age -3 and 87; Attendance -12%, 108%, 135%; Midterm -8 and 145; Final -5 and 132.
- Inconsistent text categories and one numeric column (`math_score`) stored as text.

**Distributions** (`03_score_distributions.png`, `02_attendance_distribution.png`)
- Every score is almost perfectly **uniform**: attendance and Total_Score between 50 and 100, Midterm
  and Final between 40 and 100, participation between 0 and 10. Real marks usually cluster around an average.

**Attendance vs marks** (`07_attendance_vs_total.png`, `08_attendance_vs_final.png`, `10_attendance_bands.png`)
- **No relationship.** Pearson r = -0.011 (Total_Score) and -0.012 (Final_Score). Average Total_Score is
  ~75 in every attendance range, from <60% to 85-100%.

**Correlation heatmap** (`09_correlation_heatmap.png`)
- Every pair of columns has |r| < 0.11. Only the four subject scores correlate weakly with each other.

**Total_Score and Grade** (`12_component_correlations.png`, `05_grade_vs_total.png`)
- Total_Score is **not** built from the components: every component has |r| < 0.02 with it.
- Grade contradicts Total_Score: the median Total_Score is 74.5-75.8 for *every* grade A-F. Grade is not used.

**Test-prep course and departments** (`11_prep_course.png`, `06_department_performance.png`)
- Total_Score is 75.0 with the course and 75.1 without. Departments average between 74.7 and 75.2.

**Interpretation.** The columns look as if they were generated independently of each other.
That is an important finding in itself: attendance and marks are separate warning signs in this data,
and nothing in the data can be used to estimate Total_Score. A correlation would not prove causation
anyway, and here there are no correlations to interpret.

## 8. Data cleaning and preprocessing

**Cleaning (`clean_data`)**: rules that don't learn anything from the data, so they can run on the
full dataset:

| Problem | Action | Why |
|---|---|---|
| 30 duplicate rows | removed | a duplicated student would count twice |
| `" FEMALE "`, `"CS"`, `" engineering "` ... | trimmed and unified | one category per real group |
| `"\t41"` in `math_score` | strip, then convert to number | the value is valid, only the formatting is wrong |
| Values outside valid ranges (e.g. attendance 135%) | set to **missing** | we cannot know the true value, so we don't guess or clip |
| Names, emails, Student_ID | dropped; replaced by `Anon_ID` (`STU-00001`...) | privacy |

**Preprocessing pipeline (`build_preprocessor`)**: steps that *learn* from data, so they live in
a scikit-learn `Pipeline` + `ColumnTransformer` and are fitted on the **training set only**:

- Numeric features: missing values filled with the training **median** (robust to outliers), then
  `StandardScaler` for Logistic Regression only (trees split one feature at a time and don't need scaling;
  skipping it keeps the tree's rules readable in real units).
- `test_preparation_course`: most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`).

**Split:** stratified 80/20 train/test split (7,838 / 1,960 students) *before* fitting anything.
The 202 students with no attendance value can't be labelled, so they are left out of training.

**Demographics:** Gender, Age and Department are **not** model features. There is no educational
reason to judge risk by them, they could introduce bias, and the EDA shows they carry no signal anyway.

## 9. How risk categories are defined

The rule (`src/risk_labeling.py`, thresholds in `src/config.py`) is checked **in this order**, so a
student who meets a High condition can never be labelled Medium:

| Level | Condition |
|---|---|
| **High** | attendance < 65% **or** Total_Score < **55** |
| **Medium** | not High, and attendance < 75% **or** Total_Score < 65 |
| **Low** | attendance ≥ 75% **and** Total_Score ≥ 65 |

**Why these thresholds?** 75% is a common minimum-attendance requirement, and 65% marks a serious
shortfall. A total below 65 is a weak result, and the lowest band is treated as high risk.

**Adjustment from the brief (and why).** The brief proposed *Total_Score < 50* for High risk. In this
dataset the lowest Total_Score is **50.02**, so that condition matches **0 students** and High risk would
silently become attendance-only. We raised it to **55** (roughly the bottom 10% of Total_Score):

| Policy | Low | Medium | High |
|---|---|---|---|
| Original (Total < 50) | 3,476 (35.5%) | 3,460 (35.3%) | 2,862 (29.2%) |
| **Adjusted (Total < 55)** - used | 3,476 (35.5%) | 2,762 (28.2%) | 3,560 (36.3%) |

To use the original rule instead, change `"high_total"` in `RISK_THRESHOLDS` (`src/config.py`) and retrain.

> **Methodological limitation:** these labels are built from attendance and Total_Score that already
> exist. They are **proxy labels**, not independently observed failure or the result of an intervention.
> A model trained on them learns to imitate this policy; it cannot prove it predicts future failure.

## 10. Feature selection and leakage prevention

The goal is to flag students **before the final assessment**, so the model may only use information
available by then.

**Used as features:** Attendance (%), Midterm_Score, Assignments_Avg, Quizzes_Avg, Participation_Score,
Projects_Score, test_preparation_course.

| Excluded column | Reason |
|---|---|
| Student_ID, First_Name, Last_Name, Email | personal identifiers, no predictive meaning |
| **Total_Score** | used to build the label -> direct **leakage** |
| Grade | derived from the final result (and inconsistent with Total_Score) |
| Final_Score | not known before the final assessment |
| math/reading/writing/science_score | unclear when they are measured (they look like they came from another source dataset), so they can't safely be assumed to exist at prediction time |
| Gender, Age, Department | demographic; fairness risk and no justification |

**Attendance** is both a feature and part of the label. That is intended: attendance is known during
the term, and the risk policy explicitly depends on it. It also means part of the label is *directly
visible* to the model, which matters when interpreting the scores below.

**Leakage demonstration:** the same Random Forest *with* Total_Score added scores 0.9995 test accuracy.
That near-perfect score just means the model can see the answer; it is reported only as a warning.

## 11. Models, training and evaluation

### Approaches compared (same split, same preprocessing)

| Approach | Why it is included |
|---|---|
| Most-frequent-class baseline | the score of "no model at all" |
| **Attendance-only rule** baseline | the risk rule restricted to information known before the final - the transparent alternative to ML |
| Logistic Regression | simple, linear, interpretable coefficients |
| Decision Tree (max_depth 4) | readable if-then rules |
| Random Forest (200 trees, max_depth 8, min_samples_leaf 20) | many trees vote; usually stronger, less interpretable |

All models use `class_weight="balanced"`. Classes are only mildly imbalanced (36/28/36%), but the
weighting makes sure the smallest class (Medium) isn't neglected. Hyperparameters were set once, by
reasoning, and not tuned.

### How the final model was chosen

Selection uses **5-fold cross-validation on the training set only**. The test set is touched once, for
the final report.

1. Keep every model whose CV **High-risk recall** is within 0.03 of the best.
2. Among those, choose the highest **macro F1** (ties go to the simpler model).

Why not just "highest High recall"? Logistic Regression has the best CV High recall (0.834 vs 0.812), but
only because it flags extra students almost at random: its High precision drops to 0.83 (vs 1.00) and
macro F1 to 0.69 (vs 0.83). The tolerance stops a small recall gain from causing a large loss elsewhere.

**Selected: Decision Tree.**

### Test-set results (1,960 students)

| Approach | Accuracy | Macro precision | Macro recall | Macro F1 | High precision | High recall |
|---|---|---|---|---|---|---|
| Most frequent class | 0.363 | 0.121 | 0.333 | 0.178 | 0.363 | 1.000 |
| Attendance-only rule | 0.807 | 0.852 | 0.794 | 0.800 | 1.000 | 0.773 |
| Logistic Regression | 0.688 | 0.678 | 0.675 | 0.676 | 0.834 | 0.798 |
| **Decision Tree** (selected) | **0.807** | **0.852** | **0.794** | **0.800** | **1.000** | **0.773** |
| Random Forest | 0.807 | 0.852 | 0.794 | 0.800 | 1.000 | 0.773 |
| *RF with Total_Score (leakage, reference)* | *1.000* | *0.999* | *1.000* | *1.000* | *1.000* | *1.000* |

Train vs test for the Decision Tree: macro F1 0.827 (train), 0.826 (CV), 0.800 (test). The gap is small,
so the tree isn't overfitting.

Per-class (Decision Tree, test): Low P 0.68 / R 1.00, Medium P 0.88 / R 0.61, High P 1.00 / R 0.77.

Confusion matrix (rows = true, columns = predicted):

|  | Low | Medium | High |
|---|---|---|---|
| **Low** | 695 | 0 | 0 |
| **Medium** | 216 | 337 | 0 |
| **High** | 116 | 46 | 550 |

### What this means (important for the interview)

The tree's learned rules (printed by `train_model.py`) are:

```
Attendance <= 64.99  -> High
Attendance <= 74.99  -> Medium
otherwise            -> Low
```

This is **exactly** the attendance part of the risk policy. The tree agrees with the attendance-only
rule on 100% of test students, and attendance carries 99.6% of its feature importance.

- Every student flagged High really is High (precision 1.00): attendance < 65% always means High.
- The 23% of High-risk students who are missed are those with **good attendance but Total_Score < 55**.
  No allowed feature carries any information about Total_Score (see EDA), so **no model** can find them.
- The Random Forest reaches the same result. Logistic Regression does worse because it draws smooth,
  linear boundaries and fits probabilities, which is a poor match for this sharp step-shaped rule.

**Conclusion:** on this dataset the ML models **do not beat the transparent rule**. They rediscover it.
The model is not "better" than the rule just because it scores well on labels the rule created. The
value of the ML pipeline is a reproducible, tested workflow that could learn real patterns if it were
retrained on real data where marks and engagement are related.

### Metrics in plain language

- **Accuracy**: share of all students classified correctly. Misleading on its own: a model could get
  most students right while missing many High-risk students.
- **Precision (High)**: of the students flagged High, how many really are? Low precision = unnecessary flags.
- **Recall (High)**: of the truly High-risk students, how many were flagged? Low recall = students who need
  support are **missed**.
- **F1-score**: one number that balances precision and recall.
- **Macro average**: plain average across Low/Medium/High, so each class counts equally.
- **Confusion matrix**: shows exactly which classes get mixed up.

**The trade-off:** missing a struggling student (false negative) is usually more costly than an
unnecessary check-in (false positive). Too many false alarms, though, waste advisor time and can
discourage students. That is why High recall is prioritised but not maximised at any cost.

With 1,960 test students, a metric can move by about ±2 percentage points from sampling alone, so
differences smaller than that (e.g. Decision Tree vs Random Forest) are not meaningful.

## 12. Prediction system and recommendations

**One student** (`predict_student` in `src/predict.py`, or the dashboard's *Predict a student* tab):
takes attendance, midterm, assignment, quiz, participation, project scores and prep-course status,
validates them (missing / non-numeric / out-of-range values give a clear error), and returns:

- the predicted risk level and a **confidence** (the model's probability for that class, i.e. how strongly
  the training data supports this label, **not** a probability of failing),
- a short factual summary of the indicators (e.g. *"Attendance is 62.0% (below the 65% high-risk cut-off)"*),
- recommendations.

**All students** (`predict_dataset`): predicts every row that has **all** model inputs. 1,141 rows with
any missing input are listed separately instead of being predicted from guessed values.

**Recommendations** (`src/recommendations.py`) are rule-based and depend on the risk level **and** on
which indicators are weak:

- *High*: prioritise attendance if it is low, meet a mentor/advisor, structured study plan focused on the
  weakest areas (e.g. "quizzes, midterm exam").
- *Medium*: consistent attendance, regular assignment completion, extra practice in weak areas, monthly review.
- *Low*: keep up the performance, optional small improvement area, keep preparing early.
- The test-prep course is suggested to Medium/High students who haven't taken it.

The wording is supportive and non-punitive, and it never claims to know the cause or guarantee improvement.

If `models/risk_model.pkl` is missing, every entry point stops with:
*"No trained model found ... Train it first with: python src/train_model.py"*.

## 13. Reports and dashboard

**CSV report:** `python src/predict.py` writes `reports/risk_report.csv` with exactly these columns:

```
Student,Attendance,Risk,Recommendation
STU-00001,52.29,High,"Prioritise attendance: current attendance is 52%. ..."
```

Only anonymised IDs are included. No names, emails or original IDs.

**Dashboard:** `streamlit run app.py`

| Tab | Contents |
|---|---|
| Overview | students analysed, count and % per risk level, chart, rule definition and cleaning summary |
| Data insights | key findings (computed live), attendance-vs-marks charts, distributions, heatmap |
| Model performance | comparison table, selected model's confusion matrix, learned rules, metric explanations, limitations |
| Predict a student | input form -> risk level, confidence, indicators, recommendations |
| Bulk analysis & report | predictions for all eligible students, filter, report preview, **Download risk_report.csv**, list of excluded rows |

## 14. Testing

```bat
python -m pytest -v
```

27 tests in `tests/test_pipeline.py` use small hand-made records (no dataset or saved model needed):
missing dataset file, column-name mapping, unexpected columns, the risk rule incl. boundaries and
High-over-Medium priority, cleaning (duplicates, typos, `"\t41"`, impossible values, PII removal),
invalid/empty prediction input, missing model file, single and bulk prediction, report columns and
CSV saving, and specific recommendations.

## 15. Limitations and ethical considerations

- **Proxy labels.** The target is our own rule, not real outcomes. Good scores show agreement with the
  rule, not the ability to predict failure.
- **The model adds nothing over the attendance rule here.** Total_Score is unrelated to all allowed features,
  so ~23% of High-risk students (low total, good attendance) cannot be detected before the final.
- **The data appears synthetic.** Uniform distributions, zero correlations and Grade contradicting
  Total_Score suggest the columns were generated independently. Conclusions may not carry over to real students.
- **Thresholds are a policy choice.** Different cut-offs give different labels; they should be agreed
  with academic staff.
- **Fairness.** Gender, age and department are excluded from the model. A real deployment should still
  check whether error rates differ between groups.
- **Privacy.** Names, emails and IDs are removed before analysis. Outputs use sequential pseudonymous IDs.
  Anyone holding the raw file can map them back, so reports should still be treated as confidential.
- **Human in the loop.** Outputs should start a supportive conversation, never trigger automatic penalties.

## 16. Future improvements

- Train on real data with an **observed** outcome (e.g. failing or dropping a course the next term).
- Use time-aware features (attendance and quiz trends during the term).
- Calibrate probabilities and choose the High-risk threshold together with advisors (recall vs workload).
- Add a fairness report (error rates by group) and periodic retraining/monitoring.

## 17. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pandas'` (or sklearn, streamlit...) | Activate the environment (`.venv\Scripts\activate`) and run `pip install -r requirements.txt` |
| `'streamlit' is not recognized` | Same as above, or run `python -m streamlit run app.py` |
| `ERROR: Dataset not found in '...\data'` | Put `dcs_student_data.csv` directly in `data\` (not in a sub-folder, not zipped) |
| `The dataset is missing required column(s): ...` | You are using a different CSV. Download the Kaggle file above, or rename your columns to the ones listed in [section 2](#2-dataset) |
| `No trained model found ...` | Run `python src/train_model.py` first |
| Warnings / errors when loading `risk_model.pkl` | The model was saved with scikit-learn 1.6.1. Install that version, or simply retrain |
| `ImportError: DLL load failed ... An Application Control policy has blocked this file` (Windows Smart App Control) | Windows blocked a scikit-learn binary. The pinned `scikit-learn==1.6.1` works on such systems; reinstall with `pip install --force-reinstall scikit-learn==1.6.1` |
| `python` opens the Microsoft Store | Install Python from python.org and tick "Add python.exe to PATH" |
| Dashboard shows old results after retraining | Press **R** in the browser or use the menu -> *Clear cache*, then rerun |
