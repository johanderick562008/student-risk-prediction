"""
Data cleaning and the scikit-learn preprocessing pipeline.

Two separate stages:

1. clean_data()        - rule-based fixes that do not "learn" anything from
                         the data (duplicates, typos, impossible values).
                         Safe to run on the full dataset before splitting.
2. build_preprocessor() - imputation, scaling and encoding. These LEARN
                         statistics (medians, means...), so they live inside
                         a Pipeline and are fitted on the training set only.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    PII_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
    VALID_RANGES,
)

# Inconsistent spellings found in the real data, e.g. " engineering ",
# "BUSINESS", "CS" vs "Computer Science", "Math" vs "Mathematics".
DEPARTMENT_MAP = {
    "cs": "Computer Science",
    "computer science": "Computer Science",
    "engineering": "Engineering",
    "business": "Business",
    "math": "Mathematics",
    "mathematics": "Mathematics",
}

NUMERIC_COLUMNS = [
    "Age", "Attendance (%)", "Midterm_Score", "Final_Score", "Assignments_Avg",
    "Quizzes_Avg", "Participation_Score", "Projects_Score", "Total_Score",
    "math_score", "reading_score", "writing_score", "science_score",
]


def _to_prep_flag(value):
    """Turn 0/1, 'yes'/'no', 'completed'/'none' into 1/0 (or NA if unknown)."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    if text in {"1", "1.0", "yes", "true", "completed"}:
        return 1
    if text in {"0", "0.0", "no", "false", "none", "not completed"}:
        return 0
    return pd.NA


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean the raw dataset. Returns (clean_df, log) where log records what
    was changed so it can be reported in the EDA/README.

    The cleaned frame gets an anonymous 'Anon_ID' (STU-00001, ...) and the
    personal columns (ID, names, email) are removed.
    """
    df = raw.copy()
    log = {"rows_before": len(df)}

    # 1. Exact duplicate rows: the same student recorded twice adds no
    #    information and would be over-weighted, so keep one copy.
    log["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Categorical clean-up: strip spaces and unify spelling/case.
    if "Gender" in df:
        df["Gender"] = df["Gender"].str.strip().str.title()
    if "Department" in df:
        dept = df["Department"].str.strip().str.lower()
        df["Department"] = dept.map(DEPARTMENT_MAP).fillna(dept.str.title())
    if "Grade" in df:
        df["Grade"] = df["Grade"].str.strip().str.upper()
    df["test_preparation_course"] = (
        df["test_preparation_course"].map(_to_prep_flag).astype("Float64")
    )

    # 3. Numeric conversion. Some cells contain stray characters (the real
    #    file has a math_score of "\t41"); strip them, then convert.
    #    Anything still not a number becomes NaN.
    log["invalid_values_set_to_missing"] = {}
    for col in NUMERIC_COLUMNS:
        if col not in df:
            continue
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")

        # 4. Impossible values (negative attendance, 145% midterm, age 87...)
        #    are treated as data-entry errors and set to missing rather than
        #    guessed or clipped.
        low, high = VALID_RANGES[col]
        bad = (df[col] < low) | (df[col] > high)
        if bad.any():
            log["invalid_values_set_to_missing"][col] = int(bad.sum())
            df.loc[bad, col] = float("nan")

    # 5. Anonymise: sequential IDs after de-duplication, then drop PII.
    df.insert(0, "Anon_ID", [f"STU-{i:05d}" for i in range(1, len(df) + 1)])
    df = df.drop(columns=[c for c in PII_COLUMNS if c in df])

    log["rows_after"] = len(df)
    log["missing_after_cleaning"] = {
        k: int(v) for k, v in df.isna().sum().items() if v > 0
    }
    return df, log


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """
    Preprocessing applied inside every model pipeline:

    - numeric: fill missing values with the TRAINING median (robust to
      outliers), then (optionally) standardise to mean 0, std 1.
      Scaling matters for Logistic Regression, whose coefficients and
      regularisation depend on feature scale. Trees split on one feature
      at a time and are unaffected by scaling, so we skip it for them -
      that keeps their learned rules readable in real units (e.g. 65%).
    - categorical (test-prep flag): fill missing with the most common value,
      then one-hot encode. handle_unknown='ignore' means an unexpected value
      at prediction time does not crash the model.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])
    return ColumnTransformer(
        [("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
         ("cat", categorical, CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,  # keep plain column names
    )


def split_data(df: pd.DataFrame, label_col: str = "Risk_Label"):
    """
    Stratified train/test split on the labelled rows.

    Stratifying keeps the Low/Medium/High proportions the same in both sets.
    The split happens BEFORE any imputer/scaler is fitted, which prevents
    information from the test set leaking into training.
    """
    labelled = df.dropna(subset=[label_col])
    X = labelled[MODEL_FEATURES].astype(float)
    y = labelled[label_col]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
