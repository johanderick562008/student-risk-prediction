"""
Predictions for one student or for the whole dataset, plus the CSV report.

Run:  python src/predict.py
      -> predicts every eligible student and writes reports/risk_report.csv
"""

import json
import math

import joblib
import pandas as pd

from config import (
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    REPORT_PATH,
    VALID_RANGES,
)
from recommendations import explain_indicators, generate_recommendations


class ModelNotFoundError(Exception):
    """Raised when the trained model file does not exist yet."""


class InputError(Exception):
    """Raised when prediction input is missing or invalid."""


def load_model():
    """Return (pipeline, metadata) or raise a clear ModelNotFoundError."""
    if not MODEL_PATH.exists():
        raise ModelNotFoundError(
            f"No trained model found at '{MODEL_PATH}'. "
            "Train it first with:  python src/train_model.py")
    model = joblib.load(MODEL_PATH)
    metadata = {}
    if MODEL_METADATA_PATH.exists():
        metadata = json.loads(MODEL_METADATA_PATH.read_text())
    return model, metadata


def validate_input(student: dict) -> dict:
    """
    Check one student's inputs. Returns a clean {feature: float} dict or
    raises InputError listing every problem at once.
    """
    if not student:
        raise InputError("No input provided.")
    clean, problems = {}, []
    for col in MODEL_FEATURES:
        value = student.get(col)
        if value is None or (isinstance(value, str) and not value.strip()):
            problems.append(f"'{col}' is missing")
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            problems.append(f"'{col}' must be a number (got {value!r})")
            continue
        if math.isnan(number):
            problems.append(f"'{col}' is missing")
            continue
        if col == "test_preparation_course":
            if number not in (0, 1):
                problems.append("'test_preparation_course' must be 0 (no) or 1 (yes)")
        else:
            low, high = VALID_RANGES[col]
            if not low <= number <= high:
                problems.append(f"'{col}' must be between {low} and {high} (got {number})")
        clean[col] = number
    if problems:
        raise InputError("Invalid input: " + "; ".join(problems) + ".")
    return clean


def predict_student(student: dict, model=None) -> dict:
    """
    Predict the risk level for one student.

    Returns: risk, confidence (probability of the predicted class),
    probabilities for every class, indicator summary and recommendations.
    """
    if model is None:
        model, _ = load_model()
    clean = validate_input(student)
    X = pd.DataFrame([clean], columns=MODEL_FEATURES)
    risk = model.predict(X)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0].round(3)))
    return {
        "risk": risk,
        "confidence": float(proba[risk]),
        "probabilities": {k: float(v) for k, v in proba.items()},
        "indicators": explain_indicators(clean),
        "recommendations": generate_recommendations(risk, clean),
    }


def predict_dataset(df: pd.DataFrame, model=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predict every eligible row of a CLEANED dataframe.

    Only rows with ALL model inputs present are predicted: we do not want to
    tell a real student they are at risk based on guessed (imputed) values.
    Returns (predictions, incomplete_rows).
    """
    if model is None:
        model, _ = load_model()
    missing_cols = [c for c in MODEL_FEATURES if c not in df]
    if missing_cols:
        raise InputError("Dataset is missing required feature(s): " + ", ".join(missing_cols))

    complete = df[MODEL_FEATURES].notna().all(axis=1)
    eligible = df[complete].copy()
    incomplete = df.loc[~complete, ["Anon_ID"] + MODEL_FEATURES].copy()
    incomplete["Missing"] = incomplete[MODEL_FEATURES].isna().apply(
        lambda row: ", ".join(row.index[row]), axis=1)

    X = eligible[MODEL_FEATURES].astype(float)
    eligible["Predicted_Risk"] = model.predict(X)
    eligible["Confidence"] = model.predict_proba(X).max(axis=1).round(3)
    eligible["Recommendation"] = [
        " ".join(generate_recommendations(risk, row))
        for risk, row in zip(eligible["Predicted_Risk"], X.to_dict("records"))
    ]
    return eligible, incomplete


def build_report(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report with exactly the requested columns: Student | Attendance | Risk | Recommendation."""
    return pd.DataFrame({
        "Student": predictions["Anon_ID"],
        "Attendance": predictions["Attendance (%)"].round(2),
        "Risk": predictions["Predicted_Risk"],
        "Recommendation": predictions["Recommendation"],
    }).reset_index(drop=True)


def save_report(report: pd.DataFrame, path=REPORT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    return path


def main():
    from data_loader import load_raw_data
    from preprocessing import clean_data

    model, metadata = load_model()
    df, _ = clean_data(load_raw_data())
    predictions, incomplete = predict_dataset(df, model)
    report = build_report(predictions)
    path = save_report(report)

    print(f"Model: {metadata.get('selected_model', 'unknown')}")
    print(f"Predicted {len(report)} students; skipped {len(incomplete)} incomplete rows.")
    print(report["Risk"].value_counts().reindex(["Low", "Medium", "High"]).to_string())
    print(f"\nReport saved to {path}\n")
    print(report.head(5).to_string())

    example = {"Attendance (%)": 62, "Midterm_Score": 55, "Assignments_Avg": 78,
               "Quizzes_Avg": 60, "Participation_Score": 4, "Projects_Score": 81,
               "test_preparation_course": 0}
    result = predict_student(example, model)
    print(f"\nExample student -> {result['risk']} risk (confidence {result['confidence']:.0%})")
    for line in result["indicators"]:
        print("  *", line)
    for rec in result["recommendations"]:
        print("  -", rec)


if __name__ == "__main__":
    from data_loader import DataError
    try:
        main()
    except (DataError, ModelNotFoundError, InputError) as err:
        print(f"ERROR: {err}")
        raise SystemExit(1)
