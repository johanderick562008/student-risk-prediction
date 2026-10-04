"""
Basic reliability tests using small hand-made records.

Run from the project folder:   python -m pytest -v
(The model tests train a tiny model on the fly, so they do not need the
real dataset or a saved model file.)
"""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import predict
from config import MODEL_FEATURES
from data_loader import DataError, find_csv, load_raw_data, standardize_columns, validate_columns
from predict import (
    InputError,
    ModelNotFoundError,
    build_report,
    predict_dataset,
    predict_student,
    save_report,
    validate_input,
)
from preprocessing import build_preprocessor, clean_data
from recommendations import generate_recommendations
from risk_labeling import add_risk_labels, assign_risk_label, attendance_only_rule


def make_raw(rows):
    """Build a small raw-style dataframe with every expected column."""
    base = {
        "Student_ID": "S1", "First_Name": "A", "Last_Name": "B", "Email": "a@b.com",
        "Gender": "Male", "Age": 20, "Department": "CS", "Attendance (%)": 80,
        "Midterm_Score": 70, "Final_Score": 70, "Assignments_Avg": 70,
        "Quizzes_Avg": 70, "Participation_Score": 6, "Projects_Score": 70,
        "Total_Score": 70, "Grade": "B", "test_preparation_course": 1,
        "math_score": "70", "reading_score": 70, "writing_score": 70, "science_score": 70,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


GOOD_STUDENT = {"Attendance (%)": 90, "Midterm_Score": 80, "Assignments_Avg": 85,
                "Quizzes_Avg": 82, "Participation_Score": 8, "Projects_Score": 88,
                "test_preparation_course": 1}


@pytest.fixture
def tiny_model():
    """A small pipeline trained on rule-labelled synthetic rows."""
    rows = []
    for att in range(50, 101, 2):
        rows.append({**GOOD_STUDENT, "Attendance (%)": att,
                     "Risk": attendance_only_rule(float(att))})
    data = pd.DataFrame(rows)
    pipe = Pipeline([("preprocess", build_preprocessor(scale=False)),
                     ("model", DecisionTreeClassifier(max_depth=3, random_state=0))])
    pipe.fit(data[MODEL_FEATURES], data["Risk"])
    return pipe


# --- Risk labelling ----------------------------------------------------------
@pytest.mark.parametrize("attendance,total,expected", [
    (90, 80, "Low"),
    (75, 65, "Low"),       # exactly on the thresholds counts as Low
    (70, 80, "Medium"),    # attendance below 75
    (90, 60, "Medium"),    # total below 65
    (60, 90, "High"),      # attendance below 65
    (90, 52, "High"),      # total below 55
    (60, 60, "High"),      # meets both High and Medium -> High wins
])
def test_assign_risk_label(attendance, total, expected):
    assert assign_risk_label(attendance, total) == expected


def test_missing_inputs_give_no_label():
    assert assign_risk_label(None, 70) is None
    assert assign_risk_label(float("nan"), 70) is None


# --- Loading / column checks ---------------------------------------------------
def test_missing_dataset_file(tmp_path):
    with pytest.raises(DataError, match="Dataset not found"):
        find_csv(tmp_path)


def test_anonymized_copy_is_used_when_raw_file_missing(tmp_path):
    (tmp_path / "students_anonymized.csv").write_text("a\n1\n")
    assert find_csv(tmp_path).name == "students_anonymized.csv"
    (tmp_path / "dcs_student_data.csv").write_text("a\n1\n")
    assert find_csv(tmp_path).name == "dcs_student_data.csv"  # raw file wins


def test_column_aliases_are_mapped():
    df = pd.DataFrame(columns=["Attendance", "midterm score", "Total Score"])
    renamed, mapping = standardize_columns(df)
    assert list(renamed.columns) == ["Attendance (%)", "Midterm_Score", "Total_Score"]
    assert len(mapping) == 3


def test_unexpected_columns_raise():
    with pytest.raises(DataError, match="missing required column"):
        validate_columns(pd.DataFrame(columns=["foo", "bar"]))


def test_load_raw_data_with_wrong_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
    with pytest.raises(DataError):
        load_raw_data(path)


# --- Cleaning ------------------------------------------------------------------
def test_clean_data_fixes_known_problems():
    raw = make_raw([
        {"Department": " engineering ", "Gender": " FEMALE ", "math_score": "\t41"},
        {"Attendance (%)": 135, "Age": -3},      # impossible values
        {"Attendance (%)": 135, "Age": -3},      # exact duplicate of the row above
    ])
    clean, log = clean_data(raw)
    assert log["duplicates_removed"] == 1
    assert clean.loc[0, "Department"] == "Engineering"
    assert clean.loc[0, "Gender"] == "Female"
    assert clean.loc[0, "math_score"] == 41
    assert pd.isna(clean.loc[1, "Attendance (%)"]) and pd.isna(clean.loc[1, "Age"])
    # personal data removed, anonymous ID added
    assert "Email" not in clean and "First_Name" not in clean
    assert list(clean["Anon_ID"]) == ["STU-00001", "STU-00002"]


def test_labels_added_after_cleaning():
    clean, _ = clean_data(make_raw([{"Attendance (%)": 60}, {"Attendance (%)": None}]))
    labelled = add_risk_labels(clean)
    assert labelled.loc[0, "Risk_Label"] == "High"
    assert pd.isna(labelled.loc[1, "Risk_Label"])  # no attendance -> no label


# --- Input validation ----------------------------------------------------------
def test_validate_input_accepts_good_values():
    assert validate_input({k: str(v) for k, v in GOOD_STUDENT.items()})["Midterm_Score"] == 80


@pytest.mark.parametrize("bad,message", [
    ({}, "No input"),
    ({**GOOD_STUDENT, "Attendance (%)": None}, "missing"),
    ({**GOOD_STUDENT, "Attendance (%)": ""}, "missing"),
    ({**GOOD_STUDENT, "Midterm_Score": "abc"}, "must be a number"),
    ({**GOOD_STUDENT, "Attendance (%)": 120}, "between"),
    ({**GOOD_STUDENT, "Participation_Score": 11}, "between"),
    ({**GOOD_STUDENT, "test_preparation_course": 2}, "0 \\(no\\) or 1"),
])
def test_validate_input_rejects_bad_values(bad, message):
    with pytest.raises(InputError, match=message):
        validate_input(bad)


# --- Prediction + report -------------------------------------------------------
def test_missing_model_file(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "nope.pkl")
    with pytest.raises(ModelNotFoundError, match="train_model.py"):
        predict.load_model()


def test_predict_student(tiny_model):
    good = predict_student(GOOD_STUDENT, tiny_model)
    assert good["risk"] == "Low"
    assert 0 <= good["confidence"] <= 1
    assert good["recommendations"]

    weak = predict_student({**GOOD_STUDENT, "Attendance (%)": 55}, tiny_model)
    assert weak["risk"] == "High"
    assert any("attendance" in r.lower() for r in weak["recommendations"])


def test_bulk_prediction_and_report(tiny_model, tmp_path):
    clean, _ = clean_data(make_raw([
        {"Attendance (%)": 95}, {"Attendance (%)": 55}, {"Midterm_Score": None},
    ]))
    preds, incomplete = predict_dataset(clean, tiny_model)
    assert len(preds) == 2 and len(incomplete) == 1
    assert incomplete.iloc[0]["Missing"] == "Midterm_Score"

    report = build_report(preds)
    assert list(report.columns) == ["Student", "Attendance", "Risk", "Recommendation"]
    assert report["Student"].str.startswith("STU-").all()

    path = save_report(report, tmp_path / "risk_report.csv")
    assert path.exists()
    assert len(pd.read_csv(path)) == 2


def test_recommendations_are_specific():
    weak_quiz = {**GOOD_STUDENT, "Quizzes_Avg": 50}
    recs = " ".join(generate_recommendations("Medium", weak_quiz))
    assert "quizzes" in recs
    assert "attendance" not in recs.lower()  # attendance is fine -> not prioritised
