"""
Loading the raw CSV and checking that it has the columns we need.

The functions raise DataError with a plain-English message instead of a
long traceback, so both the command-line scripts and the dashboard can
show the user what to fix.
"""

import re
from pathlib import Path

import pandas as pd

from config import CSV_CANDIDATES, DATA_DIR, REQUIRED_COLUMNS


class DataError(Exception):
    """Raised for problems the user can fix (missing file, wrong columns...)."""


# Some copies of the dataset use slightly different column spellings.
# Each alias is compared after lower-casing and removing spaces/underscores/
# brackets, and is mapped to the canonical name used in this project.
COLUMN_ALIASES = {
    "attendance": "Attendance (%)",
    "attendance%": "Attendance (%)",
    "attendancepercent": "Attendance (%)",
    "attendancepercentage": "Attendance (%)",
    "midtermscore": "Midterm_Score",
    "finalscore": "Final_Score",
    "assignmentsavg": "Assignments_Avg",
    "assignmentavg": "Assignments_Avg",
    "quizzesavg": "Quizzes_Avg",
    "quizavg": "Quizzes_Avg",
    "participationscore": "Participation_Score",
    "projectsscore": "Projects_Score",
    "projectscore": "Projects_Score",
    "totalscore": "Total_Score",
    "testpreparationcourse": "test_preparation_course",
    "testprep": "test_preparation_course",
    "studentid": "Student_ID",
    "firstname": "First_Name",
    "lastname": "Last_Name",
    "email": "Email",
    "gender": "Gender",
    "age": "Age",
    "department": "Department",
    "grade": "Grade",
    "mathscore": "math_score",
    "readingscore": "reading_score",
    "writingscore": "writing_score",
    "sciencescore": "science_score",
}


def _simplify(name: str) -> str:
    """'Attendance (%)' -> 'attendance%', ' Total_Score ' -> 'totalscore'."""
    return re.sub(r"[\s_()\-]", "", str(name)).lower()


def find_csv(data_dir: Path = DATA_DIR) -> Path:
    """Return the path of the dataset CSV inside data/, or raise DataError."""
    for name in CSV_CANDIDATES:
        path = data_dir / name
        if path.exists():
            return path

    # Fall back to "the only CSV in the folder" so a renamed file still works.
    csv_files = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []
    if len(csv_files) == 1:
        return csv_files[0]

    raise DataError(
        f"Dataset not found in '{data_dir}'. Download it from "
        "https://www.kaggle.com/datasets/ganeshkumarofficial/student-dataset, "
        "unzip it and place 'dcs_student_data.csv' in the data/ folder."
    )


def standardize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename columns to canonical names. Returns (df, {old_name: new_name})."""
    mapping = {}
    for col in df.columns:
        canonical = COLUMN_ALIASES.get(_simplify(col))
        if canonical and canonical != col:
            mapping[col] = canonical
    return df.rename(columns=mapping), mapping


def validate_columns(df: pd.DataFrame, required=REQUIRED_COLUMNS) -> None:
    """Raise DataError if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(
            "The dataset is missing required column(s): "
            + ", ".join(missing)
            + ". Found columns: "
            + ", ".join(map(str, df.columns))
        )


def load_raw_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the CSV, standardise column names and validate them."""
    path = Path(path) if path else find_csv()
    if not path.exists():
        raise DataError(f"File not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # e.g. not a CSV, unreadable encoding
        raise DataError(f"Could not read '{path.name}' as a CSV file: {exc}") from exc

    if df.empty:
        raise DataError(f"'{path.name}' contains no rows.")

    df, mapping = standardize_columns(df)
    if mapping:
        print(f"Renamed columns: {mapping}")
    validate_columns(df)
    return df


if __name__ == "__main__":
    data = load_raw_data()
    print(f"Loaded {len(data)} rows x {data.shape[1]} columns")
    print(data.dtypes)
