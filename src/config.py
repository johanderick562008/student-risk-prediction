"""
Central configuration for the Student Academic Risk Prediction System.

Everything that you might want to change (file paths, risk thresholds,
feature lists, valid value ranges) lives here, so it only has to be
changed in ONE place.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (built relative to the project root so scripts work from any folder)
# ---------------------------------------------------------------------------
# os.path.abspath (instead of Path.resolve) avoids following Windows
# redirections that can produce paths longer than the 260-character limit.
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# The Kaggle download contains "dcs_student_data.csv". "student_dataset.csv"
# is also accepted so the file can be renamed without touching code.
CSV_CANDIDATES = ["dcs_student_data.csv", "student_dataset.csv"]

MODEL_PATH = MODELS_DIR / "risk_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
REPORT_PATH = REPORTS_DIR / "risk_report.csv"

RANDOM_STATE = 42  # used everywhere randomness is involved -> reproducible
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Risk-label thresholds (the "risk policy")
# ---------------------------------------------------------------------------
# Rules (applied in this order, so High always wins over Medium):
#   High   : attendance < high_attendance   OR total < high_total
#   Medium : attendance < medium_attendance OR total < medium_total
#   Low    : everything else
#
# The task brief proposed high_total = 50. In this dataset Total_Score ranges
# from 50.02 to 99.99, so "Total_Score < 50" matches ZERO students and the
# High rule would silently become attendance-only. We therefore raise it to
# 55 (roughly the bottom 10% of Total_Score). The original value is kept
# below for reference and comparison in the EDA/README.
RISK_THRESHOLDS = {
    "high_attendance": 65.0,
    "high_total": 55.0,
    "medium_attendance": 75.0,
    "medium_total": 65.0,
}
ORIGINAL_PROPOSED_THRESHOLDS = {**RISK_THRESHOLDS, "high_total": 50.0}

RISK_ORDER = ["Low", "Medium", "High"]

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
# Personal identifiers: never used as features and never shown in outputs.
PII_COLUMNS = ["Student_ID", "First_Name", "Last_Name", "Email"]

# Columns used to BUILD the target. They must never be model inputs.
TARGET_SOURCE_COLUMNS = ["Attendance (%)", "Total_Score"]

# Model input features: information plausibly available BEFORE the final
# exam. Attendance is both a feature and part of the label (see README,
# "Feature selection and leakage prevention").
NUMERIC_FEATURES = [
    "Attendance (%)",
    "Midterm_Score",
    "Assignments_Avg",
    "Quizzes_Avg",
    "Participation_Score",
    "Projects_Score",
]
CATEGORICAL_FEATURES = ["test_preparation_course"]  # 0 = not completed, 1 = completed
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns the rest of the code expects after loading.
REQUIRED_COLUMNS = MODEL_FEATURES + ["Total_Score"]

# Valid ranges used to detect impossible values during cleaning.
VALID_RANGES = {
    "Age": (15, 60),
    "Attendance (%)": (0, 100),
    "Midterm_Score": (0, 100),
    "Final_Score": (0, 100),
    "Assignments_Avg": (0, 100),
    "Quizzes_Avg": (0, 100),
    "Participation_Score": (0, 10),  # participation is on a 0-10 scale
    "Projects_Score": (0, 100),
    "Total_Score": (0, 100),
    "math_score": (0, 100),
    "reading_score": (0, 100),
    "writing_score": (0, 100),
    "science_score": (0, 100),
}

# Friendly names used in charts, the dashboard and explanations.
DISPLAY_NAMES = {
    "Attendance (%)": "Attendance (%)",
    "Midterm_Score": "Midterm score",
    "Final_Score": "Final score",
    "Assignments_Avg": "Assignment average",
    "Quizzes_Avg": "Quiz average",
    "Participation_Score": "Participation (0-10)",
    "Projects_Score": "Project score",
    "Total_Score": "Total score",
    "test_preparation_course": "Test-prep course",
}

# Restrained colour palette. Risk colours are only ever shown next to a text
# label, so meaning never relies on colour alone.
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}
PRIMARY_COLOR = "#2a78d6"
SECONDARY_COLOR = "#eb6834"
