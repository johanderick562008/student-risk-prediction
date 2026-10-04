"""
Transparent, rule-based risk labels.

IMPORTANT: these labels are PROXY labels. They are computed from attendance
and Total_Score using a policy we chose. They are not observed future
outcomes (e.g. a student actually failing). A model trained on them learns
to reproduce this policy - it does not prove it can predict failure.
"""

import math

import pandas as pd

from config import RISK_THRESHOLDS


def assign_risk_label(attendance, total_score, thresholds=RISK_THRESHOLDS):
    """
    Return 'High', 'Medium' or 'Low' for one student, or None when either
    input is missing (we do not guess a label from incomplete data).

    The checks run in order High -> Medium -> Low, so a student who meets
    a High condition can never fall through to Medium.
    """
    if attendance is None or total_score is None:
        return None
    if (isinstance(attendance, float) and math.isnan(attendance)) or (
        isinstance(total_score, float) and math.isnan(total_score)
    ):
        return None

    t = thresholds
    if attendance < t["high_attendance"] or total_score < t["high_total"]:
        return "High"
    if attendance < t["medium_attendance"] or total_score < t["medium_total"]:
        return "Medium"
    return "Low"


def add_risk_labels(df: pd.DataFrame, thresholds=RISK_THRESHOLDS,
                    label_col: str = "Risk_Label") -> pd.DataFrame:
    """Return a copy of df with a Risk_Label column added."""
    out = df.copy()
    out[label_col] = [
        assign_risk_label(a, s, thresholds)
        for a, s in zip(out["Attendance (%)"].astype(float),
                        out["Total_Score"].astype(float))
    ]
    return out


def attendance_only_rule(attendance, thresholds=RISK_THRESHOLDS):
    """
    The risk policy restricted to the information available BEFORE the
    final assessment (attendance only, because Total_Score is not known
    yet). Used as a transparent baseline to compare the ML models against.
    """
    if attendance is None or (isinstance(attendance, float) and math.isnan(attendance)):
        return None
    if attendance < thresholds["high_attendance"]:
        return "High"
    if attendance < thresholds["medium_attendance"]:
        return "Medium"
    return "Low"


def label_distribution(labels: pd.Series) -> pd.DataFrame:
    """Counts and percentages per risk level (Low, Medium, High order)."""
    counts = labels.value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    return pd.DataFrame({
        "count": counts,
        "percent": (100 * counts / counts.sum()).round(1),
    })


if __name__ == "__main__":
    examples = [(90, 80), (70, 80), (90, 60), (60, 90), (90, 52), (None, 70)]
    for att, total in examples:
        print(f"attendance={att}, total={total} -> {assign_risk_label(att, total)}")
