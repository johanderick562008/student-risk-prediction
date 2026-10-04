"""
Rule-based, supportive recommendations.

The recommendations depend on the predicted risk level AND on which
academic indicators look weak, so two High-risk students can get different
advice. They are suggestions for a conversation with the student, not a
diagnosis: the model does not know WHY a student's numbers look the way
they do, and no recommendation guarantees improvement.
"""

from config import RISK_THRESHOLDS

# Below these values an indicator is described as an area to work on.
WEAK_LIMITS = {
    "Midterm_Score": 60,
    "Assignments_Avg": 65,
    "Quizzes_Avg": 65,
    "Projects_Score": 65,
    "Participation_Score": 5,  # 0-10 scale
}
STRONG_LIMITS = {
    "Midterm_Score": 85,
    "Assignments_Avg": 85,
    "Quizzes_Avg": 85,
    "Projects_Score": 85,
    "Participation_Score": 8,
}
AREA_NAMES = {
    "Midterm_Score": "midterm exam",
    "Assignments_Avg": "assignments",
    "Quizzes_Avg": "quizzes",
    "Projects_Score": "projects",
    "Participation_Score": "class participation",
}


def weak_areas(student: dict) -> list[str]:
    """Indicators below their 'weak' limit, weakest first (relative to limit)."""
    gaps = []
    for col, limit in WEAK_LIMITS.items():
        value = student.get(col)
        if value is not None and value < limit:
            gaps.append((value / limit, col))
    return [col for _, col in sorted(gaps)]


def explain_indicators(student: dict) -> list[str]:
    """Short, factual summary of the available indicators (no causal claims)."""
    lines = []
    att = student.get("Attendance (%)")
    if att is not None:
        if att < RISK_THRESHOLDS["high_attendance"]:
            status = f"below the {RISK_THRESHOLDS['high_attendance']:.0f}% high-risk cut-off"
        elif att < RISK_THRESHOLDS["medium_attendance"]:
            status = f"below the {RISK_THRESHOLDS['medium_attendance']:.0f}% target"
        else:
            status = f"meets the {RISK_THRESHOLDS['medium_attendance']:.0f}% target"
        lines.append(f"Attendance is {att:.1f}% ({status}).")

    for col in weak_areas(student):
        scale = "/10" if col == "Participation_Score" else "/100"
        lines.append(f"{AREA_NAMES[col].capitalize()} score is {student[col]:.1f}{scale} "
                     f"(below {WEAK_LIMITS[col]}{scale}).")

    strengths = [AREA_NAMES[c] for c, lim in STRONG_LIMITS.items()
                 if student.get(c) is not None and student[c] >= lim]
    if strengths:
        lines.append("Strengths: " + ", ".join(strengths) + ".")

    prep = student.get("test_preparation_course")
    if prep is not None:
        lines.append("Test-preparation course: " + ("completed." if prep == 1 else "not completed."))
    return lines


def generate_recommendations(risk: str, student: dict) -> list[str]:
    """Return a short list of supportive, practical recommendations."""
    recs = []
    att = student.get("Attendance (%)")
    low_attendance = att is not None and att < RISK_THRESHOLDS["medium_attendance"]
    weak = [AREA_NAMES[c] for c in weak_areas(student)]

    if risk == "High":
        if low_attendance:
            recs.append(f"Prioritise attendance: current attendance is {att:.0f}%. "
                        "Aim for at least 75% and talk to a mentor about anything "
                        "that makes attending difficult.")
        recs.append("Arrange a meeting with a faculty mentor or academic advisor "
                    "to agree on a support plan.")
        if weak:
            recs.append("Build a structured weekly study plan focused on: "
                        + ", ".join(weak[:3]) + ".")
        else:
            recs.append("Build a structured weekly study plan and check progress "
                        "with the mentor every two weeks.")

    elif risk == "Medium":
        if low_attendance:
            recs.append(f"Attend classes consistently (currently {att:.0f}%; target 75%+).")
        recs.append("Keep up regular assignment completion and submit work on time.")
        if weak:
            recs.append("Do additional practice in: " + ", ".join(weak[:3]) + ".")
        recs.append("Schedule a short progress review with a mentor each month.")

    else:  # Low
        recs.append("Keep up the current, consistent performance.")
        if weak:
            recs.append("Small improvement opportunity: " + ", ".join(weak[:2]) + ".")
        recs.append("Continue participating in class and prepare early for "
                    "upcoming assessments.")

    if risk != "Low" and student.get("test_preparation_course") == 0:
        recs.append("Consider joining the test-preparation course if it is available.")
    return recs


if __name__ == "__main__":
    example = {"Attendance (%)": 58, "Midterm_Score": 52, "Assignments_Avg": 80,
               "Quizzes_Avg": 61, "Participation_Score": 3.5, "Projects_Score": 88,
               "test_preparation_course": 0}
    print("\n".join(explain_indicators(example)))
    print()
    print("\n".join("- " + r for r in generate_recommendations("High", example)))
