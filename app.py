"""
Streamlit dashboard for the Student Academic Risk Prediction System.

Run:  streamlit run app.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import matplotlib

matplotlib.use("Agg")  # render charts off-screen for Streamlit

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import eda
from config import (
    METRICS_PATH,
    RISK_COLORS,
    RISK_ORDER,
    RISK_THRESHOLDS,
)
from data_loader import DataError, load_raw_data
from evaluate_model import plot_confusion_matrix
from predict import (
    InputError,
    ModelNotFoundError,
    build_report,
    load_model,
    predict_dataset,
    predict_student,
    save_report,
)
from preprocessing import clean_data
from risk_labeling import add_risk_labels, label_distribution

st.set_page_config(page_title="Student Academic Risk", page_icon="🎓", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders (run once, then reused on every interaction)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning the dataset...")
def get_data():
    raw = load_raw_data()
    clean, log = clean_data(raw)
    return add_risk_labels(clean), log


@st.cache_resource(show_spinner="Loading the trained model...")
def get_model():
    return load_model()


@st.cache_data
def get_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


@st.cache_data(show_spinner="Predicting risk for every eligible student...")
def get_bulk_predictions(_model, df):
    return predict_dataset(df, _model)


def show_fig(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def risk_badge(risk: str) -> str:
    """Coloured label; the text itself always states the level."""
    return (f"<span style='background:{RISK_COLORS[risk]};color:#111;padding:4px 14px;"
            f"border-radius:6px;font-weight:600;font-size:1.3rem'>{risk} risk</span>")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎓 Student Academic Risk Prediction")
st.info(
    "**Data-driven support indicator, not a final judgement of any student.** "
    "Risk levels are based on a transparent rule (attendance and total score) and are "
    "meant to start a supportive conversation. They are *proxy labels*, not observed "
    "future outcomes.",
    icon="ℹ️",
)

try:
    df, cleaning_log = get_data()
except DataError as err:
    st.error(f"**Could not load the dataset.** {err}")
    st.stop()

model, metadata, model_error = None, {}, None
try:
    model, metadata = get_model()
except ModelNotFoundError as err:
    model_error = str(err)

metrics = get_metrics()

tab_overview, tab_insights, tab_model, tab_predict, tab_bulk = st.tabs(
    ["Overview", "Data insights", "Model performance", "Predict a student", "Bulk analysis & report"])


# ---------------------------------------------------------------------------
# A. Overview
# ---------------------------------------------------------------------------
with tab_overview:
    labelled = df["Risk_Label"].dropna()
    dist = label_distribution(labelled)

    st.subheader("Students analysed")
    cols = st.columns(4)
    cols[0].metric("Students (after cleaning)", f"{len(df):,}")
    for col, level in zip(cols[1:], RISK_ORDER):
        col.metric(f"{level} risk",
                   f"{dist.loc[level, 'count']:,} ({dist.loc[level, 'percent']}%)")

    invalid_text = ", ".join(f"{col}: {n}" for col, n in
                             cleaning_log["invalid_values_set_to_missing"].items())
    st.caption(f"Percentages are of the {len(labelled):,} students with enough data to be labelled.")

    left, right = st.columns([3, 2])
    with left:
        show_fig(eda.plot_risk_distribution(labelled, "Rule-based risk levels"))
    with right:
        t = RISK_THRESHOLDS
        st.markdown(f"""
**How risk is defined** (checked in this order):

- **High**: attendance < {t['high_attendance']:.0f}% **or** total score < {t['high_total']:.0f}
- **Medium**: attendance < {t['medium_attendance']:.0f}% **or** total score < {t['medium_total']:.0f}
- **Low**: everything else

The brief suggested *total score < 50* for High risk, but in this dataset the lowest
total score is **{df['Total_Score'].min():.2f}**, so that rule would flag nobody.
It was raised to 55 (about the bottom 10% of total scores).

**Data cleaning:** {cleaning_log['duplicates_removed']} duplicate rows removed,
impossible values set to missing ({invalid_text}),
{int(df['Risk_Label'].isna().sum())} students without attendance could not be labelled.
""")


# ---------------------------------------------------------------------------
# B. Data insights
# ---------------------------------------------------------------------------
with tab_insights:
    st.subheader("Key observations from the dataset")
    for finding in eda.key_findings(df):
        st.markdown(f"- {finding}")

    st.subheader("Attendance vs marks")
    c1, c2 = st.columns(2)
    with c1:
        show_fig(eda.plot_attendance_vs(df, "Total_Score"))
    with c2:
        show_fig(eda.plot_attendance_vs(df, "Final_Score"))
    show_fig(eda.plot_attendance_bands(df))

    st.subheader("Score distributions")
    show_fig(eda.plot_score_distributions(df))
    c1, c2 = st.columns(2)
    with c1:
        show_fig(eda.plot_attendance_distribution(df))
    with c2:
        show_fig(eda.plot_grade_vs_total(df))

    st.subheader("Correlation heatmap")
    c1, _ = st.columns([3, 1])
    with c1:
        show_fig(eda.plot_correlation_heatmap(df))


# ---------------------------------------------------------------------------
# C. Model performance
# ---------------------------------------------------------------------------
with tab_model:
    if metrics is None:
        st.warning("No metrics found. Train the model first:  `python src/train_model.py`")
    else:
        selected = metrics["selected_model"]
        st.subheader(f"Model comparison (test set) - selected: {selected}")

        rows = []
        for name, res in metrics["models"].items():
            row = {"Approach": name}
            row.update({k.replace("_", " ").title(): v for k, v in res["test"].items()})
            rows.append(row)
        table = pd.DataFrame(rows).set_index("Approach")
        st.dataframe(table.style.format("{:.3f}"), width="stretch")
        st.caption(
            "All numbers are on the held-out test set "
            f"({metadata.get('test_rows', '?')} students). The final model was chosen with "
            "5-fold cross-validation on the training set: models within 0.03 of the best "
            "High-risk recall were shortlisted, then the highest macro F1 was picked. "
            "The last row deliberately includes Total_Score to show what leakage looks like.")

        c1, c2 = st.columns([2, 3])
        with c1:
            show_fig(plot_confusion_matrix(metrics["models"][selected]["confusion_test"],
                                           f"{selected} - test set"))
        with c2:
            sel = metrics["models"][selected]
            st.markdown(f"""
**Train vs test** (macro F1): {sel['train']['macro_f1']:.3f} train,
{sel['test']['macro_f1']:.3f} test. The small gap means the model is not memorising.

**Agreement with the attendance-only rule:** {sel['agreement_with_rule_test']:.1%} of test predictions.

**What the model learned:** the deciding feature is attendance
(importance {sel['feature_importance'].get('Attendance (%)', 0):.2f}). The other
academic scores carry no measurable signal in this dataset.
""")
            if metrics.get("tree_rules"):
                with st.expander("Show the decision rules learned by the tree"):
                    st.code(metrics["tree_rules"])

        with st.expander("What do these metrics mean?"):
            st.markdown("""
- **Accuracy** - share of all students classified correctly. Can look good even when one
  class is handled badly, so it is never used alone.
- **Precision (High)** - of the students flagged as High risk, how many really are.
  Low precision means students are flagged unnecessarily.
- **Recall (High)** - of the students who really are High risk, how many were flagged.
  Low recall means students who need support are **missed**. This matters most here.
- **F1-score** - balances precision and recall in one number.
- **Macro average** - plain average over Low/Medium/High, so every class counts equally.

**Trade-off:** missing a struggling student (false negative) usually costs more than an
unnecessary check-in (false positive), but flagging too many students wastes advisor time
and can discourage students. That is why High recall is prioritised but not at any price.
""")

        missed = 1 - metrics["models"][selected]["test"]["high_recall"]
        st.warning(f"""
**Limitations**
- Labels are produced by our own rule, so the model learns to imitate that rule - high scores do
  **not** prove it predicts real future failure.
- Total_Score is part of the rule but cannot be used as an input (leakage). It is unrelated to every
  other column in this dataset, so the part of the risk that comes from Total_Score cannot be predicted:
  about {missed:.0%} of High-risk students (those flagged only by a low Total_Score) are missed on the test set.
- The model therefore adds no accuracy over the simple attendance rule; its value here is a
  reproducible pipeline that can be retrained on richer, real data.
""")


# ---------------------------------------------------------------------------
# D. Single-student prediction
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict the risk level for one student")
    if model_error:
        st.warning(model_error)
    else:
        with st.form("student_form"):
            c1, c2, c3 = st.columns(3)
            attendance = c1.number_input("Attendance (%)", 0.0, 100.0, 70.0, 0.5)
            midterm = c2.number_input("Midterm score (0-100)", 0.0, 100.0, 65.0, 0.5)
            assignments = c3.number_input("Assignment average (0-100)", 0.0, 100.0, 70.0, 0.5)
            quizzes = c1.number_input("Quiz average (0-100)", 0.0, 100.0, 70.0, 0.5)
            participation = c2.number_input("Participation (0-10)", 0.0, 10.0, 5.0, 0.1)
            projects = c3.number_input("Project score (0-100)", 0.0, 100.0, 70.0, 0.5)
            prep = c1.selectbox("Test-preparation course", ["Not completed", "Completed"])
            submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            student = {
                "Attendance (%)": attendance, "Midterm_Score": midterm,
                "Assignments_Avg": assignments, "Quizzes_Avg": quizzes,
                "Participation_Score": participation, "Projects_Score": projects,
                "test_preparation_course": 1 if prep == "Completed" else 0,
            }
            try:
                result = predict_student(student, model)
            except InputError as err:
                st.error(str(err))
            else:
                st.markdown(risk_badge(result["risk"]), unsafe_allow_html=True)
                st.caption(
                    f"Model confidence: {result['confidence']:.0%} - how strongly the training "
                    "data supports this label (weighted for class balance). It is not a "
                    "probability that the student will fail.")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Key indicators**")
                    for line in result["indicators"]:
                        st.markdown(f"- {line}")
                with c2:
                    st.markdown("**Suggested next steps**")
                    for rec in result["recommendations"]:
                        st.markdown(f"- {rec}")
                st.caption("Recommendations are supportive suggestions; they do not imply a "
                           "known cause and cannot guarantee improvement.")


# ---------------------------------------------------------------------------
# E. Bulk analysis + CSV report
# ---------------------------------------------------------------------------
with tab_bulk:
    st.subheader("Predict all eligible students and export a report")
    if model_error:
        st.warning(model_error)
    else:
        predictions, incomplete = get_bulk_predictions(model, df)
        report = build_report(predictions)

        c1, c2, c3 = st.columns(3)
        c1.metric("Students predicted", f"{len(report):,}")
        c2.metric("Skipped (incomplete data)", f"{len(incomplete):,}")
        c3.metric("Predicted High risk", f"{(report['Risk'] == 'High').sum():,}")

        choice = st.multiselect("Show risk levels", RISK_ORDER, default=RISK_ORDER)
        view = predictions[predictions["Predicted_Risk"].isin(choice)]
        st.dataframe(
            view[["Anon_ID", "Attendance (%)", "Midterm_Score", "Predicted_Risk",
                  "Confidence", "Recommendation"]].rename(columns={
                      "Anon_ID": "Student", "Midterm_Score": "Midterm",
                      "Predicted_Risk": "Risk"}),
            width="stretch", hide_index=True, height=380,
            column_config={"Confidence": st.column_config.NumberColumn(format="percent")})

        st.markdown("**Report preview** (`Student | Attendance | Risk | Recommendation`)")
        st.dataframe(report.head(10), width="stretch", hide_index=True)

        b1, b2 = st.columns(2)
        b1.download_button("Download risk_report.csv", report.to_csv(index=False),
                           file_name="risk_report.csv", mime="text/csv", type="primary")
        if b2.button("Save report to reports/risk_report.csv"):
            path = save_report(report)
            st.success(f"Saved {len(report)} rows to {path.name}")

        with st.expander(f"Rows excluded because of missing inputs ({len(incomplete)})"):
            st.write("A risk level is only reported when every model input is present; "
                     "we do not label a student from guessed values.")
            st.dataframe(incomplete[["Anon_ID", "Missing"]].rename(columns={"Anon_ID": "Student"}),
                         width="stretch", hide_index=True)

st.caption("Student IDs are anonymised (STU-xxxxx). Names, e-mails and original IDs are "
           "removed before any analysis.")
