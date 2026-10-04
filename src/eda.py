"""
Exploratory Data Analysis (EDA).

Every chart is a small function that returns a Matplotlib figure, so the
same code is reused by this script and the Streamlit app.

Run:  python src/eda.py
      -> prints a dataset overview + findings, saves charts to reports/figures/
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    DISPLAY_NAMES,
    FIGURES_DIR,
    ORIGINAL_PROPOSED_THRESHOLDS,
    PRIMARY_COLOR,
    RISK_COLORS,
    RISK_THRESHOLDS,
    SECONDARY_COLOR,
)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "axes.edgecolor": "#cccccc",
    "grid.color": "#e6e6e6",
    "axes.titleweight": "bold",
    "figure.dpi": 100,
})

SCORE_COLUMNS = ["Midterm_Score", "Final_Score", "Assignments_Avg",
                 "Quizzes_Avg", "Projects_Score", "Participation_Score"]
CORR_COLUMNS = ["Attendance (%)", "Midterm_Score", "Final_Score", "Assignments_Avg",
                "Quizzes_Avg", "Participation_Score", "Projects_Score", "Total_Score",
                "math_score", "reading_score", "writing_score", "science_score",
                "test_preparation_course", "Age"]
ATTENDANCE_BANDS = [0, 60, 65, 75, 85, 100]
BAND_LABELS = ["<60%", "60-65%", "65-75%", "75-85%", "85-100%"]


def _name(col):
    return DISPLAY_NAMES.get(col, col.replace("_", " "))


# ---------------------------------------------------------------------------
# 1. Dataset overview (numbers, not charts)
# ---------------------------------------------------------------------------
def dataset_overview(raw: pd.DataFrame) -> dict:
    """Basic facts about the RAW dataset (before cleaning)."""
    return {
        "rows": raw.shape[0],
        "columns": raw.shape[1],
        "dtypes": raw.dtypes.astype(str).to_dict(),
        "missing": raw.isna().sum()[lambda s: s > 0].to_dict(),
        "duplicate_rows": int(raw.duplicated().sum()),
    }


# ---------------------------------------------------------------------------
# 2. Academic performance
# ---------------------------------------------------------------------------
def plot_attendance_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df["Attendance (%)"].dropna(), bins=30, color=PRIMARY_COLOR, ax=ax)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)  # headroom for the labels
    top = ax.get_ylim()[1]
    for value, label, height in [(RISK_THRESHOLDS["high_attendance"], "High-risk cut-off", 0.95),
                                 (RISK_THRESHOLDS["medium_attendance"], "Medium-risk cut-off", 0.87)]:
        ax.axvline(value, color="#555555", linestyle="--", linewidth=1)
        ax.text(value + 0.5, top * height, f"{label} ({value:.0f}%)", fontsize=9,
                color="#333333", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.set(title="Distribution of attendance", xlabel="Attendance (%)", ylabel="Number of students")
    fig.tight_layout()
    return fig


def plot_score_distributions(df):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, SCORE_COLUMNS):
        sns.histplot(df[col].dropna(), bins=30, color=PRIMARY_COLOR, ax=ax)
        ax.set(title=_name(col), xlabel="Score", ylabel="Students")
    fig.suptitle("Distributions of assessment scores", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_total_and_grade(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["Total_Score"], bins=30, color=PRIMARY_COLOR, ax=axes[0])
    axes[0].set(title="Distribution of Total_Score", xlabel="Total score", ylabel="Students")
    order = sorted(df["Grade"].dropna().unique())
    sns.countplot(x="Grade", data=df, order=order, color=PRIMARY_COLOR, ax=axes[1])
    axes[1].set(title="Number of students per grade", xlabel="Grade", ylabel="Students")
    fig.tight_layout()
    return fig


def plot_grade_vs_total(df):
    """Checks whether Grade is consistent with Total_Score (it should be!)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = sorted(df["Grade"].dropna().unique())
    sns.boxplot(x="Grade", y="Total_Score", data=df, order=order, color=PRIMARY_COLOR,
                width=0.5, ax=ax)
    ax.set(title="Total_Score by Grade (data-consistency check)",
           xlabel="Grade", ylabel="Total score")
    fig.tight_layout()
    return fig


def plot_department_performance(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = df.groupby("Department")["Total_Score"].median().sort_values().index
    sns.boxplot(x="Department", y="Total_Score", data=df, order=order,
                color=PRIMARY_COLOR, width=0.5, ax=axes[0])
    axes[0].set(title="Total score by department", xlabel="", ylabel="Total score")
    sns.boxplot(x="Department", y="Attendance (%)", data=df, order=order,
                color=SECONDARY_COLOR, width=0.5, ax=axes[1])
    axes[1].set(title="Attendance by department", xlabel="", ylabel="Attendance (%)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Attendance analysis
# ---------------------------------------------------------------------------
def plot_attendance_vs(df, target="Total_Score", sample=3000, seed=42):
    """Scatter of attendance vs a score, with a trend line and Pearson r."""
    data = df[["Attendance (%)", target]].dropna()
    r = data["Attendance (%)"].corr(data[target])
    shown = data.sample(min(sample, len(data)), random_state=seed)  # avoid overplotting
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.regplot(x="Attendance (%)", y=target, data=shown, ax=ax,
                scatter_kws={"s": 10, "alpha": 0.35, "color": PRIMARY_COLOR},
                line_kws={"color": "#333333", "linewidth": 2})
    ax.set(title=f"Attendance vs {_name(target)}  (Pearson r = {r:.3f}, n = {len(data)})",
           xlabel="Attendance (%)", ylabel=_name(target))
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    cols = [c for c in CORR_COLUMNS if c in df]
    corr = df[cols].astype(float).corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # hide the diagonal + upper half
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Pearson correlation", "shrink": 0.7},
                annot_kws={"size": 8})
    ax.set_xticklabels([_name(c) for c in cols], rotation=45, ha="right")
    ax.set_yticklabels([_name(c) for c in cols], rotation=0)
    ax.set_title("Correlation between numeric columns")
    fig.tight_layout()
    return fig


def attendance_band_table(df):
    """Average scores per attendance range."""
    bands = pd.cut(df["Attendance (%)"], ATTENDANCE_BANDS, labels=BAND_LABELS,
                   include_lowest=True)
    return (df.assign(Attendance_Band=bands)
              .groupby("Attendance_Band", observed=True)
              [["Midterm_Score", "Final_Score", "Total_Score"]]
              .agg(["mean", "count"]).round(2))


def plot_attendance_bands(df):
    bands = pd.cut(df["Attendance (%)"], ATTENDANCE_BANDS, labels=BAND_LABELS,
                   include_lowest=True)
    long = (df.assign(Band=bands)
              .melt(id_vars="Band", value_vars=["Midterm_Score", "Final_Score", "Total_Score"],
                    var_name="Score", value_name="Value")
              .dropna())
    long["Score"] = long["Score"].map(_name)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(x="Band", y="Value", hue="Score", data=long, errorbar=("ci", 95),
                palette=[PRIMARY_COLOR, SECONDARY_COLOR, "#1baf7a"], ax=ax)
    ax.set(title="Average score by attendance range (bars show 95% CI)",
           xlabel="Attendance range", ylabel="Average score", ylim=(0, 100))
    ax.legend(title="", loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Other insights
# ---------------------------------------------------------------------------
def plot_prep_course(df):
    cols = ["Total_Score", "math_score", "reading_score", "writing_score"]
    long = df.melt(id_vars="test_preparation_course", value_vars=cols,
                   var_name="Score", value_name="Value").dropna()
    long["Prep course"] = long["test_preparation_course"].map({0: "Not completed", 1: "Completed"})
    long["Score"] = long["Score"].map(_name)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(x="Score", y="Value", hue="Prep course", data=long, width=0.6,
                palette=[SECONDARY_COLOR, PRIMARY_COLOR], ax=ax)
    ax.set(title="Scores with and without the test-preparation course", xlabel="", ylabel="Score")
    ax.legend(title="Prep course")
    fig.tight_layout()
    return fig


def plot_component_correlations(df):
    """How strongly each assessment component is associated with Total_Score."""
    comps = ["Attendance (%)", "Midterm_Score", "Final_Score", "Assignments_Avg",
             "Quizzes_Avg", "Participation_Score", "Projects_Score"]
    r = df[comps].corrwith(df["Total_Score"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([_name(c) for c in r.index], r.values, color=PRIMARY_COLOR, height=0.6)
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_xlim(-1, 1)
    for i, v in enumerate(r.values):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.set(title="Correlation of each component with Total_Score",
           xlabel="Pearson correlation (-1 to 1)")
    fig.tight_layout()
    return fig


def plot_raw_outliers(raw):
    """Boxplots of the RAW values, showing impossible entries before cleaning."""
    cols = ["Age", "Attendance (%)", "Midterm_Score", "Final_Score"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    for ax, col in zip(axes, cols):
        sns.boxplot(y=pd.to_numeric(raw[col], errors="coerce"), color=PRIMARY_COLOR,
                    width=0.4, ax=ax)
        ax.set(title=_name(col), ylabel="")
    fig.suptitle("Raw values before cleaning (points far outside the box are suspicious)",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def plot_risk_distribution(labels, title="Risk-level distribution"):
    counts = labels.value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(counts.index, counts.values, color=[RISK_COLORS[k] for k in counts.index],
                  width=0.55)
    total = counts.sum()
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v} ({100 * v / total:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set(title=title, xlabel="Risk level", ylabel="Students")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Findings computed from the data (so the text is always true for the data)
# ---------------------------------------------------------------------------
def key_findings(df) -> list[str]:
    att_total = df["Attendance (%)"].corr(df["Total_Score"])
    att_final = df["Attendance (%)"].corr(df["Final_Score"])
    comps = ["Midterm_Score", "Final_Score", "Assignments_Avg", "Quizzes_Avg",
             "Participation_Score", "Projects_Score"]
    max_comp_r = df[comps].corrwith(df["Total_Score"]).abs().max()
    grade_medians = df.groupby("Grade")["Total_Score"].median()
    prep = df.groupby("test_preparation_course")["Total_Score"].mean()
    dept = df.groupby("Department")["Total_Score"].mean()

    return [
        f"Attendance and marks show almost no linear relationship: r = {att_total:.3f} "
        f"with Total_Score and r = {att_final:.3f} with Final_Score.",
        f"Total_Score is not explained by its components: the strongest correlation "
        f"between any assessment component and Total_Score is only |r| = {max_comp_r:.3f}.",
        f"Grade is inconsistent with Total_Score: median Total_Score is "
        f"{grade_medians.min():.1f}-{grade_medians.max():.1f} for every grade from A to F, "
        f"so Grade is treated as unreliable and not used.",
        f"Total_Score ranges from {df['Total_Score'].min():.2f} to "
        f"{df['Total_Score'].max():.2f}; attendance ranges from "
        f"{df['Attendance (%)'].min():.1f}% to {df['Attendance (%)'].max():.1f}% "
        f"(after removing impossible values). Both are close to uniformly spread.",
        f"The test-prep course makes no visible difference to Total_Score "
        f"({prep.get(1, float('nan')):.1f} with vs {prep.get(0, float('nan')):.1f} without).",
        f"Average Total_Score is similar across departments "
        f"({dept.min():.1f} to {dept.max():.1f}).",
        "These near-zero relationships suggest the columns were generated largely "
        "independently (the Kaggle page says the data was modified for this "
        "recruitment task). Correlation is not causation in any case, and here "
        "even the correlations are absent.",
    ]


def save_figure(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def run_eda():
    from data_loader import load_raw_data
    from preprocessing import clean_data
    from risk_labeling import add_risk_labels, label_distribution

    raw = load_raw_data()
    overview = dataset_overview(raw)
    print(f"Raw data: {overview['rows']} rows x {overview['columns']} columns")
    print(f"Duplicate rows: {overview['duplicate_rows']}")
    print("Missing values per column:", overview["missing"])

    df, log = clean_data(raw)
    print("\nCleaning log:", log)

    figures = {
        "01_raw_outliers": plot_raw_outliers(raw),
        "02_attendance_distribution": plot_attendance_distribution(df),
        "03_score_distributions": plot_score_distributions(df),
        "04_total_and_grade": plot_total_and_grade(df),
        "05_grade_vs_total": plot_grade_vs_total(df),
        "06_department_performance": plot_department_performance(df),
        "07_attendance_vs_total": plot_attendance_vs(df, "Total_Score"),
        "08_attendance_vs_final": plot_attendance_vs(df, "Final_Score"),
        "09_correlation_heatmap": plot_correlation_heatmap(df),
        "10_attendance_bands": plot_attendance_bands(df),
        "11_prep_course": plot_prep_course(df),
        "12_component_correlations": plot_component_correlations(df),
    }

    print("\nScores by attendance range:\n", attendance_band_table(df))

    original = add_risk_labels(df, ORIGINAL_PROPOSED_THRESHOLDS)["Risk_Label"]
    adjusted = add_risk_labels(df)["Risk_Label"]
    print("\nRisk labels with ORIGINAL proposed thresholds (Total < 50):\n",
          label_distribution(original))
    print("High-risk students flagged by 'Total_Score < 50':",
          int((df["Total_Score"] < 50).sum()))
    print("\nRisk labels with ADJUSTED thresholds (Total < 55):\n",
          label_distribution(adjusted))
    figures["13_risk_distribution"] = plot_risk_distribution(
        adjusted, "Risk labels (adjusted policy, all labelled students)")

    for name, fig in figures.items():
        save_figure(fig, name)
    print(f"\nSaved {len(figures)} figures to {FIGURES_DIR}")

    print("\nKey findings:")
    for finding in key_findings(df):
        print(" -", finding)


if __name__ == "__main__":
    from data_loader import DataError
    matplotlib.use("Agg")  # script mode: draw straight to files, no window
    try:
        run_eda()
    except DataError as err:
        print(f"ERROR: {err}")
        raise SystemExit(1)
