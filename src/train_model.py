"""
Train, compare and save the risk-classification models.

Steps
  1. Load + clean the data, create the rule-based (proxy) risk labels.
  2. Stratified 80/20 train/test split (before fitting any preprocessing).
  3. Baselines: "always predict the most common class" and the
     attendance-only version of the risk rule.
  4. Train Logistic Regression, Decision Tree and Random Forest inside
     identical preprocessing pipelines.
  5. Pick the final model with 5-fold cross-validation on the TRAINING set
     (High-risk recall first, macro F1 second - see select_model). The test
     set is only used for the final, honest report.
  6. Save the chosen pipeline (models/risk_model.pkl) and all metrics.

Run:  python src/train_model.py
"""

import json
import sys
import warnings
from datetime import datetime

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from config import (
    COMPARISON_PATH,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    RISK_THRESHOLDS,
)
from data_loader import DataError, load_raw_data
from evaluate_model import compute_metrics, confusion, per_class_report, plot_confusion_matrix
from preprocessing import build_preprocessor, clean_data, split_data
from risk_labeling import add_risk_labels, attendance_only_rule, label_distribution

# Newer SciPy versions emit a harmless "Unknown solver options: iprint"
# warning from scikit-learn 1.6's Logistic Regression; hide that one only.
warnings.filterwarnings("ignore", message="Unknown solver options")

RECALL_TOLERANCE = 0.03  # see select_model()


def build_models() -> dict:
    """
    The three candidate models. Settings are deliberately simple:

    - class_weight="balanced": classes are only mildly imbalanced
      (~36% / 28% / 36%), but weighting makes sure the smaller class is not
      ignored and costs nothing here.
    - Decision Tree max_depth=4: shallow enough to read as rules and to
      avoid memorising noise.
    - Random Forest max_depth=8, min_samples_leaf=20: limits overfitting,
      because most features turn out to carry no signal (see EDA).
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }


def select_model(cv_scores: dict, recall_tolerance: float = RECALL_TOLERANCE) -> str:
    """
    Choose the final model from CROSS-VALIDATION scores (never the test set).

    1. High-risk recall comes first: keep every model whose CV High recall
       is within `recall_tolerance` of the best one.
    2. Among those, pick the highest macro F1 (rounded to 2 decimals so
       meaningless differences do not decide); ties go to the model listed
       first in build_models(), i.e. the simpler one.

    Why not simply "highest High recall"? A model can raise High recall by
    flagging extra students almost at random, which lowers precision and
    hurts every other class. The tolerance stops a tiny recall gain from
    outweighing a large loss everywhere else.
    """
    best_recall = max(s["high_recall"] for s in cv_scores.values())
    shortlist = [n for n, s in cv_scores.items()
                 if s["high_recall"] >= best_recall - recall_tolerance]
    return max(shortlist, key=lambda n: round(cv_scores[n]["macro_f1"], 2))


def make_pipeline(model) -> Pipeline:
    # Same imputation + encoding for every model; only Logistic Regression
    # needs scaling (see build_preprocessor).
    scale = isinstance(model, LogisticRegression)
    return Pipeline([("preprocess", build_preprocessor(scale=scale)), ("model", model)])


def feature_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


def feature_importance(pipeline: Pipeline) -> dict:
    """Tree importances, or mean |coefficient| for Logistic Regression."""
    model = pipeline.named_steps["model"]
    names = feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = abs(model.coef_).mean(axis=0)
    series = pd.Series(values, index=names).sort_values(ascending=False)
    return series.round(4).to_dict()


def leaky_reference(X_train, X_test, y_train, y_test, df) -> dict:
    """
    Reference ONLY: the same Random Forest but with Total_Score added as a
    feature. Total_Score is used to build the label, so this model can
    'cheat'. Its near-perfect score shows why leakage must be avoided.
    """
    cols = MODEL_FEATURES + ["Total_Score"]
    Xtr = df.loc[X_train.index, cols].astype(float)
    Xte = df.loc[X_test.index, cols].astype(float)
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), cols)])
    pipe = Pipeline([("preprocess", pre), ("model", build_models()["Random Forest"])])
    pipe.fit(Xtr, y_train)
    return {"train": compute_metrics(y_train, pipe.predict(Xtr)),
            "test": compute_metrics(y_test, pipe.predict(Xte))}


def main():
    # 1. Data + proxy labels ---------------------------------------------------
    raw = load_raw_data()
    df, cleaning_log = clean_data(raw)
    df = add_risk_labels(df)
    unlabelled = int(df["Risk_Label"].isna().sum())
    print(f"Rows after cleaning: {len(df)} | without a label (missing attendance): {unlabelled}")
    print("Label distribution:\n", label_distribution(df["Risk_Label"].dropna()))

    # 2. Split -----------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows (stratified)")

    results = {}

    # 3. Baselines ---------------------------------------------------------------
    majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    results["Baseline: most frequent class"] = {
        "train": compute_metrics(y_train, majority.predict(X_train)),
        "test": compute_metrics(y_test, majority.predict(X_test)),
        "confusion_test": confusion(y_test, majority.predict(X_test)),
    }
    rule_train = [attendance_only_rule(a) for a in X_train["Attendance (%)"]]
    rule_test = [attendance_only_rule(a) for a in X_test["Attendance (%)"]]
    results["Baseline: attendance-only rule"] = {
        "train": compute_metrics(y_train, rule_train),
        "test": compute_metrics(y_test, rule_test),
        "confusion_test": confusion(y_test, rule_test),
    }

    # 4 + 5. Models, cross-validation and selection ------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted = {}
    for name, model in build_models().items():
        pipe = make_pipeline(model)
        cv_pred = cross_val_predict(pipe, X_train, y_train, cv=cv)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        test_pred = pipe.predict(X_test)
        results[name] = {
            "cv": compute_metrics(y_train, cv_pred),
            "train": compute_metrics(y_train, pipe.predict(X_train)),
            "test": compute_metrics(y_test, test_pred),
            "per_class_test": per_class_report(y_test, test_pred),
            "confusion_test": confusion(y_test, test_pred),
            "agreement_with_rule_test": round(
                float((pd.Series(test_pred, index=y_test.index) == pd.Series(rule_test, index=y_test.index)).mean()), 4),
            "feature_importance": feature_importance(pipe),
        }
        cvm = results[name]["cv"]
        print(f"{name:20s} CV High recall={cvm['high_recall']:.3f}  CV macro F1={cvm['macro_f1']:.3f}")

    selected = select_model({n: results[n]["cv"] for n in fitted})
    print(f"\nSelected model: {selected}")

    results["Reference only: Random Forest WITH Total_Score (leakage)"] = leaky_reference(
        X_train, X_test, y_train, y_test, df)

    # 6. Save ----------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[selected], MODEL_PATH)

    tree_rules = None
    if selected == "Decision Tree":
        tree_rules = export_text(fitted[selected].named_steps["model"],
                                 feature_names=feature_names(fitted[selected]))

    metadata = {
        "selected_model": selected,
        "features": MODEL_FEATURES,
        "risk_thresholds": RISK_THRESHOLDS,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    metrics = {
        "selected_model": selected,
        "cleaning_log": cleaning_log,
        "label_distribution": label_distribution(df["Risk_Label"].dropna())["count"].to_dict(),
        "unlabelled_rows": unlabelled,
        "models": results,
        "tree_rules": tree_rules,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    table = pd.DataFrame(
        [{"approach": n, "split": s, **r[s]} for n, r in results.items()
         for s in ("cv", "train", "test") if s in r])
    table.to_csv(COMPARISON_PATH, index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name in list(fitted) + ["Baseline: attendance-only rule"]:
        fig = plot_confusion_matrix(results[name]["confusion_test"], f"{name} (test set)")
        slug = name.lower().replace("baseline: ", "").replace(" ", "_").replace("-", "_")
        fig.savefig(FIGURES_DIR / f"cm_{slug}.png", dpi=120, bbox_inches="tight")

    print("\nTest-set results:")
    print(table[table["split"] == "test"].drop(columns="split").set_index("approach").to_string())
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH} and {COMPARISON_PATH}")
    if tree_rules:
        print("\nDecision rules learned by the tree:\n" + tree_rules)


if __name__ == "__main__":
    try:
        main()
    except DataError as err:
        print(f"ERROR: {err}")
        raise SystemExit(1)
