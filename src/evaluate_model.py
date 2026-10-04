"""
Evaluation helpers + a script to re-evaluate the saved model.

Metric cheat-sheet (for one class, e.g. "High"):
  - Precision: of the students we FLAGGED as High, how many really are High?
               (low precision = many students flagged unnecessarily)
  - Recall:    of the students who really ARE High, how many did we flag?
               (low recall = students who need support are missed)
  - F1-score:  harmonic mean of precision and recall (balances the two).
  - Accuracy:  share of all predictions that are correct. Can hide poor
               performance on one class, so we never use it alone.
  - Macro average: the plain average over Low/Medium/High, so every class
               counts equally regardless of its size.

Run:  python src/evaluate_model.py
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import FIGURES_DIR, METRICS_PATH, RISK_ORDER


def compute_metrics(y_true, y_pred) -> dict:
    """Headline metrics, with special focus on the High-risk class."""
    y_true, y_pred = list(y_true), list(y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=RISK_ORDER, average="macro", zero_division=0)
    hp, hr, hf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["High"], average=None, zero_division=0)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1, 4),
        "high_precision": round(float(hp[0]), 4),
        "high_recall": round(float(hr[0]), 4),
        "high_f1": round(float(hf[0]), 4),
    }


def per_class_report(y_true, y_pred) -> dict:
    return classification_report(y_true, y_pred, labels=RISK_ORDER,
                                 output_dict=True, zero_division=0)


def confusion(y_true, y_pred) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=RISK_ORDER).tolist()


def plot_confusion_matrix(matrix, title="Confusion matrix"):
    """Rows = true label, columns = predicted label. Diagonal = correct."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(pd.DataFrame(matrix, index=RISK_ORDER, columns=RISK_ORDER),
                annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1,
                linecolor="white", annot_kws={"size": 12}, ax=ax)
    ax.set(title=title, xlabel="Predicted risk", ylabel="True (rule-based) risk")
    fig.tight_layout()
    return fig


def print_comparison(metrics: dict) -> None:
    rows = []
    for name, m in metrics["models"].items():
        rows.append({"model": name, **m["test"]})
    table = pd.DataFrame(rows).set_index("model")
    print(table.to_string())


def main():
    from data_loader import load_raw_data
    from predict import load_model
    from preprocessing import clean_data, split_data
    from risk_labeling import add_risk_labels

    model, metadata = load_model()
    df = add_risk_labels(clean_data(load_raw_data())[0])
    X_train, X_test, y_train, y_test = split_data(df)  # same random_state as training

    print(f"Selected model: {metadata['selected_model']}")
    for split_name, X, y in [("TRAIN", X_train, y_train), ("TEST", X_test, y_test)]:
        pred = model.predict(X)
        print(f"\n=== {split_name} set ({len(y)} students) ===")
        print(compute_metrics(y, pred))
        print(classification_report(y, pred, labels=RISK_ORDER, zero_division=0))
        print("Confusion matrix (rows = true, cols = predicted, order Low/Medium/High):")
        print(pd.DataFrame(confusion(y, pred), index=RISK_ORDER, columns=RISK_ORDER))

    if METRICS_PATH.exists():
        print("\n=== Test-set comparison of all approaches (from training run) ===")
        print_comparison(json.loads(METRICS_PATH.read_text()))

    fig = plot_confusion_matrix(confusion(y_test, model.predict(X_test)),
                                f"{metadata['selected_model']} - test set")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "cm_selected_model.png", dpi=120, bbox_inches="tight")
    print(f"\nSaved confusion matrix to {FIGURES_DIR / 'cm_selected_model.png'}")


if __name__ == "__main__":
    from data_loader import DataError
    from predict import ModelNotFoundError
    try:
        main()
    except (DataError, ModelNotFoundError) as err:
        print(f"ERROR: {err}")
        raise SystemExit(1)
