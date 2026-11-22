"""Train, tune, evaluate and save all five classifiers.

Usage (from the repository root):
    python model/train_models.py
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import (  # noqa: E402
    SEED, TARGET, POSITIVE, MODEL_DIR, RESULTS_DIR, TEST_CSV,
    load_raw_data, clean_data, split_data, evaluate_model,
)
from models import MODEL_SPECS, make_pipeline  # noqa: E402

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def save_test_data(X_test, y_test):
    test_df = X_test.copy()
    test_df[TARGET] = y_test.map({1: POSITIVE, 0: "<=50K"})
    test_df.to_csv(TEST_CSV, index=False)


def tune_and_fit(name, X_train, y_train):
    pipe, grid = make_pipeline(name)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def plot_metrics(results_df):
    long_df = results_df.reset_index().melt(
        id_vars="Model", value_vars=METRIC_ORDER, var_name="Metric", value_name="Score")
    plt.figure(figsize=(13, 6))
    ax = sns.barplot(data=long_df, x="Metric", y="Score", hue="Model", palette="Set2")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=1)
    ax.set_ylim(0, 1.12)
    ax.set_title("Evaluation metrics of all models on the test set (UCI Adult)")
    ax.legend(loc="upper center", ncol=5, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "metrics_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    sns.heatmap(results_df[METRIC_ORDER], annot=True, fmt=".4f", cmap="YlGnBu",
                vmin=0, vmax=1, cbar_kws={"label": "score"})
    plt.title("Metric heatmap (test set)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "metrics_heatmap.png", dpi=150)
    plt.close()


def plot_roc(roc_data, y_test):
    plt.figure(figsize=(7, 6))
    for name, (scores, auc) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves (test set)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close()


def plot_confusion(cms):
    fig, axes = plt.subplots(1, len(cms), figsize=(4 * len(cms), 3.8))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df, clean_report = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = split_data(df)
    save_test_data(X_test, y_test)
    print(f"Cleaning: {clean_report}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, "
          f"positive rate train={y_train.mean():.4f} test={y_test.mean():.4f}")

    rows, cms, roc_data, configs = [], {}, {}, {}
    for name in MODEL_SPECS:
        start = time.time()
        search = tune_and_fit(name, X_train, y_train)
        model = search.best_estimator_
        metrics, cm, scores = evaluate_model(model, X_test, y_test)
        elapsed = time.time() - start

        file_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
        joblib.dump(model, MODEL_DIR / file_name, compress=3)

        rows.append({"Model": name, **metrics})
        cms[name] = cm
        roc_data[name] = (scores, metrics["AUC"])
        configs[name] = {
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_roc_auc": round(search.best_score_, 4),
            "train_and_tune_seconds": round(elapsed, 1),
            "saved_as": f"model/saved_models/{file_name}",
        }
        print(f"{name:26s} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
              + f"  best={configs[name]['best_params']}  ({elapsed:.0f}s)")

    results_df = pd.DataFrame(rows).set_index("Model")[METRIC_ORDER]
    results_df.round(4).to_csv(RESULTS_DIR / "metrics.csv")
    (RESULTS_DIR / "metrics_table.md").write_text(results_df.round(4).to_markdown())
    (RESULTS_DIR / "model_configs.json").write_text(json.dumps(configs, indent=2))
    plot_metrics(results_df)
    plot_roc(roc_data, y_test)
    plot_confusion(cms)
    print("\n" + results_df.round(4).to_string())
    return results_df, configs


if __name__ == "__main__":
    main()
