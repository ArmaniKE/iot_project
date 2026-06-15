from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import PATHS
from .modeling import get_feature_importance, load_model_artifact, train_and_evaluate_models
from .preprocessing import load_clean_dataset


def _ensure_metrics() -> dict:
    if not PATHS.metrics_json.exists() or not PATHS.best_model.exists():
        return train_and_evaluate_models()
    with PATHS.metrics_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_confusion_matrix() -> str:
    metrics = _ensure_metrics()
    labels = pd.read_csv(PATHS.label_mapping)["attack_type"].tolist()
    cm = metrics["confusion_matrix"]

    PATHS.output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"{metrics['best_model']} Confusion Matrix")
    plt.ylabel("True Attack Type")
    plt.xlabel("Predicted Attack Type")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PATHS.confusion_matrix_png, dpi=160, bbox_inches="tight")
    plt.close()
    return str(PATHS.confusion_matrix_png)


def plot_feature_importance(top_k: int = 15) -> str:
    metrics = _ensure_metrics()
    artifact = load_model_artifact()
    importances = get_feature_importance(artifact).head(top_k)

    PATHS.output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 7))
    sns.barplot(
        data=importances,
        x="importance",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
    )
    plt.title(f"Top {top_k} Network Features ({metrics['best_model']})")
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(PATHS.feature_importance_png, dpi=160, bbox_inches="tight")
    plt.close()
    return str(PATHS.feature_importance_png)


def plot_class_distribution() -> str:
    df = load_clean_dataset()
    counts = df["Attack_type"].value_counts().sort_values(ascending=True)

    PATHS.output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 7))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, palette="mako", legend=False)
    plt.title("Cleaned Dataset Class Distribution")
    plt.xlabel("Number of Records")
    plt.ylabel("Attack Type")
    plt.tight_layout()
    plt.savefig(PATHS.class_distribution_png, dpi=160, bbox_inches="tight")
    plt.close()
    return str(PATHS.class_distribution_png)


def plot_model_comparison() -> str:
    metrics = _ensure_metrics()
    rows = []
    for model_name, values in metrics["models"].items():
        rows.append({"model": model_name, "metric": "accuracy", "score": values["accuracy"]})
        rows.append({"model": model_name, "metric": "macro_f1", "score": values["macro_f1"]})
        rows.append({"model": model_name, "metric": "weighted_f1", "score": values["weighted_f1"]})

    comparison = pd.DataFrame(rows)
    PATHS.output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=comparison, x="model", y="score", hue="metric", palette="Set2")
    plt.title("Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(PATHS.model_comparison_png, dpi=160, bbox_inches="tight")
    plt.close()
    return str(PATHS.model_comparison_png)


def plot_per_class_f1() -> str:
    _ensure_metrics()
    report = pd.read_csv(PATHS.classification_report_csv, index_col=0)
    ignored_rows = {"accuracy", "macro avg", "weighted avg"}
    class_report = report.loc[[index for index in report.index if index not in ignored_rows]].copy()
    class_report = class_report.sort_values("f1-score", ascending=True)

    PATHS.output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 7))
    sns.barplot(
        data=class_report,
        x="f1-score",
        y=class_report.index,
        hue=class_report.index,
        palette="rocket",
        legend=False,
    )
    plt.title("Per-Class F1 Score")
    plt.xlabel("F1 Score")
    plt.ylabel("Attack Type")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(PATHS.per_class_f1_png, dpi=160, bbox_inches="tight")
    plt.close()
    return str(PATHS.per_class_f1_png)


def plot_all_evaluation_figures() -> dict:
    return {
        "class_distribution": plot_class_distribution(),
        "model_comparison": plot_model_comparison(),
        "confusion_matrix": plot_confusion_matrix(),
        "feature_importance": plot_feature_importance(),
        "per_class_f1": plot_per_class_f1(),
    }
