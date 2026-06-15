from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import PATHS
from .modeling import get_feature_importance, load_model_artifact, train_and_evaluate_models


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

