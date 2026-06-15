from __future__ import annotations

from .evaluation import plot_confusion_matrix, plot_feature_importance
from .modeling import train_and_evaluate_models
from .preprocessing import clean_dataset, explore_raw_dataset, transform_dataset, verify_clean_dataset


def run_full_pipeline() -> dict:
    exploration = explore_raw_dataset()
    cleaning = clean_dataset()
    verification = verify_clean_dataset()
    transformation = transform_dataset()
    training = train_and_evaluate_models()
    confusion_matrix_path = plot_confusion_matrix()
    feature_importance_path = plot_feature_importance()

    return {
        "exploration": exploration,
        "cleaning": cleaning,
        "verification": verification,
        "transformation": transformation,
        "training": training,
        "plots": {
            "confusion_matrix": confusion_matrix_path,
            "feature_importance": feature_importance_path,
        },
    }

