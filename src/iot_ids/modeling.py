from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from .config import PATHS, RANDOM_STATE, TEST_SIZE
from .preprocessing import load_clean_dataset, split_features_target
from .utils import ensure_parent, ensure_project_dirs, save_json


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = [column for column in X.columns if column not in categorical_features]

    return ColumnTransformer(
        transformers=[
            ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_candidates() -> dict[str, object]:
    return {
        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=120,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def make_pipeline(X: pd.DataFrame, estimator: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("model", estimator),
        ]
    )


def train_and_evaluate_models() -> dict:
    ensure_project_dirs(PATHS)
    df = load_clean_dataset()
    X, y_text = split_features_target(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    metrics = {}
    trained_models = {}
    for name, estimator in model_candidates().items():
        pipeline = make_pipeline(X_train, estimator)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        }
        trained_models[name] = pipeline

    best_model_name = max(metrics, key=lambda name: metrics[name]["macro_f1"])
    best_model = trained_models[best_model_name]
    best_predictions = best_model.predict(X_test)
    class_names = label_encoder.classes_

    artifact = {
        "model_name": best_model_name,
        "pipeline": best_model,
        "label_encoder": label_encoder,
        "feature_columns": X.columns.tolist(),
        "metrics": metrics[best_model_name],
    }
    ensure_parent(PATHS.best_model)
    joblib.dump(artifact, PATHS.best_model)

    report = classification_report(
        y_test,
        best_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(PATHS.classification_report_csv)

    predictions_df = pd.DataFrame(
        {
            "true_label": y_test,
            "predicted_label": best_predictions,
            "true_attack_type": label_encoder.inverse_transform(y_test),
            "predicted_attack_type": label_encoder.inverse_transform(best_predictions),
        }
    )
    predictions_df.to_csv(PATHS.predictions_csv, index=False)

    label_mapping = pd.DataFrame(
        {
            "encoded_label": range(len(class_names)),
            "attack_type": class_names,
        }
    )
    label_mapping.to_csv(PATHS.label_mapping, index=False)

    metrics_payload = {
        "best_model": best_model_name,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "samples": {
            "train": int(X_train.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "models": metrics,
        "confusion_matrix": confusion_matrix(y_test, best_predictions).tolist(),
    }
    save_json(metrics_payload, PATHS.metrics_json)
    return metrics_payload


def load_model_artifact() -> dict:
    if not PATHS.best_model.exists():
        return {}
    return joblib.load(PATHS.best_model)


def get_feature_importance(artifact: dict) -> pd.DataFrame:
    pipeline = artifact["pipeline"]
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError("The selected model does not expose feature importances.")

    feature_names = preprocessor.get_feature_names_out()
    importances = np.asarray(model.feature_importances_)
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

