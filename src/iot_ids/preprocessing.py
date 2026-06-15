from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import (
    CORRUPTED_NUMERIC_COLUMNS,
    LEAKAGE_COLUMNS,
    PATHS,
    TARGET_COLUMN,
    TARGET_COLUMNS,
)
from .utils import ensure_parent, require_file, save_json


def load_raw_dataset() -> pd.DataFrame:
    require_file(PATHS.raw_data, "Raw dataset")
    return pd.read_csv(PATHS.raw_data, low_memory=False)


def load_clean_dataset() -> pd.DataFrame:
    require_file(PATHS.clean_data, "Cleaned dataset")
    return pd.read_csv(PATHS.clean_data, low_memory=False)


def explore_raw_dataset() -> dict:
    df = load_raw_dataset()
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "attack_type_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "categorical_columns": df.select_dtypes(include=["object"]).columns.tolist(),
    }
    return summary


def clean_dataset() -> dict:
    df = load_raw_dataset()
    initial_rows = int(df.shape[0])
    initial_columns = int(df.shape[1])

    for column in CORRUPTED_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    dropped_columns = [column for column in LEAKAGE_COLUMNS if column in df.columns]
    df = df.drop(columns=dropped_columns, errors="ignore")
    df = df.fillna(0)

    rows_before_dedupe = int(df.shape[0])
    df = df.drop_duplicates()
    removed_duplicates = rows_before_dedupe - int(df.shape[0])

    ensure_parent(PATHS.clean_data)
    df.to_csv(PATHS.clean_data, index=False)

    summary = {
        "initial_rows": initial_rows,
        "initial_columns": initial_columns,
        "final_rows": int(df.shape[0]),
        "final_columns": int(df.shape[1]),
        "dropped_columns": dropped_columns,
        "removed_duplicates": int(removed_duplicates),
        "missing_values_after_cleaning": int(df.isna().sum().sum()),
        "attack_type_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
    }
    save_json(summary, PATHS.data_quality_json)
    return summary


def verify_clean_dataset() -> dict:
    df = load_clean_dataset()
    leaked_columns = [column for column in LEAKAGE_COLUMNS if column in df.columns]
    verification = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "leaked_columns": leaked_columns,
        "critical_dtypes": {
            column: str(df[column].dtype)
            for column in CORRUPTED_NUMERIC_COLUMNS
            if column in df.columns
        },
    }
    return verification


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    existing_targets = [column for column in TARGET_COLUMNS if column in df.columns]
    if TARGET_COLUMN not in existing_targets:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from cleaned data.")
    X = df.drop(columns=existing_targets)
    y = df[TARGET_COLUMN].astype(str)
    return X, y


def transform_dataset() -> dict:
    df = load_clean_dataset()
    X, y = split_features_target(df)

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    feature_encoders = {}
    for column in categorical_features:
        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column].astype(str))
        feature_encoders[column] = int(len(encoder.classes_))

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    ensure_parent(PATHS.feature_matrix)
    X.to_csv(PATHS.feature_matrix, index=False)
    np.save(PATHS.target_vector, y_encoded)

    mapping = pd.DataFrame(
        {
            "encoded_label": range(len(target_encoder.classes_)),
            "attack_type": target_encoder.classes_,
        }
    )
    ensure_parent(PATHS.label_mapping)
    mapping.to_csv(PATHS.label_mapping, index=False)

    return {
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "categorical_features_encoded": feature_encoders,
        "classes": mapping.to_dict(orient="records"),
    }

