import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from iot_ids.config import PATHS
from iot_ids.modeling import load_model_artifact


st.set_page_config(
    page_title="IoT/IIoT Intrusion Detection Dashboard",
    layout="wide",
)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics() -> dict:
    return load_json(PATHS.metrics_json)


@st.cache_data
def load_quality_report() -> dict:
    return load_json(PATHS.data_quality_json)


@st.cache_data
def load_classification_report() -> pd.DataFrame:
    if not PATHS.classification_report_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(PATHS.classification_report_csv, index_col=0)


@st.cache_resource
def load_artifact() -> dict:
    return load_model_artifact()


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def prepare_uploaded_features(uploaded_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    data = uploaded_df.copy()
    for target_col in ("Result", "Attack_label", "Attack_type"):
        if target_col in data.columns:
            data = data.drop(columns=[target_col])

    missing_columns = [column for column in feature_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Uploaded file is missing required feature columns: "
            + ", ".join(missing_columns[:10])
            + (" ..." if len(missing_columns) > 10 else "")
        )

    return data[feature_columns]


st.title("IoT/IIoT Intrusion Detection Dashboard")
st.caption("A visual final product for the machine learning IDS pipeline.")

metrics = load_metrics()
quality = load_quality_report()
report = load_classification_report()
artifact = load_artifact()

if not metrics or not artifact:
    st.error(
        "Model artifacts are missing. Run `python src/run_pipeline.py all` before opening the dashboard."
    )
    st.stop()

best_model = metrics["best_model"]
best_metrics = metrics["models"][best_model]

overview_tab, evaluation_tab, prediction_tab, architecture_tab = st.tabs(
    ["Overview", "Evaluation", "Try Prediction", "Architecture"]
)

with overview_tab:
    st.subheader("Project Goal")
    st.write(
        "This project detects cyber attacks in IoT and industrial IoT network traffic. "
        "The final product is a trained intrusion detection model plus this dashboard, "
        "where we can inspect results and test new traffic records."
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Best model", best_model.replace("_", " ").title())
    metric_cols[1].metric("Accuracy", format_percent(best_metrics["accuracy"]))
    metric_cols[2].metric("Macro F1", format_percent(best_metrics["macro_f1"]))
    metric_cols[3].metric("Classes", "15")

    st.subheader("Data Quality")
    quality_cols = st.columns(4)
    quality_cols[0].metric("Raw rows", f"{quality.get('initial_rows', 0):,}")
    quality_cols[1].metric("Clean rows", f"{quality.get('final_rows', 0):,}")
    quality_cols[2].metric("Removed duplicates", f"{quality.get('removed_duplicates', 0):,}")
    quality_cols[3].metric("Missing values", quality.get("missing_values_after_cleaning", 0))

    st.subheader("Model Comparison")
    model_rows = []
    for model_name, values in metrics["models"].items():
        model_rows.append(
            {
                "model": model_name,
                "accuracy": values["accuracy"],
                "balanced_accuracy": values["balanced_accuracy"],
                "macro_f1": values["macro_f1"],
                "weighted_f1": values["weighted_f1"],
            }
        )
    model_df = pd.DataFrame(model_rows).set_index("model")
    st.dataframe(model_df.style.format("{:.4f}"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Class Distribution")
        if PATHS.class_distribution_png.exists():
            st.image(str(PATHS.class_distribution_png), use_container_width=True)
        else:
            st.warning("Class distribution plot was not found.")
    with right:
        st.subheader("Model Comparison Plot")
        if PATHS.model_comparison_png.exists():
            st.image(str(PATHS.model_comparison_png), use_container_width=True)
        else:
            st.bar_chart(model_df[["accuracy", "macro_f1", "weighted_f1"]])

with evaluation_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion Matrix")
        if PATHS.confusion_matrix_png.exists():
            st.image(str(PATHS.confusion_matrix_png), use_container_width=True)
        else:
            st.warning("Confusion matrix plot was not found.")

    with right:
        st.subheader("Feature Importance")
        if PATHS.feature_importance_png.exists():
            st.image(str(PATHS.feature_importance_png), use_container_width=True)
        else:
            st.warning("Feature importance plot was not found.")

    st.subheader("Per-Class F1 Score")
    if PATHS.per_class_f1_png.exists():
        st.image(str(PATHS.per_class_f1_png), use_container_width=True)
    else:
        st.warning("Per-class F1 plot was not found.")

    st.subheader("Per-Class Classification Report")
    if report.empty:
        st.warning("Classification report was not found.")
    else:
        st.dataframe(report.style.format("{:.4f}"), use_container_width=True)

with prediction_tab:
    st.subheader("Upload Traffic CSV")
    st.write(
        "Upload rows with the same feature columns as the cleaned dataset. "
        "The app will remove target columns if they are present and predict the attack type."
    )

    uploaded_file = st.file_uploader("CSV file", type=["csv"])
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded rows")
        st.dataframe(uploaded_df.head(10), use_container_width=True)

        try:
            feature_columns = artifact["feature_columns"]
            prediction_input = prepare_uploaded_features(uploaded_df, feature_columns)
            predictions = artifact["pipeline"].predict(prediction_input)
            predicted_attack_types = artifact["label_encoder"].inverse_transform(predictions)

            results = uploaded_df.copy()
            results["predicted_attack_type"] = predicted_attack_types
            st.success("Prediction complete.")
            st.dataframe(results.head(100), use_container_width=True)

            st.subheader("Prediction Distribution")
            st.bar_chart(results["predicted_attack_type"].value_counts())
        except Exception as exc:
            st.error(str(exc))
    else:
        st.info("Tip: you can upload a small sample from `data/cleaned_vectors.csv`.")

with architecture_tab:
    st.subheader("Why This Dashboard Exists")
    st.write(
        "The pipeline creates the machine learning model. The dashboard turns that work into a visible IDS prototype. "
        "It helps a user understand the dataset, compare models, inspect mistakes, and try predictions."
    )

    st.code(
        """Raw Edge-IIoTset CSV
  -> clean leakage-prone columns
  -> verify data quality
  -> train/test split
  -> fit preprocessing and models
  -> choose best model by macro F1
  -> save model, reports, and plots
  -> show results in Streamlit dashboard""",
        language="text",
    )

    st.subheader("Saved Artifacts")
    st.write(
        {
            "model": str(PATHS.best_model),
            "metrics": str(PATHS.metrics_json),
            "classification_report": str(PATHS.classification_report_csv),
            "confusion_matrix": str(PATHS.confusion_matrix_png),
            "feature_importance": str(PATHS.feature_importance_png),
            "class_distribution": str(PATHS.class_distribution_png),
            "model_comparison": str(PATHS.model_comparison_png),
            "per_class_f1": str(PATHS.per_class_f1_png),
        }
    )
