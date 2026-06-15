from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    data_dir: Path = BASE_DIR / "data"
    output_dir: Path = BASE_DIR / "output"
    models_dir: Path = BASE_DIR / "models"
    reports_dir: Path = BASE_DIR / "reports"
    raw_data: Path = BASE_DIR / "data" / "ML-EdgeIIoT-dataset.csv"
    clean_data: Path = BASE_DIR / "data" / "cleaned_vectors.csv"
    feature_matrix: Path = BASE_DIR / "data" / "X.csv"
    target_vector: Path = BASE_DIR / "data" / "y.npy"
    label_mapping: Path = BASE_DIR / "reports" / "label_mapping.csv"
    best_model: Path = BASE_DIR / "models" / "best_intrusion_model.joblib"
    metrics_json: Path = BASE_DIR / "reports" / "model_metrics.json"
    predictions_csv: Path = BASE_DIR / "reports" / "test_predictions.csv"
    classification_report_csv: Path = BASE_DIR / "reports" / "classification_report.csv"
    data_quality_json: Path = BASE_DIR / "reports" / "data_quality_report.json"
    confusion_matrix_png: Path = BASE_DIR / "output" / "confusion_matrix.png"
    feature_importance_png: Path = BASE_DIR / "output" / "feature_importance.png"
    class_distribution_png: Path = BASE_DIR / "output" / "class_distribution.png"
    model_comparison_png: Path = BASE_DIR / "output" / "model_comparison.png"
    per_class_f1_png: Path = BASE_DIR / "output" / "per_class_f1.png"


PATHS = ProjectPaths()

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMNS = ["Result", "Attack_label", "Attack_type"]
TARGET_COLUMN = "Attack_type"

CORRUPTED_NUMERIC_COLUMNS = ["tcp.srcport", "tcp.dstport", "dns.qry.name.len"]

LEAKAGE_COLUMNS = [
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "http.request.uri.query",
    "http.referer",
    "tcp.payload",
    "mqtt.msg",
]
