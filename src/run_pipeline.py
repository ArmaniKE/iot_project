import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from iot_ids.evaluation import plot_confusion_matrix, plot_feature_importance
from iot_ids.modeling import train_and_evaluate_models
from iot_ids.pipeline import run_full_pipeline
from iot_ids.preprocessing import clean_dataset, explore_raw_dataset, transform_dataset, verify_clean_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="IoT intrusion detection project pipeline")
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=["explore", "clean", "verify", "transform", "train", "plots", "all"],
        help="Pipeline step to run",
    )
    args = parser.parse_args()

    if args.step == "explore":
        result = explore_raw_dataset()
    elif args.step == "clean":
        result = clean_dataset()
    elif args.step == "verify":
        result = verify_clean_dataset()
    elif args.step == "transform":
        result = transform_dataset()
    elif args.step == "train":
        result = train_and_evaluate_models()
    elif args.step == "plots":
        result = {
            "confusion_matrix": plot_confusion_matrix(),
            "feature_importance": plot_feature_importance(),
        }
    else:
        result = run_full_pipeline()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
