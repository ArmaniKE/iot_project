import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from iot_ids.modeling import train_and_evaluate_models


def main():
    print("Training model candidates and saving the best pipeline...")
    metrics = train_and_evaluate_models()
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()

