import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from iot_ids.preprocessing import transform_dataset


def main():
    print("Creating encoded X.csv, y.npy, and label mapping...")
    summary = transform_dataset()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

