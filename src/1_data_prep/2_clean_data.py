import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from iot_ids.preprocessing import clean_dataset


def main():
    print("Cleaning raw dataset and removing leakage-prone columns...")
    summary = clean_dataset()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

