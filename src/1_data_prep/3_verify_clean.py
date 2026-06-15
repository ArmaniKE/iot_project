import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from iot_ids.preprocessing import verify_clean_dataset


def main():
    print("Verifying cleaned dataset integrity...")
    verification = verify_clean_dataset()
    print(json.dumps(verification, indent=2, default=str))


if __name__ == "__main__":
    main()

