import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from iot_ids.evaluation import plot_feature_importance


def main():
    plot_path = plot_feature_importance()
    print(f"Success: Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
