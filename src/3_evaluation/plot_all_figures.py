import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from iot_ids.evaluation import plot_all_evaluation_figures


def main():
    plot_paths = plot_all_evaluation_figures()
    print(json.dumps(plot_paths, indent=2, default=str))


if __name__ == "__main__":
    main()
