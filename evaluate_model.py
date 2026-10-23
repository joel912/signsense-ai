"""Re-evaluate the saved model on its held-out test split.

Usage:  python evaluate_model.py
"""
import sys

from signsense.training import TrainingError, evaluate_saved


def main() -> int:
    try:
        evaluate_saved()
    except TrainingError as exc:
        print(f"ERROR: {exc}")
        return 1
    print("Saved: models/classification_report.txt and models/confusion_matrix.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
