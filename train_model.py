"""Train the sign classifier from data/landmarks.csv.

Usage:  python train_model.py
Creates: models/sign_model.joblib, models/label_map.json, models/metrics.json,
         models/classification_report.txt, models/confusion_matrix.png
"""
import argparse
import sys

from signsense import config
from signsense.training import TrainingError, train_and_save


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the SignSense AI classifier.")
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()
    try:
        metrics = train_and_save(test_size=args.test_size, seed=args.seed)
    except TrainingError as exc:
        print(f"\nERROR: {exc}")
        return 1
    print(f"\nDone. Model: {metrics['model']} | held-out accuracy: {metrics['test_accuracy']:.1%}")
    if metrics["test_accuracy"] < 0.90:
        print("Tip: accuracy is below 90%. Collect more samples with more variety "
              "(distance, tilt, lighting) and make sure each sign looks clearly different.")
    print("Next step:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
