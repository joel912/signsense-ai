"""Record training samples for each sign using your webcam.

Only the 63 normalised landmark numbers are saved to data/landmarks.csv.
No video, image or frame is ever written to disk.

Usage:
    python collect_data.py                 # record every sign that still needs samples
    python collect_data.py --only A B 1    # record only these signs
    python collect_data.py --redo A        # delete old samples of A and record again
    python collect_data.py --samples 300   # samples per sign (default 200)

Keys in the window:  SPACE = start recording   S = skip this sign   Q / ESC = quit
"""
import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

from signsense import config
from signsense.capture import CameraCapture, HandDetector
from signsense.normalizer import normalize_landmarks
from signsense.training import feature_columns
from signsense.vocabulary import load_vocabulary

WINDOW = "SignSense AI - data collection"
GREEN, RED, YELLOW, WHITE = (0, 200, 0), (0, 0, 220), (0, 220, 220), (255, 255, 255)


def read_counts() -> Counter:
    counts: Counter = Counter()
    if config.LANDMARK_CSV.exists():
        with open(config.LANDMARK_CSV, "r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if row:
                    counts[row[0]] += 1
    return counts


def append_rows(rows) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not config.LANDMARK_CSV.exists()
    with open(config.LANDMARK_CSV, "a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["label"] + feature_columns())
        writer.writerows(rows)


def remove_label(label: str) -> None:
    if not config.LANDMARK_CSV.exists():
        return
    with open(config.LANDMARK_CSV, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], [r for r in rows[1:] if r and r[0] != label]
    with open(config.LANDMARK_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(body)


def put(frame, text, y, color=WHITE, scale=0.7):
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def window_closed() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def record_sign(cam, detector, label, position, total, target) -> str:
    """Returns 'done', 'skip' or 'quit'."""
    # --- waiting screen ---
    while True:
        frame = cam.read()
        if frame is None:
            print("Lost the camera.")
            return "quit"
        det = detector.detect(frame)
        if det is not None:
            detector.draw(frame, det)
        put(frame, f"Sign {position}/{total}:  {label}", 30, YELLOW, 0.9)
        put(frame, "Make the sign, then press SPACE to record", 60)
        put(frame, "S = skip   Q = quit", 88)
        put(frame, "Hand visible" if det is not None else "No hand detected", 460,
            GREEN if det is not None else RED)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if window_closed() or key in (ord("q"), 27):
            return "quit"
        if key == ord("s"):
            return "skip"
        if key == ord(" "):
            break

    # --- countdown ---
    end = time.time() + 3
    while time.time() < end:
        frame = cam.read()
        if frame is None:
            return "quit"
        put(frame, f"Recording {label} in {int(end - time.time()) + 1}...", 40, YELLOW, 1.0)
        put(frame, "Keep your hand in view", 75)
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27) or window_closed():
            return "quit"

    # --- recording ---
    rows, last = [], 0.0
    while len(rows) < target:
        frame = cam.read()
        if frame is None:
            return "quit"
        det = detector.detect(frame)
        now = time.time()
        if det is not None:
            detector.draw(frame, det)
            features = normalize_landmarks(det.points)
            if features is not None and now - last >= config.SAMPLE_INTERVAL_SECONDS:
                rows.append([label] + [f"{v:.6f}" for v in features])
                last = now
        put(frame, f"RECORDING {label}: {len(rows)}/{target}", 30, RED, 0.9)
        put(frame, "Slowly move / tilt / shift your hand a little", 60)
        cv2.rectangle(frame, (12, 75), (12 + int(300 * len(rows) / target), 90), GREEN, -1)
        cv2.rectangle(frame, (12, 75), (312, 90), WHITE, 1)
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27) or window_closed():
            print(f"  Cancelled - nothing saved for {label}.")
            return "quit"
    append_rows(rows)
    print(f"  Saved {len(rows)} samples for {label}.")
    return "done"


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect landmark samples for SignSense AI.")
    parser.add_argument("--samples", type=int, default=config.SAMPLES_PER_SIGN)
    parser.add_argument("--only", nargs="+", metavar="SIGN", help="record only these signs")
    parser.add_argument("--redo", nargs="+", metavar="SIGN", help="delete and re-record these signs")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = parser.parse_args()

    vocab = [v["label"] for v in load_vocabulary()]
    unknown = [s for s in (args.only or []) + (args.redo or []) if s not in vocab]
    if unknown:
        print(f"Not in vocabulary.json: {', '.join(unknown)}\nAvailable: {', '.join(vocab)}")
        return 1

    for label in args.redo or []:
        remove_label(label)
        print(f"Removed old samples for {label}.")

    counts = read_counts()
    wanted = args.only or args.redo or vocab
    todo = [l for l in wanted if counts[l] < args.samples or l in (args.redo or [])]
    if not todo:
        print("All signs already have enough samples. Next step:  python train_model.py")
        return 0
    print(f"Signs to record ({len(todo)}): {', '.join(todo)}")

    cam = CameraCapture(args.camera)
    if not cam.open():
        print("Could not open the webcam. See 'Troubleshooting' in README.md.")
        return 1
    detector = HandDetector()
    try:
        for i, label in enumerate(todo, 1):
            need = args.samples - counts[label] if label not in (args.redo or []) else args.samples
            status = record_sign(cam, detector, label, i, len(todo), max(need, 1))
            if status == "quit":
                break
    finally:
        detector.close()
        cam.release()
        cv2.destroyAllWindows()

    final = read_counts()
    print("\nSamples collected so far:")
    for label in vocab:
        print(f"  {label:<12} {final[label]}")
    print("\nNext step:  python train_model.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
