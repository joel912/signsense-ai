"""Quick self-test: checks your installation and the whole code path WITHOUT a webcam.

Usage:  python selftest.py

It trains a throw-away model on SYNTHETIC hand-like data in a temporary folder.
That proves the software works; it says nothing about real sign accuracy - for that,
record your own samples with collect_data.py.
"""
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

from signsense.normalizer import NUM_LANDMARKS, normalize_landmarks
from signsense.output import SpeechGate
from signsense.quiz import PHASE_FEEDBACK, QuizSession

RESULTS = []


def check(name):
    def wrap(fn):
        try:
            fn()
            RESULTS.append((name, True, ""))
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, False, str(exc)))
            print(f"FAIL  {name}: {exc}")
            traceback.print_exc()
        return fn
    return wrap


def fake_hand(rng, base):
    return base + rng.normal(0, 0.03, base.shape)


@check("normaliser ignores position and size")
def _():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, (NUM_LANDMARKS, 3)).astype(np.float32)
    a = normalize_landmarks(pts)
    b = normalize_landmarks(pts * 3.7 + np.array([120.0, -40.0, 5.0], dtype=np.float32))
    assert a is not None and a.shape == (63,)
    assert np.allclose(a, b, atol=1e-4), "features changed when the hand moved/resized"
    assert normalize_landmarks(np.zeros((21, 3))) is None


@check("speech gate speaks once per steady sign")
def _():
    gate = SpeechGate(stable_frames=3, release_frames=2)
    out = [gate.update("A") for _ in range(5)]
    assert out == [None, None, "A", None, None]
    gate.update(None); gate.update(None)
    assert [gate.update("A") for _ in range(3)][-1] == "A"


@check("quiz scoring, timeout and feedback")
def _():
    t = [0.0]
    q = QuizSession(["A", "B", "C"], hold_frames=3, timeout_seconds=5, feedback_seconds=1, clock=lambda: t[0])
    target = q.target
    res = [q.update(target) for _ in range(3)][-1]
    assert res is not None and res.correct and q.score == 1 and q.phase == PHASE_FEEDBACK
    t[0] += 1.1
    q.update(None)
    assert q.phase != PHASE_FEEDBACK and q.target != target
    wrong = next(l for l in q.labels if l not in (q.target, target))
    res = [q.update(wrong) for _ in range(3)][-1]
    assert res is not None and not res.correct and q.score == 1 and q.attempts == 2
    t[0] += 1.1; q.update(None)
    t[0] += 6
    res = q.update(None)
    assert res is not None and res.predicted is None and not res.correct


@check("train -> save -> load -> predict (synthetic data)")
def _():
    import csv
    from signsense import training
    from signsense.classifier import SignClassifier
    from signsense.training import feature_columns

    rng = np.random.default_rng(1)
    labels = ["A", "B", "C", "1"]
    bases = {l: rng.uniform(-1, 1, (NUM_LANDMARKS, 3)) for l in labels}
    for b in bases.values():
        b[0] = 0
        b[9] = [0, -1, 0]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv_path = tmp / "landmarks.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["label"] + feature_columns())
            for l in labels:
                for _ in range(80):
                    pts = fake_hand(rng, bases[l]) * rng.uniform(40, 200) + rng.uniform(0, 500, 3)
                    w.writerow([l] + list(normalize_landmarks(pts)))
        metrics = training.train_and_save(
            csv_path, tmp / "m.joblib", tmp / "lm.json", tmp / "metrics.json",
            tmp / "r.txt", tmp / "c.png", log=lambda s: None)
        assert metrics["test_accuracy"] > 0.9, metrics
        clf = SignClassifier(tmp / "m.joblib", tmp / "lm.json")
        hits = 0
        for l in labels:
            pts = fake_hand(rng, bases[l]) * 90 + 300
            hits += clf.predict(normalize_landmarks(pts)).label == l
        assert hits >= 3, f"only {hits}/4 correct"
        assert training.evaluate_saved(csv_path, tmp / "m.joblib", tmp / "lm.json", tmp / "metrics.json",
                                       tmp / "r.txt", tmp / "c.png", log=lambda s: None) > 0.9


@check("MediaPipe hand detector loads and handles an empty frame")
def _():
    from signsense.capture import HandDetector
    det = HandDetector()
    try:
        assert det.detect(np.zeros((480, 640, 3), dtype=np.uint8)) is None
    finally:
        det.close()


@check("pyttsx3 is importable (speech is optional)")
def _():
    import pyttsx3  # noqa: F401


@check("Flask is importable (for the custom web UI)")
def _():
    import flask  # noqa: F401


failed = [r for r in RESULTS if not r[1]]
print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed.")
if failed:
    print("Some checks failed - see README.md > Troubleshooting.")
sys.exit(1 if failed else 0)
