"""Dataset loading, augmentation, model selection, training and evaluation."""
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .normalizer import FEATURE_COUNT, NUM_LANDMARKS, SCALE_REF_INDEX
from .vocabulary import load_vocabulary


class TrainingError(RuntimeError):
    pass


def feature_columns() -> List[str]:
    return [f"f{i}" for i in range(FEATURE_COUNT)]


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not Path(csv_path).exists():
        raise TrainingError(f"No training data at {csv_path}. Run 'python collect_data.py' first.")
    df = pd.read_csv(csv_path)
    cols = feature_columns()
    missing = [c for c in ["label"] + cols if c not in df.columns]
    if missing:
        raise TrainingError("landmarks.csv has the wrong columns. Delete it and collect again.")
    df = df.dropna(subset=["label"] + cols)
    if df.empty:
        raise TrainingError("landmarks.csv is empty. Run 'python collect_data.py' first.")
    return df[cols].to_numpy(np.float32), df["label"].astype(str).to_numpy()


def build_label_map(present_labels, vocabulary) -> Dict[int, Dict[str, str]]:
    order = {v["label"]: i for i, v in enumerate(vocabulary)}
    category = {v["label"]: v["category"] for v in vocabulary}
    ordered = sorted(set(present_labels), key=lambda l: (order.get(l, 10 ** 6), l))
    return {i: {"label": l, "category": category.get(l, "other")} for i, l in enumerate(ordered)}


def encode_labels(labels: np.ndarray, label_map: Dict[int, Dict[str, str]]) -> np.ndarray:
    to_idx = {v["label"]: k for k, v in label_map.items()}
    return np.array([to_idx[l] for l in labels], dtype=np.int64)


def split_dataset(X, y, test_size: float, seed: int):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def _rotation_matrices(ax, ay, az) -> np.ndarray:
    n = len(ax)
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    rx = np.zeros((n, 3, 3)); ry = np.zeros((n, 3, 3)); rz = np.zeros((n, 3, 3))
    rx[:, 0, 0] = 1; rx[:, 1, 1] = cx; rx[:, 1, 2] = -sx; rx[:, 2, 1] = sx; rx[:, 2, 2] = cx
    ry[:, 0, 0] = cy; ry[:, 0, 2] = sy; ry[:, 1, 1] = 1; ry[:, 2, 0] = -sy; ry[:, 2, 2] = cy
    rz[:, 0, 0] = cz; rz[:, 0, 1] = -sz; rz[:, 1, 0] = sz; rz[:, 1, 1] = cz; rz[:, 2, 2] = 1
    return rz @ ry @ rx


def augment(X: np.ndarray, y: np.ndarray, copies: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Add slightly rotated / noisy copies so the model tolerates small pose changes."""
    rng = np.random.default_rng(seed)
    pts = X.reshape(-1, NUM_LANDMARKS, 3).astype(np.float64)
    xs, ys = [X], [y]
    for _ in range(copies):
        n = len(pts)
        rot = _rotation_matrices(np.deg2rad(rng.uniform(-8, 8, n)),
                                 np.deg2rad(rng.uniform(-8, 8, n)),
                                 np.deg2rad(rng.uniform(-15, 15, n)))
        new = np.einsum("nij,nkj->nki", rot, pts)
        new = new + rng.normal(0.0, 0.02, new.shape)
        new[:, 0, :] = 0.0
        scale = np.linalg.norm(new[:, SCALE_REF_INDEX, :], axis=1).reshape(-1, 1, 1)
        new = new / np.maximum(scale, 1e-6)
        xs.append(new.reshape(n, -1).astype(np.float32))
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def candidate_models(seed: int) -> Dict[str, object]:
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "SVM (RBF)": make_pipeline(StandardScaler(),
                                   SVC(C=10, gamma="scale", probability=True, random_state=seed)),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, random_state=seed)),
    }


def select_model(X, y, seed: int, log: Callable[[str], None]):
    folds = max(2, min(5, int(np.bincount(y).min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: Dict[str, float] = {}
    for name, model in candidate_models(seed).items():
        s = cross_val_score(model, X, y, cv=cv, n_jobs=1)
        scores[name] = float(s.mean())
        log(f"  {name:<14} {folds}-fold CV accuracy: {s.mean():.3f}")
    best = max(scores, key=scores.get)
    return best, scores


def save_confusion_matrix(y_true, y_pred, names: List[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    size = max(6, 0.55 * len(names) + 3)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (held-out test set)")
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _evaluate(model, X_test, y_test, label_map, report_path: Path, confusion_path: Path, log):
    pred = model.predict(X_test)
    names = [label_map[i]["label"] for i in sorted(label_map)]
    acc = float(accuracy_score(y_test, pred))
    report = classification_report(y_test, pred, labels=list(range(len(names))),
                                   target_names=names, zero_division=0)
    Path(report_path).write_text(report, encoding="utf-8")
    save_confusion_matrix(y_test, pred, names, confusion_path)
    log("\nHeld-out test report:\n" + report)
    log(f"Held-out test accuracy: {acc:.3f}")
    return acc


def train_and_save(csv_path=config.LANDMARK_CSV, model_path=config.MODEL_PATH,
                   label_map_path=config.LABEL_MAP_PATH, metrics_path=config.METRICS_PATH,
                   report_path=config.REPORT_PATH, confusion_path=config.CONFUSION_PATH,
                   test_size=config.TEST_SIZE, seed=config.RANDOM_SEED,
                   augment_copies=config.AUGMENT_COPIES, log: Callable[[str], None] = print) -> dict:
    X, labels = load_dataset(csv_path)
    vocab = load_vocabulary()
    label_map = build_label_map(labels, vocab)
    if len(label_map) < 2:
        raise TrainingError("Need samples for at least 2 different signs.")
    y = encode_labels(labels, label_map)

    counts = np.bincount(y)
    log("Samples per sign:")
    for i, c in enumerate(counts):
        log(f"  {label_map[i]['label']:<12} {int(c)}")
    too_few = [label_map[i]["label"] for i, c in enumerate(counts) if c < config.MIN_SAMPLES_PER_CLASS]
    if too_few:
        raise TrainingError(f"Need at least {config.MIN_SAMPLES_PER_CLASS} samples per sign. "
                            f"Too few for: {', '.join(too_few)}. Run collect_data.py again.")
    absent = [v["label"] for v in vocab if v["label"] not in set(labels)]
    if absent:
        log(f"\nNote: no data yet for {', '.join(absent)} - these signs will NOT be recognised.")

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size, seed)
    log("\nComparing models (cross-validation on the training split)...")
    best_name, cv_scores = select_model(X_train, y_train, seed, log)
    log(f"Best model: {best_name}")

    X_aug, y_aug = augment(X_train, y_train, augment_copies, seed)
    model = candidate_models(seed)[best_name]
    model.fit(X_aug, y_aug)
    if isinstance(model, RandomForestClassifier):
        model.set_params(n_jobs=1)   # faster for one-frame-at-a-time prediction

    acc = _evaluate(model, X_test, y_test, label_map, report_path, confusion_path, log)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    with open(label_map_path, "w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in label_map.items()}, fh, indent=2)
    metrics = {"model": best_name, "test_accuracy": acc, "cv_accuracy": cv_scores,
               "n_samples": int(len(y)), "n_classes": len(label_map),
               "test_size": test_size, "seed": seed, "augment_copies": augment_copies}
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log(f"\nSaved model -> {model_path}\nSaved label map -> {label_map_path}")
    return metrics


def evaluate_saved(csv_path=config.LANDMARK_CSV, model_path=config.MODEL_PATH,
                   label_map_path=config.LABEL_MAP_PATH, metrics_path=config.METRICS_PATH,
                   report_path=config.REPORT_PATH, confusion_path=config.CONFUSION_PATH,
                   log: Callable[[str], None] = print) -> float:
    for p in (model_path, label_map_path, metrics_path):
        if not Path(p).exists():
            raise TrainingError("No trained model found. Run 'python train_model.py' first.")
    model = joblib.load(model_path)
    label_map = {int(k): v for k, v in json.loads(Path(label_map_path).read_text(encoding="utf-8")).items()}
    meta = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    X, labels = load_dataset(csv_path)
    unknown = set(labels) - {v["label"] for v in label_map.values()}
    if unknown:
        raise TrainingError(f"Data contains signs the model has never seen ({', '.join(sorted(unknown))}). "
                            "Re-run train_model.py.")
    y = encode_labels(labels, label_map)
    if len(y) != meta["n_samples"]:
        log("WARNING: the data changed since training, so this 'held-out' set may include samples the "
            "model trained on. Re-run train_model.py for a trustworthy number.")
    _, X_test, _, y_test = split_dataset(X, y, meta["test_size"], meta["seed"])
    return _evaluate(model, X_test, y_test, label_map, report_path, confusion_path, log)
