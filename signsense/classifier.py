"""Classification module: loads the trained model once and predicts a sign per frame."""
import json
from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np

from . import config


class ModelNotFoundError(RuntimeError):
    """Raised when the model has not been trained yet."""


@dataclass(frozen=True)
class Prediction:
    label: str
    category: str
    confidence: float
    accepted: bool   # True when confidence >= threshold


class SignClassifier:
    def __init__(self, model_path=config.MODEL_PATH, label_map_path=config.LABEL_MAP_PATH):
        if not model_path.exists() or not label_map_path.exists():
            raise ModelNotFoundError(
                "No trained model found. Collect data with 'python collect_data.py' "
                "and then train with 'python train_model.py'."
            )
        self._model = joblib.load(model_path)
        with open(label_map_path, "r", encoding="utf-8") as fh:
            self._label_map: Dict[int, Dict[str, str]] = {int(k): v for k, v in json.load(fh).items()}

    @property
    def labels(self) -> List[str]:
        return [self._label_map[i]["label"] for i in sorted(self._label_map)]

    def predict(self, features: np.ndarray, threshold: float = config.CONFIDENCE_THRESHOLD) -> Prediction:
        proba = self._model.predict_proba(np.asarray(features, dtype=np.float32).reshape(1, -1))[0]
        best = int(np.argmax(proba))
        class_index = int(self._model.classes_[best])
        info = self._label_map[class_index]
        confidence = float(proba[best])
        return Prediction(info["label"], info["category"], confidence, confidence >= threshold)
