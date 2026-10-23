"""Landmark normalisation: wrist-relative and scale-normalised (SDS 4.3)."""
from typing import Optional

import numpy as np

NUM_LANDMARKS = 21
FEATURE_COUNT = NUM_LANDMARKS * 3
WRIST_INDEX = 0
SCALE_REF_INDEX = 9  # middle-finger knuckle (MCP); wrist -> this point is the hand's scale


def normalize_landmarks(points) -> Optional[np.ndarray]:
    """Turn 21 raw (x, y, z) points into a 63-value feature vector.

    1. The wrist becomes the origin (removes where the hand is in the frame).
    2. Everything is divided by the wrist->middle-knuckle distance
       (removes how big / how close the hand is).
    Returns None if the hand is degenerate (scale ~ 0).
    """
    pts = np.asarray(points, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
    rel = pts - pts[WRIST_INDEX]
    scale = float(np.linalg.norm(rel[SCALE_REF_INDEX]))
    if not np.isfinite(scale) or scale < 1e-6:
        return None
    return (rel / scale).reshape(-1).astype(np.float32)
