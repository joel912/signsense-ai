"""Capture and Detection module: webcam access (OpenCV) + 21-point hand landmarks (MediaPipe)."""
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from . import config


class CameraCapture:
    """Thin wrapper around cv2.VideoCapture. Frames are mirrored (selfie view)."""

    def __init__(self, index: int = config.CAMERA_INDEX,
                 width: int = config.FRAME_WIDTH, height: int = config.FRAME_HEIGHT):
        self.index = int(index)
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return False
        self._cap = cap
        return True

    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame (mirrored) or None if the camera failed."""
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return cv2.flip(frame, 1)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


@dataclass
class HandDetection:
    points: np.ndarray   # (21, 3) in pixel units (x*width, y*height, z*width)
    _proto: object       # MediaPipe landmark object, used only for drawing


class HandDetector:
    """MediaPipe Hands wrapper: one hand, 21 landmarks per frame."""

    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._drawing = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[HandDetection]:
        """Return the hand's landmarks, or None when no hand is visible."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        proto = result.multi_hand_landmarks[0]
        h, w = frame_bgr.shape[:2]
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in proto.landmark], dtype=np.float32)
        return HandDetection(points=pts, _proto=proto)

    def draw(self, frame_bgr: np.ndarray, detection: HandDetection) -> None:
        """Draw the landmark skeleton on the frame (display only, never saved)."""
        self._drawing.draw_landmarks(
            frame_bgr, detection._proto, self._mp_hands.HAND_CONNECTIONS,
            self._styles.get_default_hand_landmarks_style(),
            self._styles.get_default_hand_connections_style(),
        )

    def close(self) -> None:
        self._hands.close()
