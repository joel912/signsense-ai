"""Engine: runs the webcam -> landmarks -> classify loop in a background thread and keeps
the latest state (prediction, quiz, history) for the web UI. Frames live only in memory."""
import threading
from typing import Optional

import cv2

from . import config
from .capture import CameraCapture, HandDetector
from .classifier import SignClassifier
from .normalizer import normalize_landmarks
from .output import Speaker, SpeechGate
from .quiz import PHASE_FEEDBACK, QuizSession

MODE_LIVE = "live"
MODE_QUIZ = "quiz"


class Engine:
    def __init__(self, speaker: Speaker):
        self.speaker = speaker
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._jpeg: Optional[bytes] = None
        self._spoken_round = None
        self.classifier: Optional[SignClassifier] = None
        self.quiz: Optional[QuizSession] = None
        self.model_error: Optional[str] = None
        self.mode = MODE_LIVE
        self.speak = True
        self.threshold = float(config.CONFIDENCE_THRESHOLD)
        self.running = False
        self.camera_error: Optional[str] = None
        self.status = "Camera is off"
        self.hand = False
        self.prediction = None
        self.history = []
        self.load_model()

    # ---------------------------------------------------------------- model
    def load_model(self) -> Optional[str]:
        try:
            clf = SignClassifier()
        except Exception as exc:  # missing or unreadable model
            with self._lock:
                self.classifier, self.quiz, self.model_error = None, None, str(exc)
            return str(exc)
        with self._lock:
            self.classifier, self.model_error = clf, None
            self.quiz = None
            if len(clf.labels) >= 2:
                self.quiz = QuizSession(clf.labels, config.QUIZ_HOLD_FRAMES,
                                        config.QUIZ_TIMEOUT_SECONDS, config.QUIZ_FEEDBACK_SECONDS)
        return None

    # -------------------------------------------------------------- control
    def start(self, camera_index: int) -> Optional[str]:
        with self._lock:
            if self.running:
                return None
            if self.classifier is None:
                return self.model_error or "No trained model yet."
            cam = CameraCapture(camera_index)
            if not cam.open():
                self.camera_error = ("Could not open the webcam. Close other apps using it "
                                     "or try another camera number.")
                return self.camera_error
            self.camera_error = None
            if self.quiz is not None and self.mode == MODE_QUIZ:
                self.quiz.start_round()
            self._stop.clear()
            self.running = True
            self.status = "Starting..."
            self._thread = threading.Thread(target=self._loop, args=(cam,), daemon=True)
            self._thread.start()
        return None

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_LIVE, MODE_QUIZ):
            return
        with self._lock:
            self.mode = mode
            if mode == MODE_QUIZ and self.quiz is not None:
                self.quiz.start_round()

    def reset_quiz(self) -> None:
        with self._lock:
            if self.quiz is not None:
                self.quiz.reset()

    # ----------------------------------------------------------------- loop
    def _loop(self, cam: CameraCapture) -> None:
        detector = HandDetector()
        gate = SpeechGate(config.STABLE_FRAMES)
        try:
            while not self._stop.is_set():
                frame = cam.read()
                if frame is None:
                    with self._lock:
                        self.camera_error = "Lost the webcam feed."
                    break
                clf = self.classifier
                if clf is None:
                    break
                detection = detector.detect(frame)
                prediction, status = None, "No hand detected"
                if detection is not None:
                    features = normalize_landmarks(detection.points)
                    if features is None:
                        status = "Hand not clear enough - move it a little"
                    else:
                        prediction = clf.predict(features, self.threshold)
                        status = ("Sign recognised" if prediction.accepted
                                  else "Low confidence - hold the sign steadier")
                    detector.draw(frame, detection)
                accepted = prediction.label if (prediction and prediction.accepted) else None

                with self._lock:
                    self.hand = detection is not None
                    self.prediction = prediction
                    self.status = status
                    if self.mode == MODE_LIVE or self.quiz is None:
                        spoken = gate.update(accepted)
                        if spoken:
                            self.history.append(spoken)
                            del self.history[:-30]
                            if self.speak:
                                self.speaker.say(spoken)
                    else:
                        quiz = self.quiz
                        result = quiz.update(accepted)
                        if self.speak and self._spoken_round != quiz.round_id and quiz.phase != PHASE_FEEDBACK:
                            self._spoken_round = quiz.round_id
                            self.speaker.say(f"Show the sign {quiz.target}")
                        if result is not None and self.speak:
                            self.speaker.say("Correct" if result.correct else "Incorrect")
                        if quiz.phase == PHASE_FEEDBACK and quiz.last_result is not None:
                            color = (80, 200, 60) if quiz.last_result.correct else (60, 60, 230)
                            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 12)
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        self._jpeg = buf.tobytes()   # in memory only, replaced every frame
        except Exception as exc:  # show unexpected errors in the UI instead of dying silently
            with self._lock:
                self.camera_error = f"Something went wrong: {exc}"
        finally:
            detector.close()
            cam.release()
            with self._lock:
                self.running = False
                self._jpeg = None
                self.hand = False
                self.prediction = None
                self.status = "Camera is off"

    # ---------------------------------------------------------------- views
    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def snapshot(self) -> dict:
        with self._lock:
            pred = None
            if self.prediction is not None:
                p = self.prediction
                pred = {"label": p.label, "category": p.category,
                        "confidence": round(p.confidence, 3), "accepted": p.accepted}
            quiz = None
            if self.quiz is not None:
                q, lr = self.quiz, self.quiz.last_result
                quiz = {"target": q.target, "phase": q.phase, "time_left": round(q.time_left, 1),
                        "timeout": q.timeout_seconds, "score": q.score, "attempts": q.attempts,
                        "accuracy": round(q.accuracy, 3), "round_id": q.round_id,
                        "last_result": None if lr is None else
                        {"correct": lr.correct, "predicted": lr.predicted, "target": lr.target}}
            return {"model_ready": self.classifier is not None, "model_error": self.model_error,
                    "labels": self.classifier.labels if self.classifier else [],
                    "running": self.running, "mode": self.mode, "speak": self.speak,
                    "threshold": self.threshold, "camera_error": self.camera_error,
                    "status": self.status, "hand": self.hand, "prediction": pred,
                    "history": list(self.history[-10:]), "quiz": quiz,
                    "speech_error": None if self.speaker.available else self.speaker.error}
