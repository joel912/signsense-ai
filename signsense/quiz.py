"""Practice and Quiz module: pure logic (no camera, no UI) so it is easy to test."""
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

PHASE_ATTEMPT = "attempt"
PHASE_FEEDBACK = "feedback"


@dataclass(frozen=True)
class QuizResult:
    target: str
    predicted: Optional[str]   # None = time ran out
    correct: bool


class QuizSession:
    """Shows a target sign, judges the user's held attempt by exact match, keeps a score."""

    def __init__(self, labels: List[str], hold_frames: int = 10, timeout_seconds: float = 12.0,
                 feedback_seconds: float = 1.5, rng: Optional[random.Random] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.labels = list(dict.fromkeys(labels))
        if len(self.labels) < 2:
            raise ValueError("A quiz needs at least 2 signs.")
        self.hold_frames = hold_frames
        self.timeout_seconds = timeout_seconds
        self.feedback_seconds = feedback_seconds
        self._rng = rng or random.Random()
        self._clock = clock
        self.score = 0
        self.attempts = 0
        self.round_id = 0
        self.target: Optional[str] = None
        self.phase = PHASE_ATTEMPT
        self.last_result: Optional[QuizResult] = None
        self._ignore: Optional[str] = None
        self._non_ignore_count = 0
        self._streak_label: Optional[str] = None
        self._streak = 0
        self._round_started = 0.0
        self._feedback_until = 0.0
        self.start_round()

    # ---- state helpers -------------------------------------------------
    def start_round(self) -> None:
        choices = [l for l in self.labels if l != self.target]
        self.target = self._rng.choice(choices)
        self.round_id += 1
        self.phase = PHASE_ATTEMPT
        self._round_started = self._clock()
        self._streak_label = None
        self._streak = 0

    def reset(self) -> None:
        self.score = 0
        self.attempts = 0
        self.last_result = None
        self._ignore = None
        self.target = None
        self.start_round()

    @property
    def time_left(self) -> float:
        if self.phase != PHASE_ATTEMPT:
            return 0.0
        return max(0.0, self.timeout_seconds - (self._clock() - self._round_started))

    @property
    def accuracy(self) -> float:
        return self.score / self.attempts if self.attempts else 0.0

    # ---- per-frame update ---------------------------------------------
    def update(self, predicted: Optional[str]) -> Optional[QuizResult]:
        """Feed the accepted label of the current frame (None if no confident sign).
        Returns a QuizResult on the frame an attempt is judged, otherwise None."""
        now = self._clock()
        if self.phase == PHASE_FEEDBACK:
            if now >= self._feedback_until:
                self.start_round()
            return None

        # Ignore the previous round's sign while the user is still holding it.
        if self._ignore is not None:
            if predicted == self._ignore:
                predicted = None
                self._non_ignore_count = 0
            else:
                self._non_ignore_count += 1
                if self._non_ignore_count >= 5:
                    self._ignore = None

        if predicted is None:
            self._streak_label = None
            self._streak = 0
        elif predicted == self._streak_label:
            self._streak += 1
        else:
            self._streak_label = predicted
            self._streak = 1

        if predicted is not None and self._streak >= self.hold_frames:
            return self._finish(predicted, now)
        if now - self._round_started >= self.timeout_seconds:
            return self._finish(None, now)
        return None

    def _finish(self, predicted: Optional[str], now: float) -> QuizResult:
        result = QuizResult(self.target, predicted, predicted == self.target)
        self.attempts += 1
        if result.correct:
            self.score += 1
        self.last_result = result
        self.phase = PHASE_FEEDBACK
        self._feedback_until = now + self.feedback_seconds
        self._ignore = predicted
        self._non_ignore_count = 0
        return result
