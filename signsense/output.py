"""Output module: text-to-speech (pyttsx3) in a background thread, plus a speech gate."""
import queue
import threading
from typing import Optional

from . import config


class Speaker:
    """Speaks short phrases without blocking the video loop."""

    def __init__(self, rate: int = config.TTS_RATE, volume: float = config.TTS_VOLUME,
                 voice_id: Optional[str] = config.TTS_VOICE_ID):
        self.available = True
        self.error: Optional[str] = None
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=4)
        self._settings = (rate, volume, voice_id)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            rate, volume, voice_id = self._settings
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)
            if voice_id:
                engine.setProperty("voice", voice_id)
        except Exception as exc:  # no speech engine installed / unsupported platform
            self.available = False
            self.error = str(exc)
            return
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                self.available = False
                self.error = str(exc)
                break

    def say(self, text: str) -> None:
        if not self.available:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass  # drop: a newer sign matters more than a stale one

    def stop(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass


class SpeechGate:
    """Decides WHEN to speak: a sign must be steady for N frames, and the same sign
    is not repeated until the hand has gone away (or the sign changes)."""

    def __init__(self, stable_frames: int = config.STABLE_FRAMES, release_frames: int = 10):
        self.stable_frames = stable_frames
        self.release_frames = release_frames
        self._candidate: Optional[str] = None
        self._streak = 0
        self._none_count = 0
        self._last_spoken: Optional[str] = None

    def update(self, label: Optional[str]) -> Optional[str]:
        """Feed the accepted label of each frame (None if nothing). Returns a label to speak or None."""
        if label is None:
            self._candidate = None
            self._streak = 0
            self._none_count += 1
            if self._none_count >= self.release_frames:
                self._last_spoken = None
            return None
        self._none_count = 0
        if label == self._candidate:
            self._streak += 1
        else:
            self._candidate = label
            self._streak = 1
        if self._streak >= self.stable_frames and label != self._last_spoken:
            self._last_spoken = label
            return label
        return None
