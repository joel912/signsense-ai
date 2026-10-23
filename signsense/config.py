"""Central configuration, read once at startup (SDS section 4.4)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

VOCAB_PATH = ROOT / "vocabulary.json"
LANDMARK_CSV = DATA_DIR / "landmarks.csv"
MODEL_PATH = MODEL_DIR / "sign_model.joblib"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
REPORT_PATH = MODEL_DIR / "classification_report.txt"
CONFUSION_PATH = MODEL_DIR / "confusion_matrix.png"

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# MediaPipe Hands
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5

# Recognition
CONFIDENCE_THRESHOLD = 0.70   # predictions below this are shown as "low confidence"
STABLE_FRAMES = 8             # frames a sign must be steady before it is spoken

# Data collection / training
SAMPLES_PER_SIGN = 200
SAMPLE_INTERVAL_SECONDS = 0.06
MIN_SAMPLES_PER_CLASS = 30
TEST_SIZE = 0.2
RANDOM_SEED = 42
AUGMENT_COPIES = 3

# Quiz
QUIZ_HOLD_FRAMES = 10         # frames the attempt must be held to be judged
QUIZ_TIMEOUT_SECONDS = 12.0
QUIZ_FEEDBACK_SECONDS = 1.5

# Text-to-speech (pyttsx3)
TTS_RATE = 160
TTS_VOLUME = 1.0
TTS_VOICE_ID = None           # None = system default voice
