"""SignSense AI - Streamlit app (live recognition + practice/quiz).

Run with:  streamlit run app.py
"""
import html

import cv2
import streamlit as st

from signsense import config
from signsense.capture import CameraCapture, HandDetector
from signsense.classifier import ModelNotFoundError, SignClassifier
from signsense.normalizer import normalize_landmarks
from signsense.output import Speaker, SpeechGate
from signsense.quiz import PHASE_FEEDBACK, QuizSession

MODE_LIVE = "Live recognition"
MODE_QUIZ = "Practice / Quiz"

st.set_page_config(page_title="SignSense AI", page_icon="🤟", layout="wide")


@st.cache_resource(show_spinner="Loading trained model...")
def load_classifier():
    try:
        return SignClassifier(), None
    except ModelNotFoundError as exc:
        return None, str(exc)


@st.cache_resource
def get_speaker():
    return Speaker()


def big(text: str, size: str = "3.2rem") -> str:
    return f"<div style='font-size:{size};font-weight:700;line-height:1.15'>{html.escape(text)}</div>"


def render_live_panel(placeholder, status, prediction, history):
    parts = ["<b>Recognised sign</b>"]
    if prediction is not None and prediction.accepted:
        parts.append(big(prediction.label))
        parts.append(f"{prediction.category} &middot; confidence {prediction.confidence:.0%}")
    else:
        parts.append(big("-"))
        if prediction is not None:
            parts.append(f"best guess: {html.escape(prediction.label)} ({prediction.confidence:.0%})")
    parts.append(f"<br><b>Status:</b> {html.escape(status)}")
    if history:
        parts.append("<br><b>Recent signs</b> (newest first)<br>" + html.escape(" · ".join(reversed(history[-8:]))))
    placeholder.markdown("<br>".join(parts), unsafe_allow_html=True)


def render_quiz_panel(placeholder, quiz, status, prediction):
    parts = ["<b>Show this sign</b>", big(quiz.target)]
    if quiz.phase == PHASE_FEEDBACK and quiz.last_result is not None:
        r = quiz.last_result
        if r.correct:
            parts.append("<span style='color:#1a9c3a;font-size:1.6rem;font-weight:700'>✅ Correct!</span>")
        elif r.predicted is None:
            parts.append("<span style='color:#d62828;font-size:1.6rem;font-weight:700'>⏱ Time's up</span>")
        else:
            parts.append("<span style='color:#d62828;font-size:1.6rem;font-weight:700'>"
                         f"❌ Incorrect - that looked like {html.escape(r.predicted)}</span>")
    else:
        parts.append(f"Time left: {quiz.time_left:0.0f}s &middot; hold the sign steady")
    parts.append(f"<br><b>Score:</b> {quiz.score} / {quiz.attempts}"
                 + (f" ({quiz.accuracy:.0%})" if quiz.attempts else ""))
    seen = "-" if prediction is None else f"{prediction.label} ({prediction.confidence:.0%})"
    parts.append(f"<b>Camera sees:</b> {html.escape(seen)}")
    parts.append(f"<b>Status:</b> {html.escape(status)}")
    placeholder.markdown("<br>".join(parts), unsafe_allow_html=True)


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("🤟 SignSense AI")
mode = st.sidebar.radio("Mode", [MODE_LIVE, MODE_QUIZ])
camera_on = st.sidebar.toggle("Camera on", value=False, help="Turn off to release the webcam.")
speak = st.sidebar.checkbox("Speak predictions", value=True)
threshold = st.sidebar.slider("Confidence threshold", 0.30, 0.95, float(config.CONFIDENCE_THRESHOLD), 0.05)
camera_index = int(st.sidebar.number_input("Camera index", 0, 5, config.CAMERA_INDEX))

if st.sidebar.button("Reload model"):
    st.cache_resource.clear()
    st.rerun()

classifier, model_error = load_classifier()
speaker = get_speaker()

# --------------------------------------------------------------------------- main page
st.title("SignSense AI")
st.caption("Recognises a fixed vocabulary of single, held ISL signs from one hand using a webcam. "
           "It does not translate continuous signing or sentences. Video frames are never saved.")

if classifier is None:
    st.error(model_error)
    st.markdown("**Do these steps in your terminal (inside the project folder):**")
    st.code("python collect_data.py\npython train_model.py\nstreamlit run app.py", language="bash")
    st.stop()

if not speaker.available and speaker.error:
    st.sidebar.warning(f"Speech unavailable: {speaker.error}")
st.sidebar.caption(f"Model knows {len(classifier.labels)} signs:")
st.sidebar.caption(", ".join(classifier.labels))

quiz = None
if mode == MODE_QUIZ:
    if len(classifier.labels) < 2:
        st.error("The quiz needs a model trained on at least 2 signs.")
        st.stop()
    quiz = st.session_state.get("quiz")
    if quiz is None or quiz.labels != classifier.labels:
        quiz = QuizSession(classifier.labels, config.QUIZ_HOLD_FRAMES,
                           config.QUIZ_TIMEOUT_SECONDS, config.QUIZ_FEEDBACK_SECONDS)
        st.session_state["quiz"] = quiz
    if st.sidebar.button("Reset score"):
        quiz.reset()

history = st.session_state.setdefault("history", [])
col_video, col_info = st.columns([3, 2])
video_ph = col_video.empty()
info_ph = col_info.empty()

if not camera_on:
    video_ph.info("Switch **Camera on** in the sidebar to start. Your browser/OS may ask for webcam permission.")
    if quiz is not None:
        render_quiz_panel(info_ph, quiz, "Camera is off", None)
    else:
        render_live_panel(info_ph, "Camera is off", None, history)
    st.stop()

cam = CameraCapture(camera_index)
if not cam.open():
    st.error("Could not open the webcam. Close other apps using it, try another camera index, "
             "or see Troubleshooting in README.md.")
    st.stop()

detector = HandDetector()
gate = SpeechGate(config.STABLE_FRAMES)
spoken_round = None
try:
    while True:
        frame = cam.read()
        if frame is None:
            st.error("Lost the webcam feed.")
            break

        detection = detector.detect(frame)
        prediction = None
        if detection is None:
            status = "No hand detected"
        else:
            features = normalize_landmarks(detection.points)
            if features is None:
                status = "Hand not clear enough - move it a little"
            else:
                prediction = classifier.predict(features, threshold)
                status = "Sign recognised" if prediction.accepted else "Low confidence - hold the sign steadier"
            detector.draw(frame, detection)
        accepted = prediction.label if (prediction is not None and prediction.accepted) else None

        if quiz is None:
            to_speak = gate.update(accepted)
            if to_speak:
                history.append(to_speak)
                del history[:-30]
                if speak:
                    speaker.say(to_speak)
            render_live_panel(info_ph, status, prediction, history)
        else:
            result = quiz.update(accepted)
            if speak and spoken_round != quiz.round_id and quiz.phase != PHASE_FEEDBACK:
                spoken_round = quiz.round_id
                speaker.say(f"Show the sign {quiz.target}")
            if result is not None and speak:
                speaker.say("Correct" if result.correct else "Incorrect")
            if quiz.phase == PHASE_FEEDBACK and quiz.last_result is not None:
                color = (0, 200, 0) if quiz.last_result.correct else (0, 0, 220)
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 10)
            render_quiz_panel(info_ph, quiz, status, prediction)

        video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB",
                       output_format="JPEG", use_container_width=True)
finally:
    detector.close()
    cam.release()
