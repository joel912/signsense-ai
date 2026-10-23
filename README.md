# SignSense AI

Webcam-based Indian Sign Language (ISL) recognition + practice/quiz mode.
Python · OpenCV · MediaPipe Hands (21 landmarks) · scikit-learn · Streamlit · pyttsx3

**Scope (per the SRS/SDS):** one hand, one webcam, a fixed vocabulary of single held signs
(alphabets, numbers, words). It does **not** translate continuous signing or sentences, and it
ignores facial expressions. Video frames are never saved - only 63 landmark numbers per sample
are written, and only while you run `collect_data.py`.

## 1. Folder structure
```
signsense-ai/
├── flask_app.py         # Flask web app with the custom UI (recommended)
├── app.py               # Streamlit app (simpler alternative)
├── templates/index.html # the frontend (HTML + CSS + JS)
├── collect_data.py      # record training samples from your webcam
├── train_model.py       # train + evaluate + save the model
├── evaluate_model.py    # re-check the saved model on its held-out split
├── selftest.py          # checks your install, no webcam needed
├── vocabulary.json      # the signs you want (edit me)
├── requirements.txt
├── signsense/           # capture, normalizer, classifier, output, quiz, engine, training, config
├── data/                # landmarks.csv is created here
└── models/              # sign_model.joblib, label_map.json, reports are created here
```

## 2. Setup (once)
You need **Python 3.10, 3.11 or 3.12** (not 3.13) - check with `python --version`
(on macOS/Linux use `python3`).

**Windows (PowerShell, inside the signsense-ai folder):**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python selftest.py
```
**macOS / Linux (Terminal, inside the signsense-ai folder):**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python selftest.py
```
Linux only, for speech: `sudo apt install espeak-ng`.
`selftest.py` should end with `6/6 checks passed`. Each time you open a new terminal,
activate the venv again (the `activate` line) before running anything.

## 3. Collect training samples (no dataset is included)
```
python collect_data.py
```
- A window opens. For each sign: make the sign, press **SPACE**, wait for the 3-second countdown,
  then slowly move/tilt/shift your hand a little while it records 200 samples (~15 s).
- **S** skips a sign, **Q** quits. Progress is saved per sign, so you can stop and resume later.
- Re-record one sign: `python collect_data.py --redo A`. Only some signs: `python collect_data.py --only A B 1`.
- Tips: use the **same hand** here and in the app; vary distance and angle; keep the hand well lit;
  make every sign look clearly different from the others.

**Choosing the signs:** edit `vocabulary.json` (alphabets / numbers / words) before collecting.
The starter list is A-E, 1-5, Hello, Yes, No, Thank You, Help. This project does **not** ship an ISL
dictionary - perform each sign as shown in an official ISL reference (e.g. ISLRTC's dictionary).
Because the model sees one hand as a still pose, use signs that can be held as a hand shape;
many ISL letters and words use two hands or movement, which this design cannot capture.

## 4. Train and evaluate
```
python train_model.py
python evaluate_model.py
```
Training compares RandomForest, SVM and a small neural net, keeps the best, and prints
held-out accuracy and a per-sign report. Files appear in `models/`
(`classification_report.txt`, `confusion_matrix.png`). Re-run `train_model.py` any time you
collect more data.

## 5. Run the app
```
streamlit run app.py
```
Your browser opens (or go to http://localhost:8501). In the sidebar switch **Camera on**.
- **Live recognition:** the sign appears as text; steady signs are spoken aloud. "Recent signs" is just a list, not a sentence.
- **Practice / Quiz:** a target sign is shown - hold it steady. You get immediate ✅/❌ feedback
  and a running score (exact-match judging, 12 s per round). Switching **Camera off** releases the webcam.

Settings (camera index, thresholds, speech rate/voice, quiz timing) are in `signsense/config.py`.

### 5b. Prettier interface (Flask) - recommended
The PDF allows "Streamlit **or** Flask". `flask_app.py` serves a custom-designed page
(`templates/index.html`: dark theme, confidence ring, quiz timer, score cards) using the exact same
recognition code. Run:
```
python flask_app.py
```
It opens http://127.0.0.1:5000 by itself (only your own computer can reach it). Press
**Start camera**, pick **Live recognition** or **Practice / Quiz**. Stop it with Ctrl+C in the terminal.
To change the look, edit the `<style>` section at the top of `templates/index.html`.

### 5c. Deployment

The Flask app is designed for a computer with a local webcam. It is **not directly
deployable to Vercel**: Vercel functions are short-lived serverless processes and
cannot access the user's webcam through OpenCV or keep the `Engine` camera thread
running. The trained scikit-learn/MediaPipe runtime also exceeds the practical
serverless deployment shape.

This repository is ready to deploy the complete application on a service that
supports a persistent Python process, such as Render, Railway, Fly.io, or a
small VPS. Use:

```
python flask_app.py
```

For a Vercel deployment, the frontend must first be separated from the Flask
runtime and recognition must be moved to a browser-compatible model (for example,
TensorFlow.js) or to a separately hosted API. Do not add a `vercel.json` pointing
at `flask_app.py` - that creates a deployment that may load the page but cannot
run webcam recognition reliably.

## 6. Troubleshooting
| Problem | Fix |
|---|---|
| `pip install` fails on mediapipe | You probably have Python 3.13 or a 32-bit Python. Install Python 3.12 (64-bit), recreate the venv. |
| `No module named ...` | The venv isn't active. Run the `activate` line again. |
| Webcam won't open | Close Zoom/Teams/other browser tabs using it; try Camera index 1 in the sidebar (`--camera 1` for collect_data). **Windows:** Settings > Privacy > Camera > allow desktop apps. **macOS:** System Settings > Privacy & Security > Camera > allow Terminal (or your editor), then restart it. |
| Window doesn't appear on Linux | `sudo apt install libgl1` and make sure you're on a desktop session (not SSH). |
| No speech | Untick "Speak predictions" if it hangs. Linux: `sudo apt install espeak-ng`. The sidebar shows the error if the engine can't start. |
| "No trained model found" | Do steps 3 and 4 first, then click **Reload model** in the sidebar. |
| Wrong or jumpy predictions | Collect more samples with more variety, re-train, raise the confidence threshold, check lighting, and make sure two signs aren't nearly identical. |
| Quiz says "no hand detected" | Keep your whole hand in view with a plain background, about arm's length away. |
| Reset everything | Delete `data/landmarks.csv` and the files in `models/`. |
