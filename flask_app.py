"""SignSense AI - Flask web app with a custom frontend.

Run with:  python flask_app.py     (opens http://127.0.0.1:5000)
"""
import atexit
import os
import threading
import time
import webbrowser

from flask import Flask, Response, jsonify, render_template, request

from signsense.engine import Engine
from signsense.output import Speaker

URL = "http://127.0.0.1:5000"
app = Flask(__name__)
engine = Engine(Speaker())
atexit.register(engine.stop)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    return jsonify(engine.snapshot())


@app.post("/api/start")
def api_start():
    data = request.get_json(silent=True) or {}
    try:
        index_ = int(data.get("camera_index", 0))
    except (TypeError, ValueError):
        index_ = 0
    error = engine.start(index_)
    return jsonify({"error": error})


@app.post("/api/stop")
def api_stop():
    engine.stop()
    return jsonify({"ok": True})


@app.post("/api/mode")
def api_mode():
    engine.set_mode((request.get_json(silent=True) or {}).get("mode", ""))
    return jsonify({"ok": True})


@app.post("/api/settings")
def api_settings():
    data = request.get_json(silent=True) or {}
    if "speak" in data:
        engine.speak = bool(data["speak"])
    if "threshold" in data:
        try:
            engine.threshold = min(0.95, max(0.3, float(data["threshold"])))
        except (TypeError, ValueError):
            pass
    return jsonify({"ok": True})


@app.post("/api/quiz/reset")
def api_quiz_reset():
    engine.reset_quiz()
    return jsonify({"ok": True})


@app.post("/api/reload")
def api_reload():
    return jsonify({"error": engine.load_model()})


@app.get("/video_feed")
def video_feed():
    def stream():
        last = None
        while engine.running:
            jpg = engine.get_jpeg()
            if jpg is not None and jpg is not last:
                last = jpg
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.01)

    resp = Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"SignSense AI running at http://{host}:{port}  (press Ctrl+C to quit)")
    threading.Timer(1.2, lambda: webbrowser.open(URL)).start()
    app.run(host=host, port=port, threaded=True, debug=False)
