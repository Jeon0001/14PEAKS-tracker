import json
import os
import re
import secrets
import tempfile
import threading
import time
from functools import wraps
from pathlib import Path

import cv2
import pytesseract
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash


BASE_DIR = Path(__file__).resolve().parent
TESSDATA_DIR = BASE_DIR / "Tesseract Data"

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_BYTES", 1_000_000_000))
# Resource note: each upload streams ~1 GB into RAM while OpenCV reads the temp file,
# plus disk space for the temp file itself.  A single replica can comfortably handle
# 1–2 concurrent 1 GB uploads before memory pressure becomes a concern.  Scale to
# multiple replicas (or increase Railway memory limits) for production workloads with
# many simultaneous users.
EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1080
EXPECTED_RESOLUTION_LABEL = "1080p (1920x1080)"
ACCESS_PASSWORD_HASH = os.environ.get(
    "ACCESS_PASSWORD_HASH",
    "scrypt:32768:8:1$Th5CJULbcz4l1cLD$95464b5d94fbfaf853bf870a222012fe24349ddc736c8cbde6e0d0c2e9eab1a0a8f4ebdcc7236b5d56dd28a82267c1b2cbe4c411e40ae85db395c17c8682b595",
)

ALTITUDE_TO_STUDS = float(os.environ.get("ALTITUDE_TO_STUDS", "3.57"))
MIN_ALTITUDE_METERS = int(os.environ.get("MIN_ALTITUDE_METERS", "3000"))
MAX_ALTITUDE_METERS = int(os.environ.get("MAX_ALTITUDE_METERS", "9000"))
MIN_ALTITUDE_STUDS = round(MIN_ALTITUDE_METERS * ALTITUDE_TO_STUDS)
MAX_ALTITUDE_STUDS = round(MAX_ALTITUDE_METERS * ALTITUDE_TO_STUDS)
MAX_X_CHANGE = int(os.environ.get("MAX_X_CHANGE", "300"))
MAX_Y_CHANGE = int(os.environ.get("MAX_Y_CHANGE", "300"))
MAX_Z_CHANGE = int(os.environ.get("MAX_Z_CHANGE", "300"))

# Original live-capture crop was measured from a 2560x1440 display.
REFERENCE_WIDTH = 2560
REFERENCE_HEIGHT = 1440
REFERENCE_HUD_CROP = {
    "left": 1824,
    "top": 1267,
    "width": 222,
    "height": 116,
}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"

JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 15 * 60


def prune_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS

    with JOBS_LOCK:
        stale_ids = [
            job_id
            for job_id, job in JOBS.items()
            if job.get("updatedAt", 0) < cutoff
        ]

        for job_id in stale_ids:
            JOBS.pop(job_id, None)


def update_job(job_id, **updates):
    if not job_id:
        return

    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        job.update(updates)
        job["updatedAt"] = time.time()


def get_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def job_cancelled(job_id):
    if not job_id:
        return False

    with JOBS_LOCK:
        return bool(JOBS.get(job_id, {}).get("cancelled"))


def authenticated():
    return session.get("authenticated") is True


def require_auth(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if authenticated():
            return view(*args, **kwargs)

        if request.path.startswith("/api/"):
            return jsonify({"error": "Password required."}), 401

        return redirect(url_for("login"))

    return wrapped


def configure_tesseract():
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    else:
        for path in (
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ):
            if path.exists():
                pytesseract.pytesseract.tesseract_cmd = str(path)
                break

    if TESSDATA_DIR.exists():
        os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)


def tesseract_config():
    return "--psm 6"


def coord_model_name():
    configured = os.environ.get("COORD_MODEL")
    if configured:
        return configured

    for traineddata in TESSDATA_DIR.glob("*.traineddata"):
        if traineddata.stem.lower() == "2kor":
            return traineddata.stem

    return "eng"


COORD_MODEL = coord_model_name()
configure_tesseract()
OCR_CONFIG = tesseract_config()


def clean_number(value):
    value = value.upper()
    value = value.replace("O", "0")
    value = value.replace("I", "1")
    value = value.replace("L", "1")
    value = re.sub(r"[^0-9-]", "", value)

    if value in ("", "-"):
        return None

    return int(value)


def parse_altitude(text):
    match = re.search(r"(\d{4,5})\s*M", text)

    if not match:
        return None

    altitude_m = clean_number(match.group(1))

    if altitude_m is None:
        return None

    if altitude_m < MIN_ALTITUDE_METERS or altitude_m > MAX_ALTITUDE_METERS:
        return None

    return round(altitude_m * ALTITUDE_TO_STUDS)


def parse_coords(text):
    match = re.search(
        r"X\s*=?\s*(-?\d+).*?[Z27]\s*=?\s*(-?\d+)",
        text,
        re.DOTALL,
    )

    if not match:
        return None

    x = clean_number(match.group(1))
    z = clean_number(match.group(2))

    if x is None or z is None:
        return None

    return x, z


def valid_against_last(points, point):
    if not points:
        return True

    last = points[-1]

    return (
        abs(point["x"] - last["x"]) <= MAX_X_CHANGE
        and abs(point["y"] - last["y"]) <= MAX_Y_CHANGE
        and abs(point["z"] - last["z"]) <= MAX_Z_CHANGE
    )


def clean_route(raw):
    clean = []

    for point in raw:
        last = clean[-1] if clean else None
        last_x = last["x"] if last else None
        last_y = last["y"] if last else None
        last_z = last["z"] if last else None

        fixed = {
            "x": fix_x(point["x"], last_x),
            "y": fix_y(point["y"], last_y),
            "z": fix_z(point["z"], last_z),
        }

        if not clean or fixed != clean[-1]:
            clean.append(fixed)

    return clean


def fix_y(y, last_y):
    if MIN_ALTITUDE_STUDS <= y <= MAX_ALTITUDE_STUDS:
        return y

    if last_y is not None:
        return last_y

    return y


def fix_x(x, last_x):
    if last_x is not None and abs(x - last_x) > 800:
        return last_x

    return x


def fix_z(z, last_z):
    if last_z is not None and abs(z - last_z) > 300:
        return last_z

    return z


def scaled_default_crop(video_width, video_height):
    return {
        "left": round(REFERENCE_HUD_CROP["left"] * video_width / REFERENCE_WIDTH),
        "top": round(REFERENCE_HUD_CROP["top"] * video_height / REFERENCE_HEIGHT),
        "width": round(REFERENCE_HUD_CROP["width"] * video_width / REFERENCE_WIDTH),
        "height": round(REFERENCE_HUD_CROP["height"] * video_height / REFERENCE_HEIGHT),
    }


def request_crop(video_width, video_height):
    crop = scaled_default_crop(video_width, video_height)

    for key in ("left", "top", "width", "height"):
        value = request.form.get(f"crop_{key}", "").strip()
        if value:
            crop[key] = int(value)

    crop["left"] = max(0, min(crop["left"], video_width - 1))
    crop["top"] = max(0, min(crop["top"], video_height - 1))
    crop["width"] = max(1, min(crop["width"], video_width - crop["left"]))
    crop["height"] = max(1, min(crop["height"], video_height - crop["top"]))

    return crop


def extract_route(video_path, sample_interval_seconds, crop, progress_callback=None, cancel_callback=None):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Could not open the uploaded video.")

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
            raise ValueError(
                f"Only {EXPECTED_RESOLUTION_LABEL} videos are accepted. "
                f"This file is {width}x{height}."
            )

        sample_every = max(1, round(fps * sample_interval_seconds))
        raw_points = []
        readable_frames = 0
        sampled_frames = 0
        last_reported_percent = -1

        if progress_callback:
            progress_callback(0, "Preparing video frames...")

        frame_indices = range(0, frame_count, sample_every) if frame_count else None

        if frame_indices is not None:
            iterable = frame_indices
        else:
            iterable = iter(int, 1)

        for frame_index in iterable:
            if cancel_callback and cancel_callback():
                raise ValueError("Route generation was cancelled.")

            if frame_indices is not None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            ok, frame = capture.read()
            if not ok:
                break

            sampled_frames += 1
            x1 = crop["left"]
            y1 = crop["top"]
            x2 = x1 + crop["width"]
            y2 = y1 + crop["height"]
            hud = frame[y1:y2, x1:x2]
            text = pytesseract.image_to_string(hud, lang=COORD_MODEL, config=OCR_CONFIG)

            y = parse_altitude(text)
            coords = parse_coords(text)

            if y is not None and coords is not None:
                readable_frames += 1
                x, z = coords
                point = {"x": x, "y": y, "z": z}

                if valid_against_last(raw_points, point):
                    if not raw_points or point != raw_points[-1]:
                        raw_points.append(point)

            if progress_callback and frame_count:
                percent = min(99, round((frame_index + 1) * 100 / frame_count))
                if percent != last_reported_percent:
                    last_reported_percent = percent
                    progress_callback(
                        percent,
                        f"Reading HUD coordinates... {sampled_frames} frames sampled",
                    )

            if frame_indices is None:
                current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                next_frame = current_frame + sample_every - 1
                capture.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

        if progress_callback:
            progress_callback(99, "Cleaning route points...")

        route = clean_route(raw_points)
        duration = frame_count / fps if fps else None

        if progress_callback:
            progress_callback(100, "Route calculation complete.")

        return {
            "route": route,
            "stats": {
                "rawPoints": len(raw_points),
                "cleanPoints": len(route),
                "sampledFrames": sampled_frames,
                "readableFrames": readable_frames,
                "fps": round(fps, 3),
                "durationSeconds": round(duration, 2) if duration else None,
                "crop": crop,
                "coordModel": COORD_MODEL,
            },
        }
    finally:
        capture.release()


def allowed_video(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.get("/login")
def login():
    if authenticated():
        return redirect(url_for("index"))

    return render_template("login.html", error=None)


@app.post("/login")
def login_submit():
    password = request.form.get("password", "")

    if check_password_hash(ACCESS_PASSWORD_HASH, password):
        session.clear()
        session["authenticated"] = True
        return redirect(url_for("index"))

    return render_template("login.html", error="Incorrect password."), 401


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/")
@require_auth
def index():
    return render_template(
        "index.html",
        max_upload_mb=round(MAX_CONTENT_LENGTH / 1024 / 1024),
        expected_width=EXPECTED_WIDTH,
        expected_height=EXPECTED_HEIGHT,
        default_crop=scaled_default_crop(EXPECTED_WIDTH, EXPECTED_HEIGHT),
    )


@app.get("/images/<path:filename>")
@require_auth
def images(filename):
    return send_from_directory(BASE_DIR / "images", filename)


@app.post("/api/process")
@require_auth
def process_video():
    prune_jobs()
    job_id = request.form.get("job_id", "").strip()
    update_job(
        job_id,
        stage="processing",
        percent=0,
        message="Upload complete. Preparing route calculation...",
        done=False,
        error=None,
    )

    upload = request.files.get("video")
    if upload is None or upload.filename == "":
        update_job(job_id, stage="error", percent=0, message="Upload a video file first.", done=True, error=True)
        return jsonify({"error": "Upload a video file first."}), 400

    if not allowed_video(upload.filename):
        update_job(job_id, stage="error", percent=0, message="Unsupported video type.", done=True, error=True)
        return jsonify({"error": "Unsupported video type."}), 400

    upload.stream.seek(0, 2)
    file_size = upload.stream.tell()
    upload.stream.seek(0)
    if file_size > 1_000_000_000:
        message = "File is too large. Maximum size is 1GB."
        update_job(job_id, stage="error", percent=0, message=message, done=True, error=True)
        return jsonify({"error": message}), 413

    sample_interval = float(request.form.get("sample_interval", "0.5"))
    sample_interval = max(0.1, min(sample_interval, 5.0))

    temp_path = None

    try:
        suffix = Path(upload.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            upload.save(temp_file)

        capture = cv2.VideoCapture(temp_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()

        if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
            message = (
                f"Only {EXPECTED_RESOLUTION_LABEL} videos are accepted. "
                f"This file is {width}x{height}."
            )
            update_job(job_id, stage="error", percent=0, message=message, done=True, error=True)
            return jsonify(
                {
                    "error": message
                }
            ), 400

        crop = request_crop(width, height)
        result = extract_route(
            temp_path,
            sample_interval,
            crop,
            progress_callback=lambda percent, message: update_job(
                job_id,
                stage="processing",
                percent=percent,
                message=message,
                done=False,
                error=None,
            ),
            cancel_callback=lambda: job_cancelled(job_id),
        )
        update_job(
            job_id,
            stage="done",
            percent=100,
            message="Route generated.",
            done=True,
            error=False,
        )
        return app.response_class(
            response=json.dumps(result),
            status=200,
            mimetype="application/json",
        )
    except ValueError as error:
        is_cancelled = job_cancelled(job_id)
        update_job(
            job_id,
            stage="cancelled" if is_cancelled else "error",
            percent=0,
            message=str(error),
            done=True,
            error=not is_cancelled,
        )
        return jsonify({"error": str(error)}), 400
    except pytesseract.TesseractNotFoundError:
        update_job(
            job_id,
            stage="error",
            percent=0,
            message="Tesseract OCR is not installed or is not in PATH.",
            done=True,
            error=True,
        )
        return jsonify({"error": "Tesseract OCR is not installed or is not in PATH."}), 500
    except pytesseract.TesseractError as error:
        update_job(job_id, stage="error", percent=0, message=f"Tesseract OCR failed: {error}", done=True, error=True)
        return jsonify({"error": f"Tesseract OCR failed: {error}"}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/progress/<job_id>")
@require_auth
def job_progress(job_id):
    job = get_job(job_id)

    if not job:
        return jsonify(
            {
                "stage": "waiting",
                "percent": 0,
                "message": "Waiting for upload to finish...",
                "done": False,
                "error": None,
            }
        )

    return jsonify(job)


@app.post("/api/cancel/<job_id>")
@require_auth
def cancel_job(job_id):
    update_job(
        job_id,
        cancelled=True,
        stage="cancelled",
        message="Route generation cancelled.",
        done=True,
        error=False,
    )
    return jsonify({"ok": True})


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": f"Upload is too large. Limit is {round(MAX_CONTENT_LENGTH / 1024 / 1024)} MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
