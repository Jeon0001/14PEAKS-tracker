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
import numpy as np
import pytesseract
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash


BASE_DIR = Path(__file__).resolve().parent
TESSDATA_DIR = BASE_DIR / "Tesseract Data"
UPLOAD_TMP_DIR = Path(os.environ.get("UPLOAD_TMP_DIR", tempfile.gettempdir()))

ALLOWED_EXTENSIONS = {".mp4"}
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_BYTES", 1_000_000_000))
TEMP_UPLOAD_PREFIX = "14peaks-upload-"
UPLOAD_TMP_DIR_CONFIGURED = "UPLOAD_TMP_DIR" in os.environ
STALE_UPLOAD_FILE_SECONDS = int(os.environ.get("STALE_UPLOAD_FILE_SECONDS", 3 * 60 * 60))
UPLOAD_CLEANUP_INTERVAL_SECONDS = int(os.environ.get("UPLOAD_CLEANUP_INTERVAL_SECONDS", 30 * 60))
# Resource note: each upload streams ~1 GB into RAM while OpenCV reads the temp file,
# plus disk space for the temp file itself.  A single replica can comfortably handle
# 1–2 concurrent 1 GB uploads before memory pressure becomes a concern.  Scale to
# multiple replicas (or increase Railway memory limits) for production workloads with
# many simultaneous users.
RECOMMENDED_RESOLUTION_LABEL = "720p or 1080p"
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
UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)

JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 15 * 60

# Tracks in-progress chunked uploads until processing starts.
CHUNK_UPLOADS = {}
CHUNK_UPLOADS_LOCK = threading.Lock()
CHUNK_UPLOAD_TTL_SECONDS = 30 * 60
CHUNK_SIZE = int(os.environ.get("UPLOAD_CHUNK_SIZE_BYTES", 5 * 1024 * 1024))
ACTIVE_PROCESSING_PATHS = set()
ACTIVE_PROCESSING_PATHS_LOCK = threading.Lock()


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


def delete_file(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def cleanup_stale_upload_files(max_age_seconds=STALE_UPLOAD_FILE_SECONDS):
    cutoff = time.time() - max_age_seconds
    active_paths = protected_upload_paths()

    for path in UPLOAD_TMP_DIR.iterdir():
        try:
            stat = path.stat()
        except OSError:
            continue

        if (
            path.is_file()
            and managed_temp_upload_file(path)
            and str(path) not in active_paths
            and path.suffix.lower() in ALLOWED_EXTENSIONS
            and stat.st_mtime < cutoff
        ):
            delete_file(path)


def managed_temp_upload_file(path):
    return path.name.startswith(TEMP_UPLOAD_PREFIX)


def protected_upload_paths():
    with CHUNK_UPLOADS_LOCK:
        chunk_paths = {
            upload_state.get("temp_path")
            for upload_state in CHUNK_UPLOADS.values()
            if upload_state.get("temp_path")
        }

    with ACTIVE_PROCESSING_PATHS_LOCK:
        processing_paths = set(ACTIVE_PROCESSING_PATHS)

    return chunk_paths | processing_paths


def start_upload_cleanup_thread():
    cleanup_stale_upload_files()

    def cleanup_loop():
        while True:
            time.sleep(UPLOAD_CLEANUP_INTERVAL_SECONDS)
            cleanup_stale_upload_files()

    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()


def cleanup_chunk_upload(job_id):
    with CHUNK_UPLOADS_LOCK:
        upload_state = CHUNK_UPLOADS.pop(job_id, None)

    if upload_state:
        delete_file(upload_state.get("temp_path"))


def prune_chunk_uploads():
    cutoff = time.time() - CHUNK_UPLOAD_TTL_SECONDS

    with CHUNK_UPLOADS_LOCK:
        stale_uploads = [
            (job_id, upload_state)
            for job_id, upload_state in CHUNK_UPLOADS.items()
            if upload_state.get("updatedAt", 0) < cutoff
        ]

        for job_id, _upload_state in stale_uploads:
            CHUNK_UPLOADS.pop(job_id, None)

    for _job_id, upload_state in stale_uploads:
        delete_file(upload_state.get("temp_path"))


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


def traineddata_model_name(env_name, preferred_name, fallback="eng"):
    configured = os.environ.get(env_name)
    if configured:
        return configured

    for traineddata in TESSDATA_DIR.glob("*.traineddata"):
        if traineddata.stem.lower() == preferred_name.lower():
            return traineddata.stem

    return fallback


COORD_MODEL = traineddata_model_name("COORD_MODEL", "2Kor")
ALTITUDE_MODEL = traineddata_model_name("ALTITUDE_MODEL", "2Kalti", fallback=COORD_MODEL)
configure_tesseract()
COORD_OCR_CONFIG = "--psm 6"
COORD_LINE_OCR_CONFIG = "--psm 13"
ALTITUDE_OCR_CONFIG = "--psm 13"


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

        if width <= 0 or height <= 0:
            raise ValueError("Could not read the uploaded video's resolution.")

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
            ocr_result = read_hud_text(hud)
            y = ocr_result["altitude"]
            coords = ocr_result["coords"]

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
                "altitudeModel": ALTITUDE_MODEL,
            },
        }
    finally:
        capture.release()


def read_hud_text(hud):
    coord_texts = read_coord_text_candidates(hud)
    altitude_texts = read_altitude_text_candidates(hud)
    altitude = first_parsed_altitude(altitude_texts + coord_texts)
    coords = first_parsed_coords(coord_texts)

    return {
        "text": "\n".join(text.strip() for text in [*coord_texts, *altitude_texts] if text.strip()),
        "altitude": altitude,
        "coords": coords,
    }


def read_coord_text_candidates(hud):
    height, width = hud.shape[:2]
    candidates = []

    # Coordinates usually sit at the bottom of the selected HUD crop. On 720p
    # footage, extra altitude/direction text in the full crop can confuse 2Kor,
    # so try coordinate-heavy slices before falling back to the full crop.
    regions = (
        hud[max(0, round(height * 0.55)) :, :],
        hud[max(0, round(height * 0.48)) :, :],
        hud[max(0, round(height * 0.40)) :, :],
        hud,
    )

    for index, region in enumerate(regions):
        if region.size == 0:
            continue
        config = COORD_LINE_OCR_CONFIG if index < len(regions) - 1 else COORD_OCR_CONFIG
        text = pytesseract.image_to_string(region, lang=COORD_MODEL, config=config)
        candidates.append(text)

    return candidates


def read_altitude_text_candidates(hud):
    height, width = hud.shape[:2]
    candidates = []

    # The user crop usually contains altitude near the top, direction in the
    # middle, and X/Z coordinates near the bottom. Kalti works best as a single
    # text-line recognizer, so try altitude-heavy slices instead of the full HUD.
    regions = (
        hud[: max(1, round(height * 0.55)), :],
        hud[: max(1, round(height * 0.45)), :],
        hud[max(0, round(height * 0.12)) : max(1, round(height * 0.58)), :],
        hud,
    )

    for region in regions:
        if region.size == 0:
            continue
        text = pytesseract.image_to_string(region, lang=ALTITUDE_MODEL, config=ALTITUDE_OCR_CONFIG)
        candidates.append(text)

    return candidates


def first_parsed_altitude(texts):
    for text in texts:
        altitude = parse_altitude(text)
        if altitude is not None:
            return altitude

    return None


def first_parsed_coords(texts):
    for text in texts:
        coords = parse_coords(text)
        if coords is not None:
            return coords

    return None


def allowed_video(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


start_upload_cleanup_thread()


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
        upload_chunk_size=CHUNK_SIZE,
        recommended_resolution_label=RECOMMENDED_RESOLUTION_LABEL,
        default_crop=scaled_default_crop(1920, 1080),
    )


@app.get("/images/<path:filename>")
@require_auth
def images(filename):
    return send_from_directory(BASE_DIR / "images", filename)


@app.post("/api/upload-chunk")
@require_auth
def upload_chunk():
    prune_chunk_uploads()
    """Receive a single chunk of a chunked video upload.

    Expected form fields:
      - job_id:       unique job identifier
      - chunk_index:  0-based index of this chunk
      - total_chunks: total number of chunks
      - filename:     original filename (used for extension)
      - chunk:        the binary chunk data (file field)
    """
    job_id = request.form.get("job_id", "").strip()
    if not job_id:
        return jsonify({"error": "Missing job_id."}), 400

    try:
        chunk_index = int(request.form.get("chunk_index", -1))
        total_chunks = int(request.form.get("total_chunks", 0))
        total_size = int(request.form.get("total_size", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid chunk upload parameters."}), 400

    if chunk_index < 0 or total_chunks <= 0 or chunk_index >= total_chunks:
        return jsonify({"error": "Invalid chunk parameters."}), 400

    if total_size <= 0 or total_size > MAX_CONTENT_LENGTH:
        return jsonify({"error": f"File is too large. Limit is {round(MAX_CONTENT_LENGTH / 1024 / 1024)} MB."}), 413

    if total_chunks > ((MAX_CONTENT_LENGTH + CHUNK_SIZE - 1) // CHUNK_SIZE):
        return jsonify({"error": "Too many upload chunks."}), 413

    filename = request.form.get("filename", "").strip()
    if not filename or not allowed_video(filename):
        return jsonify({"error": "Unsupported or missing filename."}), 400

    chunk_file = request.files.get("chunk")
    if chunk_file is None:
        return jsonify({"error": "Missing chunk data."}), 400

    suffix = Path(filename).suffix.lower()

    with CHUNK_UPLOADS_LOCK:
        if job_id not in CHUNK_UPLOADS:
            temp_fd, temp_path = tempfile.mkstemp(
                prefix=TEMP_UPLOAD_PREFIX,
                suffix=suffix,
                dir=UPLOAD_TMP_DIR,
            )
            os.close(temp_fd)
            CHUNK_UPLOADS[job_id] = {
                "temp_path": temp_path,
                "received_indexes": set(),
                "total": total_chunks,
                "total_size": total_size,
                "suffix": suffix,
                "updatedAt": time.time(),
            }
        upload_state = CHUNK_UPLOADS[job_id]

        if upload_state["total"] != total_chunks or upload_state["total_size"] != total_size:
            return jsonify({"error": "Chunk upload metadata changed during upload."}), 400

    # Write this chunk at the correct byte offset.
    chunk_data = chunk_file.read()
    offset = chunk_index * CHUNK_SIZE

    if offset + len(chunk_data) > total_size:
        return jsonify({"error": "Chunk exceeds declared file size."}), 400

    with open(upload_state["temp_path"], "r+b") as fh:
        fh.seek(offset)
        fh.write(chunk_data)

    with CHUNK_UPLOADS_LOCK:
        upload_state["received_indexes"].add(chunk_index)
        upload_state["updatedAt"] = time.time()
        received = len(upload_state["received_indexes"])

    update_job(
        job_id,
        stage="uploading",
        percent=round(received * 100 / total_chunks),
        message=f"Receiving chunks ({received}/{total_chunks})...",
        done=False,
        error=None,
    )

    return jsonify({"ok": True, "received": received, "total": total_chunks, "complete": received >= total_chunks})


@app.post("/api/test-ocr")
@require_auth
def test_ocr():
    image = request.files.get("image")

    if image is None:
        return jsonify({"error": "Send a cropped HUD image first."}), 400

    image_bytes = np.frombuffer(image.read(), dtype=np.uint8)
    hud = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if hud is None or hud.size == 0:
        return jsonify({"error": "Could not read the cropped HUD image."}), 400

    try:
        result = read_hud_text(hud)
    except pytesseract.TesseractNotFoundError:
        return jsonify({"error": "Tesseract OCR is not installed or is not in PATH."}), 500
    except pytesseract.TesseractError as error:
        return jsonify({"error": f"Tesseract OCR failed: {error}"}), 500

    coords = result["coords"]
    return jsonify(
        {
            "text": result["text"],
            "altitude": result["altitude"],
            "x": coords[0] if coords else None,
            "z": coords[1] if coords else None,
            "success": result["altitude"] is not None and coords is not None,
        }
    )


def _run_processing(job_id, temp_path, sample_interval, crop_params):
    """Background thread: validate video dimensions, extract route, update job state."""
    with ACTIVE_PROCESSING_PATHS_LOCK:
        ACTIVE_PROCESSING_PATHS.add(temp_path)

    try:
        capture = cv2.VideoCapture(temp_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()

        if width <= 0 or height <= 0:
            message = "Could not read the uploaded video's resolution."
            update_job(job_id, stage="error", percent=0, message=message, done=True, error=True)
            return

        crop = {
            "left": max(0, min(crop_params["left"], width - 1)),
            "top": max(0, min(crop_params["top"], height - 1)),
            "width": max(1, min(crop_params["width"], width - max(0, min(crop_params["left"], width - 1)))),
            "height": max(1, min(crop_params["height"], height - max(0, min(crop_params["top"], height - 1)))),
        }

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
            result=result,
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
    except pytesseract.TesseractNotFoundError:
        update_job(
            job_id,
            stage="error",
            percent=0,
            message="Tesseract OCR is not installed or is not in PATH.",
            done=True,
            error=True,
        )
    except pytesseract.TesseractError as error:
        update_job(job_id, stage="error", percent=0, message=f"Tesseract OCR failed: {error}", done=True, error=True)
    finally:
        with ACTIVE_PROCESSING_PATHS_LOCK:
            ACTIVE_PROCESSING_PATHS.discard(temp_path)
        delete_file(temp_path)


@app.post("/api/process")
@require_auth
def process_video():
    prune_jobs()
    prune_chunk_uploads()
    job_id = request.form.get("job_id", "").strip()

    # --- Chunked-upload path: file was already assembled via /api/upload-chunk ---
    with CHUNK_UPLOADS_LOCK:
        chunk_state = CHUNK_UPLOADS.pop(job_id, None)

    if chunk_state is not None:
        temp_path = chunk_state["temp_path"]

        if len(chunk_state["received_indexes"]) < chunk_state["total"]:
            message = "Not all chunks have been received yet."
            update_job(job_id, stage="error", percent=0, message=message, done=True, error=True)
            delete_file(temp_path)
            return jsonify({"error": message}), 400

        sample_interval = float(request.form.get("sample_interval", "0.5"))
        sample_interval = max(0.1, min(sample_interval, 5.0))

        try:
            crop_params = {
                "left": int(request.form.get("crop_left", 0)),
                "top": int(request.form.get("crop_top", 0)),
                "width": int(request.form.get("crop_width", 100)),
                "height": int(request.form.get("crop_height", 100)),
            }
        except (TypeError, ValueError):
            crop_params = {"left": 0, "top": 0, "width": 100, "height": 100}

        update_job(
            job_id,
            stage="processing",
            percent=0,
            message="Upload complete. Preparing route calculation...",
            done=False,
            error=None,
        )

        thread = threading.Thread(
            target=_run_processing,
            args=(job_id, temp_path, sample_interval, crop_params),
            daemon=True,
        )
        thread.start()

        # Long-poll: wait for the background thread to finish (up to 10 min)
        thread.join(timeout=600)

        job = get_job(job_id)
        if job is None or not job.get("done"):
            update_job(
                job_id,
                cancelled=True,
                stage="error",
                percent=0,
                message="Processing timed out.",
                done=True,
                error=True,
            )
            return jsonify({"error": "Processing timed out."}), 504

        if job.get("error"):
            return jsonify({"error": job.get("message", "Route generation failed.")}), 400

        result = job.get("result")
        if not result:
            return jsonify({"error": "Route generation produced no result."}), 500

        return app.response_class(
            response=json.dumps(result),
            status=200,
            mimetype="application/json",
        )

    # --- Legacy single-request path (kept for backwards compatibility) ---
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
    if file_size > MAX_CONTENT_LENGTH:
        message = f"File is too large. Maximum size is {round(MAX_CONTENT_LENGTH / 1024 / 1024)} MB."
        update_job(job_id, stage="error", percent=0, message=message, done=True, error=True)
        return jsonify({"error": message}), 413

    sample_interval = float(request.form.get("sample_interval", "0.5"))
    sample_interval = max(0.1, min(sample_interval, 5.0))

    temp_path = None

    try:
        suffix = Path(upload.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(
            delete=False,
            prefix=TEMP_UPLOAD_PREFIX,
            suffix=suffix,
            dir=UPLOAD_TMP_DIR,
        ) as temp_file:
            temp_path = temp_file.name
            upload.save(temp_file)

        with ACTIVE_PROCESSING_PATHS_LOCK:
            ACTIVE_PROCESSING_PATHS.add(temp_path)

        capture = cv2.VideoCapture(temp_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()

        if width <= 0 or height <= 0:
            message = "Could not read the uploaded video's resolution."
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
        with ACTIVE_PROCESSING_PATHS_LOCK:
            ACTIVE_PROCESSING_PATHS.discard(temp_path)
        delete_file(temp_path)


@app.get("/api/progress/<job_id>")
@require_auth
def job_progress(job_id):
    prune_chunk_uploads()
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
    cleanup_chunk_upload(job_id)
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
