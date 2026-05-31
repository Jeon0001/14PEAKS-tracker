# 14PEAKS Route Builder

Upload a 1080p (1920x1080) route video, OCR the HUD coordinates, and render the extracted route as an interactive Three.js path.

## Local Run

Install Tesseract OCR first, then install the Python dependencies:

```powershell
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

## Railway Deploy

Railway can deploy this as a Python app. The included files are the important pieces:

- `Procfile` runs Gunicorn.
- `requirements.txt` installs Flask, OpenCV, and OCR bindings.
- `nixpacks.toml` installs the system Tesseract binary.
- `Tesseract Data/` provides the custom OCR traineddata files.

The app accepts only `1080p (1920x1080)` videos. You can configure upload size and OCR model with Railway environment variables:

```text
MAX_UPLOAD_BYTES=786432000
COORD_MODEL=2Kor
```

Uploaded videos are saved only to a temporary file while OCR runs. The temp file is deleted in a `finally` block after processing succeeds or fails. Generated route points are returned directly to the browser and are not persisted on the server.

# Created by .jeon and ylcxzar