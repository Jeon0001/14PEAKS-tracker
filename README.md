# 14PEAKS Route Visualizer

Upload a 1080p (1920x1080) route video, OCR the HUD coordinates, and render the extracted route as an interactive Three.js path.

## Local Run

Install Tesseract OCR first, then install the Python dependencies:

```powershell
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

## Railway Deploy

Railway can deploy this app directly from GitHub. This repo includes a `Dockerfile`, so Railway will build a container that installs Python, Tesseract OCR, OpenCV system libraries, and the Python dependencies.

- `Dockerfile` installs Tesseract and runs Gunicorn.
- `Procfile` is kept as a fallback start command for Python/Nixpacks-style deploys.
- `requirements.txt` installs Flask, Gunicorn, OpenCV, and OCR bindings.
- `Tesseract Data/` provides the custom OCR traineddata files.
- `images/` provides the crop example image used by the UI.

### Steps

1. Push this repo to GitHub.
2. Go to Railway and create a new project.
3. Choose **Deploy from GitHub repo**.
4. Select this repository.
5. Railway should detect the `Dockerfile` and build the app.
6. After deployment, open the generated Railway domain.

### Environment Variables

The app accepts only `1080p (1920x1080)` videos. Optional Railway variables:

```text
MAX_UPLOAD_BYTES=1200000000
COORD_MODEL=2Kor
SECRET_KEY=replace-with-a-long-random-string
SESSION_COOKIE_SECURE=true
ACCESS_PASSWORD_HASH=optional-password-hash-override
```

`MAX_UPLOAD_BYTES` controls max upload size. The default is about 750 MB.

`COORD_MODEL` controls the Tesseract traineddata model. Leave it unset unless you need to force a specific file from `Tesseract Data/`.

`SECRET_KEY` signs login sessions. Set this in Railway so beta login sessions survive deploys/restarts.

`SESSION_COOKIE_SECURE=true` tells browsers to send the login cookie only over HTTPS.

`ACCESS_PASSWORD_HASH` can override the built-in beta password hash without storing a plaintext password in the repo.

Uploaded videos are saved only to a temporary file while OCR runs. The temp file is deleted in a `finally` block after processing succeeds or fails. Generated route points are returned directly to the browser and are not persisted on the server.

### Notes

- Large videos can take a while to process, so Gunicorn is configured with a longer timeout.
- The app uses one Gunicorn worker by default to avoid multiple large OCR/video jobs exhausting small Railway containers.
- If deployment fails, check the Railway build logs first for Tesseract or OpenCV package errors.

# Created by .jeon and ylcxzar
