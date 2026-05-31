import time
import re
import json
import mss
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OUTPUT = "route_raw.json"

COORD_MODEL = "2kor"

ALTITUDE_TO_STUDS = 3.57

MAX_X_CHANGE = 300
MAX_Y_CHANGE = 300
MAX_Z_CHANGE = 300

points = []

HUD_CROP = {
    "left": 1824,
    "top": 1267,
    "width": 222,
    "height": 116
}


def clean_number(s):
    s = s.upper()
    s = s.replace("O", "0")
    s = s.replace("I", "1")
    s = s.replace("L", "1")
    s = re.sub(r"[^0-9-]", "", s)

    if s in ("", "-"):
        return None

    return int(s)


def parse_altitude(text):
    match = re.search(r"(\d{4,5})\s*M", text)

    if not match:
        return None

    altitude_m = clean_number(match.group(1))

    if altitude_m is None:
        return None

    if altitude_m < 3000 or altitude_m > 9000:
        return None

    return round(altitude_m * ALTITUDE_TO_STUDS)


def parse_coords(text):
    match = re.search(
        r"X\s*=?\s*(-?\d+).*?[Z27]\s*=?\s*(-?\d+)",
        text,
        re.DOTALL
    )

    if not match:
        return None

    x = clean_number(match.group(1))
    z = clean_number(match.group(2))

    if x is None or z is None:
        return None

    return x, z


def valid_against_last(point):
    if not points:
        return True

    last = points[-1]

    dx = abs(point["x"] - last["x"])
    dy = abs(point["y"] - last["y"])
    dz = abs(point["z"] - last["z"])

    if dx > MAX_X_CHANGE:
        print("IGNORED bad X:", point, "last:", last)
        return False

    if dy > MAX_Y_CHANGE:
        print("IGNORED bad Y:", point, "last:", last)
        return False

    if dz > MAX_Z_CHANGE:
        print("IGNORED bad Z:", point, "last:", last)
        return False

    return True


def save_points():
    with open(OUTPUT, "w") as f:
        json.dump(points, f, indent=2)


with mss.mss() as sct:
    print("Tracking route. Press Ctrl+C to stop.")

    try:
        while True:
            screenshot = sct.grab(HUD_CROP)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

            img.save("debug_crop.png")

            text = pytesseract.image_to_string(
                img,
                lang=COORD_MODEL,
                config="--psm 6"
            )

            y = parse_altitude(text)
            coords = parse_coords(text)

            print("OCR:", repr(text))

            if y is None or coords is None:
                print("IGNORED unreadable")
                time.sleep(0.5)
                continue

            x, z = coords

            point = {
                "x": x,
                "y": y,
                "z": z
            }

            print("POINT:", point)

            if valid_against_last(point):
                if not points or point != points[-1]:
                    points.append(point)
                    print("SAVED:", point)
                    save_points()

            time.sleep(0.5)

    except KeyboardInterrupt:
        save_points()
        print("Saved", OUTPUT)