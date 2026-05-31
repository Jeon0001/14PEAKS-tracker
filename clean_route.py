import json

INPUT = "route_raw.json"
OUTPUT = "route_clean.json"

with open(INPUT, "r") as f:
    raw = json.load(f)

clean = []


def fix_y(y, last_y):
    # Keep normal altitude values
    if 5000 <= y <= 8999:
        return y

    # OCR giant bad values (59350, etc.)
    if y > 9000:
        if last_y is not None:
            return last_y
        return y

    # Tiny broken values
    if y < 1000:
        if last_y is not None:
            return last_y
        return y

    return y


def fix_x(x, last_x):
    if last_x is None:
        return x

    # Reject huge OCR jumps
    # Example:
    # 8524 -> 6520 -> 8495
    # 8491 -> 5486 -> 8482
    if abs(x - last_x) > 800:
        return last_x

    return x


def fix_z(z, last_z):
    if last_z is None:
        return z

    # Reject bad Z spikes
    if abs(z - last_z) > 300:
        return last_z

    return z


for p in raw:
    last = clean[-1] if clean else None

    last_x = last["x"] if last else None
    last_y = last["y"] if last else None
    last_z = last["z"] if last else None

    fixed = {
        "x": fix_x(p["x"], last_x),
        "y": fix_y(p["y"], last_y),
        "z": fix_z(p["z"], last_z),
    }

    # Avoid duplicate identical points
    if not clean or fixed != clean[-1]:
        clean.append(fixed)

with open(OUTPUT, "w") as f:
    json.dump(clean, f, indent=2)

print(f"Cleaned {len(raw)} raw points into {len(clean)} clean points.")
print(f"Saved {OUTPUT}")