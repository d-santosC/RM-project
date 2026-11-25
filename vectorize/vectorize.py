"""
Turn handwriting from a photo/webcam into pen paths (vectors) for drawing.

Pipeline (the CNC "prepare and vectorize" steps, adapted for a pen robot):
  1. clean_image   -> black/white mask of the ink (contrast, denoise, threshold)
  2. skeletonize   -> 1-pixel-wide centre line of every stroke
  3. trace_strokes -> ordered lists of points (polylines), one per pen stroke
  4. simplify      -> fewer points (Douglas-Peucker), same shape
  5. order_strokes -> reorder/flip strokes so the pen travels less while lifted
  6. export        -> SVG (open in Inkscape/browser), JSON (for the Thymio code),
                      and a PNG preview

Why centre lines and not outlines: tools like Inkscape's "Trace Bitmap" give the
OUTLINE of each letter, which is right for a CNC cutting a shape out, but a pen
would draw every letter twice (both edges of the stroke). A pen robot should
follow the middle of the stroke once, which is what the skeleton gives us.

Usage:
    python vectorize.py --image photo.jpg               # from a file
    python vectorize.py --camera 0                      # SPACE to capture, q to quit
    python vectorize.py --image photo.jpg --width-mm 150

Outputs go to ./output (paths.svg, paths.json, preview.png, mask.png).
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize

# 8 neighbours, 4-connected ones first so walks prefer straight steps
NEIGHBOURS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]


# ---------------------------------------------------------------- 1. clean
def clean_image(bgr, block=31, c=15, min_blob=40):
    """Return a binary mask (255 = ink) from a colour image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # boost local contrast so faint pencil/pen strokes stand out
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # same adaptive threshold as the OCR scripts (high C ignores paper texture)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, c)

    # remove specks, then close small gaps inside strokes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # drop small blobs (noise) by area
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob:
            keep[labels == i] = 255
    return keep


# ---------------------------------------------------------------- 2. skeleton
def skeletonize(mask):
    """1-pixel-wide centre line of the ink, as a bool array."""
    return sk_skeletonize(mask > 0)


# ---------------------------------------------------------------- 3. trace
def _neighbours(skel, y, x):
    h, w = skel.shape
    for dy, dx in NEIGHBOURS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
            yield ny, nx


def trace_strokes(skel, min_len=8):
    """Walk the skeleton into polylines [(x, y), ...] in pixel coordinates."""
    pixels = set(zip(*np.nonzero(skel)))
    degree = {p: sum(1 for _ in _neighbours(skel, *p)) for p in pixels}
    visited = set()
    strokes = []

    def walk(start):
        path = [start]
        visited.add(start)
        # join onto an already-drawn junction so branches don't leave a gap
        for n in _neighbours(skel, *start):
            if n in visited and n != start:
                path.insert(0, n)
                break
        cur = start
        while True:
            nxt = next((n for n in _neighbours(skel, *cur) if n not in visited), None)
            if nxt is None:
                break
            visited.add(nxt)
            path.append(nxt)
            cur = nxt
        return path

    # start at stroke ends first (degree 1), then whatever is left (closed loops like "o")
    starts = sorted(pixels, key=lambda p: (degree[p] != 1, p))
    for p in starts:
        if p not in visited:
            path = walk(p)
            joined = path[0] != p  # starts on an existing stroke = branch
            # short branches are skeleton "spurs" (noise); short isolated
            # pieces are real marks like the dot on an i, so keep those
            if len(path) >= min_len or not joined:
                if len(path) == 1:
                    path = path * 2  # a dot: pen down and up on the spot
                strokes.append([(x, y) for y, x in path])
    return strokes


# ---------------------------------------------------------------- 4. simplify
def simplify(strokes, epsilon=1.5):
    """Douglas-Peucker: keep the shape within `epsilon` pixels, fewer points."""
    out = []
    for s in strokes:
        pts = np.array(s, dtype=np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(pts, epsilon, closed=False).reshape(-1, 2)
        if len(approx) == 1:  # dots collapse to one point; keep them drawable
            approx = np.vstack([approx, approx])
        out.append([tuple(map(float, p)) for p in approx])
    return out


# ---------------------------------------------------------------- 5. order
def order_strokes(strokes, start=(0.0, 0.0)):
    """Greedy nearest-neighbour: pick the closest next stroke (either end)."""
    remaining = list(strokes)
    ordered, pos = [], start
    while remaining:
        best_i, best_rev, best_d = 0, False, math.inf
        for i, s in enumerate(remaining):
            for rev, end in ((False, s[0]), (True, s[-1])):
                d = math.dist(pos, end)
                if d < best_d:
                    best_i, best_rev, best_d = i, rev, d
        s = remaining.pop(best_i)
        s = s[::-1] if best_rev else s
        ordered.append(s)
        pos = s[-1]
    return ordered


def pen_up_distance(strokes, start=(0.0, 0.0)):
    pos, total = start, 0.0
    for s in strokes:
        total += math.dist(pos, s[0])
        pos = s[-1]
    return total


# ---------------------------------------------------------------- scale
def to_mm(strokes, width_mm):
    """Scale pixel paths to millimetres, origin at bottom-left, y pointing up
    (the usual robot/CNC convention; images have y pointing down)."""
    xs = [p[0] for s in strokes for p in s]
    ys = [p[1] for s in strokes for p in s]
    min_x, max_x, max_y = min(xs), max(xs), max(ys)
    min_y = min(ys)
    scale = width_mm / max(max_x - min_x, 1)
    mm = [[(round((x - min_x) * scale, 2), round((max_y - y) * scale, 2)) for x, y in s]
          for s in strokes]
    return mm, scale, (max_x - min_x) * scale, (max_y - min_y) * scale


# ---------------------------------------------------------------- 6. export
def export_svg(strokes_mm, w_mm, h_mm, path):
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_mm:.1f}mm" height="{h_mm:.1f}mm" '
        f'viewBox="0 0 {w_mm:.2f} {h_mm:.2f}">',
        '<g fill="none" stroke="black" stroke-width="0.6" stroke-linecap="round" '
        'stroke-linejoin="round">',
    ]
    for s in strokes_mm:
        # SVG y points down, so flip back for display
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.2f},{h_mm - y:.2f}" for i, (x, y) in enumerate(s))
        lines.append(f'<path d="{d}"/>')
    lines += ["</g>", "</svg>"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def export_json(strokes_mm, w_mm, h_mm, path):
    data = {
        "units": "mm",
        "origin": "bottom-left, x right, y up",
        "width_mm": round(w_mm, 2),
        "height_mm": round(h_mm, 2),
        # each stroke = pen down at the first point, move through the rest, pen up
        "strokes": [[list(p) for p in s] for s in strokes_mm],
    }
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")


def draw_preview(bgr, mask, skel, strokes_px):
    """Side by side: original | cleaned mask | paths (colour = order, grey = pen-up)."""
    h, w = mask.shape
    paths = np.full((h, w, 3), 255, np.uint8)
    pos = None
    for i, s in enumerate(strokes_px):
        pts = np.array(s, np.int32)
        if pos is not None:
            cv2.line(paths, pos, tuple(pts[0]), (200, 200, 200), 1, cv2.LINE_AA)
        hue = int(150 * i / max(len(strokes_px) - 1, 1))
        color = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.polylines(paths, [pts], False, color, 2, cv2.LINE_AA)
        cv2.circle(paths, tuple(pts[0]), 3, (0, 0, 0), -1)  # pen-down point
        pos = tuple(pts[-1])
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_bgr[skel] = (0, 0, 255)
    return np.hstack([bgr, mask_bgr, paths])


# ---------------------------------------------------------------- pipeline
def vectorize(bgr, width_mm=100.0, epsilon=1.5, out_dir="output"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mask = clean_image(bgr)
    skel = skeletonize(mask)
    strokes = trace_strokes(skel)
    if not strokes:
        print("No strokes found - is there writing in the image?")
        return None

    n_raw = sum(len(s) for s in strokes)
    strokes = simplify(strokes, epsilon)
    before = pen_up_distance(strokes)
    strokes = order_strokes(strokes)
    after = pen_up_distance(strokes)

    strokes_mm, scale, w_mm, h_mm = to_mm(strokes, width_mm)
    export_svg(strokes_mm, w_mm, h_mm, out / "paths.svg")
    export_json(strokes_mm, w_mm, h_mm, out / "paths.json")
    cv2.imwrite(str(out / "mask.png"), mask)
    cv2.imwrite(str(out / "preview.png"), draw_preview(bgr, mask, skel, strokes))

    draw_len = sum(math.dist(a, b) for s in strokes_mm for a, b in zip(s, s[1:]))
    print(f"{len(strokes)} strokes, {n_raw} -> {sum(len(s) for s in strokes)} points")
    print(f"size {w_mm:.1f} x {h_mm:.1f} mm, pen-down {draw_len:.0f} mm, "
          f"pen-up {after * scale:.0f} mm (was {before * scale:.0f} mm before ordering)")
    print(f"saved to {out.resolve()}")
    return strokes_mm


def capture_from_camera(index):
    """Show the camera with the same box as the OCR scripts; SPACE captures it."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit("Error opening video stream")
    box_w, box_h = 500, 150
    roi = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        x1, y1 = (w - box_w) // 2, (h - box_h) // 2
        display = cv2.flip(frame, 1)
        cv2.rectangle(display, (x1, y1), (x1 + box_w, y1 + box_h), (0, 255, 0), 2)
        cv2.putText(display, "SPACE: capture   q: quit", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Vectorize - capture", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            roi = frame[y1:y1 + box_h, x1:x1 + box_w].copy()
            break
        if key in (27, ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()
    return roi


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="image file with handwriting")
    src.add_argument("--camera", type=int, help="camera index; SPACE to capture")
    p.add_argument("--width-mm", type=float, default=100.0,
                   help="width of the drawing on paper")
    p.add_argument("--epsilon", type=float, default=1.5,
                   help="simplification tolerance in pixels (higher = fewer points)")
    p.add_argument("--out", default="output")
    args = p.parse_args()

    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"Could not read {args.image}")
    else:
        bgr = capture_from_camera(args.camera)
        if bgr is None:
            return

    if vectorize(bgr, args.width_mm, args.epsilon, args.out):
        preview = cv2.imread(str(Path(args.out) / "preview.png"))
        cv2.imshow("original | mask + skeleton | paths (dots = pen down)", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
