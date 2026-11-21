"""
Live webcam OCR using a local vision model served by Ollama.

The camera feed keeps running at full speed; the box region is sent to the
model in a background thread, and the latest answer is drawn on the frame.

Usage:
    python vlm_ocr.py                      # default model (gemma3:4b)
    python vlm_ocr.py --model gemma4:e2b   # any Ollama model with "vision"
    python vlm_ocr.py --manual             # only send when SPACE is pressed

Keys: q / ESC quit, SPACE send the current box now.
"""
import argparse
import base64
import json
import threading
import time
import urllib.request

import cv2

# 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 (::1) first,
# which Ollama doesn't listen on, and that adds ~2 s to every request.
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

PROMPT = (
    "Read the handwritten text in this image. "
    "Reply with ONLY the exact letters you see, no punctuation, "
    "no explanation. If there is no writing, reply with ---."
)


def ask_model(model, jpeg_bytes, timeout=60):
    payload = {
        "model": model,
        "prompt": PROMPT,
        "images": [base64.b64encode(jpeg_bytes).decode("ascii")],
        "stream": False,
        "think": False,          # ignored by models without thinking
        "keep_alive": "10m",     # keep the model loaded between requests
        # small context so the whole model fits in GPU memory (default is huge)
        "options": {"temperature": 0, "num_predict": 64, "num_ctx": 2048},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"].strip()


def draw_text_fitted(img, text, x, y_bottom, max_width,
                     scale=0.9, min_scale=0.5, color=(0, 255, 0), thickness=2):
    """Draw text so it always fits in max_width.

    Shrinks the font first; if it's still too long, wraps onto several lines.
    The last line sits at y_bottom and earlier lines stack upwards.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = lambda s, sc: cv2.getTextSize(s, font, sc, thickness)[0][0]

    while scale > min_scale and width(text, scale) > max_width:
        scale -= 0.05

    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and width(trial, scale) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)

    line_h = cv2.getTextSize("Ag", font, scale, thickness)[0][1] + 8
    y = y_bottom - line_h * (len(lines) - 1)
    for line in lines:
        cv2.putText(img, line, (x, max(y, line_h)), font, scale, color, thickness)
        y += line_h


def ink_ratio(image):
    """Fraction of the box covered by dark, stroke-like marks.

    Small vision models make up an answer (e.g. "1234567890") when the box is
    empty, so we only ask the model when there's something that looks like ink.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # a high C means only pixels clearly darker than their surroundings count,
    # so faint texture (wood grain, paper) is ignored
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 15)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ink = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 40)
    return ink / (image.shape[0] * image.shape[1])


class OcrWorker:
    """Runs one request at a time in the background; keeps the latest result."""

    def __init__(self, model):
        self.model = model
        self.text = "---"
        self.latency = 0.0
        self.busy = False
        self.error = None
        self._lock = threading.Lock()

    def submit(self, image):
        with self._lock:
            if self.busy:
                return False
            self.busy = True
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        threading.Thread(target=self._run, args=(buf.tobytes(),), daemon=True).start()
        return True

    def _run(self, jpeg_bytes):
        start = time.perf_counter()
        try:
            text = ask_model(self.model, jpeg_bytes)
            self.text = text or "---"
            self.error = None
        except Exception as e:
            self.error = str(e)
        finally:
            self.latency = time.perf_counter() - start
            self.busy = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:0.8b-q8_0")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1.0,
                        help="min seconds between automatic requests")
    parser.add_argument("--manual", action="store_true",
                        help="only send when SPACE is pressed")
    parser.add_argument("--min-ink", type=float, default=0.002,
                        help="min fraction of the box that must look like ink "
                             "before the model is asked (0.002 = 0.2%%)")
    args = parser.parse_args()

    # fail early with a clear message if the model isn't installed in Ollama
    with urllib.request.urlopen(OLLAMA_URL.replace("/generate", "/tags"), timeout=10) as resp:
        installed = [m["name"] for m in json.loads(resp.read())["models"]]
    if args.model not in installed:
        print(f"Model '{args.model}' is not installed in Ollama. Installed models:")
        print("  " + "\n  ".join(installed))
        print(f"Install it with:  ollama pull {args.model}")
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error opening video stream")
        return

    worker = OcrWorker(args.model)
    print(f"VLM OCR with {args.model} — q/ESC quit, SPACE send now")

    # warm up: load the model so the first real request isn't slow
    threading.Thread(
        target=lambda: urllib.request.urlopen(urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps({"model": args.model, "keep_alive": "10m",
                             "options": {"num_ctx": 2048}}).encode(),
            headers={"Content-Type": "application/json"}), timeout=120),
        daemon=True).start()

    last_sent = 0.0
    box_w, box_h = 500, 150

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        h, w, _ = frame.shape
        x1, y1 = (w - box_w) // 2, (h - box_h) // 2
        x2, y2 = x1 + box_w, y1 + box_h
        roi = frame[y1:y2, x1:x2].copy()   # unflipped, so text reads correctly

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

        now = time.time()
        ink = ink_ratio(roi)
        has_ink = ink >= args.min_ink
        if key == ord(' ') or (not args.manual and now - last_sent >= args.interval):
            if not has_ink:
                worker.text = "---"      # empty box: don't ask the model
                last_sent = now
            elif worker.submit(roi):
                last_sent = now

        display = cv2.flip(frame, 1)   # mirror only for display
        color = (0, 165, 255) if worker.busy else (0, 255, 0)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        draw_text_fitted(display, f"Word: {worker.text}", 10, y1 - 10,
                         max_width=w - 20)
        status = f"{args.model} | {worker.latency:.2f}s | ink {ink * 100:.1f}%"
        if worker.error:
            status = f"Error: {worker.error[:60]}"
        cv2.putText(display, status, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("VLM Writing Recognition", display)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
