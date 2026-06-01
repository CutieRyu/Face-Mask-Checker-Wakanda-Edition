# ============================================================
# app.py — Flask Backend for Face Mask Detector
#          (OpenCV Multi-Scale + MediaPipe fallback)
# ============================================================
# SETUP:
#   pip install flask tensorflow opencv-python numpy pillow flask-cors
#
# RUN:
#   python app.py
#   Then open http://localhost:5000
# ============================================================

import os
import io
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH = "mask_detector.h5"
IMG_SIZE   = (128, 128)
THRESHOLD  = 0.5

# ─── LOAD MASK MODEL ──────────────────────────────────────
print("[*] Loading mask detector model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[✓] Mask model loaded!\n")

# ─── LOAD FACE DETECTORS ──────────────────────────────────
# Two cascades — frontal + profile for better coverage
print("[*] Loading face detectors...")
frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
alt_cascade     = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
print("[✓] Face detectors loaded!\n")

# ─── HELPER: Decode base64 → numpy ────────────────────────
def decode_image(image_data):
    if isinstance(image_data, str):
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
    else:
        image_bytes = image_data
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)

# ─── HELPER: Preprocess face crop for CNN ─────────────────
def preprocess_face(face_crop_np):
    img = Image.fromarray(face_crop_np).resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ─── HELPER: Remove overlapping boxes ─────────────────────
def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []
    boxes   = np.array(boxes)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w   = np.maximum(0, xx2 - xx1)
        h   = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / areas[order[1:]]
        order   = order[np.where(overlap <= overlap_thresh)[0] + 1]
    return boxes[keep].tolist()

# ─── HELPER: Detect all faces ─────────────────────────────
def detect_faces(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = img_np.shape[:2]

    # Scale image down if too large — speeds up detection
    scale  = 1.0
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        gray  = cv2.resize(gray, (int(w * scale), int(h * scale)))

    all_faces = []

    # Try multiple scale factors to catch both close and distant faces
    for scaleFactor in [1.05, 1.1, 1.2]:
        f1 = frontal_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=4, minSize=(20, 20))
        f2 = alt_cascade.detectMultiScale    (gray, scaleFactor=scaleFactor, minNeighbors=4, minSize=(20, 20))
        f3 = profile_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=4, minSize=(20, 20))
        for faces in [f1, f2, f3]:
            if len(faces) > 0:
                for (x, y, fw, fh) in faces:
                    all_faces.append([int(x/scale), int(y/scale), int(fw/scale), int(fh/scale)])

    # Remove duplicates with NMS
    return non_max_suppression(all_faces, overlap_thresh=0.3)

# ─── ROUTES ───────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image received"}), 400

        img_np = decode_image(data["image"])
        h, w   = img_np.shape[:2]

        faces = detect_faces(img_np)

        if not faces:
            return jsonify({"faces": [], "no_mask_detected": False})

        faces_out        = []
        no_mask_detected = False

        for (x, y, fw, fh) in faces:
            pad = int(min(fw, fh) * 0.15)
            x1p = max(0, x  - pad)
            y1p = max(0, y  - pad)
            x2p = min(w, x + fw + pad)
            y2p = min(h, y + fh + pad)

            face_crop  = img_np[y1p:y2p, x1p:x2p]
            if face_crop.size == 0:
                continue

            input_data = preprocess_face(face_crop)
            raw        = float(model.predict(input_data, verbose=0)[0][0])

            if raw <= THRESHOLD:
                label      = "Mask"
                confidence = round((1 - raw) * 100, 1)
                color      = "green"
            else:
                label      = "No Mask"
                confidence = round(raw * 100, 1)
                color      = "red"
                no_mask_detected = True

            faces_out.append({
                "label":      label,
                "confidence": confidence,
                "color":      color,
                "raw":        raw,
                "box": {"x": x1p, "y": y1p, "w": x2p - x1p, "h": y2p - y1p}
            })

        return jsonify({
            "faces":            faces_out,
            "no_mask_detected": no_mask_detected
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ─── RUN ──────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running at http://localhost:{port}")
    print("   Press CTRL+C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=False)
