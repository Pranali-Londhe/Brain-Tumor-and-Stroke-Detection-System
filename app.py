"""
NeuroDetect AI - Flask backend

Serves the interactive UI and two inference endpoints:
    POST /api/predict/tumor   - brain tumor MRI classification
    POST /api/predict/stroke  - brain stroke CT classification

If a trained model file exists in saved_models/, it is loaded and used for
real inference + Grad-CAM. If not, the endpoint runs in DEMO MODE: it uses an
untrained (randomly-initialized) network of the *same real architecture* so the
UI is fully interactive end-to-end, but the response is clearly flagged
"demo_mode": true and must not be treated as a real prediction. This keeps the
app honest - see README.md for how to train real weights.
"""

import os
import io
import base64

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from PIL import Image

import tensorflow as tf

from models.tumor_cnn import build_transfer_model as build_tumor_model, IMG_SIZE as TUMOR_SIZE, CLASS_NAMES as TUMOR_CLASSES
from models.stroke_cnn import build_transfer_model as build_stroke_model, IMG_SIZE as STROKE_SIZE, CLASS_NAMES as STROKE_CLASSES
from models.gradcam import make_gradcam_heatmap, overlay_heatmap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB uploads

_model_cache = {}


def get_model(task):
    """Lazily load a trained model if present, else build an untrained one
    (demo mode) of the identical architecture used for training."""
    if task in _model_cache:
        return _model_cache[task]

    path = os.path.join(SAVED_MODELS_DIR, f"{task}_model.keras")
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        demo_mode = False
    else:
        if task == "tumor":
            model = build_tumor_model(input_shape=TUMOR_SIZE + (3,), num_classes=len(TUMOR_CLASSES))
        else:
            model = build_stroke_model(input_shape=STROKE_SIZE + (3,), num_classes=len(STROKE_CLASSES))
        demo_mode = True

    _model_cache[task] = (model, demo_mode)
    return _model_cache[task]


def read_image_from_request():
    file = request.files.get("image")
    if file is None:
        return None, None
    pil_img = Image.open(file.stream).convert("RGB")
    return pil_img, np.array(pil_img)


def prepare_batch(np_img, target_size):
    resized = cv2.resize(np_img, target_size)
    batch = np.expand_dims(resized.astype("float32"), axis=0)
    return batch, resized


def encode_image_b64(bgr_or_rgb_img, is_bgr=False):
    img = bgr_or_rgb_img
    if is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img.astype("uint8"))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_inference(task, class_names, target_size):
    pil_img, np_img = read_image_from_request()
    if np_img is None:
        return jsonify({"error": "No image uploaded. Send a file under form field 'image'."}), 400

    model, demo_mode = get_model(task)

    batch, resized_rgb = prepare_batch(np_img, target_size)
    preds = model.predict(batch, verbose=0)[0]

    pred_index = int(np.argmax(preds))
    confidences = {class_names[i]: float(preds[i]) for i in range(len(class_names))}

    # Grad-CAM overlay
    heatmap, _ = make_gradcam_heatmap(batch, model, pred_index=pred_index)
    resized_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
    overlaid_bgr = overlay_heatmap(resized_bgr, heatmap)

    response = {
        "task": task,
        "demo_mode": demo_mode,
        "predicted_class": class_names[pred_index],
        "confidences": confidences,
        "heatmap_image_b64": encode_image_b64(overlaid_bgr, is_bgr=True),
        "input_preview_b64": encode_image_b64(resized_rgb, is_bgr=False),
    }
    return jsonify(response)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict/tumor", methods=["POST"])
def predict_tumor():
    return run_inference("tumor", TUMOR_CLASSES, TUMOR_SIZE)


@app.route("/api/predict/stroke", methods=["POST"])
def predict_stroke():
    return run_inference("stroke", STROKE_CLASSES, STROKE_SIZE)


@app.route("/api/health")
def health():
    tumor_trained = os.path.exists(os.path.join(SAVED_MODELS_DIR, "tumor_model.keras"))
    stroke_trained = os.path.exists(os.path.join(SAVED_MODELS_DIR, "stroke_model.keras"))
    return jsonify({
        "status": "ok",
        "tumor_model_trained": tumor_trained,
        "stroke_model_trained": stroke_trained,
    })


if __name__ == "__main__":
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
