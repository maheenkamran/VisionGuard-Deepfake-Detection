import io
import logging
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

MODEL = tf.keras.models.load_model(os.environ.get("MODEL_PATH", "models/champion.keras"))
KEY = os.environ["API_KEY"]
TH = float(os.getenv("THRESHOLD", "0.31"))
NAME = os.getenv("MODEL_NAME", "visionguard-mobilenetv2 (ft lr 1e-4)")


@app.route("/health")
def health():
    return jsonify(status="ok", model=NAME)


@app.route("/predict", methods=["POST"])
def predict():
    if request.headers.get("X-API-Key") != KEY:
        logging.warning("401 unauthorized request")
        return jsonify(error="unauthorized"), 401
    f = request.files.get("image")
    if not f:
        return jsonify(error="no image field"), 400
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception:
        logging.warning("422 invalid image")
        return jsonify(error="invalid image"), 422
    x = tf.image.resize(np.asarray(img, dtype="float32"), (224, 224)).numpy()
    p = float(MODEL.predict(preprocess_input(x)[None], verbose=0)[0][0])
    label = "Real" if p >= TH else "AI-generated"
    conf = p if label == "Real" else 1 - p
    logging.info("200 file=%s label=%s p_real=%.3f", f.filename, label, p)
    return jsonify(label=label, confidence=conf, probability=p, model=NAME)