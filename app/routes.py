import io
import requests
from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename
from PIL import Image
from .db import db, Prediction

bp = Blueprint("main", __name__)


def error(msg, code):
    return render_template("error.html", msg=msg, code=code), code


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/health")
def health():
    try:
        db.session.execute(db.text("SELECT 1"))
        ok = True
    except Exception:
        ok = False
    return jsonify(status="ok" if ok else "degraded", db=ok,
                   model=current_app.config["MODEL_NAME"]), (200 if ok else 503)


@bp.route("/predict", methods=["POST"])
def predict():
    cfg = current_app.config
    f = request.files.get("image")
    if not f or not f.filename:
        return error("No file selected.", 400)

    name = secure_filename(f.filename)                      
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in cfg["ALLOWED_EXT"]:
        return error("Only JPG, PNG or WEBP images are allowed.", 415)

    data = f.read()                                         
    try:
        Image.open(io.BytesIO(data)).verify()              
    except Exception:
        return error("This file is not a valid image.", 422)

    try:
        if cfg["INFERENCE_MODE"] == "remote":               
            r = requests.post(cfg["API_URL"] + "/predict",
                              files={"image": (name, data)},
                              headers={"X-API-Key": cfg["API_KEY"]}, timeout=30)
            r.raise_for_status()
            j = r.json()
            label, conf, model_name = j["label"], j["confidence"], j["model"]
        else:
            label, conf = current_app.classifier.predict(data)
            model_name = cfg["MODEL_NAME"]
    except Exception:
        current_app.logger.exception("Inference failed")
        return error("The model service is unavailable. Please try again.", 502)

    db.session.add(Prediction(filename=name, label=label,
                              confidence=conf, model_name=model_name))
    db.session.commit()                                      # every result is stored
    return render_template("result.html", label=label, conf=conf * 100, filename=name)


@bp.route("/history")
def history():
    rows = Prediction.query.order_by(Prediction.created_at.desc()).limit(100).all()
    return render_template("history.html", rows=rows)