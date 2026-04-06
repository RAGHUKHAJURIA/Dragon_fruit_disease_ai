"""
Dragon Fruit AI Suite — Flask Web Application
=============================================
Routes:
    GET  /                 → Home mode selector
    GET  /disease          → Disease diagnosis page (Grad-CAM)
    GET  /quality          → Quality grading page
    GET  /detect           → YOLOv8 lesion detector page
    POST /predict_disease  → Disease inference + advisory
    POST /predict_quality  → Quality inference + market recommendation
    POST /predict_detect   → YOLOv8 lesion detection + annotated image
"""

import os
import sys
import uuid

from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image
import torch
import torch.nn as nn

# ── Project root setup ────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from xai.gradcam import (
    load_convitx_model,
    run_gradcam,
    get_target_layer_convitx,
    infer_transforms,
)
from chatbot.advisor import (
    generate_recommendation,
    get_quality_advice,
    DISEASE_KNOWLEDGE,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DISEASE_MODEL_PATH = os.path.join(ROOT, "models", "best_convitx.pth")
QUALITY_MODEL_PATH = os.path.join(ROOT, "models", "quality_convitx.pth")

DISEASE_CLASS_NAMES = [
    "Anthracnose",
    "Brown_Stem_Spot",
    "Gray_Blight",
    "Healthy",
    "Soft_Rot",
    "Stem_Canker",
]

# Keep this in sync with train_quality.py output class order.
QUALITY_CLASS_NAMES = [
    "Defect Dragon Fruit",
    "Fresh Dragon Fruit",
    "Immature Dragon Fruit",
    "Mature Dragon Fruit",
]

QUALITY_CLASS_FILE = os.path.join(ROOT, "models", "quality_classes.txt")

IMG_SIZE = 224
UPLOAD_DIR  = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "dragonfruit-secret-key"

# ── Lazy-loaded models ───────────────────────────────────────────────────────
_disease_model = None
_disease_target_layer = None
_quality_model = None
_yolo_detector = None

# YOLO class metadata (mirrors detect_disease.py)
YOLO_CLASS_NAMES = {
    0: "Anthracnose",
    1: "Stem_Canker",
    2: "Soft_Rot",
    3: "Brown_Stem_Spot",
    4: "Gray_Blight",
}
YOLO_MODEL_PATH = os.path.join(ROOT, "models", "yolo_dragon_best.pt")


def _load_quality_classes():
    if not os.path.exists(QUALITY_CLASS_FILE):
        return QUALITY_CLASS_NAMES

    classes = []
    with open(QUALITY_CLASS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                classes.append(parts[1])

    return classes or QUALITY_CLASS_NAMES


def _get_disease_artifacts():
    global _disease_model, _disease_target_layer
    if _disease_model is None:
        _disease_model = load_convitx_model(
            DISEASE_MODEL_PATH,
            num_classes=len(DISEASE_CLASS_NAMES),
        )
        _disease_target_layer = get_target_layer_convitx(_disease_model)
    return _disease_model, _disease_target_layer


def _get_quality_model():
    global _quality_model
    if _quality_model is None:
        quality_classes = _load_quality_classes()
        _quality_model = load_convitx_model(
            QUALITY_MODEL_PATH,
            num_classes=len(quality_classes),
            device=torch.device("cpu")
        )
    return _quality_model


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _get_yolo_detector():
    global _yolo_detector
    if _yolo_detector is None:
        sys.path.insert(0, ROOT)
        from detect_disease import DragonFruitDetector
        _yolo_detector = DragonFruitDetector(
            model_path=YOLO_MODEL_PATH,
            conf=0.35,
            iou=0.45,
            device="cpu",
        )
    return _yolo_detector


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Home page with dual-tool selection."""
    return render_template("index.html")


@app.route("/disease")
def disease_page():
    return render_template("disease.html", diseases=DISEASE_KNOWLEDGE)


@app.route("/quality")
def quality_page():
    return render_template("quality.html")


@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    """Accept uploaded image → inference → Grad-CAM → advisor → results."""
    if "image" not in request.files:
        flash("No file uploaded.", "error")
        return redirect(url_for("disease_page"))

    file = request.files["image"]
    if file.filename == "" or not _allowed(file.filename):
        flash("Please upload a valid image (jpg, png, webp).", "error")
        return redirect(url_for("disease_page"))

    # Save uploaded image
    uid        = uuid.uuid4().hex[:10]
    ext        = file.filename.rsplit(".", 1)[1].lower()
    img_name   = f"{uid}.{ext}"
    img_path   = os.path.join(UPLOAD_DIR, img_name)
    file.save(img_path)

    try:
        disease_model, disease_target_layer = _get_disease_artifacts()
    except FileNotFoundError:
        flash("Disease model not found. Train best_convitx.pth in models/.", "error")
        return redirect(url_for("disease_page"))

    # Run Grad-CAM inference
    cam_name  = f"{uid}_gradcam.png"
    cam_path  = os.path.join(UPLOAD_DIR, cam_name)
    result    = run_gradcam(
        model        = disease_model,
        target_layer = disease_target_layer,
        image_path   = img_path,
        class_names  = DISEASE_CLASS_NAMES,
        save_path    = cam_path,
    )

    # Save the overlay as a full-resolution image (not the tiny matplotlib panel)
    overlay_name = f"{uid}_overlay.png"
    overlay_path = os.path.join(UPLOAD_DIR, overlay_name)
    overlay_img  = Image.fromarray(result["overlay"])
    overlay_img.save(overlay_path, quality=95)

    # Advisory from knowledge base
    advisory = generate_recommendation(
        predicted_class = result["predicted_class"],
        confidence      = result["confidence"],
    )

    # Probability data for chart
    probs = result["probabilities"]

    return render_template(
        "result_disease.html",
        img_url       = url_for("static", filename=f"uploads/{img_name}"),
        cam_url       = url_for("static", filename=f"uploads/{overlay_name}"),
        prediction    = result["predicted_class"],
        confidence    = result["confidence"],
        probabilities = probs,
        advisory      = advisory,
    )


@app.route("/predict_quality", methods=["POST"])
def predict_quality():
    if "image" not in request.files:
        flash("No file uploaded.", "error")
        return redirect(url_for("quality_page"))

    file = request.files["image"]
    if file.filename == "" or not _allowed(file.filename):
        flash("Please upload a valid image (jpg, png, webp).", "error")
        return redirect(url_for("quality_page"))

    uid = uuid.uuid4().hex[:10]
    ext = file.filename.rsplit(".", 1)[1].lower()
    img_name = f"{uid}.{ext}"
    img_path = os.path.join(UPLOAD_DIR, img_name)
    file.save(img_path)

    try:
        quality_model = _get_quality_model()
    except FileNotFoundError:
        flash("Quality model not found. Train quality_convitx.pth in models/.", "error")
        return redirect(url_for("quality_page"))

    pil_img = Image.open(img_path).convert("RGB")
    image_tensor = infer_transforms(pil_img).unsqueeze(0)

    quality_classes = _load_quality_classes()

    with torch.no_grad():
        logits = quality_model(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())
    if len(quality_classes) != probs.shape[0]:
        quality_classes = [f"Quality_Class_{i}" for i in range(probs.shape[0])]

    prediction = quality_classes[pred_idx]
    probabilities = {
        cls: float(probs[i].item())
        for i, cls in enumerate(quality_classes)
    }
    quality_advice = get_quality_advice(prediction, confidence)

    return render_template(
        "result_quality.html",
        img_url=url_for("static", filename=f"uploads/{img_name}"),
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        quality_advice=quality_advice,
    )


# ── YOLO Lesion Detector Routes ──────────────────────────────────────────────
@app.route("/detect")
def detect_page():
    """YOLOv8 lesion detector upload page."""
    return render_template("detect.html", yolo_classes=YOLO_CLASS_NAMES)


@app.route("/predict_detect", methods=["POST"])
def predict_detect():
    """Run YOLOv8 on uploaded image; return annotated result + lesion stats."""
    if "image" not in request.files:
        flash("No file uploaded.", "error")
        return redirect(url_for("detect_page"))

    file = request.files["image"]
    if file.filename == "" or not _allowed(file.filename):
        flash("Please upload a valid image (jpg, png, webp).", "error")
        return redirect(url_for("detect_page"))

    uid      = uuid.uuid4().hex[:10]
    ext      = file.filename.rsplit(".", 1)[1].lower()
    img_name = f"{uid}.{ext}"
    img_path = os.path.join(UPLOAD_DIR, img_name)
    file.save(img_path)

    try:
        detector = _get_yolo_detector()
    except FileNotFoundError:
        flash(
            "YOLOv8 model not found. Train yolo_dragon_best.pt first with train_yolo_directml.py.",
            "error",
        )
        return redirect(url_for("detect_page"))

    # Run detection; save annotated image into uploads/
    result = detector.predict(
        img_path,
        save_annotated=True,
        output_dir=UPLOAD_DIR,
    )

    # Annotated image filename
    stem          = img_name.rsplit(".", 1)[0]
    annotated_name = f"{stem}_annotated.jpg"

    # Build per-class confidence summary for display
    class_summary = {}  # {class_name: {count, max_conf, avg_conf}}
    for det in result["detections"]:
        cn = det["class_name"]
        if cn not in class_summary:
            class_summary[cn] = {"count": 0, "confs": []}
        class_summary[cn]["count"] += 1
        class_summary[cn]["confs"].append(det["confidence"])

    for cn, data in class_summary.items():
        data["avg_conf"] = round(sum(data["confs"]) / len(data["confs"]) * 100, 1)
        data["max_conf"] = round(max(data["confs"]) * 100, 1)
        del data["confs"]

    return render_template(
        "result_detect.html",
        img_url        = url_for("static", filename=f"uploads/{img_name}"),
        annotated_url  = url_for("static", filename=f"uploads/{annotated_name}"),
        total_lesions  = result["total_lesions"],
        severity       = result["severity"],
        disease_counts = result["disease_counts"],
        class_summary  = class_summary,
        detections     = result["detections"],
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
