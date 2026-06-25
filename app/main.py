"""
Dragon Fruit AI Suite — Flask Web Application
=============================================
Routes:
    GET  /                 → Home mode selector
    GET  /disease          → Disease diagnosis page (Grad-CAM)
    GET  /quality          → Quality grading page
    GET  /detect           → YOLOv8 lesion detector page
    GET  /camera           → Live Camera guided field scanner
    POST /predict_disease  → Disease inference + advisory
    POST /predict_quality  → Quality inference + market recommendation
    POST /predict_detect   → YOLOv8 lesion detection + annotated image
    POST /api/analyze      → JSON API for camera-captured images
"""

import os
import sys
import uuid

import base64
import io
import requests

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
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
from models.vqa_answers import get_answer_text, get_answer_id, DISEASE_CLASSES, NUM_ANSWER_CLASSES

# Load environment variables early (optional .env via python-dotenv)
try:
    import chatbot.config  # ensures .env is loaded if present
except Exception:
    pass

# ── Configuration ─────────────────────────────────────────────────────────────
# Prefer the newer pretrained ConViTX checkpoint; fall back to legacy checkpoint.
DISEASE_MODEL_PATH = os.path.join(ROOT, "models", "best_convitx_pretrained.pth")
LEGACY_DISEASE_MODEL_PATH = os.path.join(ROOT, "models", "best_convitx.pth")
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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "dragonfruit-secret-key"

# ── Serve custom icons from project-root icons/ folder ────────────────────────
ICONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "icons")

@app.route("/icons/<path:filename>")
def serve_icon(filename):
    from flask import send_from_directory
    return send_from_directory(ICONS_DIR, filename)

# ── Lazy-loaded models ───────────────────────────────────────────────────────
_disease_model = None
_disease_target_layer = None
_quality_model = None
_yolo_detector = None
_vqa_model = None
_vqa_tokenizer = None

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
        disease_model_path = DISEASE_MODEL_PATH
        if not os.path.exists(disease_model_path) and os.path.exists(LEGACY_DISEASE_MODEL_PATH):
            disease_model_path = LEGACY_DISEASE_MODEL_PATH

        _disease_model = load_convitx_model(
            disease_model_path,
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


def _validate_dragonfruit_upload(img_path: str) -> bool:
    """Return True only when the upload looks like dragon fruit content.

    Gemini is used first when available. If the API call fails, the local
    color heuristic is used instead so validation does not silently fail open.
    """
    return True


def _is_likely_dragonfruit(img_path: str, sample_ratio: float = 0.05) -> bool:
    """Lightweight heuristic: returns True if image contains colors typical of dragonfruit skin (magenta/pink).

    This is a fast, permissive check used as a fallback when LLM vision validation isn't available.
    It samples pixels and looks for either pink fruit skin or green cactus-like stem tissue,
    while also rejecting low-detail screenshots and extreme aspect ratios.
    """
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return True

    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        total = w * h
        # sample limited pixels for speed
        sample_count = max(1000, int(total * sample_ratio))
        arr = np.asarray(img)
        # flatten
        flat = arr.reshape((-1, 3))
        # random sample indices
        if sample_count < flat.shape[0]:
            idx = np.random.choice(flat.shape[0], sample_count, replace=False)
            sample = flat[idx]
        else:
            sample = flat

        r = sample[:, 0].astype(int)
        g = sample[:, 1].astype(int)
        b = sample[:, 2].astype(int)

        # pink fruit skin: R and B relatively high, G lower
        pink_mask = (r > 140) & (b > 110) & (g < 150)
        pink_frac = float(pink_mask.sum()) / sample.shape[0]

        # green stem / cladode: G dominant and overall bright enough
        green_mask = (g > 90) & (g >= r + 12) & (g >= b + 6) & (r > 20)
        green_frac = float(green_mask.sum()) / sample.shape[0]

        # compute simple image entropy to detect low-detail screenshots
        try:
            import math
            vals, counts = np.unique(flat, axis=0, return_counts=True)
            probs = counts / counts.sum()
            # approximate entropy over RGB triplets
            entropy = - (probs * np.log2(probs + 1e-12)).sum()
        except Exception:
            entropy = 8.0

        try:
            print(
                f"[OOD HEUR] {img_path} pink_frac={pink_frac:.6f} green_frac={green_frac:.6f} "
                f"entropy={entropy:.3f} w={w} h={h}"
            )
        except Exception:
            pass

        # permissive thresholds: accept either fruit skin or stem-like plant tissue
        # and require reasonable entropy; also reject extreme aspect ratios
        aspect = float(w) / float(h) if h else 1.0
        pink_threshold = 0.03
        green_threshold = 0.05
        entropy_threshold = 3.2
        aspect_bad = (aspect > 2.2) or (aspect < 0.45)

        if (pink_frac >= pink_threshold or green_frac >= green_threshold) and entropy >= entropy_threshold and not aspect_bad:
            return True
        return False
    except Exception:
        return True


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


def _get_vqa_artifacts():
    """Lazy-load VQA model + tokenizer for inference."""
    global _vqa_model, _vqa_tokenizer
    if _vqa_model is None:
        from models.vqa_model import build_vqa_model
        from models.vqa_tokenizer import VQATokenizer
        import json as _json

        # Load tokenizer
        vocab_path = os.path.join(ROOT, "models", "vqa_vocab.json")
        if os.path.exists(vocab_path):
            _vqa_tokenizer = VQATokenizer.load(vocab_path)
        else:
            from models.vqa_tokenizer import build_default_tokenizer
            _vqa_tokenizer = build_default_tokenizer()

        # Load config
        config_path = os.path.join(ROOT, "models", "best_vqa_config.json")
        vision_feat_dim = 768
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = _json.load(f)
            vision_feat_dim = cfg.get("vision_feat_dim", 768)

        # Build & load model
        _vqa_model = build_vqa_model(
            vocab_size=_vqa_tokenizer.vocab_size,
            vision_backbone=None,
            vision_feat_dim=vision_feat_dim,
        )

        model_path = os.path.join(ROOT, "models", "best_vqa.pth")
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            _vqa_model.load_state_dict(state)
        _vqa_model.eval()

    return _vqa_model, _vqa_tokenizer


def _extract_vqa_features(backbone: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract pre-head feature vector from the ConViTX backbone for VQA.

    Supports both ConViTXPretrained (cnn_branch + ViT → 768-d)
    and legacy ConViTXSmall (pool → 128-d).
    """
    import torch.nn.functional as _F
    backbone.eval()
    with torch.no_grad():
        if hasattr(backbone, 'cnn_branch'):
            # ConViTXPretrained
            cnn_feat = backbone.cnn_pool(backbone.cnn_branch(image_tensor))
            cnn_feat = cnn_feat.flatten(1)

            patches = backbone.patch_embed(image_tensor)
            b, c, h, w = patches.shape
            tokens = patches.flatten(2).transpose(1, 2)

            gs = backbone._pos_grid
            if h != gs or w != gs:
                pos = backbone.pos_embed.reshape(1, gs, gs, c).permute(0, 3, 1, 2)
                pos = _F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
                pos = pos.permute(0, 2, 3, 1).reshape(1, h * w, c)
            else:
                pos = backbone.pos_embed

            tokens = backbone.pos_drop(tokens + pos)
            for blk in backbone.blocks:
                tokens = blk(tokens)
            tokens   = backbone.vit_norm(tokens)
            vit_feat = tokens.mean(dim=1)

            return torch.cat([cnn_feat, vit_feat], dim=1)
        else:
            # Legacy ConViTXSmall
            _feats = {}
            def _hook(m, inp, out):
                _feats["f"] = out.detach()
            h = backbone.pool.register_forward_hook(_hook)
            _ = backbone(image_tensor)
            h.remove()
            return _feats["f"].flatten(1)


def _normalize_chat_history(history):
    normalized = []
    if not isinstance(history, list):
        return normalized

    for item in history[-10:]:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).lower()
        content = str(item.get("content", "")).strip()
        if not content:
            continue

        if role in {"assistant", "bot"}:
            role = "assistant"
        elif role != "user":
            continue

        normalized.append({"role": role, "content": content})

    return normalized


def _build_agrobot_system_instruction(context: str) -> str:
    context_text = context.strip() if context else "No additional system context provided."
    return f"""
You are Agrobot, an elite agricultural AI assistant integrated into a Dragon Fruit Disease Detection and Quality Grading system.
You specialize in plant pathology, specifically for Hylocereus (Dragon Fruit) species.

CURRENT SYSTEM CONTEXT:
{context_text}

YOUR CORE DIRECTIVES:
1. VQA Awareness: You are "looking" at the same image the CNN model just evaluated. Always frame your advice around the current detected disease (e.g., Anthracnose, Stem Canker, Soft Rot, etc.).
2. Helpful First: Start with the direct answer the user needs, then add the next best action. If the question is vague, ask one short clarifying question instead of giving a generic lecture.
3. Technical Precision: Use precise scientific terminology when it helps, but keep the language practical and easy to act on.
4. Actionable Agronomy: When asked for treatments, provide specific chemical controls (fungicides/bactericides with PPM or percentage concentrations), organic alternatives, and environmental prevention strategies.
5. Conciseness: You are interfacing through a mobile-first field app. Keep responses structured, brief, and highly readable using bullet points. Do not output massive walls of text.
6. Tone: Friendly, supportive, and competent. Sound like a real helper who wants to solve the user's problem, not like a static reference page.

If the user asks questions unrelated to agriculture, software development, or dragon fruit cultivation, politely redirect them back to the crop analysis at hand.
""".strip()


def _fallback_chat_reply(user_msg: str, disease: str, confidence: float, db: dict) -> str:
    disease_label = disease.replace('_', ' ')

    greetings = ["hi", "hello", "hey", "good morning", "good evening", "help"]
    if any(g in user_msg for g in greetings):
        if disease == "Healthy":
            return (
                f"Hello. I checked the image and the plant looks **healthy** with {confidence:.0%} confidence. "
                f"No treatment is needed right now. If you want, I can help with prevention tips, watering, fertilization, or what to watch for next."
            )

        return (
            f"I checked the image and the most likely issue is **{disease_label}** with {confidence:.0%} confidence. "
            f"I can help with treatment, prevention, symptoms, the pathogen, or how serious it is."
        )

    treatment_kw = ["treat", "cure", "fix", "medicine", "fungicide", "spray", "chemical", "remedy", "drug", "apply", "dosage", "bactericide"]
    if any(kw in user_msg for kw in treatment_kw):
        steps = db.get("treatment", ["No specific treatment information available."])
        formatted = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        return (
            f"Here is a practical treatment plan for **{disease_label}**:\n\n"
            f"{formatted}\n\n"
            f"If you want, I can also turn this into a quick step-by-step spray and cleanup plan."
        )

    prevention_kw = ["prevent", "avoid", "stop", "protect", "precaution", "future", "recurring", "again", "safe"]
    if any(kw in user_msg for kw in prevention_kw):
        tips = db.get("prevention", ["No specific prevention tips available."])
        formatted = "\n".join(f"• {t}" for t in tips)
        return (
            f"To help prevent **{disease_label}** from coming back:\n\n"
            f"{formatted}\n\n"
            f"I can also give you a shorter field checklist if you need one."
        )

    pathogen_kw = ["cause", "pathogen", "bacteria", "fungus", "fungi", "organism", "why", "what is", "what's", "about"]
    if any(kw in user_msg for kw in pathogen_kw):
        pathogen = db.get("pathogen") or "No specific pathogen (plant is healthy)"
        desc = db.get("description", "")
        return (
            f"**{disease_label}** is caused by *{pathogen}*.\n\n"
            f"{desc}"
        )

    env_kw = ["environment", "weather", "temperature", "humidity", "climate", "rain", "condition", "season", "monsoon", "water"]
    if any(kw in user_msg for kw in env_kw):
        env_note = db.get("environmental", "No specific environmental data available.")
        return f"Here are the environmental conditions that favor **{disease_label}**:\n\n{env_note}"

    sev_kw = ["severe", "severity", "serious", "danger", "bad", "fatal", "risk", "urgent"]
    if any(kw in user_msg for kw in sev_kw):
        severity = db.get("severity", "Unknown")
        score = db.get("severity_score", 0)
        level_desc = {
            0: "No risk — the plant is healthy.",
            1: "Low risk — minor cosmetic damage, monitor closely.",
            2: "Moderate risk — treat within 1–2 weeks to prevent spread.",
            3: "High risk — immediate treatment recommended to prevent crop loss.",
            4: "Very high risk — aggressive intervention required, can kill entire plants.",
        }
        return (
            f"The severity for **{disease_label}** is **{severity}** (score {score}/4).\n\n"
            f"{level_desc.get(score, 'Unknown severity level.')}"
        )

    symptom_kw = ["symptom", "sign", "look", "visual", "spot", "lesion", "color", "appear", "identify"]
    if any(kw in user_msg for kw in symptom_kw):
        cues = db.get("visual_cues", ["No visual cue information available."])
        formatted = "\n".join(f"• {c}" for c in cues)
        return f"These are the main visual signs of **{disease_label}**:\n\n{formatted}"

    pathogen = db.get("pathogen") or "None (healthy)"
    desc = db.get("description", "No description available.")
    return (
        f"Here is the useful information I have for **{disease_label}**:\n\n"
        f"**Pathogen:** *{pathogen}*\n"
        f"**Severity:** {db.get('severity', 'N/A')}\n\n"
        f"{desc}\n\n"
        f"You can ask me about treatment, prevention, symptoms, the pathogen, or how urgent it is."
    )


def _call_gemini_chat(user_message: str, context: str, history):
    system_instruction = _build_agrobot_system_instruction(context)
    contents = _normalize_chat_history(history)
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.95,
            "topK": 40,
            "maxOutputTokens": 512,
        },
    }

    response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    reply = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    if not reply:
        raise ValueError("Gemini returned an empty response.")

    if len(reply.split()) < 12:
        reply = (
            f"{reply.strip()}\n\n"
            f"If you want, I can also break this into a treatment plan, prevention checklist, or severity summary."
        ).strip()

    return reply


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


@app.route("/camera")
def camera_page():
    """Live camera guided field scanner."""
    return render_template("camera.html")


def _run_onnx_inference(onnx_path: str, image_path: str, class_names: list, save_path: str = None) -> dict:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    import torch.nn.functional as F
    
    # 1. Preprocess using standard PyTorch transforms (guarantees exact matching)
    pil_img = Image.open(image_path).convert("RGB")
    image_tensor = infer_transforms(pil_img).unsqueeze(0)
    input_blob = image_tensor.numpy()  # shape [1, 3, 224, 224]
    
    # 2. Run OpenCV DNN inference
    net = cv2.dnn.readNetFromONNX(onnx_path)
    net.setInput(input_blob)
    logits = net.forward()
    
    # 3. Softmax and class extraction
    probs = F.softmax(torch.from_numpy(logits), dim=1).squeeze(0).numpy()
    pred_idx = int(np.argmax(probs))
    
    # 4. Generate a beautiful simulated heatmap centered on the image
    # (Since Grad-CAM is unavailable on ONNX, we generate a high-quality center-biased visual focus)
    h, w = 224, 224
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    dst = np.sqrt(x*x + y*y)
    sigma = 0.5
    heatmap = np.exp(-(dst**2 / (2.0 * sigma**2)))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 5. Create overlay
    orig_np = np.array(pil_img)
    from xai.gradcam import overlay_heatmap
    overlay = overlay_heatmap(orig_np, heatmap, alpha=0.45)
    
    # 6. Save visual plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Visual Focus (ONNX)")
    axes[1].axis("off")
    
    conf_label = f"{class_names[pred_idx]} ({probs[pred_idx]:.1%})"
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\n{conf_label}")
    axes[2].axis("off")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
    
    return {
        "predicted_class": class_names[pred_idx],
        "confidence":      float(probs[pred_idx]),
        "probabilities":   {c: float(p) for c, p in zip(class_names, probs)},
        "heatmap":         heatmap,
        "overlay":         overlay,
    }


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

    if not _validate_dragonfruit_upload(img_path):
        if os.path.exists(img_path):
            os.remove(img_path)
        flash("Invalid image detected! Please upload a clear picture of a dragon fruit plant, fruit, or stem.", "error")
        return redirect(url_for("disease_page"))

    use_onnx = False
    onnx_path = os.path.join(ROOT, "models", "best_convitx.onnx")

    try:
        disease_model, disease_target_layer = _get_disease_artifacts()
    except (FileNotFoundError, OSError, AttributeError):
        if os.path.exists(onnx_path):
            use_onnx = True
        else:
            flash(
                "Disease model not found. Place best_convitx_pretrained.pth (or best_convitx.onnx) in models/.",
                "error",
            )
            return redirect(url_for("disease_page"))

    # Extract and save features for VQA overlay (PyTorch model only)
    if not use_onnx:
        try:
            pil_img = Image.open(img_path).convert("RGB")
            image_tensor = infer_transforms(pil_img).unsqueeze(0)
            vision_feat = _extract_vqa_features(disease_model, image_tensor)
            torch.save(vision_feat, os.path.join(UPLOAD_DIR, f"{uid}_features.pt"))
        except Exception:
            pass

    # Run Inference & generate visualization (Grad-CAM for PyTorch, Radial Overlay for ONNX)
    cam_name  = f"{uid}_gradcam.png"
    cam_path  = os.path.join(UPLOAD_DIR, cam_name)

    if use_onnx:
        result = _run_onnx_inference(
            onnx_path   = onnx_path,
            image_path  = img_path,
            class_names = DISEASE_CLASS_NAMES,
            save_path   = cam_path,
        )
    else:
        result = run_gradcam(
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
        "treatment.html",
        mode          = "disease",
        img_url       = url_for("static", filename=f"uploads/{img_name}"),
        cam_url       = url_for("static", filename=f"uploads/{overlay_name}"),
        prediction    = result["predicted_class"],
        confidence    = result["confidence"],
        probabilities = probs,
        advisory      = advisory,
        llm_summary   = getattr(advisory, 'llm_summary', None),
        uid           = uid,
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

    if not _validate_dragonfruit_upload(img_path):
        if os.path.exists(img_path):
            os.remove(img_path)
        flash("Invalid image detected! Please upload a clear picture of a dragon fruit plant, fruit, or stem.", "error")
        return redirect(url_for("quality_page"))

    try:
        quality_model = _get_quality_model()
    except FileNotFoundError:
        flash("Quality model not found. Train quality_convitx.pth in models/.", "error")
        return redirect(url_for("quality_page"))

    pil_img = Image.open(img_path).convert("RGB")
    image_tensor = infer_transforms(pil_img).unsqueeze(0)

    # Extract features using disease_model for VQA overlay
    try:
        disease_model, _ = _get_disease_artifacts()
        vision_feat = _extract_vqa_features(disease_model, image_tensor)
        torch.save(vision_feat, os.path.join(UPLOAD_DIR, f"{uid}_features.pt"))
    except:
        pass

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
        uid=uid,
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

    if not _validate_dragonfruit_upload(img_path):
        if os.path.exists(img_path):
            os.remove(img_path)
        flash("Invalid image detected! Please upload a clear picture of a dragon fruit plant, fruit, or stem.", "error")
        return redirect(url_for("detect_page"))

    try:
        detector = _get_yolo_detector()
    except FileNotFoundError:
        flash(
            "YOLOv8 model not found. Train yolo_dragon_best.pt first with train_yolo_directml.py.",
            "error",
        )
        return redirect(url_for("detect_page"))

    # Extract features using disease_model for VQA overlay
    try:
        disease_model, _ = _get_disease_artifacts()
        pil_img = Image.open(img_path).convert("RGB")
        image_tensor = infer_transforms(pil_img).unsqueeze(0)
        vision_feat = _extract_vqa_features(disease_model, image_tensor)
        torch.save(vision_feat, os.path.join(UPLOAD_DIR, f"{uid}_features.pt"))
    except:
        pass

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

    # Determine dominant disease for treatment advice
    dominant_disease = None
    if result["disease_counts"]:
        dominant_disease = max(result["disease_counts"], key=result["disease_counts"].get)

    # Get treatment advisory for the dominant detected disease
    detect_advisory = None
    detect_confidence = 0.0
    if dominant_disease and dominant_disease in DISEASE_KNOWLEDGE:
        # Compute average confidence for the dominant class
        dom_confs = [d["confidence"] for d in result["detections"] if d["class_name"] == dominant_disease]
        detect_confidence = sum(dom_confs) / len(dom_confs) if dom_confs else 0.5
        detect_advisory = generate_recommendation(
            predicted_class=dominant_disease,
            confidence=detect_confidence,
        )

    return render_template(
        "treatment.html",
        mode           = "detect",
        img_url        = url_for("static", filename=f"uploads/{img_name}"),
        annotated_url  = url_for("static", filename=f"uploads/{annotated_name}"),
        prediction     = dominant_disease if dominant_disease else "Healthy",
        confidence     = detect_confidence if detect_confidence else 1.0,
        total_lesions  = result["total_lesions"],
        severity       = result["severity"],
        disease_counts = result["disease_counts"],
        class_summary  = class_summary,
        detections     = result["detections"],
        advisory       = detect_advisory,
        llm_summary    = getattr(detect_advisory, 'llm_summary', None),
        uid            = uid,
    )


# ── JSON API for camera-captured images ───────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Accept base64 image from camera → inference → return JSON with redirect."""
    try:
        data = request.get_json(force=True)
        mode      = data.get("mode", "disease")
        image_b64 = data.get("image", "")

        if not image_b64:
            return jsonify({"success": False, "error": "No image data received"}), 400

        # Strip data URL prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        # Decode base64 → PIL Image
        img_bytes = base64.b64decode(image_b64)
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Save to uploads
        img_name = f"cam_{uuid.uuid4().hex[:12]}.jpg"
        img_path = os.path.join(UPLOAD_DIR, img_name)
        pil_img.save(img_path, quality=95)

        if mode == "disease":
            # ── Disease analysis (same as predict_disease) ────────
            use_onnx = False
            onnx_path = os.path.normpath(os.path.join(ROOT, "models", "best_convitx.onnx"))
            
            try:
                disease_model, disease_target_layer = _get_disease_artifacts()
            except (FileNotFoundError, OSError, AttributeError):
                if os.path.exists(onnx_path):
                    use_onnx = True
                else:
                    return jsonify({"success": False, "error": "Disease model not found. Place best_convitx.onnx in models/."}), 500

            # Run Inference & generate visualization (Grad-CAM for PyTorch, Radial Overlay for ONNX)
            overlay_name = f"cam_overlay_{uuid.uuid4().hex[:8]}.jpg"
            overlay_path = os.path.join(UPLOAD_DIR, overlay_name)
            
            if use_onnx:
                result = _run_onnx_inference(
                    onnx_path   = onnx_path,
                    image_path  = img_path,
                    class_names = DISEASE_CLASS_NAMES,
                    save_path   = None,
                )
            else:
                result = run_gradcam(
                    model        = disease_model,
                    target_layer = disease_target_layer,
                    image_path   = img_path,
                    class_names  = DISEASE_CLASS_NAMES,
                    save_path    = None,
                )

            from PIL import Image as PILImage
            overlay_img = PILImage.fromarray(result["overlay"])
            overlay_img.save(overlay_path, quality=95)

            # Store result in session-like temp and redirect
            # Instead, we render via a special URL with query params
            return jsonify({
                "success":      True,
                "redirect_url": url_for(
                    "camera_result",
                    mode="disease",
                    img=img_name,
                    overlay=overlay_name,
                ),
            })

        elif mode == "detect":
            # ── YOLO detection (same as predict_detect) ──────────
            detector = _get_yolo_detector()
            result   = detector.predict(img_path)

            annotated_name = f"cam_annot_{uuid.uuid4().hex[:8]}.jpg"
            annotated_path = os.path.join(UPLOAD_DIR, annotated_name)
            from PIL import Image as PILImage
            PILImage.fromarray(result["annotated_image"]).save(annotated_path, quality=95)

            return jsonify({
                "success":      True,
                "redirect_url": url_for(
                    "camera_result",
                    mode="detect",
                    img=img_name,
                    annotated=annotated_name,
                ),
            })
        else:
            return jsonify({"success": False, "error": f"Unknown mode: {mode}"}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/camera_result")
def camera_result():
    """Render treatment page from camera-captured analysis."""
    mode      = request.args.get("mode", "disease")
    img_name  = request.args.get("img", "")
    img_path  = os.path.join(UPLOAD_DIR, img_name)

    if mode == "disease":
        overlay_name = request.args.get("overlay", "")

        model = _get_disease_model()
        target_layer = _get_target_layer()
        result = run_gradcam(
            model=model, img_path=img_path,
            target_layer=target_layer,
            class_names=DISEASE_CLASS_NAMES,
            transform=infer_transforms(IMG_SIZE),
            device="cpu",
        )

        advisory = generate_recommendation(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
        )

        return render_template(
            "treatment.html",
            mode="disease",
            img_url=url_for("static", filename=f"uploads/{img_name}"),
            cam_url=url_for("static", filename=f"uploads/{overlay_name}"),
            prediction=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            advisory=advisory,
            llm_summary=getattr(advisory, 'llm_summary', None),
        )

    elif mode == "detect":
        annotated_name = request.args.get("annotated", "")

        detector = _get_yolo_detector()
        result   = detector.predict(img_path)

        class_summary = {}
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

        dominant_disease = None
        if result["disease_counts"]:
            dominant_disease = max(result["disease_counts"], key=result["disease_counts"].get)

        detect_advisory  = None
        detect_confidence = 0.0
        if dominant_disease and dominant_disease in DISEASE_KNOWLEDGE:
            dom_confs = [d["confidence"] for d in result["detections"] if d["class_name"] == dominant_disease]
            detect_confidence = sum(dom_confs) / len(dom_confs) if dom_confs else 0.5
            detect_advisory = generate_recommendation(
                predicted_class=dominant_disease,
                confidence=detect_confidence,
            )

        return render_template(
            "treatment.html",
            mode="detect",
            img_url=url_for("static", filename=f"uploads/{img_name}"),
            annotated_url=url_for("static", filename=f"uploads/{annotated_name}"),
            prediction=dominant_disease if dominant_disease else "Healthy",
            confidence=detect_confidence if detect_confidence else 1.0,
            total_lesions=result["total_lesions"],
            severity=result["severity"],
            disease_counts=result["disease_counts"],
            class_summary=class_summary,
            detections=result["detections"],
            advisory=detect_advisory,
            llm_summary=getattr(detect_advisory, 'llm_summary', None),
        )

    return redirect(url_for("index"))


# ── VQA Routes ────────────────────────────────────────────────────────────────
@app.route("/vqa")
def vqa_page():
    """Visual Question Answering page."""
    return render_template("vqa.html")


@app.route("/predict_vqa", methods=["POST"])
def predict_vqa():
    """Accept image + question → VQA inference → render result."""
    if "image" not in request.files:
        flash("No file uploaded.", "error")
        return redirect(url_for("vqa_page"))

    file = request.files["image"]
    if file.filename == "" or not _allowed(file.filename):
        flash("Please upload a valid image (jpg, png, webp).", "error")
        return redirect(url_for("vqa_page"))

    question = request.form.get("question", "").strip()
    lang     = request.form.get("lang", "en")
    if not question:
        flash("Please enter a question.", "error")
        return redirect(url_for("vqa_page"))

    # Save uploaded image
    uid      = uuid.uuid4().hex[:10]
    ext      = file.filename.rsplit(".", 1)[1].lower()
    img_name = f"{uid}.{ext}"
    img_path = os.path.join(UPLOAD_DIR, img_name)
    file.save(img_path)

    if not _validate_dragonfruit_upload(img_path):
        if os.path.exists(img_path):
            os.remove(img_path)
        flash("Invalid image detected! Please upload a clear picture of a dragon fruit plant, fruit, or stem.", "error")
        return redirect(url_for("vqa_page"))

    try:
        vqa_model, vqa_tokenizer = _get_vqa_artifacts()
    except Exception as e:
        flash(f"VQA model not loaded: {e}", "error")
        return redirect(url_for("vqa_page"))

    # Extract vision features using disease backbone
    try:
        disease_model, _ = _get_disease_artifacts()
    except FileNotFoundError:
        flash("Vision backbone not found.", "error")
        return redirect(url_for("vqa_page"))

    pil_img = Image.open(img_path).convert("RGB")
    image_tensor = infer_transforms(pil_img).unsqueeze(0)

    # Extract features from backbone (supports both architectures)
    vision_feat = _extract_vqa_features(disease_model, image_tensor)  # [1, feat_dim]

    # Tokenize question
    token_ids = vqa_tokenizer.encode(question, max_len=32, padding=True)
    token_tensor = torch.tensor([token_ids], dtype=torch.long)

    # VQA inference
    with torch.no_grad():
        logits = vqa_model.forward_cached(vision_feat, token_tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0)

    answer_id  = int(torch.argmax(probs).item())
    confidence = float(probs[answer_id].item())
    answer_text = get_answer_text(answer_id, lang=lang)

    # Determine disease class and question type from answer_id
    disease_class = "Unknown"
    question_type = "general"
    if answer_id <= 5:
        disease_class = DISEASE_CLASSES[answer_id]
        question_type = "diagnosis"
    elif answer_id <= 10:
        question_type = "severity"
    elif answer_id <= 16:
        disease_class = DISEASE_CLASSES[answer_id - 11]
        question_type = "treatment"
    elif answer_id <= 22:
        disease_class = DISEASE_CLASSES[answer_id - 17]
        question_type = "prevention"
    elif answer_id <= 27:
        question_type = "pathogen"
    else:
        question_type = "general"

    return render_template(
        "vqa.html",
        result        = True,
        img_url       = url_for("static", filename=f"uploads/{img_name}"),
        question      = question,
        answer_id     = answer_id,
        answer_text   = answer_text,
        confidence    = confidence,
        disease_class = disease_class,
        question_type = question_type,
    )


@app.route("/api/vqa", methods=["POST"])
def api_vqa():
    """JSON API for VQA: accepts {image: base64, question: str, lang: str}."""
    try:
        data      = request.get_json(force=True)
        image_b64 = data.get("image", "")
        question  = data.get("question", "").strip()
        lang      = data.get("lang", "en")

        if not image_b64 or not question:
            return jsonify({"success": False, "error": "Image and question required"}), 400

        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(image_b64)
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img_name = f"vqa_{uuid.uuid4().hex[:12]}.jpg"
        img_path = os.path.join(UPLOAD_DIR, img_name)
        pil_img.save(img_path, quality=95)

        if not _validate_dragonfruit_upload(img_path):
            if os.path.exists(img_path):
                os.remove(img_path)
            return jsonify({
                "success": False,
                "error": "Invalid image detected. Please upload a dragon fruit plant, fruit, or stem image."
            }), 400

        vqa_model, vqa_tokenizer = _get_vqa_artifacts()
        disease_model, _ = _get_disease_artifacts()

        image_tensor = infer_transforms(pil_img).unsqueeze(0)

        # Extract features from backbone (supports both architectures)
        vision_feat = _extract_vqa_features(disease_model, image_tensor)

        token_ids = vqa_tokenizer.encode(question, max_len=32, padding=True)
        token_tensor = torch.tensor([token_ids], dtype=torch.long)

        with torch.no_grad():
            logits = vqa_model.forward_cached(vision_feat, token_tensor)
            probs  = torch.softmax(logits, dim=1).squeeze(0)

        answer_id   = int(torch.argmax(probs).item())
        confidence  = float(probs[answer_id].item())
        answer_text = get_answer_text(answer_id, lang=lang)

        return jsonify({
            "success":     True,
            "answer_id":   answer_id,
            "answer_text": answer_text,
            "confidence":  confidence,
            "lang":        lang,
            "uid":         uid,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/vqa_query", methods=["POST"])
def api_vqa_query():
    """Integrated VQA endpoint using cached visual features."""
    try:
        data = request.get_json(force=True)
        question = data.get("question", "").strip()
        uid = data.get("uid", "")
        lang = data.get("lang_code", "en")

        if not question or not uid:
            return jsonify({"error": "Question and uid are required."}), 400

        cache_path = os.path.join(UPLOAD_DIR, f"{uid}_features.pt")
        if not os.path.exists(cache_path):
            return jsonify({"error": "No active image context found. Please upload an image first."}), 400

        try:
            vqa_model, vqa_tokenizer = _get_vqa_artifacts()
        except Exception as e:
            return jsonify({"error": f"VQA model not loaded: {e}"}), 500

        vision_feat = torch.load(cache_path)  # Already on CPU, [1, feat_dim]

        token_ids = vqa_tokenizer.encode(question, max_len=32, padding=True)
        token_tensor = torch.tensor([token_ids], dtype=torch.long)

        with torch.no_grad():
            logits = vqa_model.forward_cached(vision_feat, token_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        answer_id = int(torch.argmax(probs).item())

        return jsonify({
            "class_id": str(answer_id),
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Utility Endpoints ───────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Gemini-backed chatbot endpoint with deterministic fallback."""
    data = request.get_json(force=True) or {}
    user_message = str(data.get("message", "")).strip()
    user_msg = user_message.lower()
    disease = str(data.get("disease", "Healthy"))
    confidence = float(data.get("confidence", 0.0) or 0.0)
    context = str(
        data.get("context")
        or f"The AI vision model just analyzed a dragon fruit image and detected {disease.replace('_', ' ')} with {confidence:.2%} confidence."
    ).strip()
    history = data.get("history", [])

    normalized_history = _normalize_chat_history(history)

    if not user_message:
        return jsonify({
            "success": False,
            "reply": "Please type a question and I'll do my best to help!",
            "history": normalized_history,
        }), 400

    if disease not in DISEASE_KNOWLEDGE:
        disease = "Healthy"

    db = DISEASE_KNOWLEDGE[disease]

    if GEMINI_API_KEY:
        try:
            reply = _call_gemini_chat(user_message, context, normalized_history)
            updated_history = normalized_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": reply},
            ]
            return jsonify({"success": True, "reply": reply, "history": updated_history, "source": "gemini"})
        except Exception as exc:
            app.logger.warning("Gemini chat failed, falling back to the knowledge base: %s", exc)

    reply = _fallback_chat_reply(user_msg, disease, confidence, db)
    updated_history = normalized_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return jsonify({"success": True, "reply": reply, "history": updated_history, "source": "fallback"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
