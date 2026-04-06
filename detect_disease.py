"""
detect_disease.py
═══════════════════════════════════════════════════════════════════════════════
YOLOv8 Dragon Fruit Lesion Detector

Provides:
  - DragonFruitDetector  : reusable class for Flask integration
  - CLI entrypoint       : run directly for quick inference on images

Usage (CLI):
    python detect_disease.py --source path/to/image.jpg
    python detect_disease.py --source path/to/folder/ --conf 0.35 --save

Usage (in Flask):
    from detect_disease import DragonFruitDetector
    detector = DragonFruitDetector()
    results  = detector.predict("uploads/fruit.jpg")
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "models" / "yolo_dragon_best.pt"   # set after training

CLASS_NAMES = {
    0: "Anthracnose",
    1: "Stem_Canker",
    2: "Soft_Rot",
    3: "Brown_Stem_Spot",
    4: "Gray_Blight",
}

# Distinct BGR colours per class (for OpenCV overlay)
CLASS_COLORS = {
    0: (0,   200,  50),   # Anthracnose    – green
    1: (0,   120, 255),   # Stem_Canker    – orange
    2: (60,    0, 220),   # Soft_Rot       – red
    3: (200, 100,   0),   # Brown_Stem_Spot – blue
    4: (180,  40, 180),   # Gray_Blight    – purple
}

DEFAULT_CONF = 0.35
DEFAULT_IOU  = 0.45


# ── Detector class ────────────────────────────────────────────────────────────

class DragonFruitDetector:
    """
    Wraps a trained YOLOv8 model for dragon-fruit lesion detection.

    Parameters
    ----------
    model_path : str | Path
        Path to the trained .pt weights file.
    conf       : float   Confidence threshold  (0–1)
    iou        : float   NMS IoU threshold     (0–1)
    device     : str     'cpu', '0', '0,1', …
    """

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        conf:   float = DEFAULT_CONF,
        iou:    float = DEFAULT_IOU,
        device: str   = "cpu",
    ):
        from ultralytics import YOLO
        self.model_path = Path(model_path)
        self.conf       = conf
        self.iou        = iou
        self.device     = device

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.model_path}\n"
                "Train first with: yolo task=detect mode=train "
                "model=yolov8n.pt data=data_dragon_lesions.yaml "
                "epochs=100 imgsz=640 batch=16"
            )

        print(f"  Loading YOLOv8 model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("  Model loaded ✔")

    # ── Core prediction ───────────────────────────────────────────────────────

    def predict(
        self,
        source: str | Path | np.ndarray,
        *,
        save_annotated: bool = False,
        output_dir:     str | Path | None = None,
    ) -> dict:
        """
        Run inference on a single image.

        Parameters
        ----------
        source         : File path, URL, or numpy BGR array.
        save_annotated : Write annotated image to disk if True.
        output_dir     : Directory for annotated output (defaults to source dir).

        Returns
        -------
        dict with keys:
            detections  : list of {class_id, class_name, confidence, bbox_xyxy, bbox_xywh_norm}
            disease_counts : {disease_name: count}
            severity    : "None" | "Low" | "Moderate" | "High" | "Severe"
            annotated_image_path : str | None
            raw_image_shape : [H, W, C]
        """
        results     = self.model.predict(
            source,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        result      = results[0]
        boxes       = result.boxes

        detections      = []
        disease_counts  = {name: 0 for name in CLASS_NAMES.values()}

        if boxes is not None and len(boxes):
            for box in boxes:
                cls_id  = int(box.cls[0])
                conf_sc = float(box.conf[0])
                xyxy    = box.xyxy[0].tolist()          # [x1, y1, x2, y2]
                xywhn   = box.xywhn[0].tolist()         # normalised [cx, cy, w, h]

                cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                disease_counts[cls_name] += 1

                detections.append({
                    "class_id":      cls_id,
                    "class_name":    cls_name,
                    "confidence":    round(conf_sc, 4),
                    "bbox_xyxy":     [round(v, 2) for v in xyxy],
                    "bbox_xywh_norm":[round(v, 6) for v in xywhn],
                })

        # ── Severity estimate ──────────────────────────────────────────────
        n = len(detections)
        severity = (
            "None"     if n == 0 else
            "Low"      if n <= 2 else
            "Moderate" if n <= 6 else
            "High"     if n <= 12 else
            "Severe"
        )

        # ── Optional annotated image output ───────────────────────────────
        annotated_path = None
        if save_annotated:
            annotated_path = self._save_annotated(
                result, source, output_dir
            )

        h, w = result.orig_shape
        return {
            "detections":         detections,
            "disease_counts":     {k: v for k, v in disease_counts.items() if v > 0},
            "total_lesions":      n,
            "severity":           severity,
            "annotated_image_path": str(annotated_path) if annotated_path else None,
            "raw_image_shape":    [h, w, 3],
        }

    def predict_bytes(self, image_bytes: bytes) -> dict:
        """Convenience wrapper: accepts raw image bytes (e.g. from Flask request.files)."""
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return self.predict(img)

    # ── Annotated image writer ─────────────────────────────────────────────────

    def _save_annotated(self, result, source, output_dir) -> Path:
        """Draw bounding boxes on the image and save to output_dir."""
        img = result.orig_img.copy()  # BGR numpy array

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id   = int(box.cls[0])
                conf_val = float(box.conf[0])
                color    = CLASS_COLORS.get(cls_id, (255, 255, 255))
                label    = f"{CLASS_NAMES.get(cls_id, '?')} {conf_val:.2f}"

                # Box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                # Label text
                cv2.putText(img, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Determine output path
        if output_dir:
            out_dir = Path(output_dir)
        elif isinstance(source, (str, Path)):
            out_dir = Path(source).parent
        else:
            out_dir = BASE_DIR / "runs" / "detections"
        out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(source, (str, Path)):
            stem = Path(source).stem
        else:
            stem = "detection"
        out_path = out_dir / f"{stem}_annotated.jpg"
        cv2.imwrite(str(out_path), img)
        return out_path

    # ── Batch inference ───────────────────────────────────────────────────────

    def predict_folder(self, folder: str | Path, extensions=(".jpg", ".jpeg", ".png")) -> list[dict]:
        """Run predict() on every image in a folder and return a list of result dicts."""
        folder = Path(folder)
        results = []
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() in extensions:
                r = self.predict(img_path)
                r["image"] = img_path.name
                results.append(r)
        return results


# ── Flask helper ──────────────────────────────────────────────────────────────

def get_detector(model_path: str | Path = MODEL_PATH, **kwargs) -> DragonFruitDetector:
    """
    Singleton-style factory for Flask apps.
    Call once at app startup, reuse the returned instance.

    Example (in app.py / routes):
        from detect_disease import get_detector
        detector = get_detector()

        @app.route("/detect", methods=["POST"])
        def detect():
            img_bytes = request.files["image"].read()
            result    = detector.predict_bytes(img_bytes)
            return jsonify(result)
    """
    return DragonFruitDetector(model_path=model_path, **kwargs)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dragon Fruit YOLOv8 Lesion Detector")
    parser.add_argument("--source",  required=True,  help="Image path, folder, or URL")
    parser.add_argument("--model",   default=str(MODEL_PATH), help="Path to .pt weights")
    parser.add_argument("--conf",    type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument("--iou",     type=float, default=DEFAULT_IOU,  help="NMS IoU threshold")
    parser.add_argument("--device",  default="cpu", help="'cpu' or GPU id e.g. '0'")
    parser.add_argument("--save",    action="store_true", help="Save annotated image(s)")
    parser.add_argument("--out_dir", default=None,  help="Output directory for annotated images")
    parser.add_argument("--json",    action="store_true", help="Print results as JSON")
    args = parser.parse_args()

    detector = DragonFruitDetector(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )

    source  = Path(args.source)
    results = []

    if source.is_dir():
        results = detector.predict_folder(source)
    else:
        r = detector.predict(source, save_annotated=args.save, output_dir=args.out_dir)
        r["image"] = source.name
        results = [r]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"\n── {r['image']} ──")
            print(f"   Total lesions : {r['total_lesions']}")
            print(f"   Severity      : {r['severity']}")
            if r["disease_counts"]:
                for disease, count in r["disease_counts"].items():
                    print(f"   {disease:<20}: {count} detection(s)")
            else:
                print("   No disease detected.")
            if r.get("annotated_image_path"):
                print(f"   Saved to      : {r['annotated_image_path']}")


if __name__ == "__main__":
    main()
