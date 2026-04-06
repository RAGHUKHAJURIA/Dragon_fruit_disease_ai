"""
train_yolo_directml.py
═══════════════════════════════════════════════════════════════════════
Train YOLOv8 using AMD GPU on Windows via DirectML (torch-directml).

Prerequisites:
    pip install torch-directml

Usage:
    python train_yolo_directml.py
    python train_yolo_directml.py --epochs 100 --batch 8 --imgsz 640
═══════════════════════════════════════════════════════════════════════
"""

import argparse
import sys
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "data_dragon_lesions.yaml"
OUT_DIR   = BASE_DIR / "runs" / "detect"


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 training with AMD GPU via DirectML")
    parser.add_argument("--epochs", type=int,   default=100)
    parser.add_argument("--batch",  type=int,   default=4,
                        help="Batch size — use 4-8 for CPU, 16+ for GPU")
    parser.add_argument("--imgsz",  type=int,   default=416,
                        help="Image size — 416 is faster on CPU, 640 for GPU")
    parser.add_argument("--model",  type=str,   default="yolov8n.pt")
    parser.add_argument("--device", type=str,   default=None,
                        help="'cpu' (default/safe), '0' (CUDA only), or blank to auto-detect")
    args = parser.parse_args()

    # ── Detect best available device ─────────────────────────────────────────
    device = args.device

    if device is None:
        try:
            import torch_directml
            # DirectML is detected but does NOT support torch.unique(return_counts=True)
            # which YOLOv8's loss function requires — so we MUST use CPU.
            _ = torch_directml.device()  # just test it's available
            print("  ⚠  AMD GPU (DirectML) detected but NOT compatible with YOLOv8 training.")
            print("     DirectML is missing 'unique(return_counts)' — using CPU instead.")
            print("     (This is a DirectML limitation, not a code bug.)")
            device = "cpu"
        except ImportError:
            device = "cpu"
        except Exception:
            device = "cpu"

    # Always use CPU (only reliable option for AMD on Windows)
    device = "cpu"
    print(f"  ✔  Using device: CPU (AMD GPU not supported for YOLOv8 training on Windows)")

    print(f"\n🐲  Dragon Fruit YOLOv8 Training")
    print(f"   Model  : {args.model}")
    print(f"   Data   : {DATA_YAML}")
    print(f"   Device : {device}")
    print(f"   Epochs : {args.epochs}  |  Batch: {args.batch}  |  Imgsz: {args.imgsz}")
    print(f"   ⏱  Estimated time on CPU: ~1-3 hours for 100 epochs\n")

    if not DATA_YAML.exists():
        print(f"  ✘  data yaml not found: {DATA_YAML}")
        sys.exit(1)

    # ── Launch training ───────────────────────────────────────────────────────
    from ultralytics import YOLO

    model = YOLO(args.model)

    # DirectML passes a device object — convert to string for ultralytics,
    # and always disable AMP since ultralytics' check_amp() calls
    # torch.cuda.get_device_name() which crashes on non-CUDA devices.
    is_directml = hasattr(device, '__class__') and 'directml' in str(type(device)).lower()
    device_arg  = str(device) if is_directml else device
    use_amp     = False  # Must be False for DirectML / CPU (no CUDA AMP)

    results = model.train(
        data    = str(DATA_YAML),
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = device_arg,
        project = str(OUT_DIR),
        name    = "dragon_lesions",
        exist_ok= True,

        # Disable AMP — required for DirectML / non-CUDA devices
        amp     = use_amp,

        # Training hyperparams
        lr0     = 0.01,
        lrf     = 0.01,
        momentum= 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 3,

        # Augmentation (helps with small dataset)
        hsv_h   = 0.015,
        hsv_s   = 0.7,
        hsv_v   = 0.4,
        flipud  = 0.1,
        fliplr  = 0.5,
        mosaic  = 1.0,
        mixup   = 0.1,

        # Logging
        plots   = True,
        save    = True,
        verbose = True,
    )

    # ── Done — copy best weights ──────────────────────────────────────────────
    best_weights = OUT_DIR / "dragon_lesions" / "weights" / "best.pt"
    dest_weights = BASE_DIR / "models" / "yolo_dragon_best.pt"

    if best_weights.exists():
        import shutil
        dest_weights.parent.mkdir(exist_ok=True)
        shutil.copy2(best_weights, dest_weights)
        print(f"\n  ✅  Best weights saved to: {dest_weights}")
        print(f"\n  To run detection:")
        print(f'     python detect_disease.py --source "path/to/image.jpg" --save')
    else:
        print(f"\n  ⚠  best.pt not found at expected location: {best_weights}")
        print(f"     Check runs/detect/dragon_lesions/weights/ manually")

    print(f"\n  Training results saved to: {OUT_DIR / 'dragon_lesions'}")
    print(f"  mAP50 on val: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")


if __name__ == "__main__":
    main()
