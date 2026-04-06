"""
setup_yolo_folders.py
─────────────────────────────────────────────────────────
Creates the YOLOv8 folder template for dragon-fruit
lesion detection.

Expected structure after running:
  yolo_dragon_lesions/
  ├── images/
  │   ├── train/   ← put training images here
  │   ├── val/     ← put validation images here
  │   └── test/    ← put test images here
  └── labels/
      ├── train/   ← YOLO .txt annotations for train
      ├── val/     ← YOLO .txt annotations for val
      └── test/    ← YOLO .txt annotations for test

Usage:
    python setup_yolo_folders.py
─────────────────────────────────────────────────────────
"""

from pathlib import Path

ROOT = Path(
    r"C:/Users/prith/Desktop/My Projects/Mini_Project/Dragon-fruit Disease"
    r"/mini-project/dragonfruit_disease_ai/dataset/yolo_dragon_lesions"
)

SPLITS = ["train", "val", "test"]

LABEL_README = """\
# Dragon Fruit Lesion — YOLO Label Format

Each .txt file corresponds to one image.
Each line in the file represents one bounding box:

  <class_id> <cx> <cy> <width> <height>

All values are normalised to [0, 1].

Class IDs:
  0 — Anthracnose
  1 — Stem_Canker
  2 — Soft_Rot
  3 — Brown_Stem_Spot
  4 — Gray_Blight
"""

IMAGE_README = """\
# Dragon Fruit Lesion — Image Folder

Supported formats: .jpg  .jpeg  .png  .bmp  .webp

Naming convention  (recommended):
  <disease_abbreviation>_<source_id>_<index>.jpg
  e.g.  ANTH_001_0042.jpg
"""


def create_folders():
    for split in SPLITS:
        img_dir = ROOT / "images" / split
        lbl_dir = ROOT / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✔  Created  {img_dir.relative_to(ROOT.parent.parent)}")
        print(f"  ✔  Created  {lbl_dir.relative_to(ROOT.parent.parent)}")

    # Write README files
    (ROOT / "images" / "README.md").write_text(IMAGE_README)
    (ROOT / "labels" / "README.md").write_text(LABEL_README)
    print(f"\n  ✔  README.md written to images/ and labels/")


if __name__ == "__main__":
    print("\n🐲  Setting up YOLOv8 dragon-fruit lesion dataset folders …\n")
    create_folders()
    print(f"\n✅  Done!  Root → {ROOT}\n")
    print("Next steps:")
    print("  1. Copy labelled images  into  yolo_dragon_lesions/images/<split>/")
    print("  2. Copy YOLO .txt labels into  yolo_dragon_lesions/labels/<split>/")
    print("  3. Run  python validate_yolo_dataset.py  to check everything")
    print("  4. Train with:")
    print("       yolo task=detect mode=train model=yolov8n.pt \\")
    print("            data=data_dragon_lesions.yaml epochs=100 imgsz=640 batch=16 device=0\n")
