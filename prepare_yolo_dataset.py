"""
prepare_yolo_dataset.py
═══════════════════════════════════════════════════════════════════════════════
Full pipeline: Raw annotated files  →  YOLOv8-ready train/val/test dataset

Steps performed:
  1. Scan the flat "Annotated Files" folder (images + labels co-located)
  2. Audit original class IDs used by the annotator per disease
  3. Remap class IDs to our canonical YAML mapping
  4. Stratified split into train / val / test (default 70 / 15 / 15)
  5. Copy images  → yolo_dragon_lesions/images/<split>/
     Write labels → yolo_dragon_lesions/labels/<split>/   (with remapped IDs)
  6. Print a full summary report

Usage:
    python prepare_yolo_dataset.py
    python prepare_yolo_dataset.py --train 0.7 --val 0.15 --test 0.15
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent
SOURCE_DIR  = BASE / "dataset" / "Dragon Fruit (Pitahaya)" / "Dragon Fruit (Pitahaya)" / "Annotated Files"
DEST_ROOT   = BASE / "dataset" / "yolo_dragon_lesions"

# ── Class mapping ─────────────────────────────────────────────────────────────
#
#  Our canonical YAML  (data_dragon_lesions.yaml):
#    0: Anthracnose
#    1: Stem_Canker
#    2: Soft_Rot
#    3: Brown_Stem_Spot
#    4: Gray_Blight
#
#  Annotator used different IDs — remapped below by filename prefix

DISEASE_PREFIXES = {
    "Anthracnose":    0,   # annotator used 1  → we want 0
    "Stem_Canker":    1,   # annotator used 2  → we want 1
    "Soft_Rot":       2,   # annotator used 4  → we want 2
    "Brown_Stem_Spot":3,   # annotator used 0  → we want 3
    "Gray_Blight":    4,   # annotator used 3  → we want 4
}

CLASS_NAMES = {v: k for k, v in DISEASE_PREFIXES.items()}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED      = 42

# ── Colours ───────────────────────────────────────────────────────────────────
def _c(text, code): return f"\033[{code}m{text}\033[0m"
OK   = lambda s: _c(f"  ✔  {s}", 32)
WARN = lambda s: _c(f"  ⚠  {s}", 33)
ERR  = lambda s: _c(f"  ✘  {s}", 31)
HEAD = lambda s: _c(f"\n{'═'*62}\n  {s}\n{'═'*62}", 36)
INFO = lambda s: _c(f"  ·  {s}", 37)


# ── Helpers ───────────────────────────────────────────────────────────────────

def disease_from_stem(stem: str) -> str | None:
    """Return the disease prefix that matches this filename stem."""
    for prefix in DISEASE_PREFIXES:
        if stem.startswith(prefix):
            return prefix
    return None


def remap_label(label_path: Path, correct_class: int) -> list[str]:
    """
    Read a YOLO label file, replace every class_id with correct_class,
    and return the new lines (without writing to disk).
    """
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            parts[0] = str(correct_class)   # overwrite class_id
            new_lines.append(" ".join(parts))
    return new_lines


def collect_pairs(source_dir: Path):
    """
    Collect (image_path, label_path, disease_name) tuples from the flat source folder.
    Skips images with no matching label and vice versa.
    """
    images = {f.stem: f for f in source_dir.iterdir()
              if f.suffix.lower() in IMAGE_EXTENSIONS}
    labels = {f.stem: f for f in source_dir.iterdir()
              if f.suffix == ".txt"}

    pairs         = []
    skipped_imgs  = []
    skipped_lbls  = []
    unknown_class = []

    for stem, img_path in sorted(images.items()):
        if stem not in labels:
            skipped_imgs.append(stem)
            continue
        disease = disease_from_stem(stem)
        if disease is None:
            unknown_class.append(stem)
            continue
        pairs.append((img_path, labels[stem], disease))

    for stem in sorted(labels):
        if stem not in images:
            skipped_lbls.append(stem)

    return pairs, skipped_imgs, skipped_lbls, unknown_class


def stratified_split(pairs, train_r, val_r, test_r):
    """
    Split pairs into train/val/test with stratification by disease class,
    so each split has a proportional class distribution.
    """
    rng = random.Random(RANDOM_SEED)
    by_disease = defaultdict(list)
    for p in pairs:
        by_disease[p[2]].append(p)

    train, val, test = [], [], []
    for disease, items in by_disease.items():
        rng.shuffle(items)
        n     = len(items)
        n_val  = max(1, round(n * val_r))
        n_test = max(1, round(n * test_r))
        n_train= n - n_val - n_test
        if n_train < 1:
            n_train = 1
            n_val   = max(1, (n - 1) // 2)
            n_test  = n - 1 - n_val
        train += items[:n_train]
        val   += items[n_train:n_train + n_val]
        test  += items[n_train + n_val:]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def copy_split(pairs, split_name: str, dest_root: Path):
    """Copy images and write remapped labels into the YOLO folder structure."""
    img_dir = dest_root / "images" / split_name
    lbl_dir = dest_root / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    class_counts = defaultdict(int)
    for img_path, lbl_path, disease in pairs:
        correct_class = DISEASE_PREFIXES[disease]

        # Copy image  (keep original filename)
        shutil.copy2(img_path, img_dir / img_path.name)

        # Write remapped label
        new_lines = remap_label(lbl_path, correct_class)
        (lbl_dir / lbl_path.name).write_text(
            "\n".join(new_lines) + "\n", encoding="utf-8"
        )
        class_counts[correct_class] += len(new_lines)

    return class_counts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare YOLOv8 dragon-fruit lesion dataset from raw annotated files."
    )
    parser.add_argument("--train", type=float, default=0.70, help="Train split ratio (default 0.70)")
    parser.add_argument("--val",   type=float, default=0.15, help="Val split ratio   (default 0.15)")
    parser.add_argument("--test",  type=float, default=0.15, help="Test split ratio  (default 0.15)")
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        print(ERR("train + val + test must sum to 1.0"))
        return

    print(_c("\n🐲  Dragon Fruit YOLO Dataset Preparation", 35))
    print(INFO(f"Source : {SOURCE_DIR}"))
    print(INFO(f"Dest   : {DEST_ROOT}"))
    print(INFO(f"Split  : train={args.train:.0%}  val={args.val:.0%}  test={args.test:.0%}"))

    # ── 1. Check source folder ────────────────────────────────────────────────
    if not SOURCE_DIR.exists():
        print(ERR(f"Source directory not found:\n   {SOURCE_DIR}"))
        return

    # ── 2. Collect pairs ──────────────────────────────────────────────────────
    print(HEAD("Step 1 — Scanning annotated files"))
    pairs, skip_imgs, skip_lbls, unk = collect_pairs(SOURCE_DIR)

    print(INFO(f"Valid image-label pairs found : {len(pairs)}"))
    if skip_imgs:
        print(WARN(f"Images with no label (skipped): {len(skip_imgs)}"))
        for s in skip_imgs[:5]:
            print(_c(f"     {s}", 33))
    if skip_lbls:
        print(WARN(f"Labels with no image (skipped): {len(skip_lbls)}"))
    if unk:
        print(ERR(f"Unknown disease prefix (skipped): {len(unk)} — {unk[:3]}"))

    if not pairs:
        print(ERR("No valid pairs found. Aborting."))
        return

    # ── 3. Audit original class distribution ──────────────────────────────────
    print(HEAD("Step 2 — Original class distribution (before remapping)"))
    by_disease: dict[str, int] = defaultdict(int)
    for _, lbl_path, disease in pairs:
        by_disease[disease] += sum(
            1 for ln in lbl_path.read_text().splitlines() if len(ln.strip().split()) == 5
        )
    total_boxes = sum(by_disease.values())
    print(f"  Total bounding boxes : {total_boxes}")
    for disease in sorted(by_disease):
        cnt = by_disease[disease]
        bar = "█" * int(cnt / max(by_disease.values()) * 30)
        cid = DISEASE_PREFIXES[disease]
        print(f"    [{cid}] {disease:<20} {cnt:>5} boxes  {bar}")

    # ── 4. Stratified split ───────────────────────────────────────────────────
    print(HEAD("Step 3 — Stratified split"))
    train_pairs, val_pairs, test_pairs = stratified_split(pairs, args.train, args.val, args.test)
    print(f"  Train : {len(train_pairs)} images")
    print(f"  Val   : {len(val_pairs)} images")
    print(f"  Test  : {len(test_pairs)} images")

    # ── 5. Copy + remap ───────────────────────────────────────────────────────
    print(HEAD("Step 4 — Copying images & writing remapped labels"))

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        counts = copy_split(split_pairs, split_name, DEST_ROOT)
        total  = sum(counts.values())
        print(OK(f"{split_name:>5} — {len(split_pairs):>3} images  |  {total:>5} boxes  "
                 f"  ({', '.join(f'{CLASS_NAMES[c]}:{n}' for c,n in sorted(counts.items()))})"))

    # ── 6. Final summary ──────────────────────────────────────────────────────
    print(HEAD("✅  Dataset preparation complete!"))
    print(f"  Output location : {DEST_ROOT}")
    print(f"\n  Class mapping in data_dragon_lesions.yaml:")
    for cid, name in sorted(CLASS_NAMES.items()):
        print(f"    {cid} → {name}")

    print(_c("""
┌─────────────────────────────────────────────────────────────┐
│  Next steps:                                                │
│                                                             │
│  1. Validate:                                               │
│     python validate_yolo_dataset.py                         │
│                                                             │
│  2. Train  (GPU):                                           │
│     yolo task=detect mode=train \\                           │
│          model=yolov8n.pt \\                                 │
│          data=data_dragon_lesions.yaml \\                    │
│          epochs=100 imgsz=640 batch=16 device=0             │
│                                                             │
│  2b. Train (CPU only, slower):                              │
│     yolo task=detect mode=train \\                           │
│          model=yolov8n.pt \\                                 │
│          data=data_dragon_lesions.yaml \\                    │
│          epochs=100 imgsz=640 batch=8 device=cpu            │
└─────────────────────────────────────────────────────────────┘
""", 32))


if __name__ == "__main__":
    main()
