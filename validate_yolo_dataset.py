"""
validate_yolo_dataset.py
─────────────────────────────────────────────────────────────────────────────
Pre-training sanity checker for the YOLOv8 dragon-fruit lesion dataset.

Checks performed:
  ✓ Required folder structure exists
  ✓ Every image has a matching .txt label file
  ✓ Every label file has a matching image
  ✓ Label files are non-empty and correctly formatted (class_id cx cy w h)
  ✓ Bounding-box values are in [0, 1]
  ✓ Class IDs are within the valid range (0–4)
  ✓ Per-split class distribution summary

Usage:
    python validate_yolo_dataset.py
    python validate_yolo_dataset.py --dataset_root "path/to/yolo_dragon_lesions"
─────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_DATASET_ROOT = (
    r"C:/Users/prith/Desktop/My Projects/Mini_Project/Dragon-fruit Disease"
    r"/mini-project/dragonfruit_disease_ai/dataset/yolo_dragon_lesions"
)

CLASS_NAMES = {
    0: "Anthracnose",
    1: "Stem_Canker",
    2: "Soft_Rot",
    3: "Brown_Stem_Spot",
    4: "Gray_Blight",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "val", "test"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def color(text: str, code: int) -> str:
    """ANSI colour wrapper (works in modern Windows terminals)."""
    return f"\033[{code}m{text}\033[0m"


OK   = lambda s: color(f"  ✔  {s}", 32)
WARN = lambda s: color(f"  ⚠  {s}", 33)
ERR  = lambda s: color(f"  ✘  {s}", 31)
HEAD = lambda s: color(f"\n{'─'*60}\n  {s}\n{'─'*60}", 36)


def validate_split(images_dir: Path, labels_dir: Path, split: str, errors: list, warnings: list):
    """Validate one split (train / val / test)."""
    print(HEAD(f"Split: {split.upper()}"))

    if not images_dir.exists():
        errors.append(f"[{split}] images/ dir missing: {images_dir}")
        print(ERR(f"images/ directory not found: {images_dir}"))
        return {}

    if not labels_dir.exists():
        errors.append(f"[{split}] labels/ dir missing: {labels_dir}")
        print(ERR(f"labels/ directory not found: {labels_dir}"))
        return {}

    # Collect images and labels
    images = {f.stem: f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS}
    labels = {f.stem: f for f in labels_dir.iterdir() if f.suffix == ".txt"}

    print(f"  Images found : {len(images)}")
    print(f"  Labels found : {len(labels)}")

    # ── 1. Images without labels
    no_label = set(images) - set(labels)
    if no_label:
        for stem in sorted(no_label):
            msg = f"[{split}] Image has no label: {images[stem].name}"
            warnings.append(msg)
            print(WARN(f"No label for image: {images[stem].name}"))
    else:
        print(OK("Every image has a label file"))

    # ── 2. Labels without images
    no_image = set(labels) - set(images)
    if no_image:
        for stem in sorted(no_image):
            msg = f"[{split}] Label has no image: {labels[stem].name}"
            warnings.append(msg)
            print(WARN(f"No image for label: {labels[stem].name}"))
    else:
        print(OK("Every label file has a matching image"))

    # ── 3. Validate label file content
    class_counts: dict[int, int] = defaultdict(int)
    format_errors = 0
    range_errors  = 0
    empty_labels  = 0

    matched_stems = set(images) & set(labels)
    for stem in matched_stems:
        label_path = labels[stem]
        lines = label_path.read_text().strip().splitlines()

        if not lines:
            empty_labels += 1
            warnings.append(f"[{split}] Empty label file: {label_path.name}")
            continue

        for line_no, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                format_errors += 1
                errors.append(
                    f"[{split}] {label_path.name}:{line_no} — expected 5 values, got {len(parts)}"
                )
                continue

            try:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
            except ValueError:
                format_errors += 1
                errors.append(f"[{split}] {label_path.name}:{line_no} — non-numeric values")
                continue

            # Class range check
            if cls_id not in CLASS_NAMES:
                range_errors += 1
                errors.append(
                    f"[{split}] {label_path.name}:{line_no} — class_id {cls_id} not in [0–{len(CLASS_NAMES)-1}]"
                )
            else:
                class_counts[cls_id] += 1

            # BBox range check
            for name, val in [("cx", cx), ("cy", cy), ("w", bw), ("h", bh)]:
                if not (0.0 <= val <= 1.0):
                    range_errors += 1
                    errors.append(
                        f"[{split}] {label_path.name}:{line_no} — {name}={val:.4f} out of [0,1]"
                    )

    # ── Summary for this split
    if empty_labels:
        print(WARN(f"{empty_labels} empty label file(s)"))
    else:
        print(OK("No empty label files"))

    if format_errors:
        print(ERR(f"{format_errors} label formatting error(s) — see details below"))
    else:
        print(OK("All labels are correctly formatted"))

    if range_errors:
        print(ERR(f"{range_errors} out-of-range value(s) — see details below"))
    else:
        print(OK("All bounding boxes are within [0, 1]"))

    # ── Class distribution
    if class_counts:
        print(f"\n  Class distribution ({split}):")
        for cls_id in sorted(class_counts):
            bar = "█" * min(40, class_counts[cls_id] // max(1, max(class_counts.values()) // 40))
            print(f"    [{cls_id}] {CLASS_NAMES[cls_id]:<20} {class_counts[cls_id]:>5}  {bar}")
    else:
        print(WARN("No valid annotations found in this split"))

    return class_counts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate a YOLO-format dragon-fruit dataset.")
    parser.add_argument(
        "--dataset_root",
        default=DEFAULT_DATASET_ROOT,
        help="Path to the yolo_dragon_lesions root directory",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    print(color(f"\n🐲 Dragon Fruit YOLO Dataset Validator", 35))
    print(color(f"   Root: {root}", 35))

    if not root.exists():
        print(ERR(f"Dataset root does not exist: {root}"))
        print(WARN("Run setup_yolo_folders.py first to create the folder template."))
        return

    all_errors:   list[str] = []
    all_warnings: list[str] = []
    global_counts: dict[int, int] = defaultdict(int)

    for split in SPLITS:
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        counts = validate_split(images_dir, labels_dir, split, all_errors, all_warnings)
        for cls_id, cnt in counts.items():
            global_counts[cls_id] += cnt

    # ── Overall summary ───────────────────────────────────────────────────────
    print(HEAD("OVERALL SUMMARY"))

    total_annotations = sum(global_counts.values())
    print(f"  Total annotations : {total_annotations}")
    if global_counts:
        print(f"\n  Global class distribution:")
        for cls_id in sorted(global_counts):
            pct = global_counts[cls_id] / total_annotations * 100
            bar = "█" * int(pct / 2)
            print(f"    [{cls_id}] {CLASS_NAMES[cls_id]:<20} {global_counts[cls_id]:>6}  ({pct:5.1f}%)  {bar}")

    print()
    if all_warnings:
        print(color(f"  ⚠  {len(all_warnings)} warning(s):", 33))
        for w in all_warnings[:20]:
            print(color(f"     {w}", 33))
        if len(all_warnings) > 20:
            print(color(f"     … and {len(all_warnings) - 20} more. Fix label files for full list.", 33))

    if all_errors:
        print(color(f"\n  ✘  {len(all_errors)} error(s):", 31))
        for e in all_errors[:20]:
            print(color(f"     {e}", 31))
        if len(all_errors) > 20:
            print(color(f"     … and {len(all_errors) - 20} more.", 31))
        print(color("\n  ❌  Fix errors above before launching training.", 31))
    else:
        print(color("  ✅  All checks passed — dataset looks good to train!", 32))

    print()


if __name__ == "__main__":
    main()
