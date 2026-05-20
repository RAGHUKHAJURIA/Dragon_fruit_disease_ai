"""
Build Merged Dataset for ConViTX Training
==========================================
Scans all discovered datasets, maps classes to our 6 canonical labels,
and creates a merged ImageFolder-compatible directory at:
  dataset/merged_6class/
      Anthracnose/
      Brown_Stem_Spot/
      Gray_Blight/
      Healthy/
      Soft_Rot/
      Stem_Canker/

Class mapping:
─────────────────────────────────────────────────────────────────────
SOURCE                     SOURCE CLASS        → TARGET CLASS
─────────────────────────────────────────────────────────────────────
Pitahaya (main)           Anthracnose          → Anthracnose
Pitahaya (main)           Brown_Stem_Spot      → Brown_Stem_Spot
Pitahaya (main)           Gray_Blight          → Gray_Blight
Pitahaya (main)           Healthy              → Healthy
Pitahaya (main)           Soft_Rot             → Soft_Rot
Pitahaya (main)           Stem_Canker          → Stem_Canker
archive/Fruit             Anthracnose          → Anthracnose
archive/Fruit             Brown Spot           → Brown_Stem_Spot
archive/Fruit             Soft Rot             → Soft_Rot
archive/Fruit             Fruit Rot            → Soft_Rot  (fruit rot ≈ soft rot)
archive/Fruit             Healthy              → Healthy
archive/Leaf              Anthracnose          → Anthracnose
archive/Leaf              Brown Spot           → Brown_Stem_Spot
archive/Leaf              Stem_Canker          → Stem_Canker
archive/Leaf              Twig Blight          → Gray_Blight  (twig blight ≈ gray blight)
archive/Leaf              Stem Rot             → Stem_Canker  (stem rot ≈ canker)
archive/Leaf              Healthy              → Healthy
─────────────────────────────────────────────────────────────────────
EXCLUDED (no reliable mapping):
  archive/Fruit: White Spot, White Spot → skip
  archive/Leaf:  Black Spot, Root Rot   → skip
  Bangladesh:    Bad/Good fruit/leaf    → skip (quality not disease)
─────────────────────────────────────────────────────────────────────

Usage:
  python models/build_merged_dataset.py
  python models/build_merged_dataset.py --dry-run       # preview only, don't copy
  python models/build_merged_dataset.py --output-dir dataset/merged_6class
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from collections import defaultdict


# ─── Canonical class names (must match Pitahaya classes exactly) ──────────────
CLASSES = ["Anthracnose", "Brown_Stem_Spot", "Gray_Blight", "Healthy", "Soft_Rot", "Stem_Canker"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT  = PROJECT_ROOT / "dataset"

# ─── Source definitions (path → {source_class: target_class}) ─────────────────
SOURCES = [
    # ── 1. Main Pitahaya dataset (all 6 classes) ──
    {
        "root": DATASET_ROOT / "Dragon Fruit (Pitahaya)" / "Dragon Fruit (Pitahaya)" / "Converted Images",
        "mapping": {
            "Anthracnose":   "Anthracnose",
            "Brown_Stem_Spot": "Brown_Stem_Spot",
            "Gray_Blight":   "Gray_Blight",
            "Healthy":       "Healthy",
            "Soft_Rot":      "Soft_Rot",
            "Stem_Canker":   "Stem_Canker",
        },
        "label": "Pitahaya",
    },
    # ── 2. Archive Fruit classes ──
    {
        "root": DATASET_ROOT / "archive" / "oversample" / "Fruit",
        "mapping": {
            "Anthracnose": "Anthracnose",
            "Brown Spot":  "Brown_Stem_Spot",
            "Soft Rot":    "Soft_Rot",
            "Fruit Rot":   "Soft_Rot",
            "Healthy":     "Healthy",
            # White Spot → skip
        },
        "label": "Archive/Fruit",
    },
    # ── 3. Archive Leaf classes ──
    {
        "root": DATASET_ROOT / "archive" / "oversample" / "Leaf",
        "mapping": {
            "Anthracnose": "Anthracnose",
            "Brown Spot":  "Brown_Stem_Spot",
            "Stem_Canker": "Stem_Canker",
            "Stem Rot":    "Stem_Canker",   # stem rot ≈ canker
            "Twig Blight": "Gray_Blight",   # twig blight ≈ gray blight
            "Healthy":     "Healthy",
            # Black Spot, Root Rot, White Spot → skip
        },
        "label": "Archive/Leaf",
    },
]

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(source: dict) -> list[tuple[Path, str]]:
    """Return list of (img_path, target_class) for a source definition."""
    root = Path(source["root"])
    mapping = source["mapping"]
    results = []
    if not root.exists():
        print(f"  [WARN] Source not found: {root}")
        return results
    for src_cls, tgt_cls in mapping.items():
        cls_dir = root / src_cls
        if not cls_dir.exists():
            print(f"  [WARN] Class dir not found: {cls_dir}")
            continue
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
        for img in imgs:
            results.append((img, tgt_cls))
    return results


def build_merged(output_dir: Path, dry_run: bool):
    print(f"\n{'='*60}")
    print(f"  Building Merged 6-Class Dataset")
    print(f"  Output → {output_dir}")
    if dry_run:
        print(f"  [DRY RUN] No files will be copied.")
    print(f"{'='*60}\n")

    # Collect all images
    all_images: list[tuple[Path, str, str]] = []   # (path, target_cls, source_label)
    source_counts = defaultdict(lambda: defaultdict(int))
    for src in SOURCES:
        pairs = collect_images(src)
        for img_path, tgt_cls in pairs:
            all_images.append((img_path, tgt_cls, src["label"]))
            source_counts[src["label"]][tgt_cls] += 1
        print(f"  {src['label']}: {len(pairs)} images collected")

    # Print stats
    print(f"\n{'─'*50}")
    print(f"  Per-source breakdown:")
    print(f"{'─'*50}")
    for src_lbl, cls_counts in source_counts.items():
        for cls, cnt in sorted(cls_counts.items()):
            print(f"    {src_lbl:20s}  {cls:20s}  {cnt:4d}")

    # Target counts
    target_counts = defaultdict(int)
    for _, tgt_cls, _ in all_images:
        target_counts[tgt_cls] += 1
    print(f"\n{'─'*50}")
    print(f"  Merged class totals (after mapping):")
    print(f"{'─'*50}")
    total = 0
    for cls in CLASSES:
        cnt = target_counts.get(cls, 0)
        total += cnt
        bar = "█" * (cnt // 10)
        print(f"    {cls:20s}  {cnt:4d}  {bar}")
    print(f"    {'TOTAL':20s}  {total:4d}")

    if dry_run:
        print(f"\n[DRY RUN] Would create {total} symlinks/copies in {output_dir}")
        return

    # Create output dirs
    for cls in CLASSES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    # Copy images with unique names to avoid collisions
    cls_counters = defaultdict(int)
    skipped = 0
    copied  = 0
    for img_path, tgt_cls, src_lbl in all_images:
        cls_counters[tgt_cls] += 1
        src_tag  = src_lbl.replace("/", "_").replace(" ", "_")
        dst_name = f"{src_tag}_{cls_counters[tgt_cls]:05d}{img_path.suffix.lower()}"
        dst_path = output_dir / tgt_cls / dst_name
        if dst_path.exists():
            skipped += 1
            continue
        try:
            shutil.copy2(img_path, dst_path)
            copied += 1
        except Exception as e:
            print(f"  [ERROR] {img_path} → {e}")
            skipped += 1

    print(f"\n  ✅ Done!  Copied={copied}  Skipped={skipped}")
    print(f"  Output: {output_dir}")
    print(f"\n  To train, run:")
    cmd = (
        f"  .venv\\Scripts\\python.exe models\\train_convitx_pretrained.py "
        f"--data-dir dataset\\merged_6class"
    )
    print(cmd)


def main():
    p = argparse.ArgumentParser(description="Build merged 6-class dragon fruit dataset")
    p.add_argument("--output-dir", type=str,
                   default=str(DATASET_ROOT / "merged_6class"))
    p.add_argument("--dry-run", action="store_true",
                   help="Preview without copying any files")
    args = p.parse_args()
    out = Path(args.output_dir)
    build_merged(out, args.dry_run)


if __name__ == "__main__":
    main()
