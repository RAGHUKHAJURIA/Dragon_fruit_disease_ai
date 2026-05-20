"""
Dataset audit utility for dragon-fruit disease fine-tuning.

Purpose:
- Inspect folder and zip datasets.
- Count image totals per class-like folder.
- Check compatibility with the 6-class disease taxonomy used by ConViTX.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Tuple
import zipfile

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

TARGET_CLASSES = [
    "Anthracnose",
    "Brown_Stem_Spot",
    "Gray_Blight",
    "Healthy",
    "Soft_Rot",
    "Stem_Canker",
]

ALIASES = {
    "anthracnose": "Anthracnose",
    "brown stem spot": "Brown_Stem_Spot",
    "brown_stem_spot": "Brown_Stem_Spot",
    "brown-spot": "Brown_Stem_Spot",
    "gray blight": "Gray_Blight",
    "grey blight": "Gray_Blight",
    "gray_blight": "Gray_Blight",
    "healthy": "Healthy",
    "good fruit": "Healthy",
    "good leaf": "Healthy",
    "soft rot": "Soft_Rot",
    "soft_rot": "Soft_Rot",
    "stem canker": "Stem_Canker",
    "stem_canker": "Stem_Canker",
}

COARSE_OR_INCOMPATIBLE = {
    "fruit",
    "leaf",
    "bad fruit",
    "bad leaf",
    "diseased",
    "disease",
    "good",
    "bad",
}


@dataclass
class SourceSummary:
    path: str
    kind: str
    exists: bool
    total_images: int
    discovered_classes: Dict[str, int]
    mapped_target_counts: Dict[str, int]
    unmapped_classes: Dict[str, int]
    coarse_only: bool
    notes: List[str]


def normalize_label(label: str) -> str:
    token = label.strip().lower().replace("-", " ").replace("_", " ")
    token = " ".join(token.split())
    return token


def map_label(label: str) -> str | None:
    norm = normalize_label(label)
    return ALIASES.get(norm)


def is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTS


def count_classes_from_dir(root: Path) -> Counter:
    class_counts: Counter = Counter()

    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue

        rel = p.relative_to(root)
        parts = rel.parts
        if len(parts) < 2:
            continue

        # Handle common wrappers: Augmented/<class>/..., oversample/<class>/...
        if parts[0].lower() in {"augmented", "dataset", "data", "oversample"} and len(parts) >= 3:
            cls = parts[1]
        else:
            cls = parts[0]

        class_counts[cls] += 1

    return class_counts


def count_classes_from_zip(zip_path: Path) -> Counter:
    class_counts: Counter = Counter()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/") or not is_image_name(name):
                continue
            parts = PurePosixPath(name).parts
            if len(parts) < 2:
                continue

            if len(parts) >= 3 and parts[0].lower() in {"augmented", "dataset", "data"}:
                cls = parts[1]
            else:
                cls = parts[1] if len(parts) >= 3 else parts[0]

            class_counts[cls] += 1
    return class_counts


def summarize_class_compatibility(class_counts: Counter) -> Tuple[Dict[str, int], Dict[str, int], bool, List[str]]:
    mapped = defaultdict(int)
    unmapped = {}
    coarse_tokens_seen = []

    for cls_name, n in class_counts.items():
        mapped_cls = map_label(cls_name)
        if mapped_cls is not None:
            mapped[mapped_cls] += n
        else:
            unmapped[cls_name] = n
            norm = normalize_label(cls_name)
            if norm in COARSE_OR_INCOMPATIBLE:
                coarse_tokens_seen.append(cls_name)

    for c in TARGET_CLASSES:
        mapped.setdefault(c, 0)

    coarse_only = len(unmapped) > 0 and len(mapped) > 0 and sum(mapped.values()) == mapped["Healthy"] and len(coarse_tokens_seen) == len(unmapped)

    notes: List[str] = []
    if not class_counts:
        notes.append("No image-folder class structure found.")
    elif len(unmapped) == 0:
        notes.append("All discovered classes map to the 6-class disease taxonomy.")
    else:
        notes.append("Some classes are unmapped; manual mapping/annotation is required before supervised 6-class training.")
        if coarse_tokens_seen:
            notes.append("Contains coarse labels (e.g., Fruit/Leaf/Bad/Good) that are not disease-specific.")

    return dict(mapped), unmapped, coarse_only, notes


def audit_path(path_str: str) -> SourceSummary:
    path = Path(path_str)
    if not path.exists():
        return SourceSummary(
            path=path_str,
            kind="missing",
            exists=False,
            total_images=0,
            discovered_classes={},
            mapped_target_counts={c: 0 for c in TARGET_CLASSES},
            unmapped_classes={},
            coarse_only=False,
            notes=["Path does not exist."],
        )

    if path.is_file() and path.suffix.lower() == ".zip":
        class_counts = count_classes_from_zip(path)
        mapped, unmapped, coarse_only, notes = summarize_class_compatibility(class_counts)
        return SourceSummary(
            path=path_str,
            kind="zip",
            exists=True,
            total_images=sum(class_counts.values()),
            discovered_classes=dict(class_counts),
            mapped_target_counts=mapped,
            unmapped_classes=unmapped,
            coarse_only=coarse_only,
            notes=notes,
        )

    if path.is_dir():
        # Audit direct class folders
        class_counts = count_classes_from_dir(path)

        # Also include any zip files found one level deep
        zip_paths = list(path.glob("*.zip"))
        for zp in zip_paths:
            zip_counts = count_classes_from_zip(zp)
            class_counts.update(zip_counts)

        mapped, unmapped, coarse_only, notes = summarize_class_compatibility(class_counts)
        if zip_paths:
            notes.append(f"Found {len(zip_paths)} zip file(s) and included their class counts.")

        return SourceSummary(
            path=path_str,
            kind="directory",
            exists=True,
            total_images=sum(class_counts.values()),
            discovered_classes=dict(class_counts),
            mapped_target_counts=mapped,
            unmapped_classes=unmapped,
            coarse_only=coarse_only,
            notes=notes,
        )

    return SourceSummary(
        path=path_str,
        kind="unknown",
        exists=True,
        total_images=0,
        discovered_classes={},
        mapped_target_counts={c: 0 for c in TARGET_CLASSES},
        unmapped_classes={},
        coarse_only=False,
        notes=["Unsupported source type."],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dataset paths for 6-class dragon-fruit disease training")
    parser.add_argument("paths", nargs="+", help="Dataset paths (folders or zip files)")
    parser.add_argument("--out-json", default="models/dataset_audit_report.json", help="Path to save JSON report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [audit_path(p) for p in args.paths]

    print("=" * 80)
    print("Dragon Fruit Dataset Audit")
    print("=" * 80)

    for s in summaries:
        print(f"\nSource: {s.path}")
        print(f"Type: {s.kind} | Exists: {s.exists} | Total images: {s.total_images}")
        print("Mapped target counts:")
        for k in TARGET_CLASSES:
            print(f"  - {k:<16} : {s.mapped_target_counts.get(k, 0)}")

        print("Top discovered classes:")
        top = sorted(s.discovered_classes.items(), key=lambda x: x[1], reverse=True)[:12]
        if not top:
            print("  - None")
        else:
            for cls, n in top:
                print(f"  - {cls}: {n}")

        if s.unmapped_classes:
            print("Unmapped classes (needs manual mapping/annotation):")
            for cls, n in sorted(s.unmapped_classes.items(), key=lambda x: x[1], reverse=True)[:12]:
                print(f"  - {cls}: {n}")

        print("Notes:")
        for note in s.notes:
            print(f"  - {note}")

    combined = Counter()
    for s in summaries:
        combined.update(s.mapped_target_counts)

    print("\n" + "-" * 80)
    print("Combined mapped counts across all provided sources")
    print("-" * 80)
    for k in TARGET_CLASSES:
        print(f"{k:<16} : {combined.get(k, 0)}")

    report = {
        "target_classes": TARGET_CLASSES,
        "summaries": [asdict(s) for s in summaries],
        "combined_mapped_counts": {k: combined.get(k, 0) for k in TARGET_CLASSES},
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
