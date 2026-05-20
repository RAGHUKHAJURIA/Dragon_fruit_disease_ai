"""
Synthetic VQA Dataset Generator for Dragon Fruit Disease Advisory.

Takes the existing classification dataset (folders of images by disease class)
and advisor.py rules to automatically generate (Image, Question, Answer)
triplets in JSON format.

Usage:
    python generate_vqa_dataset.py \
        --dataset-root dataset/merged_6class \
        --output       models/vqa_dataset.json \
        --max-per-class 500
"""

from __future__ import annotations
import argparse, glob, json, os, random, sys
from collections import Counter
from typing import List, Dict, Tuple

# ── Project root setup ───────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.vqa_answers import (
    DISEASE_CLASSES,
    get_answer_id,
    NUM_ANSWER_CLASSES,
)
from models.vqa_tokenizer import QUESTION_TEMPLATES


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Image Discovery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}


def discover_images(dataset_root: str) -> Dict[str, List[str]]:
    """
    Scan dataset_root for class folders and collect image paths.

    Expected structure:
        dataset_root/
            Anthracnose/
                img_001.jpg
                ...
            Brown_Stem_Spot/
                ...

    Returns:
        {class_name: [img_path, ...]}
    """
    class_images: Dict[str, List[str]] = {}

    for class_name in DISEASE_CLASSES:
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            print(f"  ⚠ Warning: class directory not found: {class_dir}")
            class_images[class_name] = []
            continue

        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(glob.glob(os.path.join(class_dir, f"*.{ext}")))
            images.extend(glob.glob(os.path.join(class_dir, f"*.{ext.upper()}")))

        # De-duplicate and sort for reproducibility
        images = sorted(set(images))
        class_images[class_name] = images
        print(f"  📂 {class_name:20s}: {len(images):>5d} images")

    return class_images


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Triplet Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def generate_triplets(
    class_images: Dict[str, List[str]],
    max_per_class: int = 500,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate (image, question, answer) triplets.

    For each image, we randomly sample questions from each category
    (diagnosis, severity, treatment, prevention, pathogen) and map
    them to the correct answer_id using the disease class.

    Args:
        class_images:  {class_name: [img_path, ...]}
        max_per_class: Max images to use per class (for balancing)
        seed:          Random seed for reproducibility

    Returns:
        List of triplet dicts with keys:
            image_path, question, answer_id, disease_class, question_type
    """
    rng = random.Random(seed)
    triplets: List[Dict] = []
    question_types = list(QUESTION_TEMPLATES.keys())

    for class_name, images in class_images.items():
        if not images:
            continue

        # Balance: cap images per class
        if len(images) > max_per_class:
            images = rng.sample(images, max_per_class)

        for img_path in images:
            # For each image, generate one question per question type
            for qtype in question_types:
                # Skip pathogen questions for Healthy class (no pathogen)
                if qtype == "pathogen" and class_name == "Healthy":
                    continue

                # Randomly select a question template
                question = rng.choice(QUESTION_TEMPLATES[qtype])

                # Get the answer class ID
                answer_id = get_answer_id(class_name, qtype)

                triplets.append({
                    "image_path":    os.path.relpath(img_path, ROOT),
                    "question":      question,
                    "answer_id":     answer_id,
                    "disease_class": class_name,
                    "question_type": qtype,
                })

    # Shuffle for training
    rng.shuffle(triplets)
    return triplets


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dataset Statistics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def print_stats(triplets: List[Dict]) -> None:
    """Print dataset statistics."""
    print(f"\n{'─' * 60}")
    print(f"  DATASET STATISTICS")
    print(f"{'─' * 60}")
    print(f"  Total triplets: {len(triplets):,}")

    # By disease class
    class_counts = Counter(t["disease_class"] for t in triplets)
    print(f"\n  By disease class:")
    for cls in DISEASE_CLASSES:
        print(f"    {cls:20s}: {class_counts.get(cls, 0):>5d}")

    # By question type
    qtype_counts = Counter(t["question_type"] for t in triplets)
    print(f"\n  By question type:")
    for qtype, count in sorted(qtype_counts.items()):
        print(f"    {qtype:15s}: {count:>5d}")

    # By answer class
    answer_counts = Counter(t["answer_id"] for t in triplets)
    print(f"\n  Answer class distribution:")
    for aid in sorted(answer_counts):
        print(f"    answer_id={aid:>2d}: {answer_counts[aid]:>5d}")

    print(f"{'─' * 60}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Train / Validation Split
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def split_dataset(
    triplets: List[Dict],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split triplets into train and validation sets.
    Stratified by answer_id to maintain class balance.
    """
    rng = random.Random(seed)

    # Group by answer_id
    by_answer: Dict[int, List[Dict]] = {}
    for t in triplets:
        by_answer.setdefault(t["answer_id"], []).append(t)

    train_set, val_set = [], []
    for aid, items in by_answer.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val_set.extend(items[:n_val])
        train_set.extend(items[n_val:])

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    return train_set, val_set


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic VQA dataset for Dragon Fruit Disease."
    )
    parser.add_argument(
        "--dataset-root", type=str,
        default=os.path.join(ROOT, "dataset", "merged_6class"),
        help="Path to classification dataset root (with class folders).",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(ROOT, "models", "vqa_dataset.json"),
        help="Output JSON path for the full dataset.",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=500,
        help="Max images per class for balancing (default: 500).",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Validation split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Dragon Fruit VQA — Synthetic Dataset Generator")
    print("=" * 60)

    # 1. Discover images
    print(f"\n📁 Scanning: {args.dataset_root}")
    class_images = discover_images(args.dataset_root)

    total_images = sum(len(v) for v in class_images.values())
    if total_images == 0:
        print("❌ No images found! Check --dataset-root path.")
        return

    # 2. Generate triplets
    print(f"\n🔧 Generating triplets (max {args.max_per_class} images/class)...")
    triplets = generate_triplets(
        class_images,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

    # 3. Print stats
    print_stats(triplets)

    # 4. Split into train / val
    train_set, val_set = split_dataset(
        triplets,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"\n  Train: {len(train_set):,}  |  Val: {len(val_set):,}")

    # 5. Save to JSON
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Save full dataset
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 Full dataset saved: {args.output}")

    # Save train / val splits
    train_path = args.output.replace(".json", "_train.json")
    val_path   = args.output.replace(".json", "_val.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    print(f"  💾 Train split saved: {train_path}")

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_set, f, ensure_ascii=False, indent=2)
    print(f"  💾 Val split saved:   {val_path}")

    print(f"\n✅ Dataset generation complete!")


if __name__ == "__main__":
    main()
