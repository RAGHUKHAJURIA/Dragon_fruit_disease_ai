"""
Step 3: Strict Held-Out Test Evaluation
========================================
Loads the best checkpoint from Step 2, evaluates ONCE on the held-out test split
(never touched during training), and produces all required artifacts.

Requires:
  models/dataset_split.json   — produced by train_step2.py
  models/step2_best_model.json — produced by train_step2.py

Outputs (all in --save-dir, default: models/):
  test_metrics.json
  classification_report.txt
  confusion_matrix_test.png
  step3_summary.md            → project root

Usage:
  python evaluate_step3.py [--ckpt PATH] [--model-name resnet50|efficientnet_b3]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    import torch_directml
    _HAS_DML = True
except ImportError:
    _HAS_DML = False

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


# --------------------------------------------------------------------------- #
#  Args
# --------------------------------------------------------------------------- #
def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.normpath(
        os.path.join(
            script_dir, "..",
            "dataset",
            "Dragon Fruit (Pitahaya)",
            "Dragon Fruit (Pitahaya)",
            "Converted Images",
        )
    )
    # Try to load defaults from step2_best_model.json
    best_info_path = os.path.join(script_dir, "step2_best_model.json")
    default_ckpt = ""
    default_model = "resnet50"
    default_split = os.path.join(script_dir, "dataset_split.json")
    if os.path.isfile(best_info_path):
        with open(best_info_path) as f:
            best_info = json.load(f)
        default_ckpt  = best_info.get("ckpt_path", "")
        default_model = best_info.get("model_name", "resnet50")
        default_split = best_info.get("split_path", default_split)

    p = argparse.ArgumentParser(description="Step 3 — Strict Held-Out Test Evaluation")
    p.add_argument("--data-dir",   type=str, default=default_data)
    p.add_argument("--save-dir",   type=str, default=script_dir)
    p.add_argument("--ckpt",       type=str, default=default_ckpt,
                   help="Path to best checkpoint (from Step 2)")
    p.add_argument("--model-name", type=str, default=default_model,
                   choices=["resnet50", "efficientnet_b3", "convitx"],
                   help="Architecture name matching the checkpoint")
    p.add_argument("--split-file", type=str, default=default_split,
                   help="Path to dataset_split.json from Step 2")
    p.add_argument("--img-size",   type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers",type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Model loader
# --------------------------------------------------------------------------- #
def load_model(model_name: str, num_classes: int, ckpt_path: str, device):
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, num_classes),
        )
    elif model_name == "efficientnet_b3":
        if not _HAS_TIMM:
            raise RuntimeError("timm not installed; cannot load EfficientNet-B3.")
        model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)
    elif model_name == "convitx":
        # Import local ConViTX builder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        import sys
        sys.path.insert(0, script_dir)
        from convitx import build_convitx_base
        model = build_convitx_base(num_classes=num_classes, enforce_budget=False)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    ckpt = torch.load(ckpt_path, map_location=device)
    # Handle checkpoints that may be bare state_dicts or have a key
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")
    return model


# --------------------------------------------------------------------------- #
#  Confusion matrix plot
# --------------------------------------------------------------------------- #
def plot_confusion(cm, class_names, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix → {out_path}")


# --------------------------------------------------------------------------- #
#  Error analysis
# --------------------------------------------------------------------------- #
def error_analysis(cm, class_names, report_dict):
    n = len(class_names)
    print("\n" + "=" * 60)
    print("  ERROR ANALYSIS")
    print("=" * 60)

    # Weakest classes by F1
    f1_scores = [(cls, report_dict.get(cls, {}).get("f1-score", 0.0)) for cls in class_names]
    f1_scores.sort(key=lambda x: x[1])
    print("\n⚠  Weakest classes (by F1):")
    for cls, f1 in f1_scores[:3]:
        print(f"   {cls:<22} F1={f1:.4f}")

    # Top confusion pairs (off-diagonal)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    print("\n🔀  Top confusion pairs (true → predicted):")
    for cnt, true_cls, pred_cls in pairs[:5]:
        print(f"   {true_cls:<22} → {pred_cls:<22}  ({cnt} samples)")

    print("\n🎯  3 Concrete actions to improve test accuracy toward 90%+:")
    actions = [
        (
            "1. Targeted augmentation for hard classes",
            f"Apply stronger MixUp / CutMix for '{f1_scores[0][0]}' and '{f1_scores[1][0]}' "
            "to simulate cross-class appearance variation."
        ),
        (
            "2. Larger backbone / full fine-tuning",
            "Unfreeze all layers (layer1–4) in ResNet50 or switch to EfficientNet-B4+ "
            "with a lower learning rate (1e-5) for deeper feature adaptation."
        ),
        (
            "3. Test-Time Augmentation (TTA)",
            "Average predictions over 5-8 horizontally/vertically flipped & slightly "
            "rotated crops at inference — typically adds 1-3% accuracy with zero retraining."
        ),
    ]
    for title, desc in actions:
        print(f"\n  {title}")
        print(f"     {desc}")
    print()
    return f1_scores, pairs


# --------------------------------------------------------------------------- #
#  Step 3 markdown writer
# --------------------------------------------------------------------------- #
def write_step3_summary(metrics, per_class, class_names, f1_scores, pairs, model_name, project_root):
    lines = [
        "# Step 3 Summary — Held-Out Test Evaluation\n",
        f"*Model: `{model_name}`*\n",
        "",
        "## Test Set Metrics\n",
        f"| Metric | Value |",
        "|--------|-------|",
        f"| Test Accuracy | **{metrics['test_acc_pct']:.2f}%** |",
        f"| Macro F1 | **{metrics['macro_f1']:.4f}** |",
        f"| Weighted F1 | **{metrics['weighted_f1']:.4f}** |",
        "",
        "## Per-Class Metrics\n",
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|----------:|-------:|---:|--------:|",
    ]
    for cls in class_names:
        m = per_class.get(cls, {})
        lines.append(
            f"| {cls} | {m.get('precision', 0):.4f} | "
            f"{m.get('recall', 0):.4f} | {m.get('f1-score', 0):.4f} | "
            f"{int(m.get('support', 0))} |"
        )

    lines += [
        "",
        "## Weakest Classes\n",
    ]
    for cls, f1 in f1_scores[:3]:
        lines.append(f"- **{cls}** — F1 = {f1:.4f}")

    lines += [
        "",
        "## Top Confusion Pairs\n",
    ]
    for cnt, tc, pc in pairs[:5]:
        lines.append(f"- `{tc}` → `{pc}` ({cnt} samples)")

    lines += [
        "",
        "## Actions to Reach 90%+\n",
        "1. **Targeted augmentation** — MixUp/CutMix for weak classes.",
        "2. **Full fine-tuning** — Unfreeze all backbone layers with LR 1e-5.",
        "3. **Test-Time Augmentation (TTA)** — Average 5–8 augmented crops at inference.",
        "",
        "## Artifacts\n",
        "- `models/test_metrics.json`",
        "- `models/classification_report.txt`",
        "- `models/confusion_matrix_test.png`",
    ]

    md_path = os.path.join(project_root, "step3_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n📄 Step 3 markdown summary → {md_path}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif _HAS_DML:
        device = torch_directml.device()
        print("Device: DirectML")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    save_dir = os.path.normpath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Validate inputs
    # ------------------------------------------------------------------ #
    if not args.ckpt or not os.path.isfile(args.ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: '{args.ckpt}'\n"
            "Run train_step2.py first, or specify --ckpt manually."
        )
    if not os.path.isfile(args.split_file):
        raise FileNotFoundError(
            f"Split file not found: '{args.split_file}'\n"
            "Run train_step2.py first, or specify --split-file manually."
        )

    # ------------------------------------------------------------------ #
    # Load split
    # ------------------------------------------------------------------ #
    with open(args.split_file) as f:
        split = json.load(f)

    test_idx    = split["test_idx"]
    class_names = split["class_names"]
    num_classes = len(class_names)
    data_dir    = split.get("data_dir", args.data_dir)
    seed        = split.get("seed", 42)

    print(f"\nSplit file   : {args.split_file}")
    print(f"Seed         : {seed}")
    print(f"Test samples : {len(test_idx)}")
    print(f"Classes      : {class_names}")
    print(f"Model        : {args.model_name}")
    print(f"Checkpoint   : {args.ckpt}")

    # ------------------------------------------------------------------ #
    # Build test loader (no augmentation, no shuffling)
    # ------------------------------------------------------------------ #
    test_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_ds = datasets.ImageFolder(data_dir, transform=test_tf)
    test_sub = Subset(full_ds, test_idx)
    test_loader = DataLoader(
        test_sub,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------ #
    # Load model & evaluate ONCE on test set
    # ------------------------------------------------------------------ #
    print("\nLoading model ...")
    model = load_model(args.model_name, num_classes, args.ckpt, device)

    print("\nEvaluating on held-out test set ...")
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # ------------------------------------------------------------------ #
    # Compute metrics
    # ------------------------------------------------------------------ #
    test_acc     = accuracy_score(all_labels, all_preds)
    macro_f1     = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    report_str   = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )
    report_dict  = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # ------------------------------------------------------------------ #
    # Print results
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print("  STEP 3 — TEST SET RESULTS")
    print(f"{'=' * 60}")
    print(f"  Test Accuracy   : {test_acc * 100:.2f}%")
    print(f"  Macro F1        : {macro_f1:.4f}")
    print(f"  Weighted F1     : {weighted_f1:.4f}")
    print(f"\nClassification Report:\n{report_str}")
    print(f"Confusion Matrix:\n{cm}")

    # ------------------------------------------------------------------ #
    # Error analysis
    # ------------------------------------------------------------------ #
    f1_scores, pairs = error_analysis(cm, class_names, report_dict)

    # ------------------------------------------------------------------ #
    # Save artifacts
    # ------------------------------------------------------------------ #
    metrics = {
        "model":         args.model_name,
        "checkpoint":    args.ckpt,
        "test_samples":  len(all_labels),
        "test_acc":      round(test_acc, 6),
        "test_acc_pct":  round(test_acc * 100, 2),
        "macro_f1":      round(macro_f1, 6),
        "weighted_f1":   round(weighted_f1, 6),
        "per_class":     {k: v for k, v in report_dict.items() if k in class_names},
        "confusion_matrix": cm.tolist(),
        "class_names":   class_names,
        "seed":          seed,
    }

    metrics_path = os.path.join(save_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  test_metrics.json → {metrics_path}")

    report_path = os.path.join(save_dir, "classification_report.txt")
    header = (
        f"Step 3 — Held-Out Test Evaluation\n"
        f"Model      : {args.model_name}\n"
        f"Checkpoint : {args.ckpt}\n"
        f"Test Acc   : {test_acc * 100:.2f}%\n"
        f"Macro F1   : {macro_f1:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n"
        f"{'=' * 60}\n\n"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(header + report_str)
    print(f"  classification_report.txt → {report_path}")

    cm_path = os.path.join(save_dir, "confusion_matrix_test.png")
    plot_confusion(
        cm, class_names,
        f"Confusion Matrix — {args.model_name} (Test Set)",
        cm_path,
    )

    # ------------------------------------------------------------------ #
    # Write markdown summary
    # ------------------------------------------------------------------ #
    project_root = os.path.normpath(os.path.join(save_dir, ".."))
    write_step3_summary(
        metrics,
        metrics["per_class"],
        class_names,
        f1_scores,
        pairs,
        args.model_name,
        project_root,
    )

    print(f"\n{'=' * 60}")
    print("  ✅ Step 3 complete.")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"  Weighted F1   : {weighted_f1:.4f}")
    print(f"{'=' * 60}")

    if test_acc >= 0.90:
        print("🎉  TARGET REACHED: Test accuracy ≥ 90%!")
    else:
        gap = (0.90 - test_acc) * 100
        print(f"   Gap to 90% target: {gap:.1f} percentage points")
        print("   See 'Actions to Reach 90%+' in step3_summary.md")


if __name__ == "__main__":
    main()
