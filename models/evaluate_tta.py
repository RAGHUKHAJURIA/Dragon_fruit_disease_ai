"""
Test-Time Augmentation (TTA) Evaluation for ConViTX-Pretrained
================================================================
Loads the best checkpoint and evaluates using TTA:
  - Original image
  - Horizontal flip
  - Vertical flip
  - 90° rotation
  - 180° rotation
  - 270° rotation
  - Center crop (tight)
  - Brightness/contrast mild shift

Averages softmax probabilities across all augmentations before argmax.
TTA typically adds +1 to +2% accuracy with NO additional training.

Usage:
  python models/evaluate_tta.py
  python models/evaluate_tta.py --ckpt models/best_convitx_pretrained.pth --data-dir dataset/merged_6class
  python models/evaluate_tta.py --tta-level 3   # 3 = max augmentations
"""

import argparse
import os
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch_directml
    _HAS_DML = True
except ImportError:
    _HAS_DML = False

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from convitx_pretrained import build_convitx_pretrained


# ─── Args ─────────────────────────────────────────────────────────────────────
def parse_args():
    default_data = os.path.normpath(os.path.join(
        _SCRIPT_DIR, "..", "dataset", "merged_6class"
    ))
    default_ckpt = os.path.join(_SCRIPT_DIR, "best_convitx_pretrained.pth")
    p = argparse.ArgumentParser(description="TTA Evaluation for ConViTX-Pretrained")
    p.add_argument("--ckpt",       type=str, default=default_ckpt)
    p.add_argument("--data-dir",   type=str, default=default_data)
    p.add_argument("--img-size",   type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-split",type=float, default=0.80)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--num-workers",type=int, default=0)
    p.add_argument("--tta-level",  type=int, default=2,
                   help="1=basic(4 aug), 2=standard(6 aug), 3=max(10 aug)")
    # Model arch (must match checkpoint)
    p.add_argument("--vit-dim",    type=int, default=192)
    p.add_argument("--vit-depth",  type=int, default=4)
    p.add_argument("--vit-heads",  type=int, default=8)
    return p.parse_args()


# ─── TTA Transform Sets ────────────────────────────────────────────────────────
def get_tta_transforms(img_size: int, level: int):
    """Return list of transform pipelines for TTA."""
    mean_ = [0.485, 0.456, 0.406]
    std_  = [0.229, 0.224, 0.225]

    def base(extra=None):
        ops = [transforms.Resize((img_size, img_size))]
        if extra:
            ops.extend(extra)
        ops += [transforms.ToTensor(), transforms.Normalize(mean_, std_)]
        return transforms.Compose(ops)

    # Level 1: original + 3 flips/rotation
    tf_list = [
        base(),                                                  # original
        base([transforms.RandomHorizontalFlip(p=1.0)]),         # H-flip
        base([transforms.RandomVerticalFlip(p=1.0)]),           # V-flip
        base([transforms.RandomRotation((90, 90))]),            # 90°
    ]

    if level >= 2:
        tf_list += [
            base([transforms.RandomRotation((180, 180))]),      # 180°
            base([transforms.RandomRotation((270, 270))]),      # 270°
        ]

    if level >= 3:
        # 5-crop: center + 4 corners
        crop_size = int(img_size * 0.9)
        tf_list += [
            transforms.Compose([
                transforms.Resize((img_size + 20, img_size + 20)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(), transforms.Normalize(mean_, std_)
            ]),
            base([transforms.RandomHorizontalFlip(p=1.0),
                  transforms.RandomRotation((90, 90))]),
            base([transforms.RandomHorizontalFlip(p=1.0),
                  transforms.RandomRotation((270, 270))]),
            base([transforms.ColorJitter(brightness=0.1, contrast=0.1)]),
        ]

    return tf_list


# ─── TTA Inference ────────────────────────────────────────────────────────────
@torch.no_grad()
def tta_inference(model, full_ds, val_idx, tta_transforms, batch_size,
                  num_workers, device):
    """Run inference with multiple augmentations and average probabilities."""
    model.eval()

    # Collect softmax probs for each TTA transform
    all_probs = []
    for i, tf in enumerate(tta_transforms):
        ds_copy = copy.copy(full_ds)
        ds_copy.transform = tf
        sub = Subset(ds_copy, val_idx)
        loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
        probs_i = []
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)
            probs_i.append(probs.cpu())
        all_probs.append(torch.cat(probs_i, dim=0))
        print(f"  TTA pass {i+1}/{len(tta_transforms)} done")

    # Average across augmentations
    avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)   # (N, C)
    preds     = avg_probs.argmax(dim=1).numpy()

    # Collect true labels
    true_labels = [full_ds.samples[i][1] for i in val_idx]

    return preds, np.array(true_labels)


# ─── Plot helpers ──────────────────────────────────────────────────────────────
def plot_cm(cm, class_names, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
    elif _HAS_DML:
        device = torch_directml.device()
        print("Device: DirectML (AMD/Intel GPU)")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    ckpt_path = args.ckpt
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(_SCRIPT_DIR, os.path.basename(args.ckpt))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    data_dir = os.path.normpath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    # ── Dataset split (same seed as training to get the same val set) ──────
    from collections import Counter
    full_ds     = datasets.ImageFolder(data_dir)
    class_names = full_ds.classes
    num_classes = len(class_names)
    all_targets = [s[1] for s in full_ds.samples]
    all_idx     = list(range(len(full_ds)))

    from sklearn.model_selection import train_test_split
    _, val_idx = train_test_split(
        all_idx, train_size=args.train_split,
        random_state=args.seed, stratify=all_targets,
    )
    print(f"\nDataset : {data_dir}  ({len(full_ds)} images)")
    print(f"Val set : {len(val_idx)} images — {dict(Counter([all_targets[i] for i in val_idx]))}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {ckpt_path}")
    model = build_convitx_pretrained(
        num_classes=num_classes,
        vit_dim=args.vit_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"Warning — missing keys: {missing[:3]}")
    model.eval()
    print("Checkpoint loaded ✅")

    # ── Baseline (no TTA) ───────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Baseline (no TTA)")
    print(f"{'─'*50}")
    base_tf     = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    preds_base, true_labels = tta_inference(
        model, full_ds, val_idx, [base_tf],
        args.batch_size, args.num_workers, device
    )
    acc_base = (preds_base == true_labels).mean()
    f1_base  = f1_score(true_labels, preds_base, average="macro", zero_division=0)
    print(f"  Accuracy : {acc_base * 100:.2f}%")
    print(f"  Macro F1 : {f1_base:.4f}")
    print(classification_report(true_labels, preds_base,
                                target_names=class_names, zero_division=0))

    # ── TTA ─────────────────────────────────────────────────────────────────
    tta_tfs = get_tta_transforms(args.img_size, args.tta_level)
    print(f"\n{'─'*50}")
    print(f"  TTA Level {args.tta_level} ({len(tta_tfs)} augmentations)")
    print(f"{'─'*50}")
    preds_tta, _ = tta_inference(
        model, full_ds, val_idx, tta_tfs,
        args.batch_size, args.num_workers, device
    )
    acc_tta = (preds_tta == true_labels).mean()
    f1_tta  = f1_score(true_labels, preds_tta, average="macro", zero_division=0)
    gain    = (acc_tta - acc_base) * 100

    print(f"\n{'='*50}")
    print(f"  RESULTS")
    print(f"{'='*50}")
    print(f"  Baseline Accuracy : {acc_base * 100:.2f}%")
    print(f"  TTA Accuracy      : {acc_tta * 100:.2f}%  (+{gain:.2f}%)")
    print(f"  Baseline Macro F1 : {f1_base:.4f}")
    print(f"  TTA Macro F1      : {f1_tta:.4f}")

    print(f"\n  Full TTA Classification Report:")
    print(classification_report(true_labels, preds_tta,
                                target_names=class_names, zero_division=0))

    cm = confusion_matrix(true_labels, preds_tta, labels=list(range(num_classes)))
    print(f"  Confusion Matrix:\n{cm}")

    # Save TTA confusion matrix
    cm_path = os.path.join(_SCRIPT_DIR, "convitx_pretrained_tta_cm.png")
    plot_cm(cm, class_names,
            f"ConViTX-Pretrained TTA L{args.tta_level} — {acc_tta*100:.2f}%",
            cm_path)

    if acc_tta >= 0.96:
        print(f"\n  🎉🎉  96%+ ACHIEVED WITH TTA!  ({acc_tta*100:.2f}%)")
    elif acc_tta >= 0.95:
        print(f"\n  🎉  95%+ achieved with TTA!  ({acc_tta*100:.2f}%)")


if __name__ == "__main__":
    main()
