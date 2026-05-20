"""
Step 2: Stronger Backbone Training & Comparison
================================================
Trains ResNet50 and EfficientNet-B3 (via timm) on the 6-class dragon-fruit
disease dataset. Produces a side-by-side comparison table vs. existing ConViTX.

Outputs (all in --save-dir, default: this script's directory):
  best_resnet50_step2.pth
  best_efficientnet_step2.pth
  resnet50_curves.png
  efficientnet_curves.png
  confusion_resnet50.png
  confusion_efficientnet.png
  step2_results.md   (written to project root)

Usage:
  python train_step2.py [--epochs 25] [--batch-size 16] [--seed 42]
"""

import argparse
import copy
import json
import os
import random
import time

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
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

# --------------------------------------------------------------------------- #
#  Try importing timm  (EfficientNet)
# --------------------------------------------------------------------------- #
try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False
    print("[WARNING] timm not installed. EfficientNet training will be skipped.")
    print("          Install with:  pip install timm")


# --------------------------------------------------------------------------- #
#  Seed helper
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
#  Argument parsing
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
    p = argparse.ArgumentParser(description="Step 2 — Stronger Backbone Training")
    p.add_argument("--data-dir",    type=str,   default=default_data)
    p.add_argument("--save-dir",    type=str,   default=script_dir)
    p.add_argument("--epochs",      type=int,   default=25)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--img-size",    type=int,   default=224)
    p.add_argument("--train-split", type=float, default=0.70)   # 70% train, 15% val, 15% test
    p.add_argument("--val-split",   type=float, default=0.15)   # applied to (train+val) remainder
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--patience",    type=int,   default=7)
    p.add_argument("--num-workers", type=int,   default=0)
    p.add_argument("--skip-effnet", action="store_true", help="Skip EfficientNet training")
    # ConViTX reference metrics for comparison table
    p.add_argument("--convitx-val-acc",  type=float, default=None,
                   help="Known ConViTX best val accuracy (0-1)")
    p.add_argument("--convitx-val-f1",   type=float, default=None,
                   help="Known ConViTX best val macro-F1 (0-1)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Transforms
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# --------------------------------------------------------------------------- #
#  Sampler / loader builder
# --------------------------------------------------------------------------- #
def build_loaders(full_ds, train_idx, val_idx, train_tf, val_tf, batch_size, num_workers):
    train_sub = Subset(copy.copy(full_ds), train_idx)
    train_sub.dataset.transform = train_tf
    val_sub   = Subset(copy.copy(full_ds), val_idx)
    val_sub.dataset.transform = val_tf

    # Weighted sampler for class balance
    targets = [full_ds.samples[i][1] for i in train_idx]
    class_counts = np.bincount(targets, minlength=len(full_ds.classes)).astype(float)
    class_counts = np.clip(class_counts, 1, None)
    class_weights = class_counts.sum() / (len(full_ds.classes) * class_counts)
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_sub, batch_size=batch_size, sampler=sampler,   num_workers=num_workers)
    val_loader   = DataLoader(val_sub,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, class_weights


# --------------------------------------------------------------------------- #
#  Model builders
# --------------------------------------------------------------------------- #
def build_resnet50(num_classes: int, device):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Unfreeze layer3, layer4, fc
    for name, param in model.named_parameters():
        if any(k in name for k in ("layer3", "layer4", "fc")):
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [ResNet50]       {trainable:>12,} / {total:>12,} params (trainable/total)")
    return model.to(device)


def build_efficientnet(num_classes: int, device):
    if not _HAS_TIMM:
        return None
    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=num_classes)
    # Freeze early blocks, fine-tune from blocks 4+
    blocks = list(model.named_children())
    freeze_names = {"conv_stem", "bn1", "blocks"}
    for name, child in model.named_children():
        if name in freeze_names:
            if name == "blocks":
                block_list = list(child.children())
                for bidx, blk in enumerate(block_list):
                    for p in blk.parameters():
                        p.requires_grad = bidx >= 4    # unfreeze blocks 4,5,6
            else:
                for p in child.parameters():
                    p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [EfficientNet-B3]{trainable:>12,} / {total:>12,} params (trainable/total)")
    return model.to(device)


# --------------------------------------------------------------------------- #
#  One-epoch train / evaluate
# --------------------------------------------------------------------------- #
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        preds = out.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    val_loss = total_loss / max(1, total)
    val_acc  = correct   / max(1, total)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return val_loss, val_acc, macro_f1, all_preds, all_labels


# --------------------------------------------------------------------------- #
#  Confusion matrix plotter
# --------------------------------------------------------------------------- #
def plot_confusion(cm, class_names, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
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
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_curves(history, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"{title} — Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"],   label="Val Acc")
    axes[1].plot(history["val_f1"],    label="Val Macro-F1")
    axes[1].set(xlabel="Epoch", ylabel="Score", title=f"{title} — Accuracy/F1")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Core training loop for one model
# --------------------------------------------------------------------------- #
def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    class_weights_np,
    num_classes,
    class_names,
    args,
    device,
    save_dir,
):
    cw_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    criterion  = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=0.1)
    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    warmup_ep  = min(3, args.epochs)
    cosine_ep  = max(1, args.epochs - warmup_ep)
    warmup_sch = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_ep
    )
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_ep)
    scheduler  = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[warmup_ep]
    )

    ckpt_path   = os.path.join(save_dir, f"best_{model_name}_step2.pth")
    curves_path = os.path.join(save_dir, f"{model_name}_curves.png")
    cm_path     = os.path.join(save_dir, f"confusion_{model_name}.png")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_metric   = -1.0
    best_val_acc  = 0.0
    best_val_f1   = 0.0
    best_epoch    = 0
    best_wts      = None
    patience_ctr  = 0
    start = time.time()

    print(f"\n{'=' * 55}")
    print(f"  Training: {model_name}  |  {args.epochs} epochs")
    print(f"{'=' * 55}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Loss {tr_loss:.4f}/{va_loss:.4f} | "
            f"Acc {tr_acc:.4f}/{va_acc:.4f} | "
            f"MacroF1 {va_f1:.4f} | LR {lr_now:.6f}"
        )

        metric = va_f1  # select on macro-F1
        if metric > best_metric:
            best_metric  = metric
            best_val_acc = va_acc
            best_val_f1  = va_f1
            best_epoch   = epoch
            best_wts     = copy.deepcopy(model.state_dict())
            torch.save(best_wts, ckpt_path)
            print(f"  ✅ Best saved  (acc={best_val_acc:.4f}  f1={best_val_f1:.4f}  epoch={epoch})")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n⏹  Early stopping at epoch {epoch}  (patience={args.patience})")
                break

    elapsed = time.time() - start
    print(f"\n  Done in {elapsed / 60:.1f} min")
    print(f"  Best Epoch      : {best_epoch}")
    print(f"  Best Val Acc    : {best_val_acc:.4f}")
    print(f"  Best Val Macro-F1: {best_val_f1:.4f}")

    # Curves
    plot_curves(history, model_name, curves_path)
    print(f"  Curves → {curves_path}")

    # Final per-class report
    if best_wts is None:
        print("  No checkpoint found; skipping report.")
        return {"model": model_name, "best_val_acc": 0, "best_val_f1": 0, "best_epoch": 0}

    model.load_state_dict(best_wts)
    _, _, final_f1, all_preds, all_labels = evaluate(model, val_loader, criterion, device)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    print(f"\n  Classification Report — {model_name}")
    print(classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    ))
    print(f"  Confusion Matrix:\n{cm}")

    plot_confusion(cm, class_names, f"Confusion Matrix — {model_name}", cm_path)
    print(f"  Confusion matrix → {cm_path}")

    return {
        "model":        model_name,
        "best_val_acc": round(best_val_acc * 100, 2),
        "best_val_f1":  round(best_val_f1, 4),
        "best_epoch":   best_epoch,
        "ckpt":         ckpt_path,
        "per_class":    {k: v for k, v in report.items() if k in class_names},
        "cm":           cm.tolist(),
    }


# --------------------------------------------------------------------------- #
#  Markdown summary writer
# --------------------------------------------------------------------------- #
def write_step2_results_md(results, class_names, project_root):
    lines = [
        "# Step 2 Results — Stronger Backbone Training\n",
        f"*Generated automatically by `train_step2.py`*\n",
        "",
        "## Executive Summary\n",
    ]

    # Find best model
    best = max(results, key=lambda r: r.get("best_val_f1", 0))
    lines.append(
        f"**Best model:** `{best['model']}` — "
        f"Val Acc: **{best['best_val_acc']:.1f}%** | "
        f"Val Macro-F1: **{best['best_val_f1']:.4f}**\n"
    )

    # Comparison table
    lines += [
        "",
        "## Model Comparison Table\n",
        "| Model | Best Val Acc (%) | Best Val Macro-F1 | Best Epoch |",
        "|-------|-----------------|-------------------|------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']:20s} | {r['best_val_acc']:>16.1f} | "
            f"{r['best_val_f1']:>17.4f} | {r['best_epoch']:>10} |"
        )

    # Per-class table for best model
    lines += [
        "",
        f"## Per-Class Metrics — {best['model']}\n",
        "| Class | Precision | Recall | F1 |",
        "|-------|----------:|-------:|---:|",
    ]
    for cls in class_names:
        m = best.get("per_class", {}).get(cls, {})
        lines.append(
            f"| {cls:20s} | {m.get('precision', 0):.4f} | "
            f"{m.get('recall', 0):.4f} | {m.get('f1-score', 0):.4f} |"
        )

    lines += [
        "",
        "## Artifacts\n",
        f"- Best checkpoint: `models/best_{best['model']}_step2.pth`",
        f"- Training curves: `models/{best['model']}_curves.png`",
        f"- Confusion matrix: `models/confusion_{best['model']}.png`",
        "",
        "## Next Step\n",
        "Run **Step 3** strict held-out evaluation:",
        "```bash",
        "python models/evaluate_step3.py",
        "```",
    ]

    md_path = os.path.join(project_root, "step2_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n📄 Step 2 markdown summary → {md_path}")
    return md_path


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif _HAS_DML:
        device = torch_directml.device()
        print("Device: DirectML")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    data_dir = os.path.normpath(args.data_dir)
    save_dir = os.path.normpath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    # ------------------------------------------------------------------ #
    # Build reproducible 70 / 15 / 15 split  (train / val / test)
    # Test indices are saved to disk for Step 3 reuse
    # ------------------------------------------------------------------ #
    train_tf, val_tf = build_transforms(args.img_size)
    full_ds = datasets.ImageFolder(data_dir)
    class_names  = full_ds.classes
    num_classes  = len(class_names)
    all_labels   = [s[1] for s in full_ds.samples]
    all_indices  = list(range(len(full_ds)))

    print(f"\nDataset: {data_dir}")
    print(f"Classes: {class_names}")
    print(f"Samples: {len(all_indices)}")
    from collections import Counter
    cnt = Counter(all_labels)
    for i, cls in enumerate(class_names):
        print(f"  {cls}: {cnt[i]}")

    # Step 1: split off test  (15%)
    trainval_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.15,
        random_state=args.seed,
        stratify=all_labels,
    )
    # Step 2: split trainval into train/val  (70% / 15% of total)
    tv_labels = [all_labels[i] for i in trainval_idx]
    # val is 15%/85% ≈ 17.6% of trainval
    val_frac = 0.15 / (1.0 - 0.15)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_frac,
        random_state=args.seed,
        stratify=tv_labels,
    )

    print(f"\nSplit (seed={args.seed}): "
          f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # Persist split to disk so Step 3 uses identical test indices
    split_path = os.path.join(save_dir, "dataset_split.json")
    with open(split_path, "w") as f:
        json.dump({
            "seed":       args.seed,
            "data_dir":   data_dir,
            "train_idx":  train_idx,
            "val_idx":    val_idx,
            "test_idx":   test_idx,
            "class_names": class_names,
        }, f)
    print(f"Split saved → {split_path}")

    # ------------------------------------------------------------------ #
    # Build loaders
    # ------------------------------------------------------------------ #
    train_labels  = [all_labels[i] for i in train_idx]
    class_counts  = np.bincount(train_labels, minlength=num_classes).astype(float)
    class_counts  = np.clip(class_counts, 1, None)
    class_weights = class_counts.sum() / (num_classes * class_counts)

    train_loader, val_loader, _ = build_loaders(
        full_ds, train_idx, val_idx,
        train_tf, val_tf,
        args.batch_size, args.num_workers,
    )

    # ------------------------------------------------------------------ #
    # Train models
    # ------------------------------------------------------------------ #
    all_results = []

    # Optional ConViTX row from arguments
    if args.convitx_val_acc is not None:
        all_results.append({
            "model":        "ConViTX",
            "best_val_acc": round(args.convitx_val_acc * 100, 2),
            "best_val_f1":  round(args.convitx_val_f1 or 0.0, 4),
            "best_epoch":   "—",
            "per_class":    {},
        })

    # ---- ResNet50 ----
    print("\n\n>>> Building ResNet50 ...")
    resnet = build_resnet50(num_classes, device)
    r50_res = train_model(
        resnet, "resnet50",
        train_loader, val_loader,
        class_weights, num_classes, class_names,
        args, device, save_dir,
    )
    all_results.append(r50_res)

    # ---- EfficientNet-B3 ----
    if not args.skip_effnet and _HAS_TIMM:
        print("\n\n>>> Building EfficientNet-B3 ...")
        effnet = build_efficientnet(num_classes, device)
        if effnet is not None:
            eff_res = train_model(
                effnet, "efficientnet_b3",
                train_loader, val_loader,
                class_weights, num_classes, class_names,
                args, device, save_dir,
            )
            all_results.append(eff_res)
    elif not _HAS_TIMM:
        print("\n[SKIP] EfficientNet skipped — timm not installed.")

    # ------------------------------------------------------------------ #
    # Final comparison table
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 65}")
    print("  STEP 2 — MODEL COMPARISON TABLE")
    print(f"{'=' * 65}")
    header = f"{'Model':<22} {'Val Acc (%)':>12} {'Macro-F1':>10} {'Best Epoch':>11}"
    print(header)
    print("-" * 65)
    for r in all_results:
        print(
            f"{r['model']:<22} "
            f"{str(r['best_val_acc']):>12} "
            f"{str(r['best_val_f1']):>10} "
            f"{str(r['best_epoch']):>11}"
        )

    # Determine best model overall
    best_step2 = max(
        (r for r in all_results if r["model"] != "ConViTX"),
        key=lambda r: r.get("best_val_f1", 0),
    )
    print(f"\n✔  Best Step-2 model: {best_step2['model']}  "
          f"(Val Acc={best_step2['best_val_acc']:.1f}%  F1={best_step2['best_val_f1']:.4f})")
    print("   → Checkpoint for Step 3:", best_step2["ckpt"])

    # Save best model name for Step 3
    best_info_path = os.path.join(save_dir, "step2_best_model.json")
    with open(best_info_path, "w") as f:
        json.dump({
            "model_name": best_step2["model"],
            "ckpt_path":  best_step2["ckpt"],
            "val_acc":    best_step2["best_val_acc"],
            "val_f1":     best_step2["best_val_f1"],
            "split_path": split_path,
        }, f, indent=2)
    print(f"Best model info → {best_info_path}")

    # ------------------------------------------------------------------ #
    # Write markdown summary
    # ------------------------------------------------------------------ #
    project_root = os.path.normpath(os.path.join(save_dir, ".."))
    write_step2_results_md(all_results, class_names, project_root)

    print("\n✅ Step 2 complete. Run evaluate_step3.py next.")


if __name__ == "__main__":
    main()
