"""
Train ResNet50 on Dragon Fruit Disease Dataset (6 classes)
===========================================================
Uses torchvision ResNet50 with ImageNet pretrained weights.

2-Phase training:
  Phase 1: Freeze backbone, train only the new FC head (5 epochs)
  Phase 2: Unfreeze all layers, full fine-tuning with differential LRs

Expected accuracy: 97-98%+ on merged_6class dataset

Usage:
  python models/train_resnet50.py
  python models/train_resnet50.py --data-dir dataset/merged_6class --epochs 40
"""

import argparse
import copy
import json
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_directml
    _HAS_DML = True
except ImportError:
    _HAS_DML = False

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Seed ─────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Args ─────────────────────────────────────────────────────────────────────
def parse_args():
    default_data = os.path.normpath(
        os.path.join(_SCRIPT_DIR, "..", "dataset", "merged_6class")
    )
    p = argparse.ArgumentParser(description="Train ResNet50 on Dragon Fruit Disease")
    p.add_argument("--data-dir",      type=str,   default=default_data)
    p.add_argument("--save-dir",      type=str,   default=_SCRIPT_DIR)
    p.add_argument("--epochs",        type=int,   default=40)
    p.add_argument("--freeze-epochs", type=int,   default=5,
                   help="Epochs to train only the head (backbone frozen)")
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-3,
                   help="LR for head (Phase 1) and base LR for Phase 2")
    p.add_argument("--train-split",   type=float, default=0.80)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--num-workers",   type=int,   default=0)
    p.add_argument("--img-size",      type=int,   default=224)
    p.add_argument("--dropout",       type=float, default=0.4)
    p.add_argument("--resume-from",   type=str,   default="",
                   help="Resume from a .pth checkpoint")
    return p.parse_args()


# ─── Transforms ───────────────────────────────────────────────────────────────
def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ─── Data ─────────────────────────────────────────────────────────────────────
def build_loaders(full_ds, train_idx, val_idx, train_tf, val_tf, batch_size, num_workers):
    train_sub = Subset(copy.copy(full_ds), train_idx)
    train_sub.dataset.transform = train_tf
    val_sub   = Subset(copy.copy(full_ds), val_idx)
    val_sub.dataset.transform   = val_tf

    # Balanced sampler
    targets = [full_ds.samples[i][1] for i in train_idx]
    nc      = len(full_ds.classes)
    counts  = np.bincount(targets, minlength=nc).astype(float)
    counts  = np.clip(counts, 1, None)
    cw      = counts.sum() / (nc * counts)
    sw      = [float(cw[t]) for t in targets]
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    train_loader = DataLoader(train_sub, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_sub,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, cw


# ─── Model ────────────────────────────────────────────────────────────────────
def build_resnet50(num_classes: int, dropout: float = 0.4) -> nn.Module:
    """ResNet50 with custom classification head."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Replace FC head with dropout + linear
    in_features = model.fc.in_features   # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module):
    """Freeze all layers except the final FC head."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad_(False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ResNet50] Backbone FROZEN — {n_trainable:,} trainable params (head only)")


def unfreeze_all(model: nn.Module):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad_(True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ResNet50] ALL UNFROZEN — {n_trainable:,} trainable params (full fine-tune)")


def get_param_groups(model: nn.Module, base_lr: float):
    """Differential LRs: backbone × 0.1, head × 1.0."""
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": backbone_params, "lr": base_lr * 0.1, "name": "backbone"},
        {"params": head_params,     "lr": base_lr,       "name": "head"},
    ]


# ─── EMA ──────────────────────────────────────────────────────────────────────
class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.ema   = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


# ─── Train / Eval ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, ema=None):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ema:
            ema.update(model)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    preds_all, labels_all = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    acc     = correct / max(1, total)
    f1      = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / max(1, total), acc, f1, preds_all, labels_all


# ─── Plotting ─────────────────────────────────────────────────────────────────
def plot_curves(history, out_path):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ep, history["train_loss"], label="Train", color="#e74c3c")
    axes[0].plot(ep, history["val_loss"],   label="Val",   color="#3498db")
    axes[0].set(title="Loss",    xlabel="Epoch", ylabel="Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["train_acc"], label="Train Acc",    color="#e74c3c")
    axes[1].plot(ep, history["val_acc"],   label="Val Acc",      color="#3498db")
    axes[1].plot(ep, history["val_f1"],    label="Val Macro-F1", color="#2ecc71", ls="--")
    axes[1].axhline(0.97, color="gold", ls=":", lw=1.5, label="97% target")
    for i, e in enumerate(history.get("events", [])):
        if e == "unfreeze":
            axes[1].axvline(i + 1, color="orange", ls="--", lw=1.2, label="Unfrozen")
            break
    axes[1].set(title="Accuracy / F1", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cm(cm, names, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(names)), yticks=range(len(names)),
           xticklabels=names, yticklabels=names,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

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

    data_dir = os.path.normpath(args.data_dir)
    save_dir = os.path.normpath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    # ── Dataset ───────────────────────────────────────────────────────────
    from collections import Counter
    train_tf, val_tf = build_transforms(args.img_size)
    full_ds     = datasets.ImageFolder(data_dir)
    class_names = full_ds.classes
    num_classes = len(class_names)
    all_targets = [s[1] for s in full_ds.samples]
    all_idx     = list(range(len(full_ds)))

    print(f"\nDataset  : {data_dir}")
    print(f"Classes  : {class_names}")
    print(f"Total    : {len(all_idx)}")
    cnt = Counter(all_targets)
    for i, c in enumerate(class_names):
        print(f"  {c}: {cnt[i]}")

    train_idx, val_idx = train_test_split(
        all_idx, train_size=args.train_split,
        random_state=args.seed, stratify=all_targets,
    )
    print(f"\nSplit    : train={len(train_idx)}  val={len(val_idx)}")

    train_loader, val_loader, cw_np = build_loaders(
        full_ds, train_idx, val_idx, train_tf, val_tf,
        args.batch_size, args.num_workers,
    )
    cw_tensor = torch.tensor(cw_np, dtype=torch.float32).to(device)

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\nBuilding ResNet50 (ImageNet pretrained) ...")
    model = build_resnet50(num_classes, args.dropout).to(device)

    # Resolve resume path
    resume_path = ""
    if args.resume_from:
        for candidate in [args.resume_from,
                          os.path.join(save_dir, args.resume_from),
                          os.path.join(save_dir, os.path.basename(args.resume_from))]:
            if os.path.isfile(candidate):
                resume_path = candidate
                break

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)
        print(f"Resumed  : {resume_path}")
        effective_freeze = 0   # skip freeze phase when resuming
    else:
        effective_freeze = args.freeze_epochs
        freeze_backbone(model)

    n_total = sum(p.numel() for p in model.parameters())
    print(f"Params   : {n_total:,} total")

    # ── Loss + EMA ────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=0.05)
    ema       = ModelEMA(model, decay=0.9995)
    if resume_path:
        ema.ema.load_state_dict(model.state_dict())

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    cnn_frozen = effective_freeze > 0

    def make_opt(frozen):
        if frozen:
            params = [p for p in model.parameters() if p.requires_grad]
            return torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        else:
            return torch.optim.AdamW(
                get_param_groups(model, args.lr * 0.3
                                 if not resume_path else args.lr),
                weight_decay=1e-4,
            )

    optimizer = make_opt(cnn_frozen)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - effective_freeze), eta_min=1e-7
    )

    # ── Paths ─────────────────────────────────────────────────────────────
    ckpt_path   = os.path.join(save_dir, "best_resnet50.pth")
    curves_path = os.path.join(save_dir, "resnet50_curves.png")
    cm_path     = os.path.join(save_dir, "resnet50_cm.png")

    # ── Training loop ─────────────────────────────────────────────────────
    history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc","val_f1","events"]}
    best_f1  = best_acc = 0.0
    best_ep  = 0
    best_wts = None
    patience_ctr = 0
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  ResNet50 Training  ({args.epochs} epochs)")
    print(f"  LR={args.lr}  Batch={args.batch_size}  Patience={args.patience}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        # Phase 2 unfreeze
        if cnn_frozen and epoch > effective_freeze:
            unfreeze_all(model)
            cnn_frozen = False
            optimizer = make_opt(cnn_frozen)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - epoch), eta_min=1e-7
            )
            print(f">>> Phase 2: full fine-tune from epoch {epoch}\n")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device, ema)
        va_loss, va_acc, va_f1, _, _ = evaluate(ema.ema, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)
        history["events"].append("unfreeze" if (epoch == effective_freeze + 1) else "")

        phase = "P1-freeze" if cnn_frozen else "P2-full"
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{phase}] Ep {epoch:03d}/{args.epochs}  |  "
              f"Loss {tr_loss:.4f}/{va_loss:.4f}  |  "
              f"Acc {tr_acc:.4f}/{va_acc:.4f}  |  "
              f"F1 {va_f1:.4f}  |  LR {lr_now:.2e}")

        if va_acc >= 0.97:
            print(f"\n  🎉  97%+ ACCURACY at epoch {epoch}!  ({va_acc*100:.2f}%)\n")

        if va_f1 > best_f1:
            best_f1  = va_f1
            best_acc = va_acc
            best_ep  = epoch
            best_wts = copy.deepcopy(ema.ema.state_dict())
            torch.save(best_wts, ckpt_path)
            print(f"  ✅ Best  acc={best_acc:.4f}  f1={best_f1:.4f}  ep={epoch}")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n⏹  Early stop at epoch {epoch} (no improvement for {args.patience} ep)")
                break

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Training done  |  {elapsed/60:.1f} min")
    print(f"  Best Epoch     : {best_ep}")
    print(f"  Best Val Acc   : {best_acc*100:.2f}%")
    print(f"  Best Macro F1  : {best_f1:.4f}")
    print(f"{'='*60}")

    plot_curves(history, curves_path)
    print(f"Curves → {curves_path}")

    if best_wts is None:
        return

    # ── Final report ──────────────────────────────────────────────────────
    ema.ema.load_state_dict(best_wts)
    ema.ema.eval()
    _, _, _, preds, labels = evaluate(ema.ema, val_loader, criterion, device)

    print(f"\n{'='*60}  FINAL REPORT  {'='*60}")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    print(f"Confusion Matrix:\n{cm}")
    plot_cm(cm, class_names, f"ResNet50 — {best_acc*100:.2f}%", cm_path)
    print(f"Confusion matrix → {cm_path}")

    # ── Save JSON summary ─────────────────────────────────────────────────
    rd = classification_report(labels, preds, target_names=class_names,
                                zero_division=0, output_dict=True)
    summary = {
        "model": "ResNet50 (ImageNet pretrained)",
        "params_total": n_total,
        "best_epoch": best_ep,
        "best_val_acc": round(best_acc * 100, 2),
        "best_val_f1":  round(best_f1, 4),
        "per_class": {k: v for k, v in rd.items() if k in class_names},
        "ckpt_path": ckpt_path,
    }
    summary_path = os.path.join(save_dir, "resnet50_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary  → {summary_path}")

    print(f"\n{'='*60}")
    print(f"  ✅ ResNet50 COMPLETE")
    print(f"  Val Acc    : {best_acc*100:.2f}%")
    print(f"  Macro F1   : {best_f1:.4f}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
