"""
Train ConViTX-Pretrained on Dragon Fruit Disease (6 classes)
=============================================================
Uses the ConViTXPretrained hybrid model:
  - CNN branch  : MobileNetV3-Small (ImageNet pretrained) — KEY improvement
  - ViT branch  : 4 transformer blocks (random init)
  - Fusion      : concat → MLP head

Training strategy (2-phase):
  Phase 1 (freeze_epochs): CNN frozen, only ViT + head trained
                           Fast convergence of attention mechanism
  Phase 2 (remaining)    : CNN unfrozen, differential LR (CNN×0.1, ViT×1, Head×2)
                           Full joint fine-tuning

No MixUp/CutMix — these hurt on tiny datasets by blurring class boundaries.
Uses mild augmentation + focal loss + EMA instead.

Usage:
  python models/train_convitx_pretrained.py
  python models/train_convitx_pretrained.py --epochs 50 --batch-size 32 --lr 3e-4
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

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from convitx_pretrained import ConViTXPretrained, build_convitx_pretrained, count_parameters


# ============================================================================ #
#  Reproducibility
# ============================================================================ #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================ #
#  Argument parsing
# ============================================================================ #
def parse_args():
    default_data = os.path.normpath(os.path.join(
        _SCRIPT_DIR, "..", "dataset",
        "Dragon Fruit (Pitahaya)", "Dragon Fruit (Pitahaya)", "Converted Images",
    ))

    p = argparse.ArgumentParser(description="Train ConViTX-Pretrained (MobileNetV3 CNN)")
    p.add_argument("--data-dir",      type=str,   default=default_data)
    p.add_argument("--save-dir",      type=str,   default=_SCRIPT_DIR)
    p.add_argument("--epochs",        type=int,   default=60)
    p.add_argument("--freeze-epochs", type=int,   default=5,
                   help="Phase 1: train ViT+head only (CNN frozen), then unfreeze. Set 0 when resuming.")
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--img-size",      type=int,   default=224)
    p.add_argument("--train-split",   type=float, default=0.80)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--num-workers",   type=int,   default=0)
    # Model
    p.add_argument("--vit-dim",       type=int,   default=192)
    p.add_argument("--vit-depth",     type=int,   default=4)
    p.add_argument("--vit-heads",     type=int,   default=8)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--drop-path",     type=float, default=0.1)
    # EMA
    p.add_argument("--ema-decay",     type=float, default=0.9995)
    # Resume
    p.add_argument("--resume-from",   type=str,   default="",
                   help="Path to a .pth checkpoint to resume from (filename or full path)")
    # Targeted class weight boost (helps fix weak classes like Stem_Canker)
    p.add_argument("--boost-class",   type=int,   default=-1,
                   help="Class index to boost (e.g. 5 = Stem_Canker). -1 = disabled.")
    p.add_argument("--boost-factor",  type=float, default=2.0,
                   help="Multiplier applied to the boosted class weight (default: 2.0)")
    # Loss type: 'focal' (default) or 'ce' (plain CrossEntropy — more stable on DirectML)
    p.add_argument("--loss-type",     type=str,   default="focal",
                   choices=["focal", "ce"],
                   help="Loss function: focal (default) or plain CE. Use 'ce' for stable resume.")
    p.add_argument("--label-smoothing", type=float, default=0.05,
                   help="Label smoothing factor (default: 0.05, set 0 for plain CE stability)")
    return p.parse_args()


# ============================================================================ #
#  Transforms — moderate augmentation, NO MixUp/CutMix
# ============================================================================ #
def build_transforms(img_size: int):
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


# ============================================================================ #
#  Data Loading
# ============================================================================ #
def build_loaders(full_ds, train_idx, val_idx, train_tf, val_tf,
                  batch_size, num_workers):
    train_sub = Subset(copy.copy(full_ds), train_idx)
    train_sub.dataset.transform = train_tf
    val_sub   = Subset(copy.copy(full_ds), val_idx)
    val_sub.dataset.transform = val_tf

    # Weighted sampler to balance classes
    targets      = [full_ds.samples[i][1] for i in train_idx]
    nc           = len(full_ds.classes)
    counts       = np.bincount(targets, minlength=nc).astype(float)
    counts       = np.clip(counts, 1, None)
    cw           = counts.sum() / (nc * counts)
    sw           = [float(cw[t]) for t in targets]
    sampler      = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    train_loader = DataLoader(train_sub, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_sub,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, cw


# ============================================================================ #
#  EMA
# ============================================================================ #
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.ema   = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


# ============================================================================ #
#  Focal Loss
# ============================================================================ #
class FocalCELoss(nn.Module):
    """Numerically stable Focal Loss.

    Uses log_softmax → nll_loss pipeline which avoids the exp() overflow
    that causes NaN on DirectML with large class weights.
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05,
                 weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.ls    = label_smoothing
        self.w     = weight

    def forward(self, logits, targets):
        # Use cross_entropy for label_smoothing support, then apply focal scaling
        ce = F.cross_entropy(logits, targets, weight=self.w,
                             label_smoothing=self.ls, reduction="none")
        # Clamp before exp to prevent NaN on DirectML (overflows at very small loss)
        ce_clamped = torch.clamp(ce, min=1e-8, max=20.0)
        # Detach pt so focal weight doesn't affect gradient flow (more stable)
        with torch.no_grad():
            pt = torch.exp(-ce_clamped).clamp(min=1e-7, max=1.0)
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


class PlainCELoss(nn.Module):
    """Standard weighted CrossEntropy — most stable on DirectML for fine-tuning."""
    def __init__(self, weight: torch.Tensor = None, label_smoothing: float = 0.0):
        super().__init__()
        self.w  = weight
        self.ls = label_smoothing

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.w,
                               label_smoothing=self.ls)


# ============================================================================ #
#  Train / Eval
# ============================================================================ #
def train_one_epoch(model, loader, criterion, optimizer, device, ema=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    val_loss = total_loss / max(1, total)
    val_acc  = correct   / max(1, total)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return val_loss, val_acc, macro_f1, all_preds, all_labels


# ============================================================================ #
#  Plotting helpers
# ============================================================================ #
def plot_curves(history, out_path):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ep, history["train_loss"], label="Train Loss", color="#e74c3c")
    axes[0].plot(ep, history["val_loss"],   label="Val Loss",   color="#3498db")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss Curve")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["train_acc"], label="Train Acc",    color="#e74c3c")
    axes[1].plot(ep, history["val_acc"],   label="Val Acc",      color="#3498db")
    axes[1].plot(ep, history["val_f1"],    label="Val Macro-F1", color="#2ecc71", ls="--")
    axes[1].axhline(0.95, color="gold", ls=":", lw=1.5, label="95% target")
    if any(e == "unfreeze" for e in history.get("events", [])):
        for i, e in enumerate(history.get("events", [])):
            if e == "unfreeze":
                axes[1].axvline(i + 1, color="orange", ls="--", lw=1, label="CNN unfrozen")
                break
    axes[1].set(xlabel="Epoch", ylabel="Score",
                title="Accuracy / Macro-F1", ylim=(0, 1))
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, class_names, title, out_path):
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


# ============================================================================ #
#  Main
# ============================================================================ #
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
        print("Device: CPU  (training will be slower)")

    data_dir = os.path.normpath(args.data_dir)
    save_dir = os.path.normpath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    from collections import Counter
    train_tf, val_tf = build_transforms(args.img_size)
    full_ds     = datasets.ImageFolder(data_dir)
    class_names = full_ds.classes
    num_classes = len(class_names)
    all_targets = [s[1] for s in full_ds.samples]
    all_idx     = list(range(len(full_ds)))

    cnt = Counter(all_targets)
    print(f"\nDataset  : {data_dir}")
    print(f"Classes  : {class_names}")
    print(f"Total    : {len(all_idx)}")
    for i, cls in enumerate(class_names):
        print(f"  {cls}: {cnt[i]}")

    train_idx, val_idx = train_test_split(
        all_idx, train_size=args.train_split,
        random_state=args.seed, stratify=all_targets,
    )
    print(f"\nSplit    : train={len(train_idx)}  val={len(val_idx)}")

    train_loader, val_loader, class_weights_np = build_loaders(
        full_ds, train_idx, val_idx, train_tf, val_tf,
        args.batch_size, args.num_workers,
    )
    # Optional class weight boost (fix weak classes like Stem_Canker)
    class_weights_np = class_weights_np.copy()
    if args.boost_class >= 0 and args.boost_class < num_classes:
        class_weights_np[args.boost_class] *= args.boost_factor
        print(f"\n⚡ Boosting class [{args.boost_class}] "
              f"{class_names[args.boost_class]} weight ×{args.boost_factor}")
    cw_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

    # ------------------------------------------------------------------ #
    # Model — when resuming, CNN should NEVER be re-frozen
    # ------------------------------------------------------------------ #
    # Resolve resume path: try as-is, then relative to save_dir
    resume_path = ""
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            resume_path = args.resume_from
        elif os.path.isfile(os.path.join(save_dir, args.resume_from)):
            resume_path = os.path.join(save_dir, args.resume_from)
        elif os.path.isfile(os.path.join(save_dir, os.path.basename(args.resume_from))):
            resume_path = os.path.join(save_dir, os.path.basename(args.resume_from))

    # If resuming, skip freeze phase (CNN already trained)
    effective_freeze = args.freeze_epochs if not resume_path else 0

    print(f"\nBuilding ConViTX-Pretrained ...")
    model = build_convitx_pretrained(
        num_classes=num_classes,
        vit_dim=args.vit_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        drop_path=args.drop_path,
        dropout=args.dropout,
        freeze_cnn=(effective_freeze > 0),  # only freeze for fresh training
    ).to(device)

    n_train = count_parameters(model)
    n_total = count_parameters(model, trainable_only=False)
    print(f"Params   : {n_train:,} trainable / {n_total:,} total")

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        # Also initialise EMA from checkpoint so it starts warm
        print(f"Resumed  : {resume_path}")
        if missing:
            print(f"  Missing : {missing[:3]}")
        print(f"Strategy : Resumed — full fine-tune (no freeze), LR={args.lr}")
    else:
        print(f"Strategy : Phase-1 freeze CNN for {effective_freeze} ep, then full fine-tune")

    # ------------------------------------------------------------------ #
    # Loss, Optimizer, Scheduler, EMA
    # ------------------------------------------------------------------ #
    if args.loss_type == "focal":
        criterion = FocalCELoss(gamma=2.0, label_smoothing=args.label_smoothing,
                                weight=cw_tensor)
        print(f"Loss     : Focal CE (γ=2.0, ls={args.label_smoothing})")
    else:
        criterion = PlainCELoss(weight=cw_tensor, label_smoothing=args.label_smoothing)
        print(f"Loss     : Plain CE (ls={args.label_smoothing}) — DirectML-stable mode")

    ema = ModelEMA(model, decay=args.ema_decay)
    # Warm-start EMA from checkpoint too
    if resume_path:
        ema.ema.load_state_dict(model.state_dict())

    def make_optimizer(model, lr, cnn_frozen):
        if cnn_frozen:
            params = [p for p in model.parameters() if p.requires_grad]
            return torch.optim.AdamW(params, lr=lr, weight_decay=5e-4)
        else:
            return torch.optim.AdamW(
                model.param_groups(lr), weight_decay=5e-4
            )

    cnn_frozen = effective_freeze > 0
    optimizer  = make_optimizer(model, args.lr, cnn_frozen)

    # Scheduler: short warmup → cosine
    if resume_path:
        # Resume: no warmup, straight cosine from args.lr
        warmup_ep = 1
        cosine_ep = max(1, args.epochs - 1)
    else:
        warmup_ep = min(3, effective_freeze if cnn_frozen else args.epochs // 10)
        cosine_ep = max(1, args.epochs - warmup_ep)
    warmup_sch = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.5 if resume_path else 0.1, total_iters=warmup_ep
    )
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_ep)
    scheduler  = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[warmup_ep]
    )

    # ------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------ #
    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_f1", "events"]}
    # When resuming, seed best_metric with prior result so we only save improvements
    best_metric   = -1.0
    best_val_acc  = 0.0
    best_val_f1   = 0.0
    best_epoch    = 0
    best_wts      = None
    patience_ctr  = 0
    phase2_entered = False

    ckpt_path   = os.path.join(save_dir, "best_convitx_pretrained.pth")
    curves_path = os.path.join(save_dir, "convitx_pretrained_curves.png")
    cm_path     = os.path.join(save_dir, "convitx_pretrained_cm.png")

    print(f"\n{'='*60}")
    print(f"  ConViTX-Pretrained Training  ({args.epochs} epochs)")
    print(f"  LR={args.lr}  Batch={args.batch_size}  Patience={args.patience}")
    print(f"{'='*60}\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- Phase 2: Unfreeze CNN ----
        if cnn_frozen and epoch > effective_freeze:
            model.unfreeze_cnn()
            cnn_frozen = False
            phase2_entered = True
            # Rebuild optimizer with differential LRs for phase 2
            optimizer = make_optimizer(model, args.lr * 0.3, cnn_frozen=False)
            warmup_ep2  = 2
            cosine_ep2  = max(1, args.epochs - epoch - warmup_ep2)
            warmup_sch2 = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.3, total_iters=warmup_ep2
            )
            cosine_sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, cosine_ep2))
            scheduler   = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sch2, cosine_sch2], milestones=[warmup_ep2]
            )
            print(f"\n>>> Phase 2 starts at epoch {epoch}: CNN unfrozen, LR×0.1 for CNN branch\n")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, ema)
        eval_model = ema.ema
        va_loss, va_acc, va_f1, _, _ = evaluate(eval_model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)
        history["events"].append("unfreeze" if (epoch == args.freeze_epochs + 1) else "")

        lr_now = optimizer.param_groups[0]["lr"]
        phase  = "P1-frozen" if (epoch <= args.freeze_epochs) else "P2-full"
        print(
            f"[{phase}] Epoch {epoch:03d}/{args.epochs}  |  "
            f"Loss {tr_loss:.4f}/{va_loss:.4f}  |  "
            f"Acc {tr_acc:.4f}/{va_acc:.4f}  |  "
            f"F1 {va_f1:.4f}  |  LR {lr_now:.2e}"
        )

        if va_acc >= 0.95:
            print(f"\n  🎉  95% ACCURACY REACHED at epoch {epoch}!\n")

        metric = va_f1
        if metric > best_metric:
            best_metric  = metric
            best_val_acc = va_acc
            best_val_f1  = va_f1
            best_epoch   = epoch
            best_wts     = copy.deepcopy(eval_model.state_dict())
            torch.save(best_wts, ckpt_path)
            print(f"  ✅ Best  (acc={best_val_acc:.4f}  f1={best_val_f1:.4f}  ep={epoch})")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n⏹  Early stopping at epoch {epoch}  (no improvement for {args.patience} epochs)")
                break

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training complete  |  {elapsed / 60:.1f} min")
    print(f"  Best Epoch        : {best_epoch}")
    print(f"  Best Val Acc      : {best_val_acc:.4f}  ({best_val_acc * 100:.2f}%)")
    print(f"  Best Val Macro-F1 : {best_val_f1:.4f}")
    print(f"{'='*60}")

    # Curves
    plot_curves(history, curves_path)
    print(f"\nCurves → {curves_path}")

    if best_wts is None:
        print("No improvements recorded.")
        return

    # ------------------------------------------------------------------ #
    # Final Evaluation
    # ------------------------------------------------------------------ #
    eval_model.load_state_dict(best_wts)
    eval_model.eval()
    _, _, final_f1, all_preds, all_labels = evaluate(eval_model, val_loader, criterion, device)

    print("\n" + "="*60)
    print("  FINAL CLASSIFICATION REPORT")
    print("="*60)
    report_str = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )
    print(report_str)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    print(f"Confusion Matrix:\n{cm}")
    plot_confusion(cm, class_names, "ConViTX-Pretrained — Val Set", cm_path)
    print(f"\nConfusion matrix → {cm_path}")

    # ------------------------------------------------------------------ #
    # Save summary
    # ------------------------------------------------------------------ #
    report_dict = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    summary = {
        "model":            "ConViTX-Pretrained (MobileNetV3-Small CNN)",
        "params_trainable": n_train,
        "params_total":     n_total,
        "seed":             args.seed,
        "best_epoch":       best_epoch,
        "best_val_acc":     round(best_val_acc * 100, 2),
        "best_val_f1":      round(best_val_f1, 4),
        "per_class":        {k: v for k, v in report_dict.items() if k in class_names},
        "ckpt_path":        ckpt_path,
    }
    summary_path = os.path.join(save_dir, "convitx_pretrained_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON → {summary_path}")

    # Markdown report
    project_root = os.path.normpath(os.path.join(save_dir, ".."))
    _write_md(summary, class_names, args, project_root)

    print(f"\n{'='*60}")
    print(f"  ✅ ConViTX-Pretrained Training COMPLETE")
    print(f"  Val Acc      : {best_val_acc * 100:.2f}%")
    print(f"  Val Macro-F1 : {best_val_f1:.4f}")
    print(f"  Checkpoint   : {ckpt_path}")
    if best_val_acc < 0.95:
        print(f"\n  To push higher, try resuming with lower LR:")
        print(f"  .venv\\Scripts\\python.exe models\\train_convitx_pretrained.py \\")
        print(f"    --epochs 30 --lr 5e-5 --freeze-epochs 0 --patience 10 \\")
        print(f"    --resume-from {os.path.basename(ckpt_path)}")
    print(f"{'='*60}")


def _write_md(summary, class_names, args, project_root):
    lines = [
        "# ConViTX-Pretrained Results\n",
        f"*Hybrid CNN (MobileNetV3-Small, pretrained) + ViT  |  "
        f"Params: {summary['params_trainable']:,} trainable*\n",
        "",
        "## Best Validation Metrics\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Accuracy** | **{summary['best_val_acc']:.2f}%** |",
        f"| **Macro F1** | **{summary['best_val_f1']:.4f}** |",
        f"| Best Epoch   | {summary['best_epoch']} |",
        "",
        "## Per-Class Metrics\n",
        "| Class | Precision | Recall | F1 |",
        "|-------|----------:|-------:|---:|",
    ]
    for cls in class_names:
        m = summary["per_class"].get(cls, {})
        lines.append(
            f"| {cls} | {m.get('precision', 0):.4f} | "
            f"{m.get('recall', 0):.4f} | {m.get('f1-score', 0):.4f} |"
        )
    lines += [
        "",
        "## Training Config\n",
        f"- CNN backbone: MobileNetV3-Small (ImageNet pretrained)",
        f"- Phase 1: CNN frozen for `{args.freeze_epochs}` epochs → only ViT+Head trained",
        f"- Phase 2: Full joint fine-tuning (CNN LR×0.1)",
        f"- LR={args.lr}  |  Batch={args.batch_size}  |  Epochs={args.epochs}",
        f"- Augmentation: RandomCrop+Flip+Rotate+ColorJitter+RandomErasing (NO MixUp/CutMix)",
        f"- Loss: Focal CE (γ=2.0) + label smoothing 0.05",
        f"- EMA decay: {args.ema_decay}",
        "",
        "## Artifacts\n",
        f"- `models/best_convitx_pretrained.pth`",
        f"- `models/convitx_pretrained_curves.png`",
        f"- `models/convitx_pretrained_cm.png`",
        f"- `models/convitx_pretrained_summary.json`",
    ]
    md_path = os.path.join(project_root, "convitx_pretrained_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown → {md_path}")


if __name__ == "__main__":
    main()
