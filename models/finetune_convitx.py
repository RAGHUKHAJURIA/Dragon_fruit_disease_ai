"""
ConViTX Fine-Tuning — Target 95%+ Accuracy
============================================
Aggressive fine-tuning of the ConViTXSmall model for maximum accuracy on the
6-class dragon-fruit disease dataset.

Key improvements over the base train_convitx.py:
  1. MixUp + CutMix augmentation (randomly alternated each batch)
  2. Stronger spatial augmentation pipeline (RandAugment-style)
  3. Optional Knowledge Distillation from ResNet50 teacher checkpoint
  4. EMA model tracking with selectable EMA eval
  5. Cosine LR with long warmup (5 ep) + final hard-restart at plateau
  6. Label smoothing + class-weighted focal loss
  7. Test-Time Augmentation (TTA) for final val accuracy boost
  8. Parameter budget NOT enforced (we expand ConViTX for fine-tuning)
  9. Best model selected on macro-F1 (robust to class imbalance)
 10. Full reproducible seed

Usage (recommended):
  python finetune_convitx.py --epochs 60 --batch-size 16 --lr 2e-4

With knowledge distillation from ResNet50 teacher:
  python finetune_convitx.py --epochs 60 --teacher-ckpt models/best_resnet50_step2.pth

Resume from existing ConViTX checkpoint:
  python finetune_convitx.py --resume-from models/best_convitx.pth --epochs 40

All outputs go to models/ by default.
"""

import argparse
import copy
import json
import math
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
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

# Import from local models directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from convitx import ConViTXConfig, ConViTXSmall, count_parameters

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


# ============================================================================ #
#  Reproducibility
# ============================================================================ #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================ #
#  Argument parsing
# ============================================================================ #
def parse_args():
    default_data = os.path.normpath(
        os.path.join(
            _SCRIPT_DIR, "..",
            "dataset",
            "Dragon Fruit (Pitahaya)",
            "Dragon Fruit (Pitahaya)",
            "Converted Images",
        )
    )
    p = argparse.ArgumentParser(description="ConViTX Fine-Tuning — Target 95%+")
    # Paths
    p.add_argument("--data-dir",     type=str, default=default_data)
    p.add_argument("--save-dir",     type=str, default=_SCRIPT_DIR)
    p.add_argument("--resume-from",  type=str, default="",
                   help="Existing ConViTX .pth checkpoint to resume from")
    p.add_argument("--teacher-ckpt", type=str, default="",
                   help="ResNet50 checkpoint for knowledge distillation (optional)")
    # Training
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--img-size",     type=int,   default=224)
    p.add_argument("--train-split",  type=float, default=0.80,
                   help="Fraction used for training (rest = val)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num-workers",  type=int,   default=0)
    # Regularization
    p.add_argument("--patience",     type=int,   default=12,
                   help="Early stopping patience (generous for long warmup)")
    p.add_argument("--ema-decay",    type=float, default=0.9995)
    p.add_argument("--mixup-alpha",  type=float, default=0.4,
                   help="MixUp alpha; 0 to disable MixUp")
    p.add_argument("--cutmix-alpha", type=float, default=1.0,
                   help="CutMix alpha; 0 to disable CutMix")
    p.add_argument("--label-smooth", type=float, default=0.05)
    p.add_argument("--drop-path",    type=float, default=0.2)
    p.add_argument("--dropout",      type=float, default=0.15)
    # KD
    p.add_argument("--kd-alpha",     type=float, default=0.5,
                   help="Weight for KD loss vs. CE loss (0 = no KD)")
    p.add_argument("--kd-temp",      type=float, default=4.0,
                   help="Temperature for knowledge distillation softening")
    # TTA
    p.add_argument("--tta-n",        type=int,   default=5,
                   help="Number of TTA crops at final evaluation (0 = disabled)")
    # ConViTX scale
    p.add_argument("--scale",        type=str,   default="large",
                   choices=["small", "base", "large"],
                   help="ConViTX scale — large lifts param budget for best accuracy")
    return p.parse_args()


# ============================================================================ #
#  ConViTX with lifted capacity (no budget constraint for fine-tuning)
# ============================================================================ #
def build_convitx_large(num_classes: int, dropout: float, drop_path_rate: float) -> ConViTXSmall:
    """
    'Large' configuration — removes the 700K budget constraint to maximise
    accuracy during fine-tuning. Still lightweight vs. ResNet50 (~4-5M params).
    """
    cfg = ConViTXConfig(
        num_classes=num_classes,
        cnn_dim=128,        # was 96 in base
        vit_dim=128,        # was 96 in base
        fuse_dim=256,       # was 192 in base
        vit_depth=6,        # was 4 in base
        heads=8,            # was 6 in base
        mlp_ratio=3.0,      # was 2.0
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        param_budget=999_999_999,   # uncapped
    )
    return ConViTXSmall(cfg)


def build_convitx_base_uncapped(num_classes: int, dropout: float, drop_path_rate: float) -> ConViTXSmall:
    cfg = ConViTXConfig(
        num_classes=num_classes,
        cnn_dim=96, vit_dim=96, fuse_dim=192, vit_depth=4, heads=6,
        mlp_ratio=2.0, dropout=dropout, drop_path_rate=drop_path_rate,
        param_budget=999_999_999,
    )
    return ConViTXSmall(cfg)


def build_convitx_small_uncapped(num_classes: int, dropout: float, drop_path_rate: float) -> ConViTXSmall:
    cfg = ConViTXConfig(
        num_classes=num_classes,
        cnn_dim=64, vit_dim=64, fuse_dim=128, vit_depth=3, heads=4,
        mlp_ratio=2.0, dropout=dropout, drop_path_rate=drop_path_rate,
        param_budget=999_999_999,
    )
    return ConViTXSmall(cfg)


SCALE_MAP = {
    "small": build_convitx_small_uncapped,
    "base":  build_convitx_base_uncapped,
    "large": build_convitx_large,
}


# ============================================================================ #
#  Teacher model (ResNet50) for KD
# ============================================================================ #
def load_teacher(ckpt_path: str, num_classes: int, device):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[KD] Teacher checkpoint not found: '{ckpt_path}'. Skipping KD.")
        return None
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    try:
        model.load_state_dict(ckpt, strict=True)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f"[KD] Teacher loaded: {ckpt_path}")
        return model
    except Exception as e:
        print(f"[KD] Failed to load teacher checkpoint ({e}). KD disabled.")
        return None


# ============================================================================ #
#  Transforms — strong augmentation
# ============================================================================ #
def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # NOTE: RandomErasing needs a tensor, so it goes after ToTensor

    # Pre-erasing transform (PIL-level) and post-erasing (tensor-level)
    train_tf_pil = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf_pil, val_tf


# ============================================================================ #
#  Data loaders with class-balanced sampling
# ============================================================================ #
def build_loaders(full_ds, train_idx, val_idx, train_tf, val_tf, batch_size, num_workers):
    train_sub = Subset(copy.copy(full_ds), train_idx)
    train_sub.dataset.transform = train_tf
    val_sub   = Subset(copy.copy(full_ds), val_idx)
    val_sub.dataset.transform = val_tf

    targets = [full_ds.samples[i][1] for i in train_idx]
    nc = len(full_ds.classes)
    counts = np.bincount(targets, minlength=nc).astype(float)
    counts = np.clip(counts, 1, None)
    cw = counts.sum() / (nc * counts)
    sw = [float(cw[t]) for t in targets]
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    train_loader = DataLoader(train_sub, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_sub,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, cw


# ============================================================================ #
#  MixUp / CutMix
# ============================================================================ #
def mixup_data(x, y, alpha=0.4, device="cpu"):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def cutmix_data(x, y, alpha=1.0, device="cpu"):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=device)
    x2 = x[idx]
    x_cut = x.clone()
    x1, y1, x2c, y2c = rand_bbox(x.size(), lam)
    x_cut[:, :, y1:y2c, x1:x2c] = x2[:, :, y1:y2c, x1:x2c]
    lam = 1 - ((x2c - x1) * (y2c - y1) / (x.size(-1) * x.size(-2)))
    return x_cut, y, y[idx], lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================ #
#  EMA
# ============================================================================ #
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
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
#  Focal loss
# ============================================================================ #
class FocalCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05,
                 weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.ls, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma * ce).mean()
        return loss


# ============================================================================ #
#  Knowledge Distillation loss
# ============================================================================ #
def kd_loss(student_logits, teacher_logits, temperature: float):
    T = temperature
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


# ============================================================================ #
#  Train one epoch
# ============================================================================ #
def train_one_epoch(
    model, loader, criterion, optimizer, device, ema,
    teacher, kd_alpha, kd_temp, mixup_alpha, cutmix_alpha,
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # --- MixUp / CutMix (randomly choose one) ---
        use_mix = (mixup_alpha > 0 or cutmix_alpha > 0)
        lam = 1.0
        if use_mix and torch.rand(1).item() > 0.3:         # 70% chance to apply
            if cutmix_alpha > 0 and torch.rand(1).item() > 0.5:
                imgs, ya, yb, lam = cutmix_data(imgs, labels, cutmix_alpha, device)
            elif mixup_alpha > 0:
                imgs, ya, yb, lam = mixup_data(imgs, labels, mixup_alpha, device)
            else:
                ya, yb = labels, labels
        else:
            ya, yb = labels, labels

        optimizer.zero_grad()
        logits = model(imgs)

        # --- Loss ---
        loss = mixed_criterion(criterion, logits, ya, yb, lam)

        # --- Knowledge Distillation ---
        if teacher is not None and kd_alpha > 0:
            with torch.no_grad():
                t_logits = teacher(imgs)
            kd = kd_loss(logits, t_logits, kd_temp)
            loss = (1 - kd_alpha) * loss + kd_alpha * kd

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * imgs.size(0)
        # Accuracy only meaningful without mixing — use hard labels
        with torch.no_grad():
            correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / max(1, total), correct / max(1, total)


# ============================================================================ #
#  Evaluate (standard)
# ============================================================================ #
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        preds = logits.argmax(1)
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
#  Test-Time Augmentation
# ============================================================================ #
@torch.no_grad()
def evaluate_tta(model, val_dataset_indices, full_ds, img_size, batch_size,
                device, num_workers, n_tta=5):
    """
    Average softmax over N augmented versions per image.
    """
    model.eval()

    tta_transforms = [
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.CenterCrop(img_size),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((img_size + 24, img_size + 24)),
            transforms.CenterCrop(img_size),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    ]
    tta_transforms = tta_transforms[:n_tta]

    # Collect all predictions averaged over TTA transforms
    all_preds_list = []   # list of (n_tta, N) arrays, then averaged
    all_labels = []

    # Use only first TTA pass to collect labels
    first_tf_ds = copy.copy(full_ds)
    first_tf_ds.transform = tta_transforms[0]
    first_sub = Subset(first_tf_ds, val_dataset_indices)
    first_loader = DataLoader(first_sub, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    logit_accum = None

    for aug_idx, tf in enumerate(tta_transforms):
        cur_ds = copy.copy(full_ds)
        cur_ds.transform = tf
        cur_sub = Subset(cur_ds, val_dataset_indices)
        loader  = DataLoader(cur_sub, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
        logits_all = []
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            logits_all.append(F.softmax(model(imgs), dim=1).cpu())
            if aug_idx == 0:
                all_labels.extend(lbls.tolist())

        logits_cat = torch.cat(logits_all, dim=0)   # (N, C)
        if logit_accum is None:
            logit_accum = logits_cat
        else:
            logit_accum = logit_accum + logits_cat

    avg_logits = logit_accum / n_tta
    all_preds  = avg_logits.argmax(1).tolist()

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, macro_f1, all_preds, all_labels


# ============================================================================ #
#  Plotting helpers
# ============================================================================ #
def plot_curves(history, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep, history["train_loss"], label="Train Loss", color="#e74c3c")
    axes[0].plot(ep, history["val_loss"],   label="Val Loss",   color="#3498db")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"{title} — Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["train_acc"], label="Train Acc",   color="#e74c3c")
    axes[1].plot(ep, history["val_acc"],   label="Val Acc",     color="#3498db")
    axes[1].plot(ep, history["val_f1"],    label="Val Macro-F1",color="#2ecc71", ls="--")
    axes[1].axhline(0.95, color="gold", ls=":", lw=1.5, label="95% target")
    axes[1].set(xlabel="Epoch", ylabel="Score", title=f"{title} — Accuracy / F1", ylim=(0, 1))
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, class_names, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)), yticks=range(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted", ylabel="True", title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================================ #
#  Main
# ============================================================================ #
def main():
    args = parse_args()
    set_seed(args.seed)

    # ---------------------------------------------------------------------- #
    # Device
    # ---------------------------------------------------------------------- #
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
    elif _HAS_DML:
        device = torch_directml.device()
        print("Device: DirectML (AMD/Intel GPU)")
    else:
        device = torch.device("cpu")
        print("Device: CPU  (training will be slow — consider --epochs 40)")

    data_dir = os.path.normpath(args.data_dir)
    save_dir = os.path.normpath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    # ---------------------------------------------------------------------- #
    # Dataset + reproducible stratified split
    # ---------------------------------------------------------------------- #
    train_tf, val_tf = build_transforms(args.img_size)
    full_ds     = datasets.ImageFolder(data_dir)
    class_names = full_ds.classes
    num_classes = len(class_names)
    all_targets = [s[1] for s in full_ds.samples]
    all_idx     = list(range(len(full_ds)))

    train_idx, val_idx = train_test_split(
        all_idx,
        train_size=args.train_split,
        random_state=args.seed,
        stratify=all_targets,
    )

    print(f"\nDataset : {data_dir}")
    print(f"Classes : {class_names}")
    print(f"Total   : {len(all_idx)}  |  Train: {len(train_idx)}  Val: {len(val_idx)}")

    train_loader, val_loader, class_weights_np = build_loaders(
        full_ds, train_idx, val_idx,
        train_tf, val_tf, args.batch_size, args.num_workers,
    )
    cw_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

    # ---------------------------------------------------------------------- #
    # Build ConViTX model
    # ---------------------------------------------------------------------- #
    build_fn = SCALE_MAP[args.scale]
    model = build_fn(num_classes, args.dropout, args.drop_path).to(device)
    n_train = count_parameters(model)
    n_total = count_parameters(model, trainable_only=False)
    print(f"\nConViTX-{args.scale.capitalize()}: {n_train:,} trainable / {n_total:,} total params")

    # ---------------------------------------------------------------------- #
    # Optional resume from checkpoint
    # ---------------------------------------------------------------------- #
    if args.resume_from:
        rp = os.path.normpath(args.resume_from)
        if os.path.isfile(rp):
            ckpt = torch.load(rp, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(f"Resumed from: {rp}")
            if missing:
                print(f"  Missing keys   : {missing[:5]}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected[:5]}")
        else:
            print(f"[WARN] Resume checkpoint not found: {rp}  — starting fresh.")

    # ---------------------------------------------------------------------- #
    # Optional teacher for KD
    # ---------------------------------------------------------------------- #
    teacher = None
    if args.teacher_ckpt and args.kd_alpha > 0:
        teacher = load_teacher(args.teacher_ckpt, num_classes, device)

    # ---------------------------------------------------------------------- #
    # Loss, Optimizer, Scheduler, EMA
    # ---------------------------------------------------------------------- #
    criterion = FocalCELoss(gamma=2.0, label_smoothing=args.label_smooth, weight=cw_tensor)

    # Differential LR: lower LR for CNN branch, higher for ViT & head
    cnn_params, vit_params, head_params = [], [], []
    for name, param in model.named_parameters():
        if "head" in name:
            head_params.append(param)
        elif "transformer_blocks" in name or "patch_embed" in name or "pos_embed" in name:
            vit_params.append(param)
        else:
            cnn_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": cnn_params,  "lr": args.lr * 0.5, "weight_decay": args.weight_decay},
        {"params": vit_params,  "lr": args.lr,        "weight_decay": args.weight_decay * 2},
        {"params": head_params, "lr": args.lr * 2.0,  "weight_decay": 1e-5},
    ])

    # Warmup (5 ep) → CosineAnnealing
    warmup_ep  = min(5, args.epochs // 6)
    cosine_ep  = args.epochs - warmup_ep
    warmup_sch = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, total_iters=warmup_ep
    )
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(cosine_ep // 2, 10), T_mult=1, eta_min=1e-6
    )
    scheduler  = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[warmup_ep]
    )

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # ---------------------------------------------------------------------- #
    # Training loop
    # ---------------------------------------------------------------------- #
    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_f1"]}
    best_metric    = -1.0
    best_val_acc   = 0.0
    best_val_f1    = 0.0
    best_epoch     = 0
    best_wts       = None
    patience_ctr   = 0
    reached_95     = False

    ckpt_path   = os.path.join(save_dir, "best_convitx_finetuned.pth")
    curves_path = os.path.join(save_dir, "convitx_finetune_curves.png")
    cm_val_path = os.path.join(save_dir, "convitx_finetune_cm_val.png")

    print(f"\n{'=' * 60}")
    print(f"  ConViTX Fine-Tuning — Target 95%+  |  {args.epochs} epochs")
    print(f"  Scale: {args.scale}  |  MixUp α={args.mixup_alpha}  CutMix α={args.cutmix_alpha}")
    print(f"  KD: {'enabled  α=' + str(args.kd_alpha) if teacher else 'disabled'}")
    print(f"  EMA decay: {args.ema_decay}  |  patience: {args.patience}")
    print(f"  LR: CNN={args.lr*0.5:.1e}  ViT={args.lr:.1e}  Head={args.lr*2:.1e}")
    print(f"{'=' * 60}\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, ema,
            teacher, args.kd_alpha, args.kd_temp, args.mixup_alpha, args.cutmix_alpha,
        )
        eval_model = ema.ema if ema is not None else model
        va_loss, va_acc, va_f1, _, _ = evaluate(eval_model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)

        lr_now = optimizer.param_groups[1]["lr"]   # ViT group
        print(
            f"Epoch {epoch:03d}/{args.epochs}  |  "
            f"Loss {tr_loss:.4f}/{va_loss:.4f}  |  "
            f"Acc {tr_acc:.4f}/{va_acc:.4f}  |  "
            f"F1 {va_f1:.4f}  |  LR {lr_now:.2e}"
        )

        if va_acc >= 0.95 and not reached_95:
            reached_95 = True
            print(f"\n  🎉 95% ACCURACY REACHED at epoch {epoch}!\n")

        metric = va_f1   # selection criterion
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
    print(f"\n{'=' * 60}")
    print(f"  Training complete  |  {elapsed / 60:.1f} min")
    print(f"  Best Epoch        : {best_epoch}")
    print(f"  Best Val Acc      : {best_val_acc:.4f}  ({best_val_acc * 100:.2f}%)")
    print(f"  Best Val Macro-F1 : {best_val_f1:.4f}")
    print(f"{'=' * 60}")

    # ---------------------------------------------------------------------- #
    # Save training curves
    # ---------------------------------------------------------------------- #
    plot_curves(history, f"ConViTX-{args.scale.capitalize()} Fine-Tune", curves_path)
    print(f"\nCurves → {curves_path}")

    if best_wts is None:
        print("No improvements found during training.")
        return

    # ---------------------------------------------------------------------- #
    # Final evaluation (standard)
    # ---------------------------------------------------------------------- #
    eval_model = ema.ema if ema is not None else model
    if ema is not None:
        # Load best EMA weights explicitly
        eval_model.load_state_dict(best_wts)
    else:
        model.load_state_dict(best_wts)
        eval_model = model

    eval_model.eval()
    _, _, final_f1, std_preds, std_labels = evaluate(eval_model, val_loader, criterion, device)

    print("\n" + "=" * 60)
    print("  STANDARD EVALUATION (no TTA)")
    print("=" * 60)
    report_str = classification_report(
        std_labels, std_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )
    print(report_str)
    cm_std = confusion_matrix(std_labels, std_preds, labels=list(range(num_classes)))
    print(f"Confusion Matrix:\n{cm_std}")
    plot_confusion(cm_std, class_names, f"ConViTX Fine-Tune — Val Set (Standard)", cm_val_path)

    # ---------------------------------------------------------------------- #
    # TTA evaluation
    # ---------------------------------------------------------------------- #
    tta_acc, tta_f1 = best_val_acc, best_val_f1
    if args.tta_n > 0:
        print(f"\n{'=' * 60}")
        print(f"  TEST-TIME AUGMENTATION  (n={args.tta_n})")
        print(f"{'=' * 60}")
        tta_acc, tta_f1, tta_preds, tta_labels = evaluate_tta(
            eval_model, val_idx, full_ds,
            args.img_size, args.batch_size, device, args.num_workers, args.tta_n,
        )
        print(f"  TTA Val Accuracy  : {tta_acc * 100:.2f}%")
        print(f"  TTA Val Macro-F1  : {tta_f1:.4f}")
        tta_report = classification_report(
            tta_labels, tta_preds,
            labels=list(range(num_classes)),
            target_names=class_names,
            zero_division=0,
        )
        print(f"\n{tta_report}")
        cm_tta = confusion_matrix(tta_labels, tta_preds, labels=list(range(num_classes)))
        tta_cm_path = os.path.join(save_dir, "convitx_finetune_cm_tta.png")
        plot_confusion(cm_tta, class_names, f"ConViTX Fine-Tune — Val Set (TTA)", tta_cm_path)
        print(f"TTA confusion matrix → {tta_cm_path}")

    # ---------------------------------------------------------------------- #
    # Save summary JSON
    # ---------------------------------------------------------------------- #
    report_dict = classification_report(
        std_labels, std_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    summary = {
        "model":         f"ConViTX-{args.scale}",
        "params_trainable": n_train,
        "params_total":     n_total,
        "seed":          args.seed,
        "best_epoch":    best_epoch,
        "best_val_acc":  round(best_val_acc * 100, 2),
        "best_val_f1":   round(best_val_f1, 4),
        "tta_val_acc":   round(tta_acc * 100, 2),
        "tta_val_f1":    round(tta_f1, 4),
        "per_class":     {k: v for k, v in report_dict.items() if k in class_names},
        "ckpt_path":     ckpt_path,
    }
    summary_path = os.path.join(save_dir, "convitx_finetune_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON → {summary_path}")

    # ---------------------------------------------------------------------- #
    # Write markdown
    # ---------------------------------------------------------------------- #
    project_root = os.path.normpath(os.path.join(save_dir, ".."))
    _write_markdown(summary, class_names, args, project_root, curves_path, cm_val_path)

    # ---------------------------------------------------------------------- #
    # Final print
    # ---------------------------------------------------------------------- #
    print(f"\n{'=' * 60}")
    print(f"  ✅ FINE-TUNING COMPLETE")
    print(f"  Standard Val Acc  : {best_val_acc * 100:.2f}%")
    if args.tta_n > 0:
        print(f"  TTA Val Acc       : {tta_acc * 100:.2f}%")
    print(f"  Checkpoint        : {ckpt_path}")
    if best_val_acc >= 0.95 or tta_acc >= 0.95:
        print(f"\n  🎉  95% TARGET ACHIEVED!")
    else:
        gap = max(0.95 - best_val_acc, 0.95 - tta_acc) * 100
        print(f"\n  Gap to 95%: {gap:.1f}pp")
        print("  Try: --epochs 80 --scale large --teacher-ckpt models/best_resnet50_step2.pth")
    print(f"{'=' * 60}")


def _write_markdown(summary, class_names, args, project_root, curves_path, cm_path):
    lines = [
        "# ConViTX Fine-Tuning Results\n",
        f"*Scale: ConViTX-{summary['model'].split('-')[1]}  |  Params: {summary['params_trainable']:,} trainable*\n",
        "",
        "## Best Validation Metrics\n",
        "| Metric | Standard | TTA |",
        "|--------|---------|-----|",
        f"| Accuracy | **{summary['best_val_acc']:.2f}%** | **{summary['tta_val_acc']:.2f}%** |",
        f"| Macro F1 | **{summary['best_val_f1']:.4f}** | **{summary['tta_val_f1']:.4f}** |",
        f"| Best Epoch | {summary['best_epoch']} | — |",
        "",
        "## Per-Class Metrics (Standard)\n",
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
        "## Training Configuration\n",
        f"- Scale: `{args.scale}`",
        f"- Epochs: `{args.epochs}`  |  Early stopping patience: `{args.patience}`",
        f"- LR (ViT group): `{args.lr}`  |  Warmup: 5 epochs → CosineAnnealingWarmRestarts",
        f"- MixUp α: `{args.mixup_alpha}`  |  CutMix α: `{args.cutmix_alpha}`",
        f"- EMA decay: `{args.ema_decay}`",
        f"- KD: `{'α=' + str(args.kd_alpha) + '  T=' + str(args.kd_temp) if args.teacher_ckpt else 'disabled'}`",
        "",
        "## Artifacts\n",
        f"- `models/best_convitx_finetuned.pth`",
        f"- `models/convitx_finetune_curves.png`",
        f"- `models/convitx_finetune_cm_val.png`",
        f"- `models/convitx_finetune_summary.json`",
    ]
    md_path = os.path.join(project_root, "convitx_finetune_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown → {md_path}")


if __name__ == "__main__":
    main()
