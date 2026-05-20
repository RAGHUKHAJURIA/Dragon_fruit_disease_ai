"""
Training Loop for Dragon Fruit VQA Model.

Features:
    • Pre-extracts and caches vision features (eliminates backbone from loop)
    • Custom PyTorch Dataset for multi-modal (image, question, answer) data
    • AdamW optimizer with cosine annealing
    • Mixed precision training (AMP)
    • Gradient clipping
    • Early stopping
    • Class-weighted CrossEntropyLoss for imbalanced answer distribution

Usage:
    python train_vqa.py \
        --dataset     models/vqa_dataset_train.json \
        --val-dataset models/vqa_dataset_val.json \
        --backbone    models/best_convitx_pretrained.pth \
        --epochs 20
"""

from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ── Project root setup ───────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.vqa_model import build_vqa_model, count_trainable_params, VQAConfig
from models.vqa_tokenizer import build_default_tokenizer, VQATokenizer
from models.vqa_answers import NUM_ANSWER_CLASSES
from xai.gradcam import infer_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Vision Feature Extractor + Cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _load_backbone(model_path: str, num_classes: int = 6) -> nn.Module:
    """Load the frozen ConViTX backbone for feature extraction."""
    from xai.gradcam import load_convitx_model
    model = load_convitx_model(model_path, num_classes=num_classes, device=DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _extract_fused_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Extract the fused (CNN+ViT) feature vector from ConViTXPretrained,
    i.e. the concatenated representation BEFORE the classification head.

    Works with both ConViTXPretrained (cnn_branch + ViT) and the legacy
    ConViTXSmall (pool-based) architectures.

    Returns:
        [B, fused_dim] tensor  (768 for Pretrained, 128 for Small)
    """
    if hasattr(backbone, 'cnn_branch'):
        # ── ConViTXPretrained path ────────────────────────────────────
        cnn_feat = backbone.cnn_pool(backbone.cnn_branch(x))  # (B, 576, 1, 1)
        cnn_feat = cnn_feat.flatten(1)                         # (B, 576)

        patches = backbone.patch_embed(x)                      # (B, C, h, w)
        b, c, h, w = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)            # (B, h*w, C)

        gs = backbone._pos_grid
        if h != gs or w != gs:
            pos = backbone.pos_embed.reshape(1, gs, gs, c).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, h * w, c)
        else:
            pos = backbone.pos_embed

        tokens = backbone.pos_drop(tokens + pos)
        for blk in backbone.blocks:
            tokens = blk(tokens)
        tokens   = backbone.vit_norm(tokens)
        vit_feat = tokens.mean(dim=1)                          # (B, vit_dim)

        fused = torch.cat([cnn_feat, vit_feat], dim=1)         # (B, 576+vit_dim)
        return fused
    else:
        # ── Legacy ConViTXSmall path (has .pool) ──────────────────────
        feats_store = {}
        def _hook(m, inp, out):
            feats_store["f"] = out.detach()
        h = backbone.pool.register_forward_hook(_hook)
        _ = backbone(x)
        h.remove()
        return feats_store["f"].flatten(1)


def extract_and_cache_features(
    backbone: nn.Module,
    image_paths: List[str],
    cache_dir: str,
    batch_size: int = 32,
) -> Dict[str, str]:
    """
    Pre-extract vision features for all images and cache to disk.

    This is the key memory optimization: by extracting features once and
    caching them as .pt files, the training loop never needs to load the
    ConViTX backbone — saving ~70 MB of GPU memory.

    Supports both ConViTXPretrained (768-d fused) and ConViTXSmall (128-d).

    Args:
        backbone:    Frozen ConViTX model.
        image_paths: List of absolute image paths.
        cache_dir:   Directory to save cached feature tensors.
        batch_size:  Batch size for feature extraction.

    Returns:
        {image_path: cached_feature_path} mapping
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_map: Dict[str, str] = {}

    # De-duplicate image paths
    unique_paths = sorted(set(image_paths))
    print(f"  Extracting features for {len(unique_paths)} unique images...")

    backbone.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_paths), batch_size), desc="Extracting"):
            batch_paths = unique_paths[i : i + batch_size]
            tensors = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    t = infer_transforms(img)
                    tensors.append(t)
                except Exception as e:
                    print(f"  ⚠ Could not load {p}: {e}")
                    tensors.append(torch.zeros(3, 224, 224))

            batch_tensor = torch.stack(tensors).to(DEVICE)
            feats = _extract_fused_features(backbone, batch_tensor).cpu()  # [B, feat_dim]

            for j, path in enumerate(batch_paths):
                cache_name = str(hash(path)) + ".pt"
                cache_path = os.path.join(cache_dir, cache_name)
                torch.save(feats[j], cache_path)
                cache_map[path] = cache_path

    print(f"  ✅ Cached {len(cache_map)} feature vectors to {cache_dir}")
    return cache_map


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Custom VQA Dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DragonFruitVQADataset(Dataset):
    """
    PyTorch Dataset for multi-modal VQA training.

    Each sample returns:
        vision_feat: [feat_dim]   pre-cached ConViTX feature vector
        token_ids:   [max_seq]    tokenized question IDs
        answer_id:   int          answer class label
    """

    def __init__(
        self,
        triplets: List[Dict],
        tokenizer: VQATokenizer,
        cache_map: Dict[str, str],
        project_root: str,
        max_seq_len: int = 32,
    ):
        self.triplets    = triplets
        self.tokenizer   = tokenizer
        self.cache_map   = cache_map
        self.project_root = project_root
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        item = self.triplets[idx]

        # 1. Load cached vision features
        img_path = os.path.join(self.project_root, item["image_path"])
        cache_path = self.cache_map.get(img_path)

        if cache_path and os.path.exists(cache_path):
            vision_feat = torch.load(cache_path, weights_only=True)
        else:
            # Fallback: zero vector (should not happen if cache is built)
            # Dimension determined dynamically from first cached sample
            vision_feat = torch.zeros(768)

        # 2. Tokenize question
        token_ids = self.tokenizer.encode(
            item["question"],
            max_len=self.max_seq_len,
            padding=True,
        )
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        # 3. Answer label
        answer_id = item["answer_id"]

        return vision_feat, token_ids, answer_id


def vqa_collate_fn(batch):
    """Custom collate: stack features, questions, and labels."""
    vision_feats, token_ids, labels = zip(*batch)
    return (
        torch.stack(vision_feats),
        torch.stack(token_ids),
        torch.tensor(labels, dtype=torch.long),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Class Weights
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_class_weights(triplets: List[Dict], num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced answer distribution.

    Classes with fewer samples get higher weight → prevents the model from
    just predicting the majority class.
    """
    counts = Counter(t["answer_id"] for t in triplets)
    total = sum(counts.values())
    weights = torch.ones(num_classes)

    for cls_id, count in counts.items():
        if cls_id < num_classes:
            weights[cls_id] = total / (num_classes * count)

    # Normalize so max weight = 3.0 (prevent extreme values)
    weights = weights.clamp(max=3.0)
    return weights


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Training + Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss   = 0.0
    correct      = 0
    total        = 0

    for vision_feats, token_ids, labels in loader:
        vision_feats = vision_feats.to(device)
        token_ids    = token_ids.to(device)
        labels       = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision forward
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model.forward_cached(vision_feats, token_ids)
            loss   = criterion(logits, labels)

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for vision_feats, token_ids, labels in loader:
        vision_feats = vision_feats.to(device)
        token_ids    = token_ids.to(device)
        labels       = labels.to(device)

        logits = model.forward_cached(vision_feats, token_ids)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main Training Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_args():
    parser = argparse.ArgumentParser(description="Train Dragon Fruit VQA model.")
    parser.add_argument(
        "--dataset", type=str,
        default=os.path.join(ROOT, "models", "vqa_dataset_train.json"),
        help="Path to training triplets JSON.",
    )
    parser.add_argument(
        "--val-dataset", type=str,
        default=os.path.join(ROOT, "models", "vqa_dataset_val.json"),
        help="Path to validation triplets JSON.",
    )
    parser.add_argument(
        "--backbone", type=str,
        default=os.path.join(ROOT, "models", "best_convitx_pretrained.pth"),
        help="Path to pre-trained ConViTX checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--cache-dir", type=str, default=os.path.join(ROOT, "models", "vqa_feat_cache"))
    parser.add_argument("--output", type=str, default=os.path.join(ROOT, "models", "best_vqa.pth"))
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Dragon Fruit VQA — Training Pipeline")
    print("=" * 60)
    print(f"  Device: {DEVICE}")

    # ── 1. Load datasets ─────────────────────────────────────────────────
    print("\n📂 Loading datasets...")
    with open(args.dataset, "r", encoding="utf-8") as f:
        train_triplets = json.load(f)
    with open(args.val_dataset, "r", encoding="utf-8") as f:
        val_triplets = json.load(f)
    print(f"  Train: {len(train_triplets):,}  |  Val: {len(val_triplets):,}")

    # ── 2. Build tokenizer ───────────────────────────────────────────────
    print("\n📝 Building tokenizer...")
    tokenizer = build_default_tokenizer()
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Save tokenizer for deployment
    tok_path = os.path.join(os.path.dirname(args.output), "vqa_vocab.json")
    tokenizer.save(tok_path)
    print(f"  Saved: {tok_path}")

    # ── 3. Load backbone & extract features ──────────────────────────────
    print(f"\n🧠 Loading backbone: {args.backbone}")

    # Collect all unique image paths
    all_img_paths = list(set(
        os.path.join(ROOT, t["image_path"]) for t in train_triplets + val_triplets
    ))
    print(f"  Unique images: {len(all_img_paths)}")

    # Check if cache already exists
    cache_map_path = os.path.join(args.cache_dir, "cache_map.json")
    if os.path.exists(cache_map_path):
        print(f"  📦 Loading existing cache from {args.cache_dir}")
        with open(cache_map_path, "r") as f:
            cache_map = json.load(f)
        # Verify cache is complete
        missing = [p for p in all_img_paths if p not in cache_map]
        if missing:
            print(f"  ⚠ {len(missing)} images missing from cache — re-extracting...")
            backbone = _load_backbone(args.backbone)
            new_cache = extract_and_cache_features(
                backbone, missing, args.cache_dir, batch_size=args.batch_size,
            )
            cache_map.update(new_cache)
            del backbone
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        backbone = _load_backbone(args.backbone)
        cache_map = extract_and_cache_features(
            backbone, all_img_paths, args.cache_dir, batch_size=args.batch_size,
        )
        # Save cache map for future runs
        with open(cache_map_path, "w") as f:
            json.dump(cache_map, f)
        del backbone
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 4. Determine vision feature dimension from cache ─────────────────
    sample_feat = torch.load(
        next(iter(cache_map.values())), weights_only=True
    )
    vision_feat_dim = sample_feat.shape[0]
    print(f"  Vision feature dim: {vision_feat_dim}")

    # ── 5. Build datasets & loaders ──────────────────────────────────────
    print("\n📊 Building DataLoaders...")
    train_ds = DragonFruitVQADataset(
        train_triplets, tokenizer, cache_map, ROOT,
    )
    val_ds = DragonFruitVQADataset(
        val_triplets, tokenizer, cache_map, ROOT,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=vqa_collate_fn, num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=vqa_collate_fn, num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── 6. Build VQA model (no backbone — cached mode) ───────────────────
    print("\n🏗️  Building VQA model...")
    model = build_vqa_model(
        vocab_size=tokenizer.vocab_size,
        vision_backbone=None,
        vision_feat_dim=vision_feat_dim,
    ).to(DEVICE)

    n_params = count_trainable_params(model)
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Estimated FP32 size:  {n_params * 4 / 1024 / 1024:.2f} MB")

    # ── 7. Loss, optimizer, scheduler ────────────────────────────────────
    class_weights = compute_class_weights(train_triplets, NUM_ANSWER_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # ── 8. Training loop ─────────────────────────────────────────────────
    print(f"\n🚀 Training for {args.epochs} epochs (patience={args.patience})...")
    print(f"   {'Epoch':>5s}  {'Train Loss':>10s}  {'Train Acc':>9s}  "
          f"{'Val Loss':>10s}  {'Val Acc':>9s}  {'LR':>10s}")
    print(f"   {'─' * 60}")

    best_val_acc    = 0.0
    patience_count  = 0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"   {epoch:>5d}  {train_loss:>10.4f}  {train_acc:>8.1%}  "
              f"{val_loss:>10.4f}  {val_acc:>8.1%}  {lr:>10.6f}  ({elapsed:.1f}s)")

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc    = val_acc
            patience_count  = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"         ★ New best: {best_val_acc:.1%}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n   ⏹ Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── 9. Save best model ───────────────────────────────────────────────
    if best_state_dict is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        torch.save(best_state_dict, args.output)
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"\n  💾 Best model saved: {args.output} ({size_mb:.2f} MB)")
        print(f"  📊 Best validation accuracy: {best_val_acc:.1%}")

    # ── 10. Save training config for reproducibility ─────────────────────
    config_path = args.output.replace(".pth", "_config.json")
    config = {
        "vocab_size":      tokenizer.vocab_size,
        "vision_feat_dim": vision_feat_dim,
        "num_answers":     NUM_ANSWER_CLASSES,
        "best_val_acc":    best_val_acc,
        "epochs_trained":  epoch,
        "lr":              args.lr,
        "batch_size":      args.batch_size,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  📋 Config saved: {config_path}")

    print(f"\n✅ Training complete!")


if __name__ == "__main__":
    main()
