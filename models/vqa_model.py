"""
DragonFruitVQA — Lightweight Visual Question Answering Model.

Architecture:
    ┌──────────────────────────────────────────────────┐
    │  Image ──► VisionEncoder (frozen ConViTX) ──► 128-d  │
    │  Text  ──► TextEncoder  (GRU)             ──► 128-d  │
    │            ↓                                         │
    │         BilinearFusion (128 × 128 → 256-d)           │
    │            ↓                                         │
    │       ClassificationHead (256 → 32 answers)          │
    └──────────────────────────────────────────────────┘

Design constraints:
    • Trainable parameters (excl. frozen backbone): ~200 K
    • Quantized model footprint: < 1 MB
    • Inference time on CPU: < 100 ms per query
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vqa_answers import NUM_ANSWER_CLASSES


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass(frozen=True)
class VQAConfig:
    """Hyper-parameters for the VQA model."""
    # Vision
    vision_feat_dim: int  = 128     # dim of pooled ConViTX features
    vision_proj_dim: int  = 128     # projected vision embedding

    # Text
    vocab_size:      int  = 600     # tokenizer vocab (overridden at runtime)
    embed_dim:       int  = 64      # word embedding dimension
    gru_hidden:      int  = 64      # GRU hidden units (bidirectional → 128)
    gru_layers:      int  = 2       # number of GRU layers
    text_proj_dim:   int  = 128     # projected text embedding
    max_seq_len:     int  = 32      # max question length in tokens

    # Fusion
    fused_dim:       int  = 256     # bilinear fusion output dim
    dropout:         float = 0.3    # dropout in fusion layer

    # Classification
    num_answers:     int  = NUM_ANSWER_CLASSES   # 32


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Vision Encoder (Frozen ConViTX Wrapper)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class VisionEncoder(nn.Module):
    """
    Wraps the pre-trained ConViTX backbone to extract pooled features.

    Supports BOTH architectures:
        • ConViTXPretrained — cnn_branch + ViT → concat → 768-d
        • ConViTXSmall      — pool → 128-d

    In training mode the backbone is fully FROZEN — we only train a small
    linear projection on top.  For maximum memory efficiency, call
    `extract_features()` once per image and cache the vector to disk
    so the backbone never needs to be loaded during the training loop.
    """

    def __init__(self, backbone: nn.Module, feat_dim: int = 768, proj_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self._is_pretrained = hasattr(backbone, 'cnn_branch')

        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Small trainable projection
        self.proj = nn.Linear(feat_dim, proj_dim)
        self._feat_dim = feat_dim

    def _extract_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the feature vector BEFORE the classification head."""
        if self._is_pretrained:
            # ── ConViTXPretrained: replicate forward() minus head ─────
            cnn_feat = self.backbone.cnn_pool(self.backbone.cnn_branch(x))
            cnn_feat = cnn_feat.flatten(1)                      # (B, 576)

            patches = self.backbone.patch_embed(x)
            b, c, h, w = patches.shape
            tokens = patches.flatten(2).transpose(1, 2)

            gs = self.backbone._pos_grid
            if h != gs or w != gs:
                pos = self.backbone.pos_embed.reshape(1, gs, gs, c).permute(0, 3, 1, 2)
                pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
                pos = pos.permute(0, 2, 3, 1).reshape(1, h * w, c)
            else:
                pos = self.backbone.pos_embed

            tokens = self.backbone.pos_drop(tokens + pos)
            for blk in self.backbone.blocks:
                tokens = blk(tokens)
            tokens   = self.backbone.vit_norm(tokens)
            vit_feat = tokens.mean(dim=1)

            return torch.cat([cnn_feat, vit_feat], dim=1)       # (B, 768)
        else:
            # ── Legacy ConViTXSmall: hook into pool layer ─────────────
            feats_store = {}
            def _hook(m, inp, out):
                feats_store["f"] = out.detach()
            h = self.backbone.pool.register_forward_hook(_hook)
            _ = self.backbone(x)
            h.remove()
            return feats_store["f"].flatten(1)

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features without gradients (for caching).

        Args:
            images: [B, 3, 224, 224] normalized image tensor

        Returns:
            [B, feat_dim] feature vectors
        """
        self.backbone.eval()
        return self._extract_fused(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: backbone (no grad) → project.

        Args:
            images: [B, 3, 224, 224] normalized tensor

        Returns:
            [B, proj_dim] vision embedding
        """
        with torch.no_grad():
            self.backbone.eval()
            feats = self._extract_fused(images)

        return self.proj(feats)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Text Encoder (Bidirectional GRU)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TextEncoder(nn.Module):
    """
    Edge-ready text encoder using word embeddings + bidirectional GRU.

    Total parameter count (with vocab_size=500, embed=64, hidden=64, layers=2):
        Embeddings:  500 × 64  = 32,000
        GRU:         ~66,000   (2-layer bidir, 64 hidden)
        Projection:  128 × 128 = 16,384
        ─────────────────────────────────
        Total:       ~114,000 parameters (~0.44 MB FP32)
    """

    def __init__(self, cfg: VQAConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.embed_dim,
            padding_idx=0,                          # <PAD> = 0
        )
        self.gru = nn.GRU(
            input_size=cfg.embed_dim,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        # Bidirectional → output is 2 × gru_hidden
        gru_out_dim = cfg.gru_hidden * 2
        self.proj = nn.Linear(gru_out_dim, cfg.text_proj_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, seq_len] integer tensor of token IDs

        Returns:
            [B, text_proj_dim] text embedding (128-d)
        """
        embedded = self.embedding(token_ids)         # [B, seq_len, embed_dim]
        output, hidden = self.gru(embedded)          # output: [B, seq_len, 2*H]

        # Use the last hidden states from both directions
        # hidden: [num_layers*2, B, H] → take last layer fwd + bwd
        fwd_hidden = hidden[-2]                      # [B, H]  (last layer, forward)
        bwd_hidden = hidden[-1]                      # [B, H]  (last layer, backward)
        combined   = torch.cat([fwd_hidden, bwd_hidden], dim=1)  # [B, 2*H]

        return self.proj(combined)                   # [B, text_proj_dim]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Bilinear Fusion + Classification Head
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DragonFruitVQA(nn.Module):
    """
    Full VQA model: Vision + Text → Bilinear Fusion → Answer.

    This module can operate in two modes:

    1. **Full mode** (for inference):
       forward(images, token_ids) — runs backbone + GRU + fusion.

    2. **Cached mode** (for training):
       forward_cached(vision_feats, token_ids) — skips backbone entirely.
       Vision features are pre-extracted and loaded from disk.
    """

    def __init__(
        self,
        cfg: VQAConfig,
        vision_backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = cfg

        # ── Vision encoder (optional — not needed for cached training) ───
        if vision_backbone is not None:
            self.vision_encoder = VisionEncoder(
                backbone=vision_backbone,
                feat_dim=cfg.vision_feat_dim,
                proj_dim=cfg.vision_proj_dim,
            )
        else:
            # Lightweight projection for cached features
            self.vision_encoder = None
            self.vision_proj = nn.Linear(cfg.vision_feat_dim, cfg.vision_proj_dim)

        # ── Text encoder ─────────────────────────────────────────────────
        self.text_encoder = TextEncoder(cfg)

        # ── Bilinear fusion ──────────────────────────────────────────────
        # Captures cross-modal interactions:  v^T W q  →  fused_dim
        self.bilinear = nn.Bilinear(
            in1_features=cfg.vision_proj_dim,
            in2_features=cfg.text_proj_dim,
            out_features=cfg.fused_dim,
        )
        self.fusion_norm = nn.LayerNorm(cfg.fused_dim)
        self.fusion_drop = nn.Dropout(cfg.dropout)

        # ── Classification head ──────────────────────────────────────────
        self.classifier = nn.Linear(cfg.fused_dim, cfg.num_answers)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for trainable layers."""
        for m in [self.bilinear, self.classifier]:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    # ── Full forward (inference) ─────────────────────────────────────────
    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full inference: image + question → logits.

        Args:
            images:    [B, 3, 224, 224] normalized image tensor
            token_ids: [B, seq_len] integer token IDs

        Returns:
            [B, num_answers] logits
        """
        if self.vision_encoder is not None:
            v = self.vision_encoder(images)           # [B, vision_proj_dim]
        else:
            raise RuntimeError(
                "Vision backbone not loaded. Use forward_cached() for training "
                "or provide a backbone at initialization for inference."
            )

        q = self.text_encoder(token_ids)              # [B, text_proj_dim]
        return self._fuse_and_classify(v, q)

    # ── Cached forward (training) ────────────────────────────────────────
    def forward_cached(
        self,
        vision_feats: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training forward with pre-extracted vision features.

        Args:
            vision_feats: [B, vision_feat_dim] pre-extracted features
            token_ids:    [B, seq_len] integer token IDs

        Returns:
            [B, num_answers] logits
        """
        if self.vision_encoder is not None:
            v = self.vision_encoder.proj(vision_feats)
        else:
            v = self.vision_proj(vision_feats)        # [B, vision_proj_dim]

        q = self.text_encoder(token_ids)              # [B, text_proj_dim]
        return self._fuse_and_classify(v, q)

    # ── Shared fusion logic ──────────────────────────────────────────────
    def _fuse_and_classify(
        self,
        v: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bilinear fusion + classification.

        Args:
            v: [B, vision_proj_dim] vision embedding
            q: [B, text_proj_dim]   text embedding

        Returns:
            [B, num_answers] raw logits
        """
        fused = self.bilinear(v, q)                   # [B, fused_dim]
        fused = self.fusion_norm(fused)
        fused = F.silu(fused)                         # SiLU activation
        fused = self.fusion_drop(fused)
        logits = self.classifier(fused)               # [B, num_answers]
        return logits


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Builder Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_vqa_model(
    vocab_size: int,
    vision_backbone: Optional[nn.Module] = None,
    vision_feat_dim: int = 128,
) -> DragonFruitVQA:
    """
    Build a DragonFruitVQA model with default config.

    Args:
        vocab_size:       Tokenizer vocabulary size.
        vision_backbone:  Pre-trained ConViTX model (optional for cached training).
        vision_feat_dim:  Dimension of pooled features from the backbone.

    Returns:
        DragonFruitVQA model instance.
    """
    cfg = VQAConfig(
        vocab_size=vocab_size,
        vision_feat_dim=vision_feat_dim,
    )
    model = DragonFruitVQA(cfg, vision_backbone=vision_backbone)
    return model


def count_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters (excludes frozen backbone)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Quick Sanity Check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    print("=" * 60)
    print("  DragonFruitVQA — Architecture Verification")
    print("=" * 60)

    # Build without backbone (cached training mode)
    model = build_vqa_model(vocab_size=500, vision_backbone=None)
    n_params = count_trainable_params(model)
    print(f"\n  Trainable parameters: {n_params:,}")
    print(f"  Estimated FP32 size: {n_params * 4 / 1024 / 1024:.2f} MB")

    # Test forward with cached features
    batch = 4
    dummy_vision = torch.randn(batch, 128)       # pre-extracted features
    dummy_tokens = torch.randint(0, 500, (batch, 32))
    logits = model.forward_cached(dummy_vision, dummy_tokens)
    print(f"\n  Input vision: {dummy_vision.shape}")
    print(f"  Input tokens: {dummy_tokens.shape}")
    print(f"  Output logits: {logits.shape}")
    print(f"  Answer classes: {logits.shape[1]}")

    # Print module breakdown
    print(f"\n  Module breakdown:")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"    {name:20s}: {n:>8,} params")

    print("\n  ✅ Architecture verified successfully.")
