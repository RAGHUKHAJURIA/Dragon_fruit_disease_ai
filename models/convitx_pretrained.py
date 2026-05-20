"""
ConViTX-Pretrained: Hybrid CNN+ViT with Pretrained CNN Branch
=============================================================
Replaces the random-init CNN branch with MobileNetV3-Small (ImageNet pretrained).
This is the KEY fix for low accuracy on small datasets — pretrained features give a
massive head start that random-init models can never recover from in ~100 epochs.

Architecture:
  - CNN Branch  : MobileNetV3-Small (pretrained, 576-ch output) → GlobalAvgPool
  - ViT Branch  : PatchEmbed(kernel=16) + 4 TransformerBlocks → GlobalAvgPool
  - Fusion      : Concat(cnn_feat, vit_feat) → Linear head
  - Params      : ~2.5M trainable (CNN unfrozen later) / 576K (CNN frozen phase)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# --------------------------------------------------------------------------- #
#  Reuse TransformerBlock from original convitx.py
# --------------------------------------------------------------------------- #
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 3.0,
                 dropout: float = 0.1, drop_path_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
        self.dp = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        attn_out, _ = self.attn(n, n, n, need_weights=False)
        x = x + self.dp(attn_out)
        x = x + self.dp(self.mlp(self.norm2(x)))
        return x


# --------------------------------------------------------------------------- #
#  ConViTX-Pretrained
# --------------------------------------------------------------------------- #
class ConViTXPretrained(nn.Module):
    """
    Hybrid CNN (MobileNetV3-Small, pretrained) + ViT model.

    CNN branch output: 576-dim vector (GlobalAvgPool of MobileNetV3 features)
    ViT branch output: vit_dim-dim vector (mean of transformer tokens)
    Fusion           : concat → dropout → linear head

    Args:
        num_classes  : number of output classes
        vit_dim      : transformer embedding dimension (default 192)
        vit_depth    : number of transformer blocks (default 4)
        vit_heads    : attention heads (default 8)
        drop_path    : stochastic depth rate
        dropout      : dropout in MLP and head
        freeze_cnn   : if True, freeze CNN branch (good for first few epochs)
    """

    CNN_OUT_DIM = 576   # MobileNetV3-Small features output channels

    def __init__(
        self,
        num_classes: int = 6,
        vit_dim: int     = 192,
        vit_depth: int   = 4,
        vit_heads: int   = 8,
        drop_path: float = 0.1,
        dropout: float   = 0.1,
        freeze_cnn: bool = False,
        img_size: int    = 224,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.vit_dim     = vit_dim
        self._img_size   = img_size

        # ------------------------------------------------------------------ #
        # CNN Branch — MobileNetV3-Small pretrained features
        # ------------------------------------------------------------------ #
        mv3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.cnn_branch = mv3.features   # (B, 576, H/32, W/32)
        self.cnn_pool   = nn.AdaptiveAvgPool2d(1)   # → (B, 576, 1, 1)

        if freeze_cnn:
            for p in self.cnn_branch.parameters():
                p.requires_grad_(False)

        # ------------------------------------------------------------------ #
        # ViT Branch — patch embed + transformer blocks
        # ------------------------------------------------------------------ #
        patch_size  = 16
        self.patch_embed = nn.Conv2d(3, vit_dim, kernel_size=patch_size, stride=patch_size)
        seq_len          = (img_size // patch_size) ** 2   # 196 for 224
        self.pos_embed   = nn.Parameter(torch.zeros(1, seq_len, vit_dim))
        self._pos_grid   = img_size // patch_size
        self.pos_drop    = nn.Dropout(dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path, vit_depth)]
        self.blocks  = nn.ModuleList([
            TransformerBlock(vit_dim, vit_heads, mlp_ratio=3.0,
                             dropout=dropout, drop_path_rate=dpr[i])
            for i in range(vit_depth)
        ])
        self.vit_norm = nn.LayerNorm(vit_dim)

        # ------------------------------------------------------------------ #
        # Fusion head
        # ------------------------------------------------------------------ #
        fuse_in = self.CNN_OUT_DIM + vit_dim
        self.head = nn.Sequential(
            nn.Linear(fuse_in, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self._init_vit_weights()

    def _init_vit_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.blocks.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- CNN Branch ----
        cnn_feat = self.cnn_pool(self.cnn_branch(x))   # (B, 576, 1, 1)
        cnn_feat = cnn_feat.flatten(1)                  # (B, 576)

        # ---- ViT Branch ----
        patches = self.patch_embed(x)                   # (B, vit_dim, h, w)
        b, c, h, w = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)     # (B, h*w, C)

        # Interpolate pos_embed if spatial size differs
        gs = self._pos_grid
        if h != gs or w != gs:
            pos = self.pos_embed.reshape(1, gs, gs, c).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, h * w, c)
        else:
            pos = self.pos_embed

        tokens = self.pos_drop(tokens + pos)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens   = self.vit_norm(tokens)
        vit_feat = tokens.mean(dim=1)                   # (B, vit_dim) — global avg

        # ---- Fusion ----
        fused = torch.cat([cnn_feat, vit_feat], dim=1)  # (B, 576+vit_dim)
        return self.head(fused)

    def freeze_cnn(self):
        """Freeze CNN branch (call during warm-up phase)."""
        for p in self.cnn_branch.parameters():
            p.requires_grad_(False)
        print("[ConViTX] CNN branch FROZEN")

    def unfreeze_cnn(self):
        """Unfreeze CNN branch (call after warm-up)."""
        for p in self.cnn_branch.parameters():
            p.requires_grad_(True)
        print("[ConViTX] CNN branch UNFROZEN — full fine-tuning")

    def param_groups(self, base_lr: float):
        """Return parameter groups with differential LRs for AdamW."""
        cnn_params  = list(self.cnn_branch.parameters())
        vit_params  = list(self.patch_embed.parameters()) + \
                      [self.pos_embed] + \
                      list(self.blocks.parameters()) + \
                      list(self.vit_norm.parameters())
        head_params = list(self.head.parameters())
        return [
            {"params": cnn_params,  "lr": base_lr * 0.1,  "name": "cnn"},
            {"params": vit_params,  "lr": base_lr,         "name": "vit"},
            {"params": head_params, "lr": base_lr * 2.0,   "name": "head"},
        ]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def build_convitx_pretrained(
    num_classes: int = 6,
    vit_dim: int     = 192,
    vit_depth: int   = 4,
    vit_heads: int   = 8,
    drop_path: float = 0.1,
    dropout: float   = 0.1,
    freeze_cnn: bool = False,
) -> ConViTXPretrained:
    return ConViTXPretrained(
        num_classes=num_classes,
        vit_dim=vit_dim,
        vit_depth=vit_depth,
        vit_heads=vit_heads,
        drop_path=drop_path,
        dropout=dropout,
        freeze_cnn=freeze_cnn,
    )
