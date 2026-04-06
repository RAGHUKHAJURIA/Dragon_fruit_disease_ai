"""
Lightweight ConViTX hybrid model for edge/remote deployment.

Design goal:
- Parallel dual-branch architecture (CNN + ViT).
- Multiscale CNN branch (3x3 and 5x5 kernels).
- Keep total trainable parameters under 0.7M.
- Fusion layer for Grad-CAM targeting.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def _conv_bn_act(in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )

class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int = 2, kernel_size: int = 3, drop_path: float = 0.):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_residual:
            y = self.drop_path(y) + x
        return y

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1, drop_path: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

@dataclass(frozen=True)
class ConViTXConfig:
    num_classes: int = 6
    cnn_dim: int = 64
    vit_dim: int = 64
    fuse_dim: int = 128
    vit_depth: int = 3
    heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1
    param_budget: int = 700_000

class ConViTXSmall(nn.Module):
    """Dual-branch CNN + ViT model for edge deployment."""

    def __init__(self, cfg: ConViTXConfig):
        super().__init__()
        self.cfg = cfg

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, cfg.vit_depth)]
        cnn_dpr = cfg.drop_path_rate * 0.5  # Slightly less aggressive on CNN early layers

        # --- CNN Branch (Local Texture) ---
        self.cnn_branch = nn.Sequential(
            _conv_bn_act(3, 24, kernel_size=3, stride=2, padding=1),             
            InvertedResidual(24, 32, stride=2, expand_ratio=2, kernel_size=3, drop_path=cnn_dpr),   
            InvertedResidual(32, 48, stride=2, expand_ratio=2, kernel_size=5, drop_path=cnn_dpr),   
            InvertedResidual(48, cfg.cnn_dim, stride=2, expand_ratio=2, kernel_size=3, drop_path=cnn_dpr), 
        )

        # --- ViT Branch (Global Context) ---
        self.patch_embed = nn.Conv2d(3, cfg.vit_dim, kernel_size=16, stride=16)
        self.pos_embed = nn.Parameter(torch.zeros(1, 14 * 14, cfg.vit_dim))
        self.pos_drop = nn.Dropout(cfg.dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=cfg.vit_dim,
                num_heads=cfg.heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                drop_path=dpr[i],
            ) for i in range(cfg.vit_depth)
        ])
        self.vit_norm = nn.LayerNorm(cfg.vit_dim)

        # --- Fusion Layer ---
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(cfg.cnn_dim + cfg.vit_dim, cfg.fuse_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.fuse_dim),
            nn.SiLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(cfg.fuse_dim, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_branch(x)

        vit_patches = self.patch_embed(x)
        b, c, h, w = vit_patches.shape
        tokens = vit_patches.flatten(2).transpose(1, 2)
        tokens = self.pos_drop(tokens + self.pos_embed)
        
        for blk in self.transformer_blocks:
            tokens = blk(tokens)
            
        tokens = self.vit_norm(tokens)
        vit_feat = tokens.transpose(1, 2).reshape(b, -1, h, w)

        fused = torch.cat([cnn_feat, vit_feat], dim=1)
        fused_map = self.fusion_conv(fused)
        
        out = self.pool(fused_map)
        out = torch.flatten(out, 1)
        out = self.head(out)
        return out


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def build_convitx_small(num_classes: int = 6, enforce_budget: bool = True) -> ConViTXSmall:
    cfg = ConViTXConfig(num_classes=num_classes)
    model = ConViTXSmall(cfg)
    if enforce_budget:
        n_params = count_parameters(model)
        if n_params > cfg.param_budget:
            raise ValueError(f"ConViTXSmall exceeds parameter budget: {n_params:,} > {cfg.param_budget:,}")
    return model


def build_convitx_base(num_classes: int = 6, enforce_budget: bool = True) -> ConViTXSmall:
    """Scaled up configuration for higher accuracy while staying under 700k parameters."""
    cfg = ConViTXConfig(
        num_classes=num_classes,
        cnn_dim=96,
        vit_dim=96,
        fuse_dim=192,
        vit_depth=4,
        heads=6,
        drop_path_rate=0.2, # Slightly stronger regularization for the larger model
    )
    model = ConViTXSmall(cfg)
    if enforce_budget:
        n_params = count_parameters(model)
        if n_params > cfg.param_budget:
            raise ValueError(f"ConViTXBase exceeds parameter budget: {n_params:,} > {cfg.param_budget:,}")
    return model
