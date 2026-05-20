"""
Grad-CAM XAI module for Dragon Fruit Disease Detection.
Highlights the visual regions that influenced the model's prediction.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")  # headless backend — no plt.show() popups
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models.convitx import build_convitx_base
from models.convitx_pretrained import build_convitx_pretrained

IMG_SIZE = 224
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── TRANSFORMS ──────────────────────────────────────────────────────────────
infer_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# TTA augmentation set — same as evaluate_tta.py (6 passes → +1% accuracy)
_TTA_TRANSFORMS = [
    transforms.Compose([  # original
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([  # H-flip
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([  # V-flip
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([  # 90°
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation((90, 90)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([  # 180°
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation((180, 180)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([  # 270°
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation((270, 270)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]

# ─── GRAD-CAM ────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Computes Grad-CAM heatmap for any CNN with a named target layer.
    Works with both timm EfficientNet and torchvision ResNet.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model.eval().to(DEVICE)
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor: torch.Tensor, class_idx: int = None):
        """
        Args:
            image_tensor: [1, 3, H, W] normalized tensor
            class_idx:    target class index (None → use predicted class)
        Returns:
            heatmap (np.ndarray, float32, shape [H, W], range [0,1])
            predicted class index (int)
            prediction probabilities (np.ndarray)
        """
        image_tensor = image_tensor.to(DEVICE).requires_grad_(True)

        # Forward pass
        logits = self.model(image_tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().detach().numpy()

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Pool gradients across channels
        weights  = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam      = (weights * self.activations).sum(dim=1).squeeze()  # [H, W]
        cam      = F.relu(cam).detach().cpu().numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probs


# ─── OVERLAY ─────────────────────────────────────────────────────────────────
def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Superimposes Grad-CAM heatmap on the original image.

    Args:
        original_image: BGR or RGB uint8 image [H, W, 3]
        heatmap:        float32 array [H', W'], values in [0, 1]
        alpha:          transparency of heatmap overlay
        colormap:       OpenCV colormap constant

    Returns:
        Overlaid image (uint8, RGB)
    """
    h, w   = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)   # BGR

    if original_image.shape[2] == 3:
        img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = original_image

    overlay = cv2.addWeighted(img_bgr, 1 - alpha, colored_heatmap, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ─── CONVENIENCE FUNCTION ────────────────────────────────────────────────────
def run_gradcam(
    model:        torch.nn.Module,
    target_layer: torch.nn.Module,
    image_path:   str,
    class_names:  list,
    save_path:    str = None,
    use_tta:      bool = True,
) -> dict:
    """
    End-to-end Grad-CAM pipeline with optional TTA for better accuracy.

    TTA (Test-Time Augmentation) averages predictions over 6 augmented views
    before selecting the final class — same technique that boosted accuracy
    from 94.62% to 95.76% in evaluation.

    Returns dict with:
        predicted_class (str)
        confidence      (float)
        probabilities   (dict {class_name: prob})
        heatmap         (np.ndarray)
        overlay         (np.ndarray, RGB)
        low_confidence  (bool)  — True if model is uncertain (<50%)
    """
    pil_img = Image.open(image_path).convert("RGB")
    orig_np = np.array(pil_img)

    # ── TTA: average probabilities over 6 augmented views ────────────────
    if use_tta:
        all_probs = []
        for tf in _TTA_TRANSFORMS:
            t = tf(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(t)
                p = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            all_probs.append(p)
        probs    = np.stack(all_probs, axis=0).mean(axis=0)   # avg over 6 passes
        pred_idx = int(np.argmax(probs))
    else:
        tensor_img = infer_transforms(pil_img).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor_img.to(DEVICE))
            probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))

    # ── Grad-CAM heatmap (single pass on original image) ─────────────────
    tensor_img = infer_transforms(pil_img).unsqueeze(0)
    gradcam    = GradCAM(model, target_layer)
    heatmap, _, _ = gradcam.generate(tensor_img, class_idx=pred_idx)
    overlay    = overlay_heatmap(orig_np, heatmap)

    low_confidence = float(probs[pred_idx]) < 0.50

    result = {
        "predicted_class": class_names[pred_idx],
        "confidence":      float(probs[pred_idx]),
        "probabilities":   {c: float(p) for c, p in zip(class_names, probs)},
        "heatmap":         heatmap,
        "overlay":         overlay,
        "low_confidence":  low_confidence,
    }

    # Visualise
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_np);   axes[0].set_title("Original Image");        axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM Map"); axes[1].axis("off")
    conf_label = f"{result['predicted_class']} ({result['confidence']:.1%})"
    if low_confidence:
        conf_label += "\n⚠ Low confidence"
    axes[2].imshow(overlay);  axes[2].set_title(f"Overlay\n{conf_label}"); axes[2].axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return result


# ─── LAYER HELPERS ───────────────────────────────────────────────────────────
def get_target_layer_efficientnet(model) -> torch.nn.Module:
    """Returns the last convolutional block of a timm EfficientNet."""
    return model.blocks[-1]




def get_target_layer_convitx(model) -> torch.nn.Module:
    """Returns an architecture-aware target layer for ConViTX Grad-CAM hooks."""
    # Old ConViTX (scratch-trained) has fusion_conv
    if hasattr(model, "fusion_conv"):
        return model.fusion_conv[0]

    # ConViTXPretrained — use the second-to-last block of MobileNetV3 features
    # (avoids the final Hardswish-only block which has no spatial gradients)
    if hasattr(model, "cnn_branch"):
        # MobileNetV3-Small: features[12] is the last InvertedResidual block
        # that has meaningful spatial gradients for Grad-CAM
        try:
            return model.cnn_branch[12]   # Last InvertedResidual in MobileNetV3-Small
        except (IndexError, TypeError):
            return model.cnn_branch[-2]   # fallback: second-to-last

    raise AttributeError("Unsupported ConViTX architecture: missing Grad-CAM target layer")





def load_convitx_model(
    model_path: str,
    num_classes: int = 6,
    device: torch.device = DEVICE,
) -> torch.nn.Module:
    """Load ConViTX checkpoint, auto-selecting architecture from checkpoint keys."""
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format in: {model_path}")

    # Pretrained ConViTX uses a sequential head (head.0/head.3) and no fusion_conv.
    is_pretrained_convitx = any(k.startswith("head.0") for k in state.keys())
    if is_pretrained_convitx:
        model = build_convitx_pretrained(num_classes=num_classes)
    else:
        model = build_convitx_base(num_classes=num_classes, enforce_budget=False)

    model.load_state_dict(state)
    model.eval().to(device)
    return model
