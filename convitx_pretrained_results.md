# ConViTX-Pretrained Results

*Hybrid CNN (MobileNetV3-Small, pretrained) + ViT  |  Params: 2,993,254 trainable*


## Best Validation Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **94.62%** |
| **Macro F1** | **0.9363** |
| Best Epoch   | 9 |

## Per-Class Metrics

| Class | Precision | Recall | F1 |
|-------|----------:|-------:|---:|
| Anthracnose | 0.7692 | 1.0000 | 0.8696 |
| Brown_Stem_Spot | 0.9841 | 0.9394 | 0.9612 |
| Gray_Blight | 0.9028 | 1.0000 | 0.9489 |
| Healthy | 1.0000 | 0.9545 | 0.9767 |
| Soft_Rot | 0.9910 | 0.9821 | 0.9865 |
| Stem_Canker | 0.9130 | 0.8400 | 0.8750 |

## Training Config

- CNN backbone: MobileNetV3-Small (ImageNet pretrained)
- Phase 1: CNN frozen for `0` epochs → only ViT+Head trained
- Phase 2: Full joint fine-tuning (CNN LR×0.1)
- LR=3e-05  |  Batch=32  |  Epochs=25
- Augmentation: RandomCrop+Flip+Rotate+ColorJitter+RandomErasing (NO MixUp/CutMix)
- Loss: Focal CE (γ=2.0) + label smoothing 0.05
- EMA decay: 0.9995

## Artifacts

- `models/best_convitx_pretrained.pth`
- `models/convitx_pretrained_curves.png`
- `models/convitx_pretrained_cm.png`
- `models/convitx_pretrained_summary.json`