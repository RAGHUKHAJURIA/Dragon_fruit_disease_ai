# ConViTX Fine-Tuning Results

*Scale: ConViTX-large  |  Params: 1,213,166 trainable*


## Best Validation Metrics

| Metric | Standard | TTA |
|--------|---------|-----|
| Accuracy | **46.21%** | **45.52%** |
| Macro F1 | **0.4119** | **0.4068** |
| Best Epoch | 41 | — |

## Per-Class Metrics (Standard)

| Class | Precision | Recall | F1 |
|-------|----------:|-------:|---:|
| Anthracnose | 0.4848 | 0.6667 | 0.5614 |
| Brown_Stem_Spot | 0.4000 | 0.3333 | 0.3636 |
| Gray_Blight | 0.0000 | 0.0000 | 0.0000 |
| Healthy | 0.5294 | 0.6667 | 0.5902 |
| Soft_Rot | 1.0000 | 0.3077 | 0.4706 |
| Stem_Canker | 0.3400 | 0.8500 | 0.4857 |

## Training Configuration

- Scale: `large`
- Epochs: `60`  |  Early stopping patience: `12`
- LR (ViT group): `0.0002`  |  Warmup: 5 epochs → CosineAnnealingWarmRestarts
- MixUp α: `0.4`  |  CutMix α: `1.0`
- EMA decay: `0.9995`
- KD: `disabled`

## Artifacts

- `models/best_convitx_finetuned.pth`
- `models/convitx_finetune_curves.png`
- `models/convitx_finetune_cm_val.png`
- `models/convitx_finetune_summary.json`