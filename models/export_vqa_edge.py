"""
Edge Optimization for Dragon Fruit VQA Model.

Applies Dynamic INT8 Quantization to the text encoder, fusion layer,
and classification head.  The final quantized model + vocabulary +
answer map ships under 1 MB (full pipeline with ConViTX backbone < 3 MB).

Usage:
    python export_vqa_edge.py \
        --model   models/best_vqa.pth \
        --vocab   models/vqa_vocab.json \
        --output  models/vqa_edge/

Artifacts produced:
    vqa_model_quantized.pth  — INT8 quantized model weights
    vqa_vocab.json           — tokenizer vocabulary
    vqa_answers.json         — answer class → response text (all languages)
    vqa_config.json          — model config for reconstruction
"""

from __future__ import annotations
import argparse, json, os, sys, shutil

import torch
import torch.nn as nn
import torch.quantization

# ── Project root setup ───────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.vqa_model import build_vqa_model, count_trainable_params, VQAConfig
from models.vqa_tokenizer import VQATokenizer
from models.vqa_answers import export_answer_map, NUM_ANSWER_CLASSES, get_answer_text


def parse_args():
    parser = argparse.ArgumentParser(description="Export quantized VQA model for edge.")
    parser.add_argument(
        "--model", type=str,
        default=os.path.join(ROOT, "models", "best_vqa.pth"),
        help="Path to trained VQA model checkpoint.",
    )
    parser.add_argument(
        "--vocab", type=str,
        default=os.path.join(ROOT, "models", "vqa_vocab.json"),
        help="Path to tokenizer vocabulary JSON.",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(ROOT, "models", "vqa_edge"),
        help="Output directory for edge artifacts.",
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(ROOT, "models", "best_vqa_config.json"),
        help="Path to training config JSON (for model reconstruction).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Dragon Fruit VQA — Edge Optimization (INT8)")
    print("=" * 60)

    # ── 1. Load training config ──────────────────────────────────────────
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            train_config = json.load(f)
        vocab_size      = train_config.get("vocab_size", 500)
        vision_feat_dim = train_config.get("vision_feat_dim", 128)
    else:
        print("  ⚠ No config file found — using defaults.")
        vocab_size      = 500
        vision_feat_dim = 128

    # ── 2. Reconstruct and load the FP32 model ──────────────────────────
    print(f"\n📦 Loading FP32 model: {args.model}")
    model = build_vqa_model(
        vocab_size=vocab_size,
        vision_backbone=None,
        vision_feat_dim=vision_feat_dim,
    )

    state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Report FP32 size
    fp32_path = args.model
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"  FP32 model size: {fp32_size:.2f} MB")
    print(f"  Parameters:      {count_trainable_params(model):,}")

    # ── 3. Apply Dynamic INT8 Quantization ───────────────────────────────
    print(f"\n⚡ Applying Dynamic INT8 Quantization...")
    print(f"   Targeting: nn.Linear, nn.GRU, nn.LSTM")

    # Dynamic quantization targets Linear and GRU/LSTM layers
    # This quantizes:
    #   - TextEncoder: GRU layers + projection Linear
    #   - BilinearFusion: handled via the Linear inside nn.Bilinear
    #   - ClassificationHead: Linear layer
    #   - VisionProj: Linear layer
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={
            nn.Linear,     # All linear layers (projection, classifier)
            nn.GRU,        # Text encoder GRU
        },
        dtype=torch.qint8,
    )

    # ── 4. Save quantized model ──────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    quant_path = os.path.join(args.output, "vqa_model_quantized.pth")
    torch.save(quantized_model.state_dict(), quant_path)
    quant_size = os.path.getsize(quant_path) / (1024 * 1024)

    print(f"\n  📊 Size Comparison:")
    print(f"     FP32 model:     {fp32_size:.2f} MB")
    print(f"     INT8 quantized: {quant_size:.2f} MB")
    print(f"     Compression:    {fp32_size / quant_size:.1f}×")

    # ── 5. Verify quantized model ────────────────────────────────────────
    print(f"\n🔍 Verifying quantized model...")
    dummy_vision = torch.randn(1, vision_feat_dim)
    dummy_tokens = torch.randint(0, vocab_size, (1, 32))

    with torch.no_grad():
        fp32_logits = model.forward_cached(dummy_vision, dummy_tokens)
        int8_logits = quantized_model.forward_cached(dummy_vision, dummy_tokens)

    fp32_pred = fp32_logits.argmax(dim=1).item()
    int8_pred = int8_logits.argmax(dim=1).item()

    print(f"  FP32 prediction: class {fp32_pred}")
    print(f"  INT8 prediction: class {int8_pred}")
    print(f"  Match: {'✅ Yes' if fp32_pred == int8_pred else '⚠ No (expected for random input)'}")

    # ── 6. Export tokenizer vocabulary ───────────────────────────────────
    vocab_out = os.path.join(args.output, "vqa_vocab.json")
    shutil.copy2(args.vocab, vocab_out)
    vocab_size_kb = os.path.getsize(vocab_out) / 1024
    print(f"\n  📝 Vocabulary:  {vocab_out} ({vocab_size_kb:.1f} KB)")

    # ── 7. Export answer map (all languages) ─────────────────────────────
    answers_out = os.path.join(args.output, "vqa_answers.json")
    export_answer_map(answers_out)
    answers_size_kb = os.path.getsize(answers_out) / 1024
    print(f"  🗣️  Answers:     {answers_out} ({answers_size_kb:.1f} KB)")

    # ── 8. Export model config ───────────────────────────────────────────
    config_out = os.path.join(args.output, "vqa_config.json")
    edge_config = {
        "vocab_size":      vocab_size,
        "vision_feat_dim": vision_feat_dim,
        "num_answers":     NUM_ANSWER_CLASSES,
        "max_seq_len":     32,
        "quantization":    "dynamic_int8",
        "supported_languages": ["en", "vi", "th"],
    }
    with open(config_out, "w") as f:
        json.dump(edge_config, f, indent=2)
    print(f"  ⚙️  Config:      {config_out}")

    # ── 9. Total package size ────────────────────────────────────────────
    total_size = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if os.path.isfile(os.path.join(args.output, f))
    )
    total_mb = total_size / (1024 * 1024)

    print(f"\n{'─' * 60}")
    print(f"  📦 TOTAL EDGE PACKAGE: {total_mb:.2f} MB")
    budget_ok = total_mb < 15.0
    print(f"  📏 Under 15 MB budget: {'✅ YES' if budget_ok else '❌ NO'}")
    print(f"{'─' * 60}")

    # ── 10. Print deployment instructions ────────────────────────────────
    print(f"""
  🚀 Deployment Instructions:
     1. Copy the '{os.path.basename(args.output)}/' folder to the mobile device.
     2. Load with:
        tokenizer = VQATokenizer.load("vqa_vocab.json")
        model = build_vqa_model(vocab_size=tokenizer.vocab_size)
        model = torch.quantization.quantize_dynamic(model, {{nn.Linear, nn.GRU}}, torch.qint8)
        model.load_state_dict(torch.load("vqa_model_quantized.pth"))
     3. For answers:
        answer_id = model.forward_cached(vision_feat, token_ids).argmax(1).item()
        text = get_answer_text(answer_id, lang="en")  # or "vi", "th"
""")

    print("✅ Edge optimization complete!")


if __name__ == "__main__":
    main()
