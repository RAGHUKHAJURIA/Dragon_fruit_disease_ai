"""
Export ConViTX model for edge deployment.
Applies dynamic quantization (FP32 -> INT8) to linear layers for compression
and exports to ONNX format.
"""
import os
import argparse
import torch
import torch.onnx
from convitx import build_convitx_base

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="best_convitx.pth", help="Path to trained ConViTX weights")
    parser.add_argument("--num-classes", type=int, default=6, help="Number of classes")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Model path {args.model_path} doesn't exist.")
        return
        
    print(f"Loading {args.model_path} ...")
    model = build_convitx_base(num_classes=args.num_classes, enforce_budget=False)
    
    # Load weights
    try:
        state_dict = torch.load(args.model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 1. Dynamic Quantization (to INT8 for Linear layers)
    print("\nApplying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    quantized_path = os.path.join(args.output_dir, "best_convitx_quantized.pth")
    torch.save(quantized_model.state_dict(), quantized_path)
    
    # Print size difference
    orig_size = os.path.getsize(args.model_path) / (1024 * 1024)
    quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
    print(f"Original size:  {orig_size:.2f} MB")
    print(f"Quantized size: {quant_size:.2f} MB")
    print(f"Saved quantized model to {quantized_path}")
    
    # 2. Export to ONNX
    print("\nExporting to ONNX...")
    onnx_path = os.path.join(args.output_dir, "best_convitx.onnx")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    try:
        torch.onnx.export(
            model,               
            dummy_input,         
            onnx_path,           
            export_params=True,  
            opset_version=14,    
            do_constant_folding=True, 
            input_names=['input'],    
            output_names=['output'],  
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Saved ONNX model to {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")

if __name__ == "__main__":
    main()
