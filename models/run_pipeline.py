"""
Master Pipeline Runner — Step 2 + Step 3
=========================================
Runs Step 2 (stronger backbone training) followed automatically by Step 3
(strict held-out test evaluation) using the best checkpoint from Step 2.

Usage:
  python run_pipeline.py [--epochs 25] [--batch-size 16] [--skip-effnet]
                         [--convitx-val-acc 0.76] [--convitx-val-f1 0.74]

All other arguments are forwarded to train_step2.py and evaluate_step3.py.
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(cmd, description):
    print(f"\n{'#' * 65}")
    print(f"  {description}")
    print(f"{'#' * 65}\n")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"\n[ERROR] '{description}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)


def parse_args():
    p = argparse.ArgumentParser(description="Run Step 2 + Step 3 pipeline end-to-end")
    p.add_argument("--epochs",           type=int,   default=25)
    p.add_argument("--batch-size",       type=int,   default=16)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--patience",         type=int,   default=7)
    p.add_argument("--img-size",         type=int,   default=224)
    p.add_argument("--num-workers",      type=int,   default=0)
    p.add_argument("--skip-effnet",      action="store_true",
                   help="Skip EfficientNet-B3 (faster, ResNet50 only)")
    p.add_argument("--convitx-val-acc",  type=float, default=None,
                   help="Your known ConViTX best val accuracy (e.g. 0.76)")
    p.add_argument("--convitx-val-f1",   type=float, default=None,
                   help="Your known ConViTX best val macro-F1  (e.g. 0.74)")
    return p.parse_args()


def main():
    args = parse_args()
    python = sys.executable

    # ---- Build Step 2 command ----
    step2_cmd = [
        python, os.path.join(SCRIPT_DIR, "train_step2.py"),
        "--epochs",      str(args.epochs),
        "--batch-size",  str(args.batch_size),
        "--lr",          str(args.lr),
        "--seed",        str(args.seed),
        "--patience",    str(args.patience),
        "--img-size",    str(args.img_size),
        "--num-workers", str(args.num_workers),
    ]
    if args.skip_effnet:
        step2_cmd.append("--skip-effnet")
    if args.convitx_val_acc is not None:
        step2_cmd += ["--convitx-val-acc", str(args.convitx_val_acc)]
    if args.convitx_val_f1 is not None:
        step2_cmd += ["--convitx-val-f1", str(args.convitx_val_f1)]

    run(step2_cmd, "STEP 2 — Stronger Backbone Training")

    # ---- Build Step 3 command  (auto-reads step2_best_model.json) ----
    step3_cmd = [
        python, os.path.join(SCRIPT_DIR, "evaluate_step3.py"),
        "--img-size",    str(args.img_size),
        "--batch-size",  str(args.batch_size),
        "--num-workers", str(args.num_workers),
    ]
    run(step3_cmd, "STEP 3 — Strict Held-Out Test Evaluation")

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print("  Outputs:")
    print("  models/best_resnet50_step2.pth")
    print("  models/best_efficientnet_b3_step2.pth   (if ran)")
    print("  models/test_metrics.json")
    print("  models/classification_report.txt")
    print("  models/confusion_matrix_test.png")
    print("  step2_results.md  (project root)")
    print("  step3_summary.md  (project root)")
    print("=" * 65)


if __name__ == "__main__":
    main()
