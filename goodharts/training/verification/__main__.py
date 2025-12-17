#!/usr/bin/env python3
"""
Model verification suite entry point.

Usage:
    python -m goodharts.training.verification
    python -m goodharts.training.verification --steps 500 --verbose
"""
import argparse
import torch

from .directional import test_directional_accuracy
from .survival import compare_behaviors


def check_gpu():
    """Check GPU availability and current setup."""
    print("=" * 60)
    print("GPU STATUS")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"[OK] CUDA available")
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Current device: {torch.cuda.current_device()}")
    else:
        print("[X] CUDA not available")
        if hasattr(torch.version, 'hip') and torch.version.hip:
            print(f"  (ROCm detected: {torch.version.hip})")
        print("  Training will use CPU")
    print()


def main():
    parser = argparse.ArgumentParser(description="Verify trained model fitness")
    parser.add_argument('--steps', type=int, default=500, help='Simulation steps per run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--skip-gpu', action='store_true', help='Skip GPU check')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  MODEL VERIFICATION SUITE")
    print("=" * 60 + "\n")
    
    # GPU status
    if not args.skip_gpu:
        check_gpu()
    
    # Test model directional accuracy
    print("=" * 60)
    print("DIRECTIONAL ACCURACY TESTS")
    print("=" * 60)
    
    gt_acc = test_directional_accuracy('models/ground_truth.pth', 'Ground Truth Model')
    proxy_acc = test_directional_accuracy('models/proxy_trained.pth', 'Proxy Model')
    
    # Behavior comparison
    compare_behaviors(steps=args.steps, verbose=args.verbose)
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    
    # Final verdict
    if gt_acc >= 0.75:
        print("[OK] Models appear to be trained correctly")
        print("  Run 'python main.py --learned' for visual demo")
    else:
        print("[WARN] Models may need retraining")
        print("  Run 'python -m goodharts.training.train_ppo --mode both --epochs 100'")

if __name__ == "__main__":
    main()
