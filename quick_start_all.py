#!/usr/bin/env python
"""
Quick start test for all models including MixedPatch.
Verifies that all models can be created and shows their parameter counts.
"""

import torch
import sys

from dlinear import DLinear
from timesnet import TimesNet
from timemixer import TimeMixer
from itransformer import iTransformer
from patchtst import PatchTST
from mixedpatch import MixedPatch


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_creation(name, model_fn, use_mixed=False):
    """Test if a model can be created and return parameter count."""
    try:
        model = model_fn()
        params = count_parameters(model)
        print(f"✓ {name:20s} - {params:>12,} parameters")
        return True
    except Exception as e:
        print(f"✗ {name:20s} - Failed: {str(e)[:50]}")
        return False


def main():
    print("=" * 70)
    print("Quick Start Test - All Models")
    print("=" * 70)
    print()
    
    all_passed = True

    # Test DLinear
    all_passed &= test_model_creation(
        "DLinear",
        lambda: DLinear(
            seq_len=96,
            pred_len=96,
            input_dim=21,
            output_dim=1,
            kernel_size=25,
            individual=False
        )
    )
    
    # Test TimesNet
    all_passed &= test_model_creation(
        "TimesNet",
        lambda: TimesNet(
            seq_len=96,
            pred_len=96,
            input_dim=21,
            output_dim=1,
            d_model=128,
            d_ff=256,
            num_kernels=6,
            top_k=5,
            e_layers=3,
            dropout=0.1
        )
    )
    
    # Test TimeMixer
    all_passed &= test_model_creation(
        "TimeMixer",
        lambda: TimeMixer(
            seq_len=96,
            pred_len=96,
            input_dim=21,
            output_dim=1,
            d_model=64,
            d_ff=128,
            e_layers=2,
            dropout=0.1
        )
    )
    
    # Test iTransformer
    seq_lens = [96] + [286] * 10 + [24] * 10
    all_passed &= test_model_creation(
        "iTransformer",
        lambda: iTransformer(
            seq_len=96,
            pred_len=96,
            input_dim=21,
            output_dim=1,
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            use_mixed_batches=True,
            seq_lens=seq_lens
        )
    )
    
    # Test PatchTST
    all_passed &= test_model_creation(
        "PatchTST",
        lambda: PatchTST(
            seq_len=96,
            pred_len=96,
            patch_len=16,
            stride=8,
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1
        )
    )
    
    # Test MixedPatch (NEW)
    all_passed &= test_model_creation(
        "MixedPatch (NEW)",
        lambda: MixedPatch(
            seq_len=96,
            pred_len=96,
            patch_len=16,
            stride=8,
            d_model=128,
            nhead=8,
            num_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            use_mixed_batches=True,
            seq_lens=[96, 286, 24]
        )
    )
    
    print()
    print("=" * 70)
    if all_passed:
        print("All models created successfully! ✓")
    else:
        print("Some models failed to create. ✗")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
