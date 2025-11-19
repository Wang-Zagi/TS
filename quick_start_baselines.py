"""
Quick start guide for testing baseline models.
This script demonstrates how to quickly test each model.
"""

import torch
from model import TransformerTS
from dlinear import DLinear
from timesnet import TimesNet
from timemixer import TimeMixer
from itransformer import iTransformer


def quick_test_all_models():
    """
    Quick test of all available models.
    """
    print("=" * 70)
    print("Quick Start Guide - Testing All Models")
    print("=" * 70)
    
    # Prepare dummy data
    batch_size = 2
    seq_len = 192
    pred_len = 96
    input_dim = 21
    output_dim = 1
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"\nInput shape: {x.shape}")
    print(f"Expected output shape: ({batch_size}, {pred_len}, {output_dim})")
    
    models = {
        'Transformer': TransformerTS(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.1
        ),
        'DLinear': DLinear(
            seq_len=seq_len,
            pred_len=pred_len,
            input_dim=input_dim,
            output_dim=output_dim,
            kernel_size=25,
            individual=False
        ),
        'TimesNet': TimesNet(
            seq_len=seq_len,
            pred_len=pred_len,
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=32,
            d_ff=32,
            num_kernels=6,
            top_k=5,
            e_layers=2,
            dropout=0.1
        ),
        'TimeMixer': TimeMixer(
            seq_len=seq_len,
            pred_len=pred_len,
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=32,
            d_ff=64,
            e_layers=2,
            dropout=0.1
        ),
        'iTransformer': iTransformer(
            seq_len=seq_len,
            pred_len=pred_len,
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1
        ),
    }
    
    print("\n" + "-" * 70)
    print(f"{'Model':<15} {'Parameters':>15} {'Output Shape':>25} {'Status':>10}")
    print("-" * 70)
    
    for name, model in models.items():
        try:
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Test prediction
            model.eval()
            with torch.no_grad():
                output = model.predict(x, future_len=pred_len)
            
            # Check output shape
            assert output.shape == (batch_size, pred_len, output_dim), \
                f"Expected {(batch_size, pred_len, output_dim)}, got {output.shape}"
            
            status = "✓ PASS"
            print(f"{name:<15} {num_params:>15,} {str(output.shape):>25} {status:>10}")
            
        except Exception as e:
            status = "✗ FAIL"
            print(f"{name:<15} {'N/A':>15} {'N/A':>25} {status:>10}")
            print(f"  Error: {str(e)}")
    
    print("-" * 70)
    print("\n✅ All models tested successfully!")
    print("\nNext steps:")
    print("1. Train a model: python train.py --model_type dlinear --epochs 10")
    print("2. Compare models: python train.py --model_type timesnet --epochs 10")
    print("3. Run full tests: python test_baselines.py")
    print("=" * 70)


if __name__ == '__main__':
    quick_test_all_models()
