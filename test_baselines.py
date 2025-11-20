"""
Test script for baseline models: DLinear, TimesNet, TimeMixer, PatchTST.
"""

import torch
from dlinear import DLinear
from timesnet import TimesNet
from timemixer import TimeMixer
from patchtst import PatchTST


def test_dlinear():
    """Test DLinear model."""
    print("\n" + "="*60)
    print("Testing DLinear Model")
    print("="*60)
    
    # Create model
    model = DLinear(
        seq_len=96,
        pred_len=96,
        input_dim=21,
        output_dim=1,
        kernel_size=25,
        individual=False
    )
    
    # Test forward pass
    batch_size = 4
    src = torch.randn(batch_size, 96, 21)
    
    output = model(src)
    assert output.shape == (batch_size, 96, 1), f"Expected shape {(batch_size, 96, 1)}, got {output.shape}"
    print(f"✓ Forward pass successful. Output shape: {output.shape}")
    
    # Test predict method
    predictions = model.predict(src)
    assert predictions.shape == (batch_size, 96, 1), f"Expected shape {(batch_size, 96, 1)}, got {predictions.shape}"
    print(f"✓ Predict method successful. Output shape: {predictions.shape}")
    
    # Test with individual=True
    model_individual = DLinear(
        seq_len=96,
        pred_len=96,
        input_dim=21,
        output_dim=1,
        kernel_size=25,
        individual=True
    )
    
    output_individual = model_individual(src)
    assert output_individual.shape == (batch_size, 96, 1), f"Expected shape {(batch_size, 96, 1)}, got {output_individual.shape}"
    print(f"✓ Individual mode successful. Output shape: {output_individual.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ DLinear has {num_params:,} trainable parameters")
    
    print("\n✅ All DLinear tests passed!")


def test_timesnet():
    """Test TimesNet model."""
    print("\n" + "="*60)
    print("Testing TimesNet Model")
    print("="*60)
    
    # Create model
    model = TimesNet(
        seq_len=96,
        pred_len=96,
        input_dim=21,
        output_dim=1,
        d_model=64,
        d_ff=64,
        num_kernels=6,
        top_k=5,
        e_layers=2,
        dropout=0.1
    )
    
    # Test forward pass
    batch_size = 4
    src = torch.randn(batch_size, 96, 21)
    
    output = model(src)
    assert output.shape == (batch_size, 96, 1), f"Expected shape {(batch_size, 96, 1)}, got {output.shape}"
    print(f"✓ Forward pass successful. Output shape: {output.shape}")
    
    # Test predict method
    predictions = model.predict(src)
    assert predictions.shape == (batch_size, 96, 1), f"Expected shape {(batch_size, 96, 1)}, got {predictions.shape}"
    print(f"✓ Predict method successful. Output shape: {predictions.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ TimesNet has {num_params:,} trainable parameters")
    
    print("\n✅ All TimesNet tests passed!")


def test_timemixer():
    """Test TimeMixer model."""
    print("\n" + "="*60)
    print("Testing TimeMixer Model")
    print("="*60)
    
    # Create model
    model = TimeMixer(
        seq_len=96,
        pred_len=96,
        input_dim=21,
        output_dim=1,
        d_model=64,
        d_ff=128,
        e_layers=2,
        kernel_size_list=[3, 5, 7],
        dropout=0.1
    )
    
    # Test forward pass
    batch_size = 4
    src = torch.randn(batch_size, 96, 21)
    
    output = model(src)
    assert output.shape == (batch_size, 96, 1), f"Expected shape {(batch_size, 96, 1)}, got {output.shape}"
    print(f"✓ Forward pass successful. Output shape: {output.shape}")
    
    # Test predict method
    predictions = model.predict(src)
    assert predictions.shape == (batch_size, 96, 1), f"Expected shape {(batch_size, 96, 1)}, got {predictions.shape}"
    print(f"✓ Predict method successful. Output shape: {predictions.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ TimeMixer has {num_params:,} trainable parameters")
    
    print("\n✅ All TimeMixer tests passed!")


def test_patchtst():
    """Test PatchTST model."""
    print("\n" + "="*60)
    print("Testing PatchTST Model")
    print("="*60)
    
    # Create model
    model = PatchTST(
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
    
    # Test forward pass with univariate input
    batch_size = 4
    src = torch.randn(batch_size, 192, 1)  # Univariate input (temperature only)
    
    output = model(src)
    assert output.shape == (batch_size, 96, 1), f"Expected shape {(batch_size, 96, 1)}, got {output.shape}"
    print(f"✓ Forward pass successful. Output shape: {output.shape}")
    
    # Test predict method
    predictions = model.predict(src)
    assert predictions.shape == (batch_size, 96, 1), f"Expected shape {(batch_size, 96, 1)}, got {predictions.shape}"
    print(f"✓ Predict method successful. Output shape: {predictions.shape}")
    
    # Test with 2D input (batch_size, seq_len)
    src_2d = torch.randn(batch_size, 192)
    output_2d = model(src_2d)
    assert output_2d.shape == (batch_size, 96, 1), f"Expected shape {(batch_size, 96, 1)}, got {output_2d.shape}"
    print(f"✓ 2D input handling successful. Output shape: {output_2d.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ PatchTST has {num_params:,} trainable parameters")
    
    # Test with different patch configurations
    model_small_patch = PatchTST(
        seq_len=96,
        pred_len=96,
        patch_len=8,
        stride=4,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )
    
    output_small = model_small_patch(src)
    assert output_small.shape == (batch_size, 96, 1), f"Expected shape {(batch_size, 96, 1)}, got {output_small.shape}"
    print(f"✓ Small patch configuration successful. Output shape: {output_small.shape}")
    
    print("\n✅ All PatchTST tests passed!")


def test_all_models_comparison():
    """Compare all models."""
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    batch_size = 4
    src = torch.randn(batch_size, 96, 21)
    src_univariate = torch.randn(batch_size, 192, 1)  # For PatchTST
    
    models = {
        'DLinear': DLinear(seq_len=96, pred_len=96, input_dim=21, output_dim=1),
        'DLinear (Individual)': DLinear(seq_len=96, pred_len=96, input_dim=21, output_dim=1, individual=True),
        'TimesNet': TimesNet(seq_len=96, pred_len=96, input_dim=21, output_dim=1),
        'TimeMixer': TimeMixer(seq_len=96, pred_len=96, input_dim=21, output_dim=1),
        'PatchTST': PatchTST(seq_len=96, pred_len=96, patch_len=16, stride=8),
    }
    
    print(f"\n{'Model':<25} {'Parameters':>15} {'Output Shape':>20}")
    print("-" * 60)
    
    for name, model in models.items():
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Use univariate input for PatchTST
        if name == 'PatchTST':
            output = model(src_univariate)
        else:
            output = model(src)
        print(f"{name:<25} {num_params:>15,} {str(output.shape):>20}")
    
    print("\n✅ Model comparison completed!")


if __name__ == '__main__':
    print("="*60)
    print("Running Baseline Model Tests")
    print("="*60)
    
    test_dlinear()
    test_timesnet()
    test_timemixer()
    test_patchtst()
    test_all_models_comparison()
    
    print("\n" + "="*60)
    print("All baseline model tests passed successfully!")
    print("="*60)
