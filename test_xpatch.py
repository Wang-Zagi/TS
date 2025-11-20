"""
Test script for xPatch model.

This test verifies that the xPatch model from import_models/xPatch can be properly
integrated and used for time series forecasting.

The xPatch model should use create_aligned_batches() for data loading because:
1. It processes multivariate input (all 21 features together)
2. It applies decomposition to the entire input
3. It follows a similar pattern to other multivariate models
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xpatch import xPatch


def test_xpatch_creation():
    """Test that xPatch model can be created."""
    print("Testing xPatch model creation...")
    
    model = xPatch(
        seq_len=192,
        pred_len=96,
        input_dim=21,
        patch_len=16,
        stride=8,
        revin=True,
        ma_type='ema'
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created successfully with {num_params:,} parameters")
    
    return model


def test_xpatch_forward():
    """Test forward pass with multivariate input."""
    print("\nTesting xPatch forward pass with multivariate input...")
    
    model = xPatch(
        seq_len=192,
        pred_len=96,
        input_dim=21,
        patch_len=16,
        stride=8
    )
    
    batch_size = 4
    
    # Create multivariate input (aligned batches format)
    # This is what create_aligned_batches() produces
    x = torch.randn(batch_size, 192, 21)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify output shape
    assert output.shape == (batch_size, 96, 1), f"Expected shape (4, 96, 1), got {output.shape}"
    print("✓ Forward pass successful!")
    
    return output


def test_xpatch_univariate():
    """Test forward pass with univariate input."""
    print("\nTesting xPatch forward pass with univariate input...")
    
    model = xPatch(
        seq_len=192,
        pred_len=96,
        input_dim=1,  # Single channel (temperature only)
        patch_len=16,
        stride=8
    )
    
    batch_size = 4
    
    # Create univariate input
    x = torch.randn(batch_size, 192, 1)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify output shape
    assert output.shape == (batch_size, 96, 1), f"Expected shape (4, 96, 1), got {output.shape}"
    print("✓ Univariate forward pass successful!")
    
    return output


def test_xpatch_different_configs():
    """Test xPatch with different configurations."""
    print("\nTesting xPatch with different configurations...")
    
    configs = [
        {'ma_type': 'reg', 'revin': False},  # No decomposition, no RevIN
        {'ma_type': 'ema', 'alpha': 0.3},     # EMA decomposition
        {'ma_type': 'dema', 'alpha': 0.2, 'beta': 0.1},  # DEMA decomposition
        {'patch_len': 8, 'stride': 4},        # Smaller patches
    ]
    
    batch_size = 4
    x = torch.randn(batch_size, 192, 21)
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        
        model = xPatch(
            seq_len=192,
            pred_len=96,
            input_dim=21,
            patch_len=config.get('patch_len', 16),
            stride=config.get('stride', 8),
            revin=config.get('revin', True),
            ma_type=config.get('ma_type', 'ema'),
            alpha=config.get('alpha', 0.2),
            beta=config.get('beta', 0.1)
        )
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 96, 1), f"Output shape mismatch in config {i+1}"
        print(f"✓ Configuration {i+1} successful!")
    
    print("\n✓ All configurations passed!")


def test_xpatch_training_step():
    """Test a training step."""
    print("\nTesting xPatch training step...")
    
    model = xPatch(
        seq_len=192,
        pred_len=96,
        input_dim=21,
        patch_len=16,
        stride=8
    )
    
    batch_size = 4
    
    # Create input and target
    x = torch.randn(batch_size, 192, 21)
    y_target = torch.randn(batch_size, 96, 1)
    
    # Training step
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y_target)
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
    print("✓ Training step successful!")
    
    return loss.item()


def test_xpatch_predict_method():
    """Test the predict method."""
    print("\nTesting xPatch predict method...")
    
    model = xPatch(
        seq_len=192,
        pred_len=96,
        input_dim=21,
        patch_len=16,
        stride=8
    )
    
    batch_size = 4
    x = torch.randn(batch_size, 192, 21)
    
    # Test predict method
    predictions = model.predict(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    assert predictions.shape == (batch_size, 96, 1), f"Expected shape (4, 96, 1), got {predictions.shape}"
    print("✓ Predict method successful!")
    
    return predictions


def print_model_summary():
    """Print a summary of the xPatch model and its data source."""
    print("\n" + "="*60)
    print("xPatch Model Summary")
    print("="*60)
    print("\n📊 Data Source: create_aligned_batches()")
    print("\nReason:")
    print("  - Processes multivariate input (all 21 features)")
    print("  - Applies decomposition to entire input")
    print("  - Similar architecture to Transformer, DLinear, TimesNet, TimeMixer")
    print("  - Expects aligned frequency data at 30-minute intervals")
    print("\n🏗️  Model Architecture:")
    print("  - Decomposition: Moving average-based (EMA/DEMA)")
    print("  - Non-linear stream: Patch-based with CNN")
    print("  - Linear stream: MLP-based trend modeling")
    print("  - Normalization: RevIN (reversible instance normalization)")
    print("\n📥 Input: (batch_size, 192, 21)")
    print("  - 192 timesteps at 30-minute intervals")
    print("  - 21 features (temperature + weather variables)")
    print("\n📤 Output: (batch_size, 96, 1)")
    print("  - 96 timesteps prediction")
    print("  - 1 feature (temperature)")
    print("="*60)


if __name__ == '__main__':
    print("=" * 60)
    print("xPatch Model Tests")
    print("=" * 60)
    
    # Print model summary
    print_model_summary()
    
    # Run tests
    test_xpatch_creation()
    test_xpatch_forward()
    test_xpatch_univariate()
    test_xpatch_different_configs()
    test_xpatch_training_step()
    test_xpatch_predict_method()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
    print("\nConclusion:")
    print("The xPatch model should use create_aligned_batches() for data loading.")
    print("This function provides multivariate data at aligned 30-minute frequency,")
    print("which matches the model's expectation for processing all features together.")
    print("=" * 60)
