"""
Test script for MixedPatch model.
Verifies that the model can be created, and a forward pass works.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixedpatch import MixedPatch


def test_mixedpatch_creation():
    """Test that MixedPatch model can be created."""
    print("Testing MixedPatch model creation...")
    
    model = MixedPatch(
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
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created successfully with {num_params:,} parameters")
    
    return model


def test_mixedpatch_forward_dict():
    """Test forward pass with dictionary input (mixed batches)."""
    print("\nTesting MixedPatch forward pass with dict input...")
    
    model = MixedPatch(
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
    
    batch_size = 4
    
    # Create mixed batch input
    X_dict = {
        'T_30min_hist': torch.randn(batch_size, 96, 1),
        'A_10min_hist': torch.randn(batch_size, 286, 10),
        'B_120min_hist': torch.randn(batch_size, 24, 10)
    }
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(X_dict)
    
    print(f"Input T shape: {X_dict['T_30min_hist'].shape}")
    print(f"Input A shape: {X_dict['A_10min_hist'].shape}")
    print(f"Input B shape: {X_dict['B_120min_hist'].shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify output shape
    assert output.shape == (batch_size, 96, 1), f"Expected shape (4, 96, 1), got {output.shape}"
    print("Forward pass successful!")
    
    return output


def test_mixedpatch_training_step():
    """Test a training step."""
    print("\nTesting MixedPatch training step...")
    
    model = MixedPatch(
        seq_len=96,
        pred_len=96,
        patch_len=16,
        stride=8,
        d_model=64,  # Smaller for faster testing
        nhead=4,
        num_layers=1,
        dim_feedforward=256,
        dropout=0.1,
        use_mixed_batches=True,
        seq_lens=[96, 286, 24]
    )
    
    batch_size = 4
    
    # Create mixed batch input
    X_dict = {
        'T_30min_hist': torch.randn(batch_size, 96, 1),
        'A_10min_hist': torch.randn(batch_size, 286, 10),
        'B_120min_hist': torch.randn(batch_size, 24, 10)
    }
    y_target = torch.randn(batch_size, 96, 1)
    
    # Training step
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    optimizer.zero_grad()
    output = model(X_dict)
    loss = criterion(output, y_target)
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
    print("Training step successful!")
    
    return loss.item()


if __name__ == '__main__':
    print("=" * 60)
    print("MixedPatch Model Tests")
    print("=" * 60)
    
    # Run tests
    test_mixedpatch_creation()
    test_mixedpatch_forward_dict()
    test_mixedpatch_training_step()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
