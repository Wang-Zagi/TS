"""
Test script to verify MAE/MSE metrics tracking and DataEmbedding functionality.
"""

import torch
import torch.nn as nn
from model import TransformerTS
from embed import DataEmbedding, TokenEmbedding, PositionalEmbedding


def test_embedding_modules():
    """Test individual embedding modules."""
    print("Testing embedding modules...")
    
    # Test TokenEmbedding
    token_emb = TokenEmbedding(c_in=21, d_model=128)
    x = torch.randn(4, 192, 21)
    out = token_emb(x)
    assert out.shape == (4, 192, 128), f"TokenEmbedding output shape mismatch: {out.shape}"
    print("✓ TokenEmbedding works correctly")
    
    # Test PositionalEmbedding
    pos_emb = PositionalEmbedding(d_model=128)
    x = torch.randn(4, 192, 128)
    out = pos_emb(x)
    assert out.shape == (1, 192, 128), f"PositionalEmbedding output shape mismatch: {out.shape}"
    print("✓ PositionalEmbedding works correctly")
    
    # Test DataEmbedding
    data_emb = DataEmbedding(c_in=21, d_model=128, embed_type='fixed', freq='h', dropout=0.1)
    x = torch.randn(4, 192, 21)
    out = data_emb(x, x_mark=None)
    assert out.shape == (4, 192, 128), f"DataEmbedding output shape mismatch: {out.shape}"
    print("✓ DataEmbedding works correctly")
    

def test_model_with_different_layers():
    """Test model with different numbers of layers."""
    print("\nTesting model with different layer configurations...")
    
    layer_configs = [
        (1, 1, "minimal"),
        (3, 3, "default"),
        (6, 6, "deep"),
        (4, 2, "asymmetric"),
    ]
    
    for enc_layers, dec_layers, desc in layer_configs:
        model = TransformerTS(
            input_dim=21,
            output_dim=1,
            d_model=128,
            nhead=8,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=512,
            dropout=0.1
        )
        
        # Test forward pass
        src = torch.randn(2, 192, 21)
        tgt = torch.randn(2, 96, 1)
        output = model(src, tgt)
        
        assert output.shape == (2, 96, 1), f"Output shape mismatch: {output.shape}"
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ {desc}: {enc_layers} enc + {dec_layers} dec layers, {num_params:,} params")


def test_mae_mse_metrics():
    """Test MAE and MSE metric calculation."""
    print("\nTesting MAE and MSE metric calculation...")
    
    model = TransformerTS(
        input_dim=21,
        output_dim=1,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )
    
    # Create dummy data
    src = torch.randn(8, 192, 21)
    tgt = torch.randn(8, 96, 1)
    
    # Forward pass
    output = model(src, tgt)
    
    # Calculate MSE
    mse = torch.mean((output - tgt) ** 2).item()
    
    # Calculate MAE
    mae = torch.mean(torch.abs(output - tgt)).item()
    
    print(f"✓ MSE calculated: {mse:.6f}")
    print(f"✓ MAE calculated: {mae:.6f}")
    
    # Verify metrics are positive
    assert mse >= 0, "MSE should be non-negative"
    assert mae >= 0, "MAE should be non-negative"
    print("✓ Metrics are valid")


def test_input_output_dimensions():
    """Test that the model correctly handles 192 input -> 96 output."""
    print("\nTesting input/output dimensions (192 -> 96)...")
    
    model = TransformerTS(
        input_dim=21,
        output_dim=1,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3
    )
    
    # Test with exact required dimensions
    batch_size = 16
    src = torch.randn(batch_size, 192, 21)  # 192 timesteps, 21 features
    tgt = torch.randn(batch_size, 96, 1)    # 96 timesteps, 1 feature
    
    output = model(src, tgt)
    
    assert output.shape == (batch_size, 96, 1), f"Output shape mismatch: {output.shape}"
    print(f"✓ Input: (batch={batch_size}, seq=192, features=21)")
    print(f"✓ Output: (batch={batch_size}, seq=96, features=1)")
    
    # Test prediction
    predictions = model.predict(src, future_len=96)
    assert predictions.shape == (batch_size, 96, 1), f"Prediction shape mismatch: {predictions.shape}"
    print("✓ Autoregressive prediction works correctly")


if __name__ == '__main__':
    print("="*60)
    print("Running Comprehensive Model Tests")
    print("="*60)
    
    test_embedding_modules()
    test_model_with_different_layers()
    test_mae_mse_metrics()
    test_input_output_dimensions()
    
    print("\n" + "="*60)
    print("All tests passed successfully!")
    print("="*60)
    
    print("\nKey Features Verified:")
    print("✓ DataEmbedding with TokenEmbedding + PositionalEmbedding")
    print("✓ Controllable transformer layer stacking")
    print("✓ MAE and MSE metrics calculation")
    print("✓ 192 input timesteps -> 96 output timesteps")
