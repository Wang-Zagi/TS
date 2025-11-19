"""
Simple test script to verify the Transformer model implementation.
"""

import torch
from model import TransformerTS

def test_model_forward():
    """Test forward pass of the model."""
    print("Testing model forward pass...")
    
    # Create model
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
    
    # Create dummy input
    batch_size = 4
    src_len = 192
    tgt_len = 96
    
    src = torch.randn(batch_size, src_len, 21)
    tgt = torch.randn(batch_size, tgt_len, 1)
    
    # Forward pass
    output = model(src, tgt)
    
    assert output.shape == (batch_size, tgt_len, 1), f"Expected shape {(batch_size, tgt_len, 1)}, got {output.shape}"
    print(f"✓ Forward pass successful. Output shape: {output.shape}")


def test_model_predict():
    """Test autoregressive prediction."""
    print("\nTesting autoregressive prediction...")
    
    # Create model
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
    
    # Create dummy input
    batch_size = 4
    src_len = 192
    future_len = 96
    
    src = torch.randn(batch_size, src_len, 21)
    
    # Predict
    predictions = model.predict(src, future_len=future_len)
    
    assert predictions.shape == (batch_size, future_len, 1), f"Expected shape {(batch_size, future_len, 1)}, got {predictions.shape}"
    print(f"✓ Prediction successful. Output shape: {predictions.shape}")


def test_model_parameters():
    """Test model parameter count."""
    print("\nTesting model parameters...")
    
    model = TransformerTS(
        input_dim=21,
        output_dim=1,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model has {num_params:,} trainable parameters")


if __name__ == '__main__':
    print("="*50)
    print("Running Transformer Model Tests")
    print("="*50)
    
    test_model_forward()
    test_model_predict()
    test_model_parameters()
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
