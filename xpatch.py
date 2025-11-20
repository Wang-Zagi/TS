"""
xPatch model wrapper for integration with the TS forecasting framework.

The xPatch model uses decomposition-based approach with patching and multiple streams
for time series forecasting. It should use create_aligned_batches() for data loading
as it processes multivariate input in a unified manner.

Key features:
- Moving average-based decomposition (EMA/DEMA)
- Patch-based non-linear stream with CNN
- Linear stream with MLP
- RevIN normalization
"""

import torch
import torch.nn as nn
import sys
import os

# Add the xPatch directory to path
xpatch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'import_models', 'xPatch')
sys.path.insert(0, xpatch_dir)

from models.xPatch import Model as xPatchModel


class Config:
    """Configuration class for xPatch model."""
    def __init__(self, seq_len=192, pred_len=96, enc_in=21, patch_len=16, stride=8,
                 padding_patch='end', revin=True, ma_type='ema', alpha=0.2, beta=0.1):
        self.seq_len = seq_len  # lookback window
        self.pred_len = pred_len  # prediction length
        self.enc_in = enc_in  # input channels
        self.patch_len = patch_len  # patch length
        self.stride = stride  # stride for patching
        self.padding_patch = padding_patch  # padding strategy
        self.revin = revin  # use RevIN normalization
        self.ma_type = ma_type  # moving average type: 'reg', 'ema', 'dema'
        self.alpha = alpha  # smoothing factor for EMA
        self.beta = beta  # smoothing factor for DEMA


class xPatch(nn.Module):
    """
    xPatch model wrapper.
    
    This model combines decomposition-based approach with patching for time series forecasting.
    
    Data Source: Uses create_aligned_batches() because:
    - Processes all features together in a unified manner
    - Applies decomposition to the entire multivariate input
    - Similar to other multivariate models (Transformer, DLinear, TimesNet, TimeMixer)
    
    Args:
        seq_len (int): Input sequence length (default: 192)
        pred_len (int): Prediction length (default: 96)
        input_dim (int): Number of input features (default: 21)
        patch_len (int): Length of each patch (default: 16)
        stride (int): Stride for patching (default: 8)
        padding_patch (str): Padding strategy ('end' or None, default: 'end')
        revin (bool): Whether to use RevIN normalization (default: True)
        ma_type (str): Moving average type - 'reg', 'ema', 'dema' (default: 'ema')
        alpha (float): Smoothing factor for EMA (default: 0.2)
        beta (float): Smoothing factor for DEMA (default: 0.1)
    """
    
    def __init__(self, seq_len=192, pred_len=96, input_dim=21, patch_len=16, stride=8,
                 padding_patch='end', revin=True, ma_type='ema', alpha=0.2, beta=0.1):
        super(xPatch, self).__init__()
        
        # Store parameters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        
        # Create configuration
        config = Config(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=input_dim,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
            revin=revin,
            ma_type=ma_type,
            alpha=alpha,
            beta=beta
        )
        
        # Initialize the xPatch model
        self.model = xPatchModel(config)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, pred_len, 1) for temperature prediction
        """
        # Handle 2D input (batch_size, seq_len) by adding channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # xPatch model expects [Batch, Input, Channel]
        # x is already in this format from data loader
        
        # Forward through xPatch model
        output = self.model(x)
        
        # Extract only the temperature prediction (first channel)
        # Output shape: [Batch, pred_len, Channel]
        # We only need the temperature channel (first one)
        if output.shape[-1] > 1:
            output = output[:, :, 0:1]
        
        return output
    
    def predict(self, x):
        """
        Make predictions (for evaluation).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, pred_len, 1)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


if __name__ == '__main__':
    # Test the model
    print("Testing xPatch model wrapper...")
    
    batch_size = 4
    seq_len = 192
    pred_len = 96
    input_dim = 21
    
    # Create model
    model = xPatch(
        seq_len=seq_len,
        pred_len=pred_len,
        input_dim=input_dim,
        patch_len=16,
        stride=8
    )
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Test forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {pred_len}, 1)")
    
    assert output.shape == (batch_size, pred_len, 1), f"Output shape mismatch!"
    
    print("\n✅ xPatch model wrapper test passed!")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
