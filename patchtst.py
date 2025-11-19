"""
PatchTST model for time series forecasting.
PatchTST is a channel-independent model that uses patching and Transformer for univariate forecasting.

Reference:
A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (ICLR 2023)
https://arxiv.org/abs/2211.14730

Key features:
- Channel independence: Each channel/variable is processed separately
- Patching: Divides time series into patches for more efficient computation
- Transformer encoder: Captures dependencies between patches
- Designed for univariate forecasting (single variable)
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer that divides time series into patches and projects them.
    """
    def __init__(self, patch_len, stride, d_model, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        
        # Linear projection from patch to d_model
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, 1) - univariate time series
        Returns:
            (batch_size, num_patches, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Unfold to create patches: (batch_size, num_patches, patch_len)
        # We process each channel independently, so we work on (batch, seq_len)
        x = x.squeeze(-1)  # (batch_size, seq_len)
        
        # Create patches using unfold
        # unfold(dimension, size, step)
        num_patches = (seq_len - self.patch_len) // self.stride + 1
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches: (batch_size, num_patches, patch_len)
        
        # Project patches to d_model
        x = self.projection(patches)  # (batch_size, num_patches, d_model)
        x = self.dropout(x)
        
        return x, num_patches


class PositionalEncoding(nn.Module):
    """
    Positional encoding for patch positions.
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_patches, d_model)
        Returns:
            (batch_size, num_patches, d_model)
        """
        num_patches = x.size(1)
        return x + self.pe[:num_patches, :].unsqueeze(0)


class PatchTST(nn.Module):
    """
    PatchTST: Patch Time Series Transformer for univariate forecasting.
    
    Architecture:
    - Patch embedding: Divides time series into patches
    - Positional encoding: Adds position information
    - Transformer encoder: Captures dependencies between patches
    - Projection head: Maps to prediction length
    
    Args:
        seq_len: Input sequence length (e.g., 192)
        pred_len: Prediction sequence length (e.g., 96)
        patch_len: Length of each patch (default: 16)
        stride: Stride for patching (default: 8)
        d_model: Model dimension (default: 128)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 3)
        dim_feedforward: Feedforward dimension (default: 512)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        seq_len: int = 192,
        pred_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_len, stride, d_model, dropout)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection head to prediction length
        # We flatten the patch representations and project to pred_len
        self.projection = nn.Linear(self.num_patches * d_model, pred_len)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: Input sequence (batch_size, seq_len, 1) - univariate time series
        
        Returns:
            Output sequence (batch_size, pred_len, 1)
        """
        # Ensure input is univariate (single channel)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch_size, seq_len) -> (batch_size, seq_len, 1)
        
        batch_size = x.size(0)
        
        # Patch embedding
        x, num_patches = self.patch_embedding(x)  # (batch_size, num_patches, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, num_patches, d_model)
        
        # Flatten patches
        x = x.reshape(batch_size, -1)  # (batch_size, num_patches * d_model)
        
        # Project to prediction length
        x = self.projection(x)  # (batch_size, pred_len)
        
        # Reshape to (batch_size, pred_len, 1) for consistency
        output = x.unsqueeze(-1)
        
        return output
    
    def predict(self, src, future_len: int = None):
        """
        Prediction interface for compatibility with training script.
        
        Args:
            src: Input sequence (batch_size, seq_len, 1)
            future_len: Number of future steps to predict (not used, model uses pred_len)
        
        Returns:
            Predictions (batch_size, pred_len, 1)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(src)
