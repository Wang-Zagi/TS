"""
TimeMixer model for time series forecasting.
TimeMixer uses multi-scale mixing in both past and future to model temporal patterns.

Reference:
TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting (ICLR 2024)
https://arxiv.org/abs/2405.14616
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecompMulti(nn.Module):
    """
    Multi-scale series decomposition.
    """
    def __init__(self, kernel_size_list):
        super().__init__()
        self.kernel_size_list = kernel_size_list
        self.moving_avgs = nn.ModuleList([
            nn.AvgPool1d(kernel_size=k, stride=1, padding=0)
            for k in kernel_size_list
        ])

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, features)
        Returns:
            seasonal_list: list of seasonal components
            trend_list: list of trend components
        """
        seasonal_list = []
        trend_list = []
        
        for moving_avg, kernel_size in zip(self.moving_avgs, self.kernel_size_list):
            # Padding
            front = x[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
            x_padded = torch.cat([front, x, end], dim=1)
            
            # Apply moving average
            x_padded = x_padded.permute(0, 2, 1)
            trend = moving_avg(x_padded)
            trend = trend.permute(0, 2, 1)
            
            seasonal = x - trend
            
            seasonal_list.append(seasonal)
            trend_list.append(trend)
        
        return seasonal_list, trend_list


class MixingLayer(nn.Module):
    """
    Mixing layer for temporal and feature mixing.
    """
    def __init__(self, seq_len, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Temporal mixing (operates on d_model dimension)
        self.temporal_mixing = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Feature mixing (operates on seq_len dimension)
        self.feature_mixing = nn.Sequential(
            nn.Linear(seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, seq_len),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        # Temporal mixing (operates on the last dimension)
        x_temporal = self.temporal_mixing(x)
        x = self.norm1(x + x_temporal)
        
        # Feature mixing (operates across time dimension)
        # Transpose for feature-wise operation
        x_t = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x_feature = self.feature_mixing(x_t)
        x_feature = x_feature.transpose(1, 2)  # (batch, seq_len, d_model)
        x = self.norm2(x + x_feature)
        
        return x


class PastDecomposableMixing(nn.Module):
    """
    Past decomposable mixing module.
    """
    def __init__(self, seq_len, pred_len, d_model, d_ff, kernel_size_list, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Multi-scale decomposition
        self.decomp = SeriesDecompMulti(kernel_size_list)
        
        # Mixing layers for each scale
        self.mixing_layers = nn.ModuleList([
            MixingLayer(seq_len, d_model, d_ff, dropout)
            for _ in kernel_size_list
        ])
        
        # Aggregation
        self.aggregation = nn.Linear(len(kernel_size_list) * d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        # Decompose at multiple scales
        seasonal_list, trend_list = self.decomp(x)
        
        # Process each scale
        seasonal_outputs = []
        trend_outputs = []
        
        for i, (seasonal, trend) in enumerate(zip(seasonal_list, trend_list)):
            # Mix seasonal and trend separately
            seasonal_out = self.mixing_layers[i](seasonal)
            trend_out = self.mixing_layers[i](trend)
            
            seasonal_outputs.append(seasonal_out)
            trend_outputs.append(trend_out)
        
        # Concatenate and aggregate
        seasonal_cat = torch.cat(seasonal_outputs, dim=-1)
        trend_cat = torch.cat(trend_outputs, dim=-1)
        
        seasonal_agg = self.aggregation(seasonal_cat)
        trend_agg = self.aggregation(trend_cat)
        
        # Combine seasonal and trend
        output = seasonal_agg + trend_agg
        
        return output


class TimeMixer(nn.Module):
    """
    TimeMixer: Multi-scale mixing for time series forecasting.
    
    Architecture:
    - Input embedding
    - Multi-scale decomposable mixing blocks
    - Output projection
    
    Args:
        seq_len: Input sequence length (e.g., 192)
        pred_len: Prediction sequence length (e.g., 96)
        input_dim: Number of input features (e.g., 21)
        output_dim: Number of output features (e.g., 1)
        d_model: Model dimension (default: 64)
        d_ff: FFN dimension (default: 128)
        e_layers: Number of encoder layers (default: 2)
        kernel_size_list: List of kernel sizes for multi-scale decomposition (default: [3, 5, 7])
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        seq_len: int = 192,
        pred_len: int = 96,
        input_dim: int = 21,
        output_dim: int = 1,
        d_model: int = 64,
        d_ff: int = 128,
        e_layers: int = 2,
        kernel_size_list: list = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if kernel_size_list is None:
            kernel_size_list = [3, 5, 7]
        
        # Input embedding
        self.enc_embedding = nn.Linear(input_dim, d_model)
        
        # Past decomposable mixing blocks
        self.mixing_blocks = nn.ModuleList([
            PastDecomposableMixing(seq_len, pred_len, d_model, d_ff, kernel_size_list, dropout)
            for _ in range(e_layers)
        ])
        
        # Prediction head
        self.prediction_head = nn.Linear(d_model, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
        
        Returns:
            Output sequence (batch_size, pred_len, output_dim)
        """
        # Embedding
        enc_out = self.enc_embedding(x)
        enc_out = self.dropout(enc_out)
        
        # Apply mixing blocks
        for mixing_block in self.mixing_blocks:
            enc_out = mixing_block(enc_out)
        
        # Prediction
        # Transpose to (batch, d_model, seq_len) for temporal projection
        enc_out = enc_out.transpose(1, 2)
        
        # Project from seq_len to pred_len
        dec_out = F.adaptive_avg_pool1d(enc_out, self.pred_len)
        
        # Transpose back to (batch, pred_len, d_model)
        dec_out = dec_out.transpose(1, 2)
        
        # Project to output dimension
        output = self.prediction_head(dec_out)
        
        return output

    def predict(self, src, future_len: int = None):
        """
        Prediction interface for compatibility with training script.
        
        Args:
            src: Input sequence (batch_size, seq_len, input_dim)
            future_len: Number of future steps to predict (not used, model uses pred_len)
        
        Returns:
            Predictions (batch_size, pred_len, output_dim)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(src)
