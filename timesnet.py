"""
TimesNet model for time series forecasting.
TimesNet uses 2D convolution on transformed 1D time series to capture multi-period patterns.

Reference:
TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis (ICLR 2023)
https://arxiv.org/abs/2210.02186
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Inception_Block_V1(nn.Module):
    """
    Inception block with multiple kernel sizes for multi-scale feature extraction.
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=2*i+1, padding=i)
            )
        self.kernels = nn.ModuleList(kernels)
        
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, height, width)
        Returns:
            (batch, out_channels, height, width)
        """
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class FFT_for_Period(nn.Module):
    """
    FFT-based period detection module.
    """
    def __init__(self, seq_len, k=5):
        super().__init__()
        self.seq_len = seq_len
        self.k = k

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            Top-k periods
        """
        # FFT
        xf = torch.fft.rfft(x, dim=1)
        
        # Find periods by FFT amplitude
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0  # Remove DC component
        
        # Top-k frequencies
        _, top_list = torch.topk(frequency_list, self.k)
        top_list = top_list.detach().cpu().numpy()
        
        # Convert frequency to period
        period = x.shape[1] // top_list
        
        return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    """
    TimesBlock: 2D variation modeling on temporal data.
    """
    def __init__(self, seq_len, pred_len, k, d_model, d_ff, num_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = k
        
        # Parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, N = x.shape
        
        # Period detection using FFT
        period_list, period_weight = FFT_for_Period(T, self.k)(x)
        
        res = []
        for i in range(self.k):
            period = period_list[i]
            
            # Padding to make divisible by period
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros([B, (length - T), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x
            
            # Reshape to 2D: (batch, length, d_model) -> (batch, num_periods, period, d_model)
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            
            # 2D conv: (batch, d_model, num_periods, period) -> (batch, d_model, num_periods, period)
            out = self.conv(out)
            
            # Reshape back to 1D
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :T, :])
        
        # Adaptive aggregation
        res = torch.stack(res, dim=-1)
        
        # Period-wise weights
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        # Residual connection
        res = res + x
        
        return res


class TimesNet(nn.Module):
    """
    TimesNet: Multi-period modeling for time series forecasting.
    
    Architecture:
    - Embedding layer
    - Multiple TimesBlocks for multi-period feature extraction
    - Projection layer for prediction
    
    Args:
        seq_len: Input sequence length (e.g., 96)
        pred_len: Prediction sequence length (e.g., 96)
        input_dim: Number of input features (e.g., 21)
        output_dim: Number of output features (e.g., 1)
        d_model: Model dimension (default: 64)
        d_ff: FFN dimension (default: 64)
        num_kernels: Number of kernels in Inception block (default: 6)
        top_k: Number of top periods to use (default: 5)
        e_layers: Number of encoder layers (default: 2)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        input_dim: int = 21,
        output_dim: int = 1,
        d_model: int = 64,
        d_ff: int = 64,
        num_kernels: int = 6,
        top_k: int = 5,
        e_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_type = 'TimesNet'
        
        # Embedding
        self.enc_embedding = nn.Linear(input_dim, d_model)
        
        # Encoder blocks
        self.model = nn.ModuleList([
            TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
            for _ in range(e_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Projection
        self.projection = nn.Linear(d_model, output_dim)
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
        
        # TimesBlocks
        for layer in self.model:
            enc_out = layer(enc_out)
        
        # Layer norm
        enc_out = self.layer_norm(enc_out)
        
        # For forecasting, we use a linear projection to extend sequence to pred_len
        # First transpose: (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        enc_out = enc_out.permute(0, 2, 1)
        
        # Apply adaptive pooling to get pred_len timesteps
        dec_out = F.adaptive_avg_pool1d(enc_out, self.pred_len)
        
        # Transpose back: (batch, d_model, pred_len) -> (batch, pred_len, d_model)
        dec_out = dec_out.permute(0, 2, 1)
        
        # Project to output dimension
        output = self.projection(dec_out)
        
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
