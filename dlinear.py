"""
DLinear model for time series forecasting.
DLinear uses simple linear layers with trend-seasonal decomposition.

Reference:
Are Transformers Effective for Time Series Forecasting? (AAAI 2023)
https://arxiv.org/abs/2205.13504
"""

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    """
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, features)
        Returns:
            Trend component: (batch_size, seq_len, features)
        """
        # Padding on both ends
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        # x_padded: (batch, seq_len, features) -> (batch, features, seq_len)
        x_padded = x_padded.permute(0, 2, 1)
        x_trend = self.avg(x_padded)
        # (batch, features, seq_len) -> (batch, seq_len, features)
        x_trend = x_trend.permute(0, 2, 1)
        
        return x_trend


class SeriesDecomp(nn.Module):
    """
    Series decomposition block to separate trend and seasonal components.
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, features)
        Returns:
            seasonal, trend: both (batch_size, seq_len, features)
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear: Decomposition Linear for time series forecasting.
    
    Architecture:
    - Series decomposition into trend and seasonal components
    - Separate linear layers for trend and seasonal
    - Combine predictions from both components
    
    Args:
        seq_len: Input sequence length (e.g., 96)
        pred_len: Prediction sequence length (e.g., 96)
        input_dim: Number of input features (e.g., 21)
        output_dim: Number of output features (e.g., 1)
        kernel_size: Kernel size for moving average (default: 25)
        individual: Whether to use individual linear layers for each feature (default: False)
    """
    
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        input_dim: int = 21,
        output_dim: int = 1,
        kernel_size: int = 25,
        individual: bool = False
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.individual = individual
        
        # Decomposition
        self.decomposition = SeriesDecomp(kernel_size)
        
        if individual:
            # Individual linear layers for each feature
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(input_dim)
            ])
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(input_dim)
            ])
        else:
            # Shared linear layers across features
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)
        
        # Project to output dimension if needed
        if input_dim != output_dim:
            self.output_projection = nn.Linear(input_dim, output_dim)
        else:
            self.output_projection = None

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
        
        Returns:
            Output sequence (batch_size, pred_len, output_dim)
        """
        # Decomposition
        seasonal_init, trend_init = self.decomposition(x)
        
        # seasonal_init: (batch, seq_len, features)
        # trend_init: (batch, seq_len, features)
        
        if self.individual:
            # Apply individual linear layers
            seasonal_output = torch.zeros(
                [x.size(0), self.pred_len, self.input_dim],
                dtype=x.dtype, device=x.device
            )
            trend_output = torch.zeros(
                [x.size(0), self.pred_len, self.input_dim],
                dtype=x.dtype, device=x.device
            )
            
            for i in range(self.input_dim):
                # (batch, seq_len) -> (batch, pred_len)
                seasonal_output[:, :, i] = self.Linear_Seasonal[i](
                    seasonal_init[:, :, i]
                )
                trend_output[:, :, i] = self.Linear_Trend[i](
                    trend_init[:, :, i]
                )
        else:
            # Transpose: (batch, seq_len, features) -> (batch, features, seq_len)
            seasonal_init = seasonal_init.permute(0, 2, 1)
            trend_init = trend_init.permute(0, 2, 1)
            
            # Apply linear: (batch, features, seq_len) -> (batch, features, pred_len)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            
            # Transpose back: (batch, features, pred_len) -> (batch, pred_len, features)
            seasonal_output = seasonal_output.permute(0, 2, 1)
            trend_output = trend_output.permute(0, 2, 1)
        
        # Combine seasonal and trend
        output = seasonal_output + trend_output
        
        # Project to output dimension if needed
        if self.output_projection is not None:
            output = self.output_projection(output)
        
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
