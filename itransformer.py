"""
iTransformer model for time series forecasting.
iTransformer inverts the typical Transformer architecture by treating each variable as a token.

Reference:
iTransformer: Inverted Transformers Are Effective for Time Series Forecasting (ICLR 2024)
https://arxiv.org/abs/2310.06625

Key differences from standard Transformer:
- Variables (features) are treated as tokens instead of time steps
- Each variable has its own embedding layer (to handle different frequencies)
- Attention is computed across variables instead of time
- Only predicts temperature (T) - the first output variable
"""

import torch
import torch.nn as nn
import math


class VariableEmbedding(nn.Module):
    """
    Individual embedding for each variable to handle different frequencies.
    Maps the entire time series of one variable to d_model dimension.
    """
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.linear = nn.Linear(seq_len, d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) - time series for one variable
        Returns:
            (batch_size, d_model)
        """
        return self.linear(x)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for variable positions.
    """
    def __init__(self, d_model, max_vars=100):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_vars, d_model)
        position = torch.arange(0, max_vars, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_vars, d_model)
        Returns:
            (batch_size, num_vars, d_model)
        """
        num_vars = x.size(1)
        return x + self.pe[:num_vars, :].unsqueeze(0)


class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformer for time series forecasting.
    
    Architecture:
    - Each variable's entire sequence is embedded separately (individual embeddings)
    - Variables become tokens (instead of time steps)
    - Transformer attention operates across variables
    - Only predicts temperature (output_dim=1)
    
    Args:
        seq_len: Input sequence length (e.g., 192)
        pred_len: Prediction sequence length (e.g., 96)
        input_dim: Number of input features/variables (e.g., 21)
        output_dim: Number of output features (e.g., 1 for temperature only)
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
        input_dim: int = 21,
        output_dim: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        # Individual embedding for each variable
        # This is necessary because different variables may have different frequency characteristics
        self.variable_embeddings = nn.ModuleList([
            VariableEmbedding(seq_len, d_model) for _ in range(input_dim)
        ])
        
        # Positional encoding for variable positions
        self.pos_encoder = PositionalEncoding(d_model, max_vars=input_dim)
        
        # Transformer Encoder
        # Operates across variables (not time steps)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection head for each output variable
        # Since we only predict temperature (1 variable), we need 1 projection head
        self.projection_heads = nn.ModuleList([
            nn.Linear(d_model, pred_len) for _ in range(output_dim)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
               where input_dim is the number of variables
        
        Returns:
            Output sequence (batch_size, pred_len, output_dim)
        """
        batch_size = x.size(0)
        
        # Embed each variable separately
        # x: (batch_size, seq_len, input_dim)
        # We need to process each variable's time series through its own embedding
        
        embedded_vars = []
        for i in range(self.input_dim):
            # Extract i-th variable's time series: (batch_size, seq_len)
            var_series = x[:, :, i]
            # Embed: (batch_size, seq_len) -> (batch_size, d_model)
            var_embedded = self.variable_embeddings[i](var_series)
            embedded_vars.append(var_embedded)
        
        # Stack all variables: (batch_size, input_dim, d_model)
        # Now each "token" represents a variable (not a time step)
        x_embedded = torch.stack(embedded_vars, dim=1)
        
        # Add positional encoding
        x_embedded = self.pos_encoder(x_embedded)
        x_embedded = self.dropout(x_embedded)
        
        # Apply transformer encoder across variables
        # (batch_size, input_dim, d_model) -> (batch_size, input_dim, d_model)
        transformer_out = self.transformer_encoder(x_embedded)
        
        # Project each output variable
        # We only predict temperature (output_dim=1), so we use the first variable's representation
        # or we could use a specific variable's representation (e.g., the temperature variable if known)
        outputs = []
        for i in range(self.output_dim):
            # Use the first variable's representation (assuming it's temperature)
            # or average across all variables
            # Here we use the first variable (index 0) which should correspond to temperature T
            var_repr = transformer_out[:, 0, :]  # (batch_size, d_model)
            # Project to prediction length: (batch_size, d_model) -> (batch_size, pred_len)
            pred = self.projection_heads[i](var_repr)
            outputs.append(pred)
        
        # Stack outputs: (batch_size, pred_len, output_dim)
        output = torch.stack(outputs, dim=2)
        
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
