"""
Transformer model for time series forecasting.
Input: 192 timesteps × 21 features
Output: 96 timesteps × 1 feature (T temperature)
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerTS(nn.Module):
    """
    Transformer for time series forecasting.
    
    Architecture:
    - Input projection: maps input features to d_model dimension
    - Positional encoding
    - Transformer encoder-decoder
    - Output projection: maps d_model to output feature dimension
    """
    
    def __init__(
        self,
        input_dim: int = 21,
        output_dim: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Output projection (for decoder input)
        self.output_projection = nn.Linear(output_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # We use (seq_len, batch, features) format
        )
        
        # Final output layer
        self.output_linear = nn.Linear(d_model, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src, tgt):
        """
        Args:
            src: Input sequence (batch_size, src_seq_len, input_dim)
            tgt: Target sequence (batch_size, tgt_seq_len, output_dim)
        
        Returns:
            Output sequence (batch_size, tgt_seq_len, output_dim)
        """
        # Convert to (seq_len, batch, features) format
        src = src.transpose(0, 1)  # (src_seq_len, batch_size, input_dim)
        tgt = tgt.transpose(0, 1)  # (tgt_seq_len, batch_size, output_dim)
        
        # Project inputs to d_model dimension
        src = self.input_projection(src) * math.sqrt(self.d_model)
        tgt = self.output_projection(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Generate target mask (causal)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        
        # Transformer forward pass
        output = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask
        )
        
        # Project to output dimension
        output = self.output_linear(output)
        
        # Convert back to (batch, seq_len, features) format
        output = output.transpose(0, 1)
        
        return output
    
    def predict(self, src, future_len: int = 96):
        """
        Autoregressive prediction.
        
        Args:
            src: Input sequence (batch_size, src_seq_len, input_dim)
            future_len: Number of future steps to predict
        
        Returns:
            Predictions (batch_size, future_len, output_dim)
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device
            
            # Initialize decoder input with zeros
            tgt = torch.zeros(batch_size, 1, self.output_dim).to(device)
            
            predictions = []
            
            for i in range(future_len):
                # Predict next step
                output = self.forward(src, tgt)
                
                # Take the last prediction
                next_pred = output[:, -1:, :]
                predictions.append(next_pred)
                
                # Append to target for next iteration
                tgt = torch.cat([tgt, next_pred], dim=1)
            
            # Concatenate all predictions
            predictions = torch.cat(predictions, dim=1)
            
            return predictions
