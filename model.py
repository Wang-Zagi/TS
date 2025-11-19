"""
Transformer model for time series forecasting.
Input: 192 timesteps × 21 features
Output: 96 timesteps × 1 feature (T temperature)
"""

import torch
import torch.nn as nn
import math
from embed import DataEmbedding


class TransformerTS(nn.Module):
    """
    Transformer for time series forecasting.
    
    Architecture:
    - DataEmbedding: maps input features to d_model dimension with token, positional, and temporal embeddings
    - Transformer encoder-decoder
    - Output projection: maps d_model to output feature dimension
    
    The number of encoder and decoder layers can be controlled via num_encoder_layers and num_decoder_layers.
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
        embed_type: str = 'fixed',
        freq: str = 'h',
        max_len: int = 512
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Data embedding for encoder input
        self.enc_embedding = DataEmbedding(
            c_in=input_dim,
            d_model=d_model,
            embed_type=embed_type,
            freq=freq,
            dropout=dropout
        )
        
        # Data embedding for decoder input
        self.dec_embedding = DataEmbedding(
            c_in=output_dim,
            d_model=d_model,
            embed_type=embed_type,
            freq=freq,
            dropout=dropout
        )
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Using batch_first for better performance
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
    
    def forward(self, src, tgt, src_mark=None, tgt_mark=None):
        """
        Args:
            src: Input sequence (batch_size, src_seq_len, input_dim)
            tgt: Target sequence (batch_size, tgt_seq_len, output_dim)
            src_mark: Time features for source (batch_size, src_seq_len, n_features) - optional
            tgt_mark: Time features for target (batch_size, tgt_seq_len, n_features) - optional
        
        Returns:
            Output sequence (batch_size, tgt_seq_len, output_dim)
        """
        # Apply data embeddings
        src = self.enc_embedding(src, src_mark)
        tgt = self.dec_embedding(tgt, tgt_mark)
        
        # Generate target mask (causal)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer forward pass
        output = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask
        )
        
        # Project to output dimension
        output = self.output_linear(output)
        
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
