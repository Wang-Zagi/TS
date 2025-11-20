"""
MixedPatch model for time series forecasting with mixed frequency data.

This model handles data with different variable sampling frequencies using a patch-based approach:
1. Uses main variable T as baseline to construct patches with sliding windows
2. For auxiliary variables, constructs patches with the same time range as the main variable
3. Ensures all variables have the same number of patches (tokens)
4. Uses self-attention to extract temporal features from each variable separately
5. Uses cross-attention with main variable T as Q and auxiliary variables as KV
6. Uses linear layer to predict main variable T

Architecture:
- Variable-specific patch embedding for handling different frequencies
- Self-attention for temporal feature extraction
- Cross-attention for information aggregation (T as query, others as key-value)
- Layer normalization and residual connections
- Linear projection for final prediction
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer that creates patches from time series.
    For the main variable T, uses sliding window.
    For auxiliary variables, creates patches covering the same time range as T.
    """
    def __init__(self, seq_len, patch_len, stride, d_model, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        
        # Calculate number of patches based on main variable T
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        # Linear projection from patch to d_model
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, 1) - time series for one variable
        Returns:
            (batch_size, num_patches, d_model)
        """
        batch_size, seq_len, num_vars = x.shape
        
        # Process each variable separately
        x = x.squeeze(-1)  # (batch_size, seq_len)
        
        # Create patches using unfold
        # unfold(dimension, size, step)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches: (batch_size, num_patches, patch_len)
        
        # Project patches to d_model
        x = self.projection(patches)  # (batch_size, num_patches, d_model)
        x = self.dropout(x)
        
        return x


class VariablePatchEmbedding(nn.Module):
    """
    Variable-specific patch embedding that handles different sequence lengths.
    Creates aligned patches so all variables have the same number of patches.
    """
    def __init__(self, seq_len, num_patches, d_model, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.num_patches = num_patches
        
        # Calculate patch length for this variable
        # We want num_patches from seq_len
        self.patch_len = seq_len // num_patches
        
        # Linear projection from patch to d_model
        self.projection = nn.Linear(self.patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, 1) - time series for one variable
        Returns:
            (batch_size, num_patches, d_model)
        """
        batch_size, seq_len, num_vars = x.shape
        
        x = x.squeeze(-1)  # (batch_size, seq_len)
        
        # Create patches by reshaping
        # We want to create num_patches from seq_len
        # Truncate to make it divisible
        usable_len = self.patch_len * self.num_patches
        x = x[:, :usable_len]
        
        # Reshape to patches: (batch_size, num_patches, patch_len)
        patches = x.reshape(batch_size, self.num_patches, self.patch_len)
        
        # Project patches to d_model
        x = self.projection(patches)  # (batch_size, num_patches, d_model)
        x = self.dropout(x)
        
        return x


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


class MixedPatch(nn.Module):
    """
    MixedPatch: Patch-based model for mixed frequency time series forecasting.
    
    Architecture:
    1. Patch embedding: Creates aligned patches from all variables (same number of patches)
    2. Self-attention: Extracts temporal features from each variable separately
    3. Cross-attention: Main variable T as Q, auxiliary variables as KV for information aggregation
    4. Projection: Linear layer to predict main variable T
    
    Args:
        seq_len: Input sequence length for main variable T (e.g., 96)
        pred_len: Prediction sequence length (e.g., 96)
        patch_len: Length of each patch for main variable T (default: 16)
        stride: Stride for patching main variable T (default: 8)
        d_model: Model dimension (default: 128)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of self-attention layers (default: 2)
        dim_feedforward: Feedforward dimension (default: 512)
        dropout: Dropout rate (default: 0.1)
        use_mixed_batches: Whether to use mixed frequency batches (default: True)
        seq_lens: List of sequence lengths for each variable group [T_len, A_len, B_len]
    """
    
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_mixed_batches: bool = True,
        seq_lens: list = None
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.use_mixed_batches = use_mixed_batches
        
        # Calculate number of patches based on main variable T
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        # Default sequence lengths: T(96), A(286), B(24)
        if seq_lens is None:
            seq_lens = [96, 286, 24]
        self.seq_lens = seq_lens
        
        # Patch embeddings for each variable group
        # T: use sliding window
        self.T_patch_embed = PatchEmbedding(seq_lens[0], patch_len, stride, d_model, dropout)
        
        # Group A and B: create aligned patches (same number as T)
        self.A_patch_embed = VariablePatchEmbedding(seq_lens[1], self.num_patches, d_model, dropout)
        self.B_patch_embed = VariablePatchEmbedding(seq_lens[2], self.num_patches, d_model, dropout)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches)
        
        # Self-attention layers for each variable group (temporal feature extraction)
        # We'll use separate encoder layers for T, A, and B
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.T_self_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Group A and B can share self-attention (or we can make separate ones)
        # For simplicity, we'll make them share
        encoder_layer_aux = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.aux_self_attn = nn.TransformerEncoder(encoder_layer_aux, num_layers=num_layers)
        
        # Cross-attention: T as Query, auxiliary variables as Key-Value
        # We'll use MultiheadAttention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network after cross-attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        # Projection head to prediction length
        # Flatten patch representations and project to pred_len
        self.projection = nn.Linear(self.num_patches * d_model, pred_len)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: Input can be:
               - Dict with keys ['T_30min_hist', 'A_10min_hist', 'B_120min_hist'] for mixed batches
               - Tensor of shape (batch_size, seq_len, 1) for single variable (fallback)
        
        Returns:
            Output sequence (batch_size, pred_len, 1)
        """
        # Handle mixed batches (dict input)
        if isinstance(x, dict):
            batch_size = x['T_30min_hist'].size(0)
            
            # 1. Patch embedding for each variable group
            # T: (batch_size, 96, 1) -> (batch_size, num_patches, d_model)
            T_patches = self.T_patch_embed(x['T_30min_hist'])
            T_patches = self.pos_encoder(T_patches)
            
            # Group A: process each of the 10 variables
            # A: (batch_size, 286, 10)
            A_patches_list = []
            for i in range(x['A_10min_hist'].size(2)):
                A_var = x['A_10min_hist'][:, :, i:i+1]  # (batch_size, 286, 1)
                A_var_patches = self.A_patch_embed(A_var)  # (batch_size, num_patches, d_model)
                A_var_patches = self.pos_encoder(A_var_patches)
                A_patches_list.append(A_var_patches)
            
            # Group B: process each of the 10 variables
            # B: (batch_size, 24, 10)
            B_patches_list = []
            for i in range(x['B_120min_hist'].size(2)):
                B_var = x['B_120min_hist'][:, :, i:i+1]  # (batch_size, 24, 1)
                B_var_patches = self.B_patch_embed(B_var)  # (batch_size, num_patches, d_model)
                B_var_patches = self.pos_encoder(B_var_patches)
                B_patches_list.append(B_var_patches)
            
            # 2. Self-attention for temporal feature extraction
            # T self-attention
            T_features = self.T_self_attn(T_patches)  # (batch_size, num_patches, d_model)
            
            # Auxiliary variables self-attention (process each separately)
            A_features_list = [self.aux_self_attn(A_patch) for A_patch in A_patches_list]
            B_features_list = [self.aux_self_attn(B_patch) for B_patch in B_patches_list]
            
            # Concatenate all auxiliary features
            # Stack auxiliary variables: (batch_size, num_aux_vars * num_patches, d_model)
            # where num_aux_vars = 10 + 10 = 20
            aux_features = torch.cat(A_features_list + B_features_list, dim=1)
            # aux_features: (batch_size, 20 * num_patches, d_model)
            
            # 3. Cross-attention: T as Query, auxiliary as Key-Value
            # T_features: (batch_size, num_patches, d_model) - Query
            # aux_features: (batch_size, 20 * num_patches, d_model) - Key, Value
            attn_output, _ = self.cross_attn(
                query=T_features,
                key=aux_features,
                value=aux_features
            )  # (batch_size, num_patches, d_model)
            
            # Residual connection + layer norm
            T_features = self.norm1(T_features + attn_output)
            
            # 4. Feedforward network
            ffn_output = self.ffn(T_features)
            T_features = self.norm2(T_features + ffn_output)
            
        else:
            # Fallback for single variable input (not the main use case)
            batch_size = x.size(0)
            
            # Patch embedding
            T_patches = self.T_patch_embed(x)
            T_patches = self.pos_encoder(T_patches)
            
            # Self-attention
            T_features = self.T_self_attn(T_patches)
        
        # 5. Flatten and project to prediction length
        T_features_flat = T_features.reshape(batch_size, -1)  # (batch_size, num_patches * d_model)
        output = self.projection(T_features_flat)  # (batch_size, pred_len)
        
        # Reshape to (batch_size, pred_len, 1)
        output = output.unsqueeze(-1)
        
        return output
    
    def predict(self, src, future_len: int = None):
        """
        Prediction interface for compatibility with training script.
        
        Args:
            src: Input sequence (dict or tensor)
            future_len: Number of future steps to predict (not used, model uses pred_len)
        
        Returns:
            Predictions (batch_size, pred_len, 1)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(src)
