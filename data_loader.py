"""
Data loader for time series forecasting.
Handles data loading, preprocessing, and batching.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os

# Add dataset directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'dataset', 'weather'))
from create_batches import create_aligned_batches, create_mixed_batches, create_single_batches


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(self, X, y):
        """
        Args:
            X: Input sequences (num_samples, seq_len, num_features)
            y: Target sequences (num_samples, future_len, 1)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MixedBatchDataset(Dataset):
    """Dataset for mixed frequency batches (for iTransformer)."""
    
    def __init__(self, X_dicts, y):
        """
        Args:
            X_dicts: List of dictionaries, each with keys:
                    'T_30min_hist', 'A_10min_hist', 'B_120min_hist'
            y: Target sequences (num_samples, future_len, 1)
        """
        self.X_dicts = X_dicts
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X_dicts)
    
    def __getitem__(self, idx):
        # Convert numpy arrays to tensors
        X_dict = {
            'T_30min_hist': torch.FloatTensor(self.X_dicts[idx]['T_30min_hist']),
            'A_10min_hist': torch.FloatTensor(self.X_dicts[idx]['A_10min_hist']),
            'B_120min_hist': torch.FloatTensor(self.X_dicts[idx]['B_120min_hist'])
        }
        return X_dict, self.y[idx]


def collate_mixed_batch(batch):
    """Custom collate function for mixed batch data."""
    X_dicts, y = zip(*batch)
    
    # Stack all T, A, B data separately
    T_batch = torch.stack([x['T_30min_hist'] for x in X_dicts])
    A_batch = torch.stack([x['A_10min_hist'] for x in X_dicts])
    B_batch = torch.stack([x['B_120min_hist'] for x in X_dicts])
    y_batch = torch.stack(list(y))
    
    return {
        'T_30min_hist': T_batch,
        'A_10min_hist': A_batch,
        'B_120min_hist': B_batch
    }, y_batch


def load_data(history_len=96, future_len=96, step_size=96, train_ratio=0.7, val_ratio=0.15):
    """
    Load and split data into train, validation, and test sets.
    
    Args:
        history_len: Length of input sequence
        future_len: Length of target sequence
        step_size: Step size for sliding window
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Change to weather data directory
    original_dir = os.getcwd()
    data_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'weather')
    os.chdir(data_dir)
    
    try:
        # Collect all batches
        X_list = []
        y_list = []
        
        batch_gen = create_aligned_batches(
            history_len=history_len,
            future_len=future_len,
            step_size=step_size
        )
        
        for X_batch, y_batch in batch_gen:
            X_list.append(X_batch)
            y_list.append(y_batch)
        
        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)
    finally:
        # Change back to original directory
        os.chdir(original_dir)
    
    print(f"Total samples: {len(X)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Normalize data
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True)
    X_normalized = (X - X_mean) / (X_std + 1e-8)
    
    y_mean = y.mean()
    y_std = y.std()
    y_normalized = (y - y_mean) / (y_std + 1e-8)
    
    # Split data
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    X_train = X_normalized[:n_train]
    y_train = y_normalized[:n_train]
    
    X_val = X_normalized[n_train:n_train+n_val]
    y_val = y_normalized[n_train:n_train+n_val]
    
    X_test = X_normalized[n_train+n_val:]
    y_test = y_normalized[n_train+n_val:]
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    # Store normalization parameters for later use
    norm_params = {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std
    }
    
    return train_dataset, val_dataset, test_dataset, norm_params


def load_mixed_data(history_len=96, future_len=96, step_size=96, train_ratio=0.7, val_ratio=0.15):
    """
    Load mixed frequency data for iTransformer.
    
    Args:
        history_len: Length of input sequence for T (30min)
        future_len: Length of target sequence
        step_size: Step size for sliding window
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    
    Returns:
        train_dataset, val_dataset, test_dataset, norm_params
    """
    # Change to weather data directory
    original_dir = os.getcwd()
    data_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'weather')
    os.chdir(data_dir)
    
    try:
        # Collect all batches
        X_dict_list = []
        y_list = []
        
        batch_gen = create_mixed_batches(
            history_len=history_len,
            future_len=future_len,
            step_size=step_size
        )
        
        for X_dict, y_batch in batch_gen:
            X_dict_list.append(X_dict)
            y_list.append(y_batch)
        
        # Convert y to numpy array
        y = np.array(y_list)
    finally:
        # Change back to original directory
        os.chdir(original_dir)
    
    print(f"Total samples: {len(X_dict_list)}")
    print(f"First batch structure:")
    print(f"  T_30min_hist shape: {X_dict_list[0]['T_30min_hist'].shape}")
    print(f"  A_10min_hist shape: {X_dict_list[0]['A_10min_hist'].shape}")
    print(f"  B_120min_hist shape: {X_dict_list[0]['B_120min_hist'].shape}")
    print(f"y shape: {y.shape}")
    
    # Normalize data
    # For mixed batches, we need to normalize each group separately
    # Compute statistics from all samples
    T_data = np.array([x['T_30min_hist'] for x in X_dict_list])
    A_data = np.array([x['A_10min_hist'] for x in X_dict_list])
    B_data = np.array([x['B_120min_hist'] for x in X_dict_list])
    
    T_mean = T_data.mean(axis=(0, 1), keepdims=True)
    T_std = T_data.std(axis=(0, 1), keepdims=True)
    A_mean = A_data.mean(axis=(0, 1), keepdims=True)
    A_std = A_data.std(axis=(0, 1), keepdims=True)
    B_mean = B_data.mean(axis=(0, 1), keepdims=True)
    B_std = B_data.std(axis=(0, 1), keepdims=True)
    
    y_mean = y.mean()
    y_std = y.std()
    y_normalized = (y - y_mean) / (y_std + 1e-8)
    
    # Normalize all samples
    X_dict_normalized = []
    for i in range(len(X_dict_list)):
        X_dict_normalized.append({
            'T_30min_hist': (X_dict_list[i]['T_30min_hist'] - T_mean.squeeze(0)) / (T_std.squeeze(0) + 1e-8),
            'A_10min_hist': (X_dict_list[i]['A_10min_hist'] - A_mean.squeeze(0)) / (A_std.squeeze(0) + 1e-8),
            'B_120min_hist': (X_dict_list[i]['B_120min_hist'] - B_mean.squeeze(0)) / (B_std.squeeze(0) + 1e-8)
        })
    
    # Split data
    n_samples = len(X_dict_normalized)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    X_train = X_dict_normalized[:n_train]
    y_train = y_normalized[:n_train]
    
    X_val = X_dict_normalized[n_train:n_train+n_val]
    y_val = y_normalized[n_train:n_train+n_val]
    
    X_test = X_dict_normalized[n_train+n_val:]
    y_test = y_normalized[n_train+n_val:]
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create datasets
    train_dataset = MixedBatchDataset(X_train, y_train)
    val_dataset = MixedBatchDataset(X_val, y_val)
    test_dataset = MixedBatchDataset(X_test, y_test)
    
    # Store normalization parameters for later use
    norm_params = {
        'T_mean': T_mean,
        'T_std': T_std,
        'A_mean': A_mean,
        'A_std': A_std,
        'B_mean': B_mean,
        'B_std': B_std,
        'y_mean': y_mean,
        'y_std': y_std
    }
    
    return train_dataset, val_dataset, test_dataset, norm_params


def load_single_data(history_len=96, future_len=96, step_size=96, train_ratio=0.7, val_ratio=0.15):
    """
    Load single variable (temperature) data for PatchTST.
    
    Args:
        history_len: Length of input sequence
        future_len: Length of target sequence
        step_size: Step size for sliding window
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    
    Returns:
        train_dataset, val_dataset, test_dataset, norm_params
    """
    # Change to weather data directory
    original_dir = os.getcwd()
    data_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'weather')
    os.chdir(data_dir)
    
    try:
        # Collect all batches
        X_list = []
        y_list = []
        
        batch_gen = create_single_batches(
            history_len=history_len,
            future_len=future_len,
            step_size=step_size
        )
        
        for X_batch, y_batch in batch_gen:
            X_list.append(X_batch)
            y_list.append(y_batch)
        
        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)
    finally:
        # Change back to original directory
        os.chdir(original_dir)
    
    print(f"Total samples: {len(X)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Normalize data
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True)
    X_normalized = (X - X_mean) / (X_std + 1e-8)
    
    y_mean = y.mean()
    y_std = y.std()
    y_normalized = (y - y_mean) / (y_std + 1e-8)
    
    # Split data
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    X_train = X_normalized[:n_train]
    y_train = y_normalized[:n_train]
    
    X_val = X_normalized[n_train:n_train+n_val]
    y_val = y_normalized[n_train:n_train+n_val]
    
    X_test = X_normalized[n_train+n_val:]
    y_test = y_normalized[n_train+n_val:]
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    # Store normalization parameters for later use
    norm_params = {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std
    }
    
    return train_dataset, val_dataset, test_dataset, norm_params


def get_data_loaders(batch_size=32, history_len=96, future_len=96, step_size=96, 
                     train_ratio=0.7, val_ratio=0.15):
    """
    Get data loaders for training, validation, and testing.
    
    Args:
        batch_size: Batch size for data loaders
        history_len: Length of input sequence
        future_len: Length of target sequence
        step_size: Step size for sliding window
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    
    Returns:
        train_loader, val_loader, test_loader, norm_params
    """
    train_dataset, val_dataset, test_dataset, norm_params = load_data(
        history_len=history_len,
        future_len=future_len,
        step_size=step_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, norm_params


def get_mixed_data_loaders(batch_size=32, history_len=96, future_len=96, step_size=96,
                           train_ratio=0.7, val_ratio=0.15):
    """
    Get data loaders for mixed frequency batches (for iTransformer).
    
    Args:
        batch_size: Batch size for data loaders
        history_len: Length of input sequence for T (30min)
        future_len: Length of target sequence
        step_size: Step size for sliding window
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    
    Returns:
        train_loader, val_loader, test_loader, norm_params
    """
    train_dataset, val_dataset, test_dataset, norm_params = load_mixed_data(
        history_len=history_len,
        future_len=future_len,
        step_size=step_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_mixed_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_mixed_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_mixed_batch
    )
    
    return train_loader, val_loader, test_loader, norm_params


def get_single_data_loaders(batch_size=32, history_len=96, future_len=96, step_size=96,
                            train_ratio=0.7, val_ratio=0.15):
    """
    Get data loaders for single variable (temperature) batches (for PatchTST).
    
    Args:
        batch_size: Batch size for data loaders
        history_len: Length of input sequence
        future_len: Length of target sequence
        step_size: Step size for sliding window
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    
    Returns:
        train_loader, val_loader, test_loader, norm_params
    """
    train_dataset, val_dataset, test_dataset, norm_params = load_single_data(
        history_len=history_len,
        future_len=future_len,
        step_size=step_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, norm_params


if __name__ == '__main__':
    # Test data loading
    print("Testing data loading...")
    train_loader, val_loader, test_loader, norm_params = get_data_loaders(batch_size=32)
    
    # Print first batch
    for X_batch, y_batch in train_loader:
        print(f"\nFirst batch:")
        print(f"X shape: {X_batch.shape}")
        print(f"y shape: {y_batch.shape}")
        break
    
    print("\nData loading test completed successfully!")
