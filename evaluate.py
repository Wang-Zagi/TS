"""
Evaluation and prediction script for the Transformer model.
Loads a trained model and makes predictions on test data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os

from model import TransformerTS
from data_loader import get_data_loaders


def evaluate(args):
    """Evaluate the trained model and visualize predictions."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    _, _, test_loader, norm_params = get_data_loaders(
        batch_size=args.batch_size,
        history_len=192,
        future_len=96,
        step_size=96
    )
    
    # Create model
    print("Loading model...")
    model = TransformerTS(
        input_dim=21,
        output_dim=1,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.6f}")
    
    # Evaluate on test set
    predictions = []
    targets = []
    
    print("Making predictions...")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            
            # Predict
            output = model.predict(X_batch, future_len=96)
            
            predictions.append(output.cpu().numpy())
            targets.append(y_batch.numpy())
    
    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Denormalize
    y_mean = norm_params['y_mean']
    y_std = norm_params['y_std']
    predictions = predictions * y_std + y_mean
    targets = targets * y_std + y_mean
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    print("\n" + "="*50)
    print("Test Set Results:")
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print("="*50 + "\n")
    
    # Visualize some predictions
    num_samples = min(5, len(predictions))
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Plot predictions and targets
        ax.plot(predictions[i].flatten(), label='Prediction', linewidth=2)
        ax.plot(targets[i].flatten(), label='Ground Truth', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('T (degC)')
        ax.set_title(f'Sample {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.dirname(args.model_path)
    plot_path = os.path.join(output_dir, 'predictions.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Predictions plot saved to {plot_path}")
    
    # Save metrics
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse)
    }
    
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Transformer model')
    
    # Model parameters (should match training)
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=3, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == '__main__':
    main()
