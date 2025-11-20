"""
Evaluation script for all models.
Loads trained models, calculates metrics, and generates comparison plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
from pathlib import Path

from model import TransformerTS
from dlinear import DLinear
from timesnet import TimesNet
from timemixer import TimeMixer
from itransformer import iTransformer
from patchtst import PatchTST
from mixedpatch import MixedPatch
from data_loader import get_data_loaders, get_mixed_data_loaders, get_single_data_loaders


def load_model_and_params(model_dir, model_type, device):
    """
    Load a trained model and its normalization parameters.
    
    Args:
        model_dir: Directory containing model checkpoint and norm_params.json
        model_type: Type of model to load
        device: Device to load model on
    
    Returns:
        model, norm_params, checkpoint (or None if files don't exist)
    """
    model_path = os.path.join(model_dir, 'best_model.pth')
    norm_params_path = os.path.join(model_dir, 'norm_params.json')
    
    if not os.path.exists(model_path) or not os.path.exists(norm_params_path):
        print(f"  Warning: Model or norm_params not found in {model_dir}")
        return None, None, None
    
    # Load normalization parameters
    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)
    
    # Load checkpoint to get model configuration
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model based on type
    if model_type == 'transformer':
        model = TransformerTS(
            input_dim=21,
            output_dim=1,
            d_model=128,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            dropout=0.1
        )
    elif model_type == 'dlinear':
        model = DLinear(
            seq_len=192,
            pred_len=96,
            input_dim=21,
            output_dim=1,
            kernel_size=25,
            individual=False
        )
    elif model_type == 'timesnet':
        model = TimesNet(
            seq_len=192,
            pred_len=96,
            input_dim=21,
            output_dim=1,
            d_model=128,
            d_ff=128,
            num_kernels=5,
            top_k=5,
            e_layers=2,
            dropout=0.1
        )
    elif model_type == 'timemixer':
        model = TimeMixer(
            seq_len=192,
            pred_len=96,
            input_dim=21,
            output_dim=1,
            d_model=128,
            d_ff=128,
            e_layers=3,
            dropout=0.1
        )
    elif model_type == 'itransformer':
        seq_lens = [192] + [574] * 10 + [48] * 10
        model = iTransformer(
            seq_len=192,
            pred_len=96,
            input_dim=21,
            output_dim=1,
            d_model=128,
            nhead=8,
            num_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            use_mixed_batches=True,
            seq_lens=seq_lens
        )
    elif model_type == 'patchtst':
        model = PatchTST(
            seq_len=192,
            pred_len=96,
            patch_len=16,
            stride=8,
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.1
        )
    elif model_type == 'mixedpatch':
        seq_lens = [192, 574, 48]
        model = MixedPatch(
            seq_len=192,
            pred_len=96,
            patch_len=16,
            stride=8,
            d_model=128,
            nhead=8,
            num_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            use_mixed_batches=True,
            seq_lens=seq_lens
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, norm_params, checkpoint


def evaluate_model(model, test_loader, norm_params, model_type, device):
    """
    Evaluate a model on test set and return denormalized predictions and targets.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        norm_params: Normalization parameters
        model_type: Type of model
        device: Device
    
    Returns:
        predictions, targets (both denormalized), mae, mse
    """
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Handle different input types
            if isinstance(X_batch, dict):
                X_batch = {
                    'T_30min_hist': X_batch['T_30min_hist'].to(device),
                    'A_10min_hist': X_batch['A_10min_hist'].to(device),
                    'B_120min_hist': X_batch['B_120min_hist'].to(device)
                }
            else:
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
    predictions_denorm = predictions * y_std + y_mean
    targets_denorm = targets * y_std + y_mean
    
    # Calculate metrics on denormalized data
    mse = np.mean((predictions_denorm - targets_denorm) ** 2)
    mae = np.mean(np.abs(predictions_denorm - targets_denorm))
    
    return predictions_denorm, targets_denorm, mae, mse


def plot_comparison(all_results, output_dir, num_samples=3):
    """
    Create comparison plots for all models.
    
    Args:
        all_results: Dictionary with model_name -> (predictions, targets, mae, mse)
        output_dir: Directory to save plots
        num_samples: Number of samples to plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out models that failed to load
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    # Get number of samples
    first_model = list(valid_results.keys())[0]
    total_samples = len(valid_results[first_model][0])
    num_samples = min(num_samples, total_samples)
    
    # Create individual plots for each sample
    for sample_idx in range(num_samples):
        fig, axes = plt.subplots(len(valid_results), 1, figsize=(14, 4*len(valid_results)))
        
        if len(valid_results) == 1:
            axes = [axes]
        
        for idx, (model_name, (predictions, targets, mae, mse)) in enumerate(valid_results.items()):
            ax = axes[idx]
            
            # Plot predictions and targets
            time_steps = range(96)
            pred = predictions[sample_idx].flatten()
            target = targets[sample_idx].flatten()
            
            ax.plot(time_steps, target, label='Actual', linewidth=2, alpha=0.8, color='blue')
            ax.plot(time_steps, pred, label='Predicted', linewidth=2, alpha=0.8, color='red', linestyle='--')
            
            ax.set_xlabel('Time Steps (30min intervals)', fontsize=10)
            ax.set_ylabel('Temperature (°C)', fontsize=10)
            ax.set_title(f'{model_name} - Sample {sample_idx+1} (MAE: {mae:.4f}, MSE: {mse:.4f})', fontsize=11)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'comparison_sample_{sample_idx+1}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {plot_path}")
        plt.close()
    
    # Create a summary plot with all models on one sample
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Use first sample
    sample_idx = 0
    time_steps = range(96)
    
    # Plot actual data once
    first_model = list(valid_results.keys())[0]
    target = valid_results[first_model][1][sample_idx].flatten()
    ax.plot(time_steps, target, label='Actual', linewidth=3, alpha=0.9, color='black')
    
    # Plot predictions from all models
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_results)))
    for idx, (model_name, (predictions, targets, mae, mse)) in enumerate(valid_results.items()):
        pred = predictions[sample_idx].flatten()
        ax.plot(time_steps, pred, label=f'{model_name} (MAE: {mae:.4f})', 
                linewidth=2, alpha=0.7, color=colors[idx], linestyle='--')
    
    ax.set_xlabel('Time Steps (30min intervals)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('All Models Comparison - Sample 1', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, 'all_models_comparison.png')
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {summary_plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate all trained models')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints_complete',
                        help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to plot')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Model configurations
    models_to_evaluate = {
        'DLinear': ('dlinear', 'aligned'),
        'TimeMixer': ('timemixer', 'aligned'),
        'TimesNet': ('timesnet', 'aligned'),
        'iTransformer': ('itransformer', 'mixed'),
        'PatchTST': ('patchtst', 'single'),
        'MixedPatch': ('mixedpatch', 'mixed'),
    }
    
    # Store results for all models
    all_results = {}
    metrics_summary = []
    
    print("="*70)
    print("Evaluating All Models")
    print("="*70)
    print()
    
    for model_name, (model_type, data_type) in models_to_evaluate.items():
        print(f"Evaluating {model_name}...")
        model_dir = os.path.join(args.checkpoints_dir, model_type)
        
        # Load model and params
        model, norm_params, checkpoint = load_model_and_params(model_dir, model_type, device)
        
        if model is None:
            print(f"  Skipping {model_name} (not found)\n")
            all_results[model_name] = None
            continue
        
        # Load appropriate data loader
        if data_type == 'mixed':
            _, _, test_loader, _ = get_mixed_data_loaders(
                batch_size=args.batch_size,
                history_len=192,
                future_len=96,
                step_size=96
            )
        elif data_type == 'single':
            _, _, test_loader, _ = get_single_data_loaders(
                batch_size=args.batch_size,
                history_len=192,
                future_len=96,
                step_size=96
            )
        else:  # aligned
            _, _, test_loader, _ = get_data_loaders(
                batch_size=args.batch_size,
                history_len=192,
                future_len=96,
                step_size=96
            )
        
        # Evaluate model
        predictions, targets, mae, mse = evaluate_model(
            model, test_loader, norm_params, model_type, device
        )
        
        # Store results
        all_results[model_name] = (predictions, targets, mae, mse)
        
        # Print metrics
        print(f"  MAE: {mae:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {np.sqrt(mse):.6f}")
        print()
        
        # Store for summary
        metrics_summary.append({
            'model': model_name,
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(np.sqrt(mse))
        })
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics summary
    metrics_path = os.path.join(args.output_dir, 'metrics_summary.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Metrics summary saved to {metrics_path}")
    print()
    
    # Generate plots
    print("="*70)
    print("Generating Comparison Plots")
    print("="*70)
    print()
    plot_comparison(all_results, args.output_dir, args.num_samples)
    
    # Print summary table
    print()
    print("="*70)
    print("Evaluation Summary (Denormalized Metrics)")
    print("="*70)
    print(f"{'Model':<20} {'MAE':>12} {'MSE':>12} {'RMSE':>12}")
    print("-"*70)
    for metric in metrics_summary:
        print(f"{metric['model']:<20} {metric['mae']:>12.6f} {metric['mse']:>12.6f} {metric['rmse']:>12.6f}")
    print("="*70)
    print()
    print(f"Results saved to: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
