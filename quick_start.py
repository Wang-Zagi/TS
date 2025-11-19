"""
Quick start example for training and evaluating the Transformer model.
This script demonstrates the basic usage of the forecasting pipeline.
"""

import os
import sys

def main():
    print("=" * 70)
    print("Transformer Time Series Forecasting - Quick Start")
    print("=" * 70)
    print()
    
    print("This example demonstrates how to use the Transformer model for")
    print("time series forecasting with the weather dataset.")
    print()
    
    # Step 1: Data Preparation
    print("Step 1: Data Preparation")
    print("-" * 70)
    print("First, ensure the data files are generated:")
    print("  $ cd dataset/weather")
    print("  $ python data_transform.py")
    print("  $ cd ../..")
    print()
    
    # Step 2: Training
    print("Step 2: Training the Model")
    print("-" * 70)
    print("Train with default parameters (recommended for first run):")
    print("  $ python train.py --epochs 50 --batch_size 32 --output_dir ./checkpoints")
    print()
    print("Quick test (fewer epochs):")
    print("  $ python train.py --epochs 10 --batch_size 16 --output_dir ./checkpoints")
    print()
    print("Advanced training with custom parameters:")
    print("  $ python train.py \\")
    print("      --epochs 100 \\")
    print("      --batch_size 32 \\")
    print("      --lr 0.0001 \\")
    print("      --d_model 256 \\")
    print("      --nhead 8 \\")
    print("      --num_encoder_layers 4 \\")
    print("      --num_decoder_layers 4 \\")
    print("      --dim_feedforward 1024 \\")
    print("      --dropout 0.1 \\")
    print("      --output_dir ./checkpoints")
    print()
    
    # Step 3: Evaluation
    print("Step 3: Evaluation and Visualization")
    print("-" * 70)
    print("Evaluate the trained model:")
    print("  $ python evaluate.py --model_path ./checkpoints/best_model.pth")
    print()
    print("This will:")
    print("  - Load the trained model")
    print("  - Make predictions on the test set")
    print("  - Calculate metrics (MSE, MAE, RMSE)")
    print("  - Generate visualization plots")
    print("  - Save results to the checkpoint directory")
    print()
    
    # Step 4: Understanding Results
    print("Step 4: Understanding the Results")
    print("-" * 70)
    print("After evaluation, check these files in the output directory:")
    print("  - best_model.pth        : Trained model weights")
    print("  - predictions.png       : Visualization of predictions vs ground truth")
    print("  - test_metrics.json     : Test set metrics")
    print("  - results.txt           : Training summary")
    print("  - norm_params.json      : Normalization parameters")
    print("  - logs/                 : TensorBoard logs")
    print()
    
    # Step 5: TensorBoard
    print("Step 5: Monitoring with TensorBoard (Optional)")
    print("-" * 70)
    print("To monitor training in real-time:")
    print("  $ tensorboard --logdir ./checkpoints/logs")
    print()
    print("Then open http://localhost:6006 in your browser")
    print()
    
    # Model Architecture
    print("Model Architecture")
    print("-" * 70)
    print("Input:  192 timesteps × 21 features")
    print("        - 1 target variable: T (temperature)")
    print("        - 20 external variables: weather features")
    print()
    print("Output: 96 timesteps × 1 feature (T temperature)")
    print()
    print("Architecture:")
    print("  - Transformer encoder-decoder with positional encoding")
    print("  - Default: 3 encoder layers, 3 decoder layers")
    print("  - 8 attention heads, d_model=128")
    print("  - ~1.4M trainable parameters (default config)")
    print()
    
    # Tips
    print("Tips and Best Practices")
    print("-" * 70)
    print("1. Start with default parameters for baseline")
    print("2. Use early stopping to prevent overfitting")
    print("3. Monitor validation loss during training")
    print("4. Increase model size (d_model, layers) for better accuracy")
    print("5. Adjust learning rate if training is unstable")
    print("6. Use GPU for faster training (if available)")
    print()
    
    print("=" * 70)
    print("For more information, see README.md")
    print("=" * 70)


if __name__ == '__main__':
    main()
