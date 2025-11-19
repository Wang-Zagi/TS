"""
Training script for Transformer time series forecasting.
Trains a model to predict 96 timesteps from 192 timesteps of input.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from tqdm import tqdm
import json

from model import TransformerTS
from data_loader import get_data_loaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Teacher forcing: use ground truth as decoder input
        # For the first prediction, we use zeros
        decoder_input = torch.zeros(y_batch.size(0), 1, y_batch.size(2)).to(device)
        decoder_input = torch.cat([decoder_input, y_batch[:, :-1, :]], dim=1)
        
        # Forward pass
        output = model(X_batch, decoder_input)
        
        # Calculate loss
        loss = criterion(output, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_loader, desc="Validation", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Use autoregressive prediction for validation
            output = model.predict(X_batch, future_len=y_batch.size(1))
            
            # Calculate loss
            loss = criterion(output, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train(args):
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, norm_params = get_data_loaders(
        batch_size=args.batch_size,
        history_len=192,
        future_len=96,
        step_size=96
    )
    
    # Save normalization parameters
    norm_params_save = {
        'X_mean': norm_params['X_mean'].tolist(),
        'X_std': norm_params['X_std'].tolist(),
        'y_mean': float(norm_params['y_mean']),
        'y_std': float(norm_params['y_std'])
    }
    with open(os.path.join(args.output_dir, 'norm_params.json'), 'w') as f:
        json.dump(norm_params_save, f)
    
    # Create model
    print("Creating model...")
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
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  → Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Test on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")
    
    # Save test results
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write(f"Best validation loss: {best_val_loss:.6f}\n")
        f.write(f"Test loss: {test_loss:.6f}\n")
    
    writer.close()
    print(f"\nTraining completed! Results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Transformer for time series forecasting')
    
    # Data parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=3, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
