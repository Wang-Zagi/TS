#!/usr/bin/env python
"""
Quick validation script for MixedPatch model.
Tests data loading, model creation, and a training iteration.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixedpatch import MixedPatch
from data_loader import get_mixed_data_loaders


def main():
    print("=" * 70)
    print("MixedPatch Model Validation")
    print("=" * 70)
    
    # Test 1: Load data
    print("\n1. Testing data loading...")
    try:
        train_loader, val_loader, test_loader, norm_params = get_mixed_data_loaders(
            batch_size=4,
            history_len=192,
            future_len=96,
            step_size=96
        )
        print(f"   ✓ Data loaded successfully")
        print(f"   - Train batches: {len(train_loader)}")
        print(f"   - Val batches: {len(val_loader)}")
        print(f"   - Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"   ✗ Data loading failed: {e}")
        return False
    
    # Test 2: Create model
    print("\n2. Testing model creation...")
    try:
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
            seq_lens=[192, 574, 48]
        )
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Model created successfully")
        print(f"   - Parameters: {num_params:,}")
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        return False
    
    # Test 3: Forward pass
    print("\n3. Testing forward pass...")
    try:
        for X_batch, y_batch in train_loader:
            model.eval()
            with torch.no_grad():
                output = model(X_batch)
            print(f"   ✓ Forward pass successful")
            print(f"   - Input T shape: {X_batch['T_30min_hist'].shape}")
            print(f"   - Input A shape: {X_batch['A_10min_hist'].shape}")
            print(f"   - Input B shape: {X_batch['B_120min_hist'].shape}")
            print(f"   - Output shape: {output.shape}")
            print(f"   - Target shape: {y_batch.shape}")
            break
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Training step
    print("\n4. Testing training step...")
    try:
        model.train()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            print(f"   ✓ Training step successful")
            print(f"   - Loss: {loss.item():.6f}")
            break
    except Exception as e:
        print(f"   ✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Validation step
    print("\n5. Testing validation step...")
    try:
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = criterion(output, y_batch)
                
                print(f"   ✓ Validation step successful")
                print(f"   - Val Loss: {loss.item():.6f}")
                break
    except Exception as e:
        print(f"   ✗ Validation step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("All validation tests passed! ✓")
    print("=" * 70)
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
