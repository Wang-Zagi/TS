# Mixed Batches Implementation for iTransformer

## Overview

This document describes the implementation of mixed frequency batches for the iTransformer model, allowing it to handle variables with different sampling frequencies without resampling.

## Problem Statement

The original implementation resampled all data to a common 30-minute frequency, which:
- Lost information from high-frequency data (10-minute Group A data)
- Required interpolation for low-frequency data (120-minute Group B data)
- Did not fully leverage the iTransformer's capability to handle different sequence lengths per variable

## Solution

The iTransformer now supports mixed frequency batches where each variable retains its original sampling frequency:

- **T (Temperature)**: 192 timesteps at 30-minute frequency
- **Group A variables (10)**: 574 timesteps at 10-minute frequency
- **Group B variables (10)**: 48 timesteps at 120-minute frequency

## Implementation Details

### 1. Data Loading

**New Classes in `data_loader.py`:**

- `MixedBatchDataset`: Dataset that stores data as dictionaries with different frequency components
- `collate_mixed_batch()`: Custom collate function that batches dictionaries properly

**New Functions:**

- `load_mixed_data()`: Loads and normalizes mixed frequency data
- `get_mixed_data_loaders()`: Creates data loaders for mixed batches

### 2. Model Modifications

**`itransformer.py` Changes:**

- Added `use_mixed_batches` parameter to enable mixed batch mode
- Added `seq_lens` parameter to specify sequence length for each variable
- Modified `__init__()` to create variable embeddings with appropriate sequence lengths
- Updated `forward()` to handle both dict (mixed) and tensor (aligned) inputs

**Variable Embedding Configuration:**
```python
seq_lens = [192] + [574] * 10 + [48] * 10
# 1 T variable (192 steps) + 10 A variables (574 steps) + 10 B variables (48 steps)
```

### 3. Training Script Updates

**`train.py` Changes:**

- Modified data loading to use `get_mixed_data_loaders()` for iTransformer
- Updated `train_epoch()` to handle dict inputs by moving each component to device
- Updated `validate()` to handle dict inputs similarly
- Added separate normalization parameter handling for mixed vs aligned batches

### 4. Batch Script Updates

**`test_all_models.sh` Changes:**

- Disabled early stopping for all models (removed `--use_early_stopping` flag)
- Improved TimesNet parameters:
  - d_model: 64 → 128
  - d_ff: 128 → 256
  - e_layers: 2 → 3
  - learning_rate: 0.0005 → 0.0003

## Usage

### Training iTransformer with Mixed Batches

```bash
python train.py \
    --model_type itransformer \
    --d_model 128 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --dim_feedforward 512 \
    --dropout 0.1 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.0001 \
    --output_dir ./checkpoints/itransformer
```

### Running Full Test Suite

```bash
bash test_all_models.sh
```

This will test all models including iTransformer with mixed batches.

## Backward Compatibility

All changes maintain backward compatibility:

- Other models (DLinear, TimesNet, TimeMixer, Transformer) continue using aligned batches
- iTransformer can still use aligned batches by setting `use_mixed_batches=False`
- Training and validation functions detect input type automatically

## Data Format

### Mixed Batch Dictionary Structure

```python
{
    'T_30min_hist': Tensor[batch_size, 192, 1],    # Temperature
    'A_10min_hist': Tensor[batch_size, 574, 10],   # Group A (10 variables)
    'B_120min_hist': Tensor[batch_size, 48, 10]    # Group B (10 variables)
}
```

### Aligned Batch Structure (Other Models)

```python
Tensor[batch_size, 192, 21]  # All 21 variables resampled to 30-minute frequency
```

## Benefits

1. **Preserves Information**: No data loss from downsampling or interpolation
2. **Exploits iTransformer Design**: Each variable gets appropriate embedding for its frequency
3. **Better Performance**: Model can learn frequency-specific patterns
4. **Maintains Compatibility**: Other models continue working unchanged

## Testing

Comprehensive tests verify:
- Model creation with mixed batch configuration
- Data loading and batching
- Forward pass with dictionary inputs
- Training and validation loops
- Backward compatibility with other models

All tests pass successfully.

## Files Modified

1. `dataset/weather/create_batches.py` - Fixed file path for T_DEGC_FILE
2. `data_loader.py` - Added mixed batch support
3. `itransformer.py` - Added mixed batch mode
4. `train.py` - Updated for mixed batches
5. `test_all_models.sh` - Disabled early stopping, improved TimesNet params
6. `.gitignore` - Added test_output directory

## Performance Expectations

With mixed batches, iTransformer can:
- Better capture high-frequency patterns in Group A variables
- Better understand low-frequency trends in Group B variables
- Make more accurate predictions by leveraging frequency-specific information
