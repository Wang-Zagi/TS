# xPatch Model Analysis and Integration

## Summary

This document provides analysis of the xPatch model from `import_models/xPatch/` directory and explains which data source it should use.

## Model Overview

**xPatch** is a decomposition-based time series forecasting model that combines:
- Moving average-based decomposition (EMA/DEMA)
- Patch-based non-linear stream with CNN
- Linear stream with MLP for trend modeling
- RevIN (Reversible Instance Normalization)

## Data Source Analysis

### Question: Which data source should xPatch use?

The repository provides three data loading functions in `dataset/weather/create_batches.py`:

1. **`create_single_batches()`**: For single variable (temperature only)
2. **`create_mixed_batches()`**: For mixed frequency data (T=30min, A=10min, B=120min)
3. **`create_aligned_batches()`**: For aligned frequency data (all at 30min)

### Answer: `create_aligned_batches()`

**Reasoning:**

1. **Input Structure**: The xPatch model expects input of shape `[Batch, Input, Channel]` where:
   - `Input` = sequence length (192 timesteps)
   - `Channel` = number of features (enc_in parameter)

2. **Model Architecture**:
   - The model applies decomposition to the **entire input** (all channels together)
   - It processes all features in a **unified manner**
   - The decomposition module operates on `x: [Batch, Input, Channel]`

3. **Similar Models**:
   - xPatch follows the same pattern as Transformer, DLinear, TimesNet, and TimeMixer
   - These models all use `create_aligned_batches()` for multivariate data

4. **Data Loader Configuration**:
   - The xPatch data loader (`Dataset_Custom`) supports both:
     - `features='S'` (single variable)
     - `features='M'` (multivariate)
   - This flexibility indicates it's designed for aligned frequency data

## Model Parameters

```python
class Config:
    seq_len = 192          # lookback window
    pred_len = 96          # prediction length
    enc_in = 21            # input channels (features)
    patch_len = 16         # patch length
    stride = 8             # stride for patching
    padding_patch = 'end'  # padding strategy
    revin = True           # use RevIN normalization
    ma_type = 'ema'        # moving average type: 'reg', 'ema', 'dema'
    alpha = 0.2            # smoothing factor for EMA
    beta = 0.1             # smoothing factor for DEMA
```

## Integration

The xPatch model has been integrated into the repository with:

1. **Wrapper Class** (`xpatch.py`):
   - Provides a consistent interface matching other models
   - Handles configuration and initialization
   - Supports both univariate and multivariate input

2. **Test Script** (`test_xpatch.py`):
   - Comprehensive tests for model creation
   - Forward pass tests (multivariate and univariate)
   - Different configuration tests
   - Training step verification
   - Predict method validation

## Usage Example

```python
from xpatch import xPatch

# Create model
model = xPatch(
    seq_len=192,
    pred_len=96,
    input_dim=21,      # 21 features for multivariate
    patch_len=16,
    stride=8,
    revin=True,
    ma_type='ema'
)

# Input: (batch_size, 192, 21) - aligned frequency data
x = torch.randn(4, 192, 21)

# Forward pass
output = model(x)  # Output: (4, 96, 1) - temperature prediction
```

## Data Loading

When using xPatch in the training pipeline, use `create_aligned_batches()`:

```python
from dataset.weather.create_batches import create_aligned_batches

# Generate training batches
batch_generator = create_aligned_batches(
    history_len=192,
    future_len=96,
    step_size=96
)

# Each batch contains:
# X: (192, 21) - all features aligned to 30-minute frequency
# Y: (96, 1) - temperature targets
```

## Model Architecture Details

### Decomposition Stream
- Uses EMA (Exponential Moving Average) or DEMA (Double Exponential Moving Average)
- Separates input into seasonal and trend components
- Configurable with `ma_type` parameter ('reg', 'ema', 'dema')

### Non-linear Stream (Seasonal)
1. **Patching**: Divides sequence into patches (patch_len=16, stride=8)
2. **Patch Embedding**: Linear projection to higher dimension
3. **CNN Depthwise**: Depthwise convolution for feature extraction
4. **Residual Connection**: Skip connection from patch embedding
5. **CNN Pointwise**: 1x1 convolution for channel mixing
6. **Flatten Head**: Maps patches to prediction

### Linear Stream (Trend)
1. **MLP Layers**: Series of fully connected layers
2. **Average Pooling**: Downsampling between layers
3. **Layer Normalization**: Stabilizes training

### Stream Combination
- Concatenates seasonal and trend predictions
- Final linear layer produces the forecast

## Fixes Applied

During integration, the following fixes were applied to the original xPatch code:

1. **Device Agnostic**: Changed hardcoded `.to('cuda')` to `.to(x.device)` in EMA layer
2. **DEMA Initialization**: Removed device-specific initialization in DEMA layer

These changes ensure the model works on both CPU and GPU without modification.

## Test Results

All tests pass successfully:
- ✅ Model creation (218,458 parameters)
- ✅ Forward pass with multivariate input (21 features)
- ✅ Forward pass with univariate input (1 feature)
- ✅ Different configurations (reg, ema, dema, different patch sizes)
- ✅ Training step
- ✅ Predict method

## Conclusion

**The xPatch model should use `create_aligned_batches()` for data loading.**

This is because:
1. It processes multivariate data in a unified manner
2. It applies decomposition to all features together
3. It follows the same pattern as other multivariate models in the repository
4. It expects aligned frequency data at 30-minute intervals

The model has been successfully integrated and tested, ready for use in time series forecasting tasks.
