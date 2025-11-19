# Time Series Forecasting with Multiple Baseline Models

This repository implements multiple state-of-the-art models for weather time series forecasting, including Transformer, DLinear, TimesNet, TimeMixer, iTransformer, and PatchTST.

## Task Description

- **Input**: 192 timesteps × 21 features
  - Main variable: T (temperature in °C)
  - External variables: 20 weather-related features (pressure, humidity, wind, radiation, etc.)
- **Output**: 96 timesteps × 1 feature (T temperature prediction)
- **Dataset**: Weather data at 30-minute frequency

## Project Structure

```
TS/
├── dataset/
│   └── weather/
│       ├── weather.csv           # Original weather data
│       ├── data_transform.py     # Data preprocessing script
│       └── create_batches.py     # Batch creation utilities
├── embed.py                      # Data embedding modules (TokenEmbedding, PositionalEmbedding, etc.)
├── model.py                      # Transformer model implementation
├── dlinear.py                    # DLinear model implementation
├── timesnet.py                   # TimesNet model implementation
├── timemixer.py                  # TimeMixer model implementation
├── itransformer.py               # iTransformer model implementation
├── patchtst.py                   # PatchTST model implementation
├── data_loader.py               # Data loading and preprocessing
├── train.py                     # Training script with MAE/MSE metrics
├── evaluate.py                  # Evaluation and visualization script
├── test_model.py                # Transformer model unit tests
├── test_baselines.py            # Baseline models unit tests
├── test_all_models.sh           # Bash script to test all baseline models
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Models

### 1. Transformer
Standard Transformer encoder-decoder architecture with advanced data embedding.
- **Paper**: Attention Is All You Need (NeurIPS 2017)
- **Parameters**: ~1.4M (default configuration)
- **Features**: Multi-head attention, teacher forcing, autoregressive prediction

### 2. DLinear
Simple yet effective decomposition-based linear model.
- **Paper**: Are Transformers Effective for Time Series Forecasting? (AAAI 2023)
- **Parameters**: ~37K (default configuration)
- **Features**: Trend-seasonal decomposition, individual or shared linear layers
- **Advantages**: Fast training, few parameters, strong baseline

### 3. TimesNet
Multi-period modeling using 2D convolution on transformed 1D time series.
- **Paper**: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis (ICLR 2023)
- **Parameters**: ~4.7M (default configuration)
- **Features**: FFT-based period detection, Inception blocks, multi-scale feature extraction
- **Advantages**: Captures complex temporal patterns

### 4. TimeMixer
Multi-scale mixing with decomposable mixing in both past and future.
- **Paper**: TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting (ICLR 2024)
- **Parameters**: ~424K (default configuration)
- **Features**: Multi-scale decomposition, temporal and feature mixing
- **Advantages**: Efficient multi-scale modeling

### 5. iTransformer
Inverted Transformer that treats variables as tokens instead of time steps.
- **Paper**: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting (ICLR 2024)
- **Parameters**: ~366K (default configuration)
- **Features**: Variable-wise attention, individual variable embeddings, captures inter-variable dependencies
- **Advantages**: Better multivariate modeling, handles different variable frequencies

### 6. PatchTST
Channel-independent patching-based Transformer for univariate forecasting.
- **Paper**: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (ICLR 2023)
- **Parameters**: ~243K (default configuration with patch_len=16, stride=8)
- **Features**: Patching mechanism, channel independence, univariate forecasting, positional encoding
- **Advantages**: Efficient computation through patching, works with single variable (temperature only)
- **Input**: Only temperature (T) - uses `create_single_batches()` for univariate data

## Installation

1. Install dependencies:
```bash
pip install torch numpy pandas matplotlib scikit-learn tqdm einops tensorboard
```

Or use the requirements file (note: some versions may need adjustment for Python 3.12+):
```bash
pip install -r requirements.txt
```

2. Prepare the data:
```bash
cd dataset/weather
python data_transform.py
cd ../..
```

## Quick Start

Test all models quickly:
```bash
python quick_start_baselines.py
```

This will verify all models are working correctly and show their parameter counts.

Run comprehensive tests on all baseline models:
```bash
bash test_all_models.sh
```

This will train and evaluate iTransformer, TimeMixer, DLinear, and TimesNet with reasonable configurations.

## Usage

### Training

Train with default Transformer model:
```bash
python train.py
```

Train with DLinear:
```bash
python train.py --model_type dlinear --lr 0.001
```

Train with TimesNet:
```bash
python train.py --model_type timesnet --d_model 64 --e_layers 2
```

Train with TimeMixer:
```bash
python train.py --model_type timemixer --d_model 64 --d_ff 128 --e_layers 2
```

Train with iTransformer:
```bash
python train.py --model_type itransformer --d_model 128 --nhead 8 --num_encoder_layers 3
```

Train with PatchTST (univariate - temperature only):
```bash
python train.py --model_type patchtst --d_model 128 --nhead 8 --num_encoder_layers 3 --patch_len 16 --stride 8
```

### Common Arguments

- `--model_type`: Model to use (`transformer`, `dlinear`, `timesnet`, `timemixer`, `itransformer`, `patchtst`) (default: `transformer`)
- `--batch_size`: Batch size (default: 32)
- `--d_model`: Model dimension (default: 128)
- `--dropout`: Dropout rate (default: 0.1)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.0001)
- `--patience`: Early stopping patience (default: 15)
- `--use_early_stopping`: Enable early stopping (flag)
- `--output_dir`: Output directory for checkpoints (default: ./checkpoints)

### Model-Specific Arguments

**Transformer:**
- `--nhead`: Number of attention heads (default: 8)
- `--num_encoder_layers`: Number of encoder layers (default: 3)
- `--num_decoder_layers`: Number of decoder layers (default: 3)
- `--dim_feedforward`: Feedforward dimension (default: 512)

**iTransformer:**
- `--nhead`: Number of attention heads (default: 8)
- `--num_encoder_layers`: Number of encoder layers (default: 3)
- `--dim_feedforward`: Feedforward dimension (default: 512)

**DLinear:**
- `--kernel_size`: Kernel size for moving average (default: 25)
- `--individual`: Use individual linear layers for each feature (flag)

**TimesNet:**
- `--d_ff`: FFN dimension (default: 128)
- `--num_kernels`: Number of kernels in Inception block (default: 6)
- `--top_k`: Number of top periods to use (default: 5)
- `--e_layers`: Number of encoder layers (default: 2)

**TimeMixer:**
- `--d_ff`: FFN dimension (default: 128)
- `--e_layers`: Number of encoder layers (default: 2)

**PatchTST:**
- `--nhead`: Number of attention heads (default: 8)
- `--num_encoder_layers`: Number of encoder layers (default: 3)
- `--dim_feedforward`: Feedforward dimension (default: 512)
- `--patch_len`: Patch length (default: 16)
- `--stride`: Stride for patching (default: 8)

### Examples

```bash
# Shallow Transformer (faster, fewer parameters)
python train.py --model_type transformer --num_encoder_layers 2 --num_decoder_layers 2

# Deep Transformer (more capacity)
python train.py --model_type transformer --num_encoder_layers 6 --num_decoder_layers 6

# DLinear with individual layers
python train.py --model_type dlinear --individual --lr 0.001

# Larger TimesNet
python train.py --model_type timesnet --d_model 128 --e_layers 3 --top_k 7

# PatchTST with smaller patches (univariate)
python train.py --model_type patchtst --patch_len 8 --stride 4 --d_model 64
```

### Evaluation

Evaluate the trained model:
```bash
python evaluate.py --model_path ./checkpoints/best_model.pth
```

This will:
- Load the trained model
- Make predictions on the test set
- **Calculate metrics (MSE, MAE, RMSE)**
- Generate visualization plots
- Save results to the output directory

## Model Architectures

### Transformer

The Transformer uses a standard encoder-decoder architecture with advanced data embedding:

1. **DataEmbedding Layer**: 
   - **TokenEmbedding**: 1D convolution for feature extraction
   - **PositionalEmbedding**: Sinusoidal position encoding
   - **TemporalEmbedding**: Optional time-based features (hour, day, week, month)
2. **Transformer Encoder**: Processes the input sequence (controllable number of layers)
3. **Transformer Decoder**: Generates the output sequence autoregressively (controllable number of layers)
4. **Output Projection**: Projects back to output dimension

Key features:
- **Controllable layer stacking**: Configure the number of encoder and decoder layers independently
- Autoregressive prediction during inference
- Teacher forcing during training
- Gradient clipping for stability
- Learning rate scheduling
- Early stopping
- **MAE and MSE metrics** tracked during training and validation

### DLinear

DLinear uses a simple yet effective decomposition approach:

1. **Series Decomposition**: Separates input into trend and seasonal components using moving average
2. **Linear Projection**: Applies separate linear layers to trend and seasonal components
3. **Combination**: Sums the predictions from both components

Key features:
- Very fast training and inference
- Minimal parameters
- Strong baseline performance
- Optional individual linear layers for each feature

### TimesNet

TimesNet transforms 1D time series into 2D representations to capture multi-period patterns:

1. **FFT-based Period Detection**: Identifies top-k periods in the time series
2. **2D Reshaping**: Transforms 1D sequences into 2D tensors based on detected periods
3. **Inception Blocks**: Multi-scale 2D convolution for feature extraction
4. **Adaptive Aggregation**: Combines features from different periods with learned weights

Key features:
- Captures complex temporal patterns
- Multi-scale feature extraction
- Period-aware modeling

### TimeMixer

TimeMixer uses multi-scale decomposable mixing:

1. **Multi-scale Decomposition**: Decomposes input at multiple scales (e.g., kernel sizes 3, 5, 7)
2. **Mixing Layers**: Applies temporal and feature mixing at each scale
3. **Aggregation**: Combines features from all scales
4. **Adaptive Pooling**: Projects to prediction length

Key features:
- Efficient multi-scale modeling
- Both temporal and feature mixing
- Decomposition at multiple scales

### iTransformer

iTransformer inverts the traditional Transformer architecture for time series:

1. **Variable Embedding**: Each variable gets its own embedding layer to handle different frequencies
2. **Variable-as-Token**: Treats each variable's entire time series as a single token
3. **Attention Across Variables**: Transformer attention operates across variables instead of time steps
4. **Individual Projection**: Projects each variable representation to prediction length

Key features:
- Better captures inter-variable dependencies
- Handles variables with different frequency characteristics
- More parameter-efficient than traditional Transformers
- Inverted attention mechanism

### PatchTST

PatchTST uses channel-independent patching for efficient univariate forecasting:

1. **Patching**: Divides the input time series into patches (non-overlapping or overlapping segments)
2. **Patch Embedding**: Projects each patch to a higher-dimensional embedding space
3. **Positional Encoding**: Adds position information for each patch
4. **Transformer Encoder**: Captures dependencies between patches using self-attention
5. **Projection Head**: Maps the encoded patch representations to the forecast horizon

Key features:
- Channel independence: Each variable is processed separately (univariate approach)
- Efficient computation through patching
- Works with single variable (temperature T only)
- Uses `create_single_batches()` to load only temperature data
- Reduces sequence length through patching, making Transformer attention more efficient

## Data Processing

The data processing pipeline includes:

1. **Frequency Alignment**: Resamples different frequency data to 30-minute intervals
2. **Normalization**: Standardizes features using mean and standard deviation
3. **Sliding Window**: Creates overlapping sequences with 192 input steps and 96 output steps
4. **Train/Val/Test Split**: 70% training, 15% validation, 15% test

### Data Loading Strategies

- **Aligned Batches** (`create_aligned_batches`): All variables resampled to 30-min frequency (for Transformer, DLinear, TimesNet, TimeMixer)
- **Mixed Batches** (`create_mixed_batches`): Preserves original frequencies for different variable groups (for iTransformer)
- **Single Batches** (`create_single_batches`): Only temperature variable (for PatchTST)

## Embedding Architecture

The model uses a sophisticated embedding system based on the DataEmbedding approach:

### Components

1. **TokenEmbedding**: Uses 1D convolution (kernel_size=3) to extract features from input sequences
2. **PositionalEmbedding**: Adds sinusoidal position encodings to capture temporal ordering
3. **TemporalEmbedding** (optional): Encodes time features like hour, day, week, month using learned or fixed embeddings

### DataEmbedding Module

The `DataEmbedding` class combines all three embedding types:
```python
output = value_embedding + positional_embedding + temporal_embedding
```

This rich representation helps the model better understand:
- The raw input values (via TokenEmbedding)
- The position in the sequence (via PositionalEmbedding)
- Time-based patterns (via TemporalEmbedding, if time features are provided)

## Metrics Tracking

The training script tracks and reports multiple metrics after each epoch:

- **Loss**: MSE-based training loss
- **MSE** (Mean Squared Error): Primary evaluation metric
- **MAE** (Mean Absolute Error): Alternative evaluation metric showing average absolute deviation

Both training and validation metrics are computed and logged to:
- Console output (printed each epoch)
- TensorBoard logs (for visualization)
- Model checkpoints (saved with best model)

Example epoch output:
```
Epoch 1/100 | Train Loss: 0.524618 | Val Loss: 0.489123 | Train MSE: 0.524618 | Val MSE: 0.489123 | Train MAE: 0.562341 | Val MAE: 0.541287 | Time: 45.32s
```

## Testing

### Unit Tests

Run tests for the Transformer model:
```bash
python test_model.py
```

Run tests for all baseline models:
```bash
python test_baselines.py
```

Run comprehensive tests:
```bash
python test_comprehensive.py
```

Quick start test (all models):
```bash
python quick_start_baselines.py
```

## Results

Training results are saved to the output directory:
- `best_model.pth`: Best model checkpoint (includes MSE and MAE metrics)
- `norm_params.json`: Normalization parameters
- `results.txt`: Training summary with MSE and MAE
- `test_metrics.json`: Test set metrics
- `predictions.png`: Visualization of predictions
- `logs/`: TensorBoard logs (includes MSE/MAE tracking)

## Requirements

Core dependencies:
- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm
- einops (for TimesNet)
- tensorboard (for training visualization)

See `requirements.txt` for complete list.
