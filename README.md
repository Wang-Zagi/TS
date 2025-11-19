# Time Series Forecasting with Transformer

This repository implements a Transformer-based model for weather time series forecasting.

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
├── data_loader.py               # Data loading and preprocessing
├── train.py                     # Training script with MAE/MSE metrics
├── evaluate.py                  # Evaluation and visualization script
├── test_model.py                # Model unit tests
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare the data:
```bash
cd dataset/weather
python data_transform.py
cd ../..
```

## Usage

### Training

Train the model with default parameters:
```bash
python train.py
```

Train with custom parameters:
```bash
python train.py --epochs 100 --batch_size 32 --lr 0.0001 --d_model 128 --nhead 8 --output_dir ./checkpoints
```

Available arguments:
- `--batch_size`: Batch size (default: 32)
- `--d_model`: Model dimension (default: 128)
- `--nhead`: Number of attention heads (default: 8)
- `--num_encoder_layers`: **Number of encoder layers - controllable stacking** (default: 3)
- `--num_decoder_layers`: **Number of decoder layers - controllable stacking** (default: 3)
- `--dim_feedforward`: Feedforward dimension (default: 512)
- `--dropout`: Dropout rate (default: 0.1)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.0001)
- `--patience`: Early stopping patience (default: 15)
- `--output_dir`: Output directory for checkpoints (default: ./checkpoints)

**Note**: You can control the model depth by adjusting `num_encoder_layers` and `num_decoder_layers`. For example:
```bash
# Shallow model (faster, fewer parameters)
python train.py --num_encoder_layers 2 --num_decoder_layers 2

# Deep model (more capacity)
python train.py --num_encoder_layers 6 --num_decoder_layers 6
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

## Model Architecture

The model uses a standard Transformer encoder-decoder architecture with advanced data embedding:

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

## Data Processing

The data processing pipeline includes:

1. **Frequency Alignment**: Resamples different frequency data to 30-minute intervals
2. **Normalization**: Standardizes features using mean and standard deviation
3. **Sliding Window**: Creates overlapping sequences with 192 input steps and 96 output steps
4. **Train/Val/Test Split**: 70% training, 15% validation, 15% test

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

## Results

Training results are saved to the output directory:
- `best_model.pth`: Best model checkpoint (includes MSE and MAE metrics)
- `norm_params.json`: Normalization parameters
- `results.txt`: Training summary with MSE and MAE
- `test_metrics.json`: Test set metrics
- `predictions.png`: Visualization of predictions
- `logs/`: TensorBoard logs (includes MSE/MAE tracking)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

See `requirements.txt` for complete list.
