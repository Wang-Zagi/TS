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
├── model.py                      # Transformer model implementation
├── data_loader.py               # Data loading and preprocessing
├── train.py                     # Training script
├── evaluate.py                  # Evaluation and visualization script
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
- `--num_encoder_layers`: Number of encoder layers (default: 3)
- `--num_decoder_layers`: Number of decoder layers (default: 3)
- `--dim_feedforward`: Feedforward dimension (default: 512)
- `--dropout`: Dropout rate (default: 0.1)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.0001)
- `--patience`: Early stopping patience (default: 15)
- `--output_dir`: Output directory for checkpoints (default: ./checkpoints)

### Evaluation

Evaluate the trained model:
```bash
python evaluate.py --model_path ./checkpoints/best_model.pth
```

This will:
- Load the trained model
- Make predictions on the test set
- Calculate metrics (MSE, MAE, RMSE)
- Generate visualization plots
- Save results to the output directory

## Model Architecture

The model uses a standard Transformer encoder-decoder architecture:

1. **Input Projection**: Projects input features to model dimension
2. **Positional Encoding**: Adds positional information to sequences
3. **Transformer Encoder**: Processes the input sequence
4. **Transformer Decoder**: Generates the output sequence autoregressively
5. **Output Projection**: Projects back to output dimension

Key features:
- Autoregressive prediction during inference
- Teacher forcing during training
- Gradient clipping for stability
- Learning rate scheduling
- Early stopping

## Data Processing

The data processing pipeline includes:

1. **Frequency Alignment**: Resamples different frequency data to 30-minute intervals
2. **Normalization**: Standardizes features using mean and standard deviation
3. **Sliding Window**: Creates overlapping sequences with 192 input steps and 96 output steps
4. **Train/Val/Test Split**: 70% training, 15% validation, 15% test

## Results

Training results are saved to the output directory:
- `best_model.pth`: Best model checkpoint
- `norm_params.json`: Normalization parameters
- `results.txt`: Training summary
- `test_metrics.json`: Test set metrics
- `predictions.png`: Visualization of predictions
- `logs/`: TensorBoard logs

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

See `requirements.txt` for complete list.
