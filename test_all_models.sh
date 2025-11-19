#!/bin/bash

# Script to test all baseline models with reasonable configurations
# Tests iTransformer, TimeMixer, DLinear, and TimesNet

echo "========================================================================="
echo "Testing All Baseline Models for Time Series Forecasting"
echo "========================================================================="
echo ""
echo "This script will train and test the following models:"
echo "  1. iTransformer - Inverted Transformer with variable-wise attention"
echo "  2. TimeMixer    - Multi-scale mixing for time series"
echo "  3. DLinear      - Simple decomposition-based linear model"
echo "  4. TimesNet     - Multi-period 2D convolution model"
echo ""
echo "All models predict temperature (T) from 192 timesteps to 96 timesteps"
echo "========================================================================="
echo ""

# Create output directories
mkdir -p checkpoints_test

# Set common parameters
EPOCHS=50
BATCH_SIZE=32
USE_EARLY_STOPPING="--use_early_stopping"
PATIENCE=10

# =============================================================================
# Test 1: iTransformer
# =============================================================================
echo "========================================================================="
echo "Test 1: iTransformer"
echo "========================================================================="
echo "Configuration:"
echo "  - Model dimension (d_model): 128"
echo "  - Attention heads (nhead): 8"
echo "  - Encoder layers: 3"
echo "  - Feedforward dimension: 512"
echo "  - Learning rate: 0.0001"
echo "  - Dropout: 0.1"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Early stopping: Enabled (patience: ${PATIENCE})"
echo ""

python train.py \
    --model_type itransformer \
    --d_model 128 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --dim_feedforward 512 \
    --dropout 0.1 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 0.0001 \
    ${USE_EARLY_STOPPING} \
    --patience ${PATIENCE} \
    --output_dir ./checkpoints_test/itransformer

echo ""
echo "iTransformer test completed. Results saved to ./checkpoints_test/itransformer"
echo ""

# =============================================================================
# Test 2: TimeMixer
# =============================================================================
echo "========================================================================="
echo "Test 2: TimeMixer"
echo "========================================================================="
echo "Configuration:"
echo "  - Model dimension (d_model): 64"
echo "  - Feedforward dimension (d_ff): 128"
echo "  - Encoder layers: 2"
echo "  - Learning rate: 0.0005"
echo "  - Dropout: 0.1"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Early stopping: Enabled (patience: ${PATIENCE})"
echo ""

python train.py \
    --model_type timemixer \
    --d_model 64 \
    --d_ff 128 \
    --e_layers 2 \
    --dropout 0.1 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 0.0005 \
    ${USE_EARLY_STOPPING} \
    --patience ${PATIENCE} \
    --output_dir ./checkpoints_test/timemixer

echo ""
echo "TimeMixer test completed. Results saved to ./checkpoints_test/timemixer"
echo ""

# =============================================================================
# Test 3: DLinear
# =============================================================================
echo "========================================================================="
echo "Test 3: DLinear"
echo "========================================================================="
echo "Configuration:"
echo "  - Kernel size: 25"
echo "  - Individual linear layers: No"
echo "  - Learning rate: 0.001"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Early stopping: Enabled (patience: ${PATIENCE})"
echo ""

python train.py \
    --model_type dlinear \
    --kernel_size 25 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 0.001 \
    ${USE_EARLY_STOPPING} \
    --patience ${PATIENCE} \
    --output_dir ./checkpoints_test/dlinear

echo ""
echo "DLinear test completed. Results saved to ./checkpoints_test/dlinear"
echo ""

# =============================================================================
# Test 4: TimesNet
# =============================================================================
echo "========================================================================="
echo "Test 4: TimesNet"
echo "========================================================================="
echo "Configuration:"
echo "  - Model dimension (d_model): 64"
echo "  - Feedforward dimension (d_ff): 128"
echo "  - Encoder layers: 2"
echo "  - Top-k periods: 5"
echo "  - Number of kernels: 6"
echo "  - Learning rate: 0.0005"
echo "  - Dropout: 0.1"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Early stopping: Enabled (patience: ${PATIENCE})"
echo ""

python train.py \
    --model_type timesnet \
    --d_model 64 \
    --d_ff 128 \
    --e_layers 2 \
    --top_k 5 \
    --num_kernels 6 \
    --dropout 0.1 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 0.0005 \
    ${USE_EARLY_STOPPING} \
    --patience ${PATIENCE} \
    --output_dir ./checkpoints_test/timesnet

echo ""
echo "TimesNet test completed. Results saved to ./checkpoints_test/timesnet"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "========================================================================="
echo "All Tests Completed!"
echo "========================================================================="
echo ""
echo "Results Summary:"
echo "----------------"
echo ""

# Function to extract and display results
display_results() {
    model_name=$1
    results_file=$2
    
    if [ -f "$results_file" ]; then
        echo "$model_name:"
        cat "$results_file"
        echo ""
    else
        echo "$model_name: Results file not found"
        echo ""
    fi
}

display_results "iTransformer" "./checkpoints_test/itransformer/results.txt"
display_results "TimeMixer" "./checkpoints_test/timemixer/results.txt"
display_results "DLinear" "./checkpoints_test/dlinear/results.txt"
display_results "TimesNet" "./checkpoints_test/timesnet/results.txt"

echo "========================================================================="
echo "All model checkpoints and results are saved in ./checkpoints_test/"
echo "========================================================================="
