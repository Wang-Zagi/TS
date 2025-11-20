#!/bin/bash

# Comprehensive script to test all models for time series forecasting
# Tests all available models: Transformer, DLinear, TimesNet, TimeMixer, iTransformer, PatchTST, and MixedPatch

echo "========================================================================="
echo "Comprehensive Testing of All Time Series Forecasting Models"
echo "========================================================================="
echo ""
echo "This script will train and test the following models:"
echo "  1. Transformer  - Standard encoder-decoder architecture"
echo "  2. DLinear      - Simple decomposition-based linear model"
echo "  3. TimesNet     - Multi-period 2D convolution model"
echo "  4. TimeMixer    - Multi-scale mixing for time series"
echo "  5. iTransformer - Inverted Transformer with variable-wise attention"
echo "  6. PatchTST     - Patch-based Transformer for univariate forecasting"
echo "  7. MixedPatch   - Patch-based model for mixed frequency data (NEW)"
echo ""
echo "All models predict temperature (T) from 192 timesteps to 96 timesteps"
echo "========================================================================="
echo ""

# Create output directories
mkdir -p checkpoints_all
mkdir -p logs

# Set common parameters
EPOCHS=50
BATCH_SIZE=32
PATIENCE=10

# Log file
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"
echo "Training log will be saved to: $LOG_FILE"
echo ""

# =============================================================================
# Test 1: Transformer (Standard)
# =============================================================================
echo "========================================================================="
echo "Test 1: Transformer (Standard)"
echo "========================================================================="
echo "Configuration:"
echo "  - Model dimension (d_model): 128"
echo "  - Attention heads (nhead): 8"
echo "  - Encoder layers: 3"
echo "  - Decoder layers: 3"
echo "  - Feedforward dimension: 512"
echo "  - Learning rate: 0.0001"
echo "  - Dropout: 0.1"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo ""

python train.py \
    --model_type transformer \
    --d_model 128 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --dim_feedforward 512 \
    --dropout 0.1 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 0.0001 \
    --patience ${PATIENCE} \
    --output_dir ./checkpoints_all/transformer 2>&1 | tee -a ${LOG_FILE}

echo ""
echo "Transformer test completed. Results saved to ./checkpoints_all/transformer"
echo ""

# =============================================================================
# Test 2: DLinear
# =============================================================================
echo "========================================================================="
echo "Test 2: DLinear"
echo "========================================================================="
echo "Configuration:"
echo "  - Kernel size: 25"
echo "  - Individual linear layers: No"
echo "  - Learning rate: 0.001"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo ""

python train.py \
    --model_type dlinear \
    --kernel_size 25 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 0.001 \
    --patience ${PATIENCE} \
    --output_dir ./checkpoints_all/dlinear 2>&1 | tee -a ${LOG_FILE}

echo ""
echo "DLinear test completed. Results saved to ./checkpoints_all/dlinear"
echo ""

# =============================================================================
# Test 3: TimesNet
# =============================================================================
echo "========================================================================="
echo "Test 3: TimesNet"
echo "========================================================================="
echo "Configuration:"
echo "  - Model dimension (d_model): 128"
echo "  - Feedforward dimension (d_ff): 256"
echo "  - Encoder layers: 3"
echo "  - Top-k periods: 5"
echo "  - Number of kernels: 6"
echo "  - Learning rate: 0.0003"
echo "  - Dropout: 0.1"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo ""

python train.py \
    --model_type timesnet \
    --d_model 128 \
    --d_ff 256 \
    --e_layers 2 \
    --top_k 5 \
    --num_kernels 5 \
    --dropout 0.1 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 0.0003 \
    --patience ${PATIENCE} \
    --output_dir ./checkpoints_all/timesnet 2>&1 | tee -a ${LOG_FILE}

echo ""
echo "TimesNet test completed. Results saved to ./checkpoints_all/timesnet"
echo ""

# =============================================================================
# Test 4: TimeMixer
# =============================================================================
echo "========================================================================="
echo "Test 4: TimeMixer"
echo "========================================================================="
echo "Configuration:"
echo "  - Model dimension (d_model): 64"
echo "  - Feedforward dimension (d_ff): 128"
echo "  - Encoder layers: 2"
echo "  - Learning rate: 0.0005"
echo "  - Dropout: 0.1"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
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
    --patience ${PATIENCE} \
    --output_dir ./checkpoints_all/timemixer 2>&1 | tee -a ${LOG_FILE}

echo ""
echo "TimeMixer test completed. Results saved to ./checkpoints_all/timemixer"
echo ""

# =============================================================================
# Test 5: iTransformer
# =============================================================================
echo "========================================================================="
echo "Test 5: iTransformer"
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
echo "  - Uses mixed frequency batches"
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
    --patience ${PATIENCE} \
    --output_dir ./checkpoints_all/itransformer 2>&1 | tee -a ${LOG_FILE}

echo ""
echo "iTransformer test completed. Results saved to ./checkpoints_all/itransformer"
echo ""

# =============================================================================
# Test 6: PatchTST
# =============================================================================
echo "========================================================================="
echo "Test 6: PatchTST"
echo "========================================================================="
echo "Configuration:"
echo "  - Model dimension (d_model): 128"
echo "  - Attention heads (nhead): 8"
echo "  - Encoder layers: 3"
echo "  - Feedforward dimension: 512"
echo "  - Patch length: 16"
echo "  - Stride: 8"
echo "  - Learning rate: 0.0001"
echo "  - Dropout: 0.1"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Uses single variable (temperature only)"
echo ""

python train.py \
    --model_type patchtst \
    --d_model 128 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --dim_feedforward 512 \
    --patch_len 16 \
    --stride 8 \
    --dropout 0.1 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 0.0001 \
    --patience ${PATIENCE} \
    --output_dir ./checkpoints_all/patchtst 2>&1 | tee -a ${LOG_FILE}

echo ""
echo "PatchTST test completed. Results saved to ./checkpoints_all/patchtst"
echo ""

# =============================================================================
# Test 7: MixedPatch (NEW)
# =============================================================================
echo "========================================================================="
echo "Test 7: MixedPatch (NEW - Patch-based for Mixed Frequency Data)"
echo "========================================================================="
echo "Configuration:"
echo "  - Model dimension (d_model): 128"
echo "  - Attention heads (nhead): 8"
echo "  - Self-attention layers: 2"
echo "  - Feedforward dimension: 512"
echo "  - Patch length: 16"
echo "  - Stride: 8"
echo "  - Learning rate: 0.0001"
echo "  - Dropout: 0.1"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Uses mixed frequency batches with patch-based architecture"
echo "  - Self-attention for temporal features + Cross-attention for aggregation"
echo ""

python train.py \
    --model_type mixedpatch \
    --d_model 128 \
    --nhead 8 \
    --num_encoder_layers 2 \
    --dim_feedforward 512 \
    --patch_len 16 \
    --stride 8 \
    --dropout 0.1 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 0.0001 \
    --patience ${PATIENCE} \
    --output_dir ./checkpoints_all/mixedpatch 2>&1 | tee -a ${LOG_FILE}

echo ""
echo "MixedPatch test completed. Results saved to ./checkpoints_all/mixedpatch"
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

display_results "Transformer" "./checkpoints_all/transformer/results.txt"
display_results "DLinear" "./checkpoints_all/dlinear/results.txt"
display_results "TimesNet" "./checkpoints_all/timesnet/results.txt"
display_results "TimeMixer" "./checkpoints_all/timemixer/results.txt"
display_results "iTransformer" "./checkpoints_all/itransformer/results.txt"
display_results "PatchTST" "./checkpoints_all/patchtst/results.txt"
display_results "MixedPatch (NEW)" "./checkpoints_all/mixedpatch/results.txt"

echo "========================================================================="
echo "All model checkpoints and results are saved in ./checkpoints_all/"
echo "Training log saved to: $LOG_FILE"
echo "========================================================================="
