#!/bin/bash

# Complete script to test all baseline models and new MixedPatch model
# Tests: iTransformer, TimeMixer, DLinear, TimesNet, PatchTST, and MixedPatch

echo "========================================================================="
echo "Testing All Models for Time Series Forecasting"
echo "========================================================================="
echo ""
echo "This script will train and test the following models:"
echo "  Baseline Models:"
echo "    1. iTransformer - Inverted Transformer with variable-wise attention"
echo "    2. TimeMixer    - Multi-scale mixing for time series"
echo "    3. DLinear      - Simple decomposition-based linear model"
echo "    4. TimesNet     - Multi-period 2D convolution model"
echo "    5. PatchTST     - Patch-based Time Series Transformer"
echo "  New Model:"
echo "    6. MixedPatch   - Mixed frequency patch-based transformer (NEW)"
echo ""
echo "All models predict temperature (T) from 192 timesteps to 96 timesteps"
echo "========================================================================="
echo ""

# Create output directories
mkdir -p checkpoints_complete

# Set common parameters
EPOCHS=100
BATCH_SIZE=32


# =============================================================================
# Test 1: DLinear (Baseline)
# =============================================================================
echo "========================================================================="
echo "Test 1: DLinear (Baseline)"
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
    --epochs 200 \
    --lr 0.001 \
    --output_dir ./checkpoints_complete/dlinear

echo ""
echo "DLinear test completed. Results saved to ./checkpoints_complete/dlinear"
echo ""

# =============================================================================
# Test 2: TimeMixer (Baseline)
# =============================================================================
echo "========================================================================="
echo "Test 2: TimeMixer (Baseline)"
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
    --d_model 128 \
    --d_ff 128 \
    --e_layers 3 \
    --dropout 0.1 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 0.0001 \
    --output_dir ./checkpoints_complete/timemixer

echo ""
echo "TimeMixer test completed. Results saved to ./checkpoints_complete/timemixer"
echo ""

# =============================================================================
# Test 3: TimesNet (Baseline)
# =============================================================================
echo "========================================================================="
echo "Test 3: TimesNet (Baseline)"
echo "========================================================================="
echo "Configuration:"
echo "  - Model dimension (d_model): 512"
echo "  - Feedforward dimension (d_ff): 128"
echo "  - Encoder layers: 5"
echo "  - Top-k periods: 3"
echo "  - Number of kernels: 6"
echo "  - Learning rate: 0.0001"
echo "  - Dropout: 0.05"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo ""

python train.py \
    --model_type timesnet \
    --d_model 128 \
    --d_ff 128 \
    --e_layers 2 \
    --top_k 5 \
    --num_kernels 5 \
    --dropout 0.1 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 0.0001 \
    --output_dir ./checkpoints_complete/timesnet

echo ""
echo "TimesNet test completed. Results saved to ./checkpoints_complete/timesnet"
echo ""

# =============================================================================
# Test 4: iTransformer (Baseline)
# =============================================================================
echo "========================================================================="
echo "Test 4: iTransformer (Baseline)"
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
echo "  - Uses mixed batches with different sequence lengths per variable"
echo ""

python train.py \
    --model_type itransformer \
    --d_model 128 \
    --nhead 8 \
    --num_encoder_layers 2 \
    --dim_feedforward 512 \
    --dropout 0.1 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 0.001 \
    --output_dir ./checkpoints_complete/itransformer

echo ""
echo "iTransformer test completed. Results saved to ./checkpoints_complete/itransformer"
echo ""

# =============================================================================
# Test 5: PatchTST (Baseline)
# =============================================================================
echo "========================================================================="
echo "Test 5: PatchTST (Baseline)"
echo "========================================================================="
echo "Configuration:"
echo "  - Model dimension (d_model): 128"
echo "  - Attention heads (nhead): 8"
echo "  - Encoder layers: 3"
echo "  - Feedforward dimension: 256"
echo "  - Patch length: 16"
echo "  - Stride: 8"
echo "  - Learning rate: 0.0001"
echo "  - Dropout: 0.2"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Uses single variable (temperature) input"
echo ""

python train.py \
    --model_type patchtst \
    --d_model 128 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --dim_feedforward 256 \
    --patch_len 16 \
    --stride 8 \
    --dropout 0.1 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 0.0001 \
    --output_dir ./checkpoints_complete/patchtst

echo ""
echo "PatchTST test completed. Results saved to ./checkpoints_complete/patchtst"
echo ""

# =============================================================================
# Test 6: MixedPatch (NEW MODEL)
# =============================================================================
echo "========================================================================="
echo "Test 6: MixedPatch (Proposed MODEL)"
echo "========================================================================="
echo "Configuration:"
echo "  - Model dimension (d_model): 128"
echo "  - Attention heads (nhead): 8"
echo "  - Encoder layers: 2"
echo "  - Feedforward dimension: 512"
echo "  - Patch length: 16"
echo "  - Stride: 8"
echo "  - Learning rate: 0.0001"
echo "  - Dropout: 0.1"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Uses mixed frequency batches with patching"
echo ""
echo "MixedPatch Key Features:"
echo "  - Combines patch-based architecture with mixed frequency data"
echo "  - Different sequence lengths: T(192), A(574), B(48)"
echo "  - Channel independence with patching"
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
    --output_dir ./checkpoints_complete/mixedpatch

echo ""
echo "MixedPatch test completed. Results saved to ./checkpoints_complete/mixedpatch"
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

echo "=== BASELINE MODELS ==="
echo ""
display_results "DLinear" "./checkpoints_complete/dlinear/results.txt"
display_results "TimeMixer" "./checkpoints_complete/timemixer/results.txt"
display_results "TimesNet" "./checkpoints_complete/timesnet/results.txt"
display_results "iTransformer" "./checkpoints_complete/itransformer/results.txt"
display_results "PatchTST" "./checkpoints_complete/patchtst/results.txt"

echo "=== NEW MODEL ==="
echo ""
display_results "MixedPatch" "./checkpoints_complete/mixedpatch/results.txt"

echo "========================================================================="
echo "Model Comparison"
echo "========================================================================="
echo ""
echo "Baseline Models:"
echo ""
echo "1. TimeMixer:"
echo "   - Multi-scale temporal mixing"
echo "   - Lightweight and efficient"
echo "   - Good for capturing different temporal patterns"
echo ""
echo "2. DLinear:"
echo "   - Simple decomposition-based linear model"
echo "   - Very fast training and inference"
echo "   - Strong baseline for many datasets"
echo ""
echo "3. TimesNet:"
echo "   - Multi-period 2D convolution"
echo "   - Captures intra-period and inter-period variations"
echo "   - Good for data with multiple periodicities"
echo ""
echo "4. iTransformer:"
echo "   - Inverted architecture (variable-wise attention)"
echo "   - Uses mixed batches with different sequence lengths"
echo "   - Good for multivariate forecasting"
echo ""
echo "5. PatchTST:"
echo "   - Patch-based transformer"
echo "   - Channel independence with patching"
echo "   - State-of-the-art on many benchmarks"
echo ""
echo "Proposed Model:"
echo ""
echo "6. MixedPatch:"
echo "   - Combines PatchTST architecture with mixed frequency data"
echo "   - Patch-based processing for efficiency"
echo "   - Handles variables with different sampling rates"
echo "   - Uses mixed batches like iTransformer"
echo "   - Expected to perform well on multi-frequency time series"
echo ""
echo "All model checkpoints and results are saved in ./checkpoints_complete/"
echo "========================================================================="

# =============================================================================
# Evaluation: Generate comparison plots for all models
# =============================================================================
echo ""
echo "========================================================================="
echo "Generating Comparison Plots with Denormalized Data"
echo "========================================================================="
echo ""
echo "Running evaluation script to generate:"
echo "  1. MAE and MSE metrics for all models"
echo "  2. Line plots comparing predictions vs actual data (denormalized)"
echo "  3. Individual plots for each model"
echo "  4. Combined comparison plot for all models"
echo ""

python evaluate_all_models.py \
    --checkpoints_dir ./checkpoints_complete \
    --output_dir ./evaluation_results \
    --num_samples 3 \
    --batch_size 32

echo ""
echo "========================================================================="
echo "Evaluation Complete!"
echo "========================================================================="
echo ""
echo "Check the following directory for results:"
echo "  - Plots: ./evaluation_results/"
echo "  - Metrics: ./evaluation_results/metrics_summary.json"
echo ""
echo "========================================================================="