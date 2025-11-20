#!/bin/bash

# Quick test script to train models for a few epochs and test evaluation

echo "========================================================================="
echo "Quick Test: Train models and test evaluation script"
echo "========================================================================="
echo ""

# Create output directories
mkdir -p checkpoints_test

# Train DLinear for a few epochs
echo "Training DLinear (5 epochs)..."
python train.py \
    --model_type dlinear \
    --kernel_size 25 \
    --batch_size 16 \
    --epochs 5 \
    --lr 0.001 \
    --output_dir ./checkpoints_test/dlinear

echo ""
echo "Training TimeMixer (5 epochs)..."
python train.py \
    --model_type timemixer \
    --d_model 64 \
    --d_ff 128 \
    --e_layers 2 \
    --dropout 0.1 \
    --batch_size 16 \
    --epochs 5 \
    --lr 0.0001 \
    --output_dir ./checkpoints_test/timemixer

echo ""
echo "Training PatchTST (5 epochs)..."
python train.py \
    --model_type patchtst \
    --d_model 64 \
    --nhead 4 \
    --num_encoder_layers 2 \
    --dim_feedforward 256 \
    --patch_len 16 \
    --stride 8 \
    --dropout 0.1 \
    --batch_size 16 \
    --epochs 5 \
    --lr 0.0001 \
    --output_dir ./checkpoints_test/patchtst

echo ""
echo "========================================================================="
echo "Testing Evaluation Script"
echo "========================================================================="
echo ""

# Run evaluation
python evaluate_all_models.py \
    --checkpoints_dir ./checkpoints_test \
    --output_dir ./evaluation_test \
    --num_samples 3 \
    --batch_size 16

echo ""
echo "========================================================================="
echo "Test Complete!"
echo "========================================================================="
echo ""
echo "Check results in ./evaluation_test/"
echo ""
