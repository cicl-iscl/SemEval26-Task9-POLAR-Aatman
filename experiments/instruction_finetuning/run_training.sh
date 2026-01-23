#!/bin/bash
# Run script for Llama-Guard-3-8B instruction fine-tuning
# Usage: ./run_training.sh

set -e  # Exit on error

echo "=========================================="
echo "Llama-Guard-3-8B Training Script"
echo "=========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file with your HF_TOKEN"
    echo "You can copy .env.example and fill in your token:"
    echo "  cp .env.example .env"
    echo "  # Then edit .env and add your token"
    exit 1
fi

# Check if data directories exist
if [ ! -d "../../subtask1/train" ]; then
    echo "Error: Training data directory not found: ../../subtask1/train"
    exit 1
fi

if [ ! -d "../../subtask1/dev" ]; then
    echo "Error: Dev data directory not found: ../../subtask1/dev"
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q -r requirements.txt

# Run training
echo ""
echo "Starting training..."
echo "Logs will be saved to training_YYYYMMDD_HHMMSS.log"
echo ""

python train_llama_guard.py

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Model saved to: ./llama-guard-3-8b-polarization/"
echo "Predictions saved to: ./predictions/"
