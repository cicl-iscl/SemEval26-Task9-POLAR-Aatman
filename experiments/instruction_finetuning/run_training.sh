#!/bin/bash
#SBATCH --job-name=LlamaGuard_Polar
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=aatman-vrundavan.vaidya@student.uni-tuebingen.de

# NOTE: Llama-Guard-3-8B instruction fine-tuning for polarization detection
# 8B model with 4-bit quantization fits on single A100 (80GB)

echo "=========================================="
echo "Llama-Guard-3-8B Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# 1. Load Modules
echo "Loading modules..."
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"
echo "Python: $(which python)"
echo ""

# 2. Project Setup
echo "Setting up project environment..."
PROJECT_ROOT=/home/tu/tu_tu/tu_zxord71/SemEval26-Task9-POLAR-Aatman
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT/experiments/instruction_finetuning || exit 1
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

echo ""

# 4. Check for .env file
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with your HF_TOKEN"
    exit 1
fi
echo ".env file found ✓"
echo ""

# 5. Data paths configuration
TRAIN_DATA_PATH="../../data/dev_phase/subtask1/train"
DEV_DATA_PATH="../../data/dev_phase/subtask1/dev"

# Verify data paths exist
if [ ! -d "$TRAIN_DATA_PATH" ]; then
    echo "ERROR: Training data directory not found: $TRAIN_DATA_PATH"
    exit 1
fi

if [ ! -d "$DEV_DATA_PATH" ]; then
    echo "ERROR: Dev data directory not found: $DEV_DATA_PATH"
    exit 1
fi

echo "Data paths verified ✓"
echo "  Train: $TRAIN_DATA_PATH"
echo "  Dev: $DEV_DATA_PATH"
echo ""

# 6. Execute training with uv
echo "=========================================="
echo "Starting Llama-Guard-3-8B training..."
echo "=========================================="
echo ""

uv run train_llama_guard.py \
    --train_data_path "$TRAIN_DATA_PATH" \
    --dev_data_path "$DEV_DATA_PATH" \
    --output_dir "./llama-guard-3-8b-polarization" \
    --predictions_dir "./predictions" \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --max_seq_length 512 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    # --test_sample_size 500

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training job completed"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo "Model saved to: ./llama-guard-3-8b-polarization/"
    echo "Predictions saved to: ./predictions/"
else
    echo "✗ Training failed with exit code $EXIT_CODE"
    echo "Check logs for details: logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
