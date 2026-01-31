#!/bin/bash
#SBATCH --job-name=Eval_mDeBERTa
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=aatman-vrundavan.vaidya@student.uni-tuebingen.de

# NOTE: Evaluating mDeBERTa on Subtask 1 Dev Set

echo "=========================================="
echo "mDeBERTa Evaluation Job"
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
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source $PROJECT_ROOT/.venv/bin/activate
else
    echo "Warning: .venv not found at $PROJECT_ROOT/.venv, trying local .venv"
    source .venv/bin/activate
fi

# Navigate to the script directory
cd $PROJECT_ROOT/experiments/transfer_learning/finetune || exit 1
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

echo ""

# 5. Data paths configuration
DATA_DIR="../../../data/subtask1/dev"
MODEL_PATH="./finetune_results/final_model"

# Verify paths exist
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    echo "Please run the finetuning step first."
    exit 1
fi

echo "Data path: $DATA_DIR"
echo "Model path: $MODEL_PATH"
echo ""

# 6. Execute evaluation
echo "=========================================="
echo "Starting Evaluation..."
echo "=========================================="
echo ""

# Using uv run if available, otherwise python
if command -v uv &> /dev/null; then
    RUN_CMD="uv run"
else
    RUN_CMD="python"
fi

$RUN_CMD evaluate_finetuned_model.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --batch_size 32

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Evaluation job completed"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

exit $EXIT_CODE
