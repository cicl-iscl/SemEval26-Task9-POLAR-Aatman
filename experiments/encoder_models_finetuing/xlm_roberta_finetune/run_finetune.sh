#!/bin/bash
#SBATCH --job-name=Finetune_XLM_RoBERTa
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=aatman-vrundavan.vaidya@student.uni-tuebingen.de

# NOTE: Finetuning XLM-RoBERTa-large on Subtask 1

echo "=========================================="
echo "XLM-RoBERTa-large Finetuning Job"
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
cd $PROJECT_ROOT/experiments/encoder_models_finetuing/xlm_roberta_finetune || exit 1
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

echo ""

# 5. Data paths configuration
# Adjust these paths relative to the script location or absolute paths
DATA_DIR="../../../data/subtask1"

# Verify data paths exist
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

echo "Data path: $DATA_DIR"
echo ""

# 6. Execute training
echo "=========================================="
echo "Starting XLM-RoBERTa-large finetuning..."
echo "=========================================="
echo ""

# Using uv run if available, otherwise python
if command -v uv &> /dev/null; then
    RUN_CMD="uv run"
else
    RUN_CMD="python"
fi

$RUN_CMD finetune_xlm_roberta.py \
    --model_name "FacebookAI/xlm-roberta-large" \
    --data_dir "$DATA_DIR" \
    --output_dir "./finetune_results_xlm_roberta" \
    --submission_dir "./subtask_1_xlm_roberta" \
    --batch_size 32 \
    --epochs 30 \
    --lr 2e-5 \
    --gradient_accumulation_steps 2 \
    --max_length 256

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
    echo "Results saved to: ./finetune_results_xlm_roberta"
    echo "Submission zip created: ./subtask_1_xlm_roberta.zip"
else
    echo "✗ Training failed with exit code $EXIT_CODE"
    echo "Check logs for details: logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
