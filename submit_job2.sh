#!/bin/bash
#SBATCH --job-name=basd_training
#SBATCH --output=basd_training_%j.out
#SBATCH --error=basd_training_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:1           # Request 2 GPUs
#SBATCH --cpus-per-task=8      # Adjust based on your needs
#SBATCH --mem=32G              # Adjust based on your needs
#SBATCH --time=24:00:00        # Max time for the job

# Load necessary modules (uncomment and modify as needed)
# module load cuda/11.2
# module load cudnn/8.1

# Set wandb to offline mode
export WANDB_MODE=offline
export WANDB_DIR="$PWD/wandb"
echo "Setting wandb to offline mode. Logs will be stored locally."

# Activate your Conda environment (modify paths as needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate basd_env

# Set the base directory to the current directory where the script is run
BASE_DIR="$PWD"
DATA_DIR="$BASE_DIR/data"
BASD_DIR="$BASE_DIR/basd"
OUTPUT_DIR="$BASD_DIR/output"
WANDB_DIR="$BASE_DIR/wandb"

# Create output directories if they don't exist
mkdir -p $OUTPUT_DIR
mkdir -p $WANDB_DIR

echo "Starting BASD training job..."
echo "Base directory: $BASE_DIR"
echo "Data directory: $DATA_DIR"
echo "BASD directory: $BASD_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Wandb directory: $WANDB_DIR"

# Move to the BASD directory
cd $BASD_DIR

# Run the training script
python main.py \
    --data "$DATA_DIR/release_train_patients.zip" \
    --eval_data "$DATA_DIR/release_validate_patients.zip" \
    --cuda_idx 0 \
    --seed 2919 \
    --n_workers 4 \
    --thres 0.5 \
    --min_rec_ratio 1.0 \
    --output $OUTPUT_DIR \
    --run_mode train \
    --evi_meta_path "$DATA_DIR/release_evidences.json" \
    --patho_meta_path "$DATA_DIR/release_conditions.json" \
    --eval_batch_size 500 \
    --batch_size 500 \
    --num_epochs 10 \
    --interaction_length 30 \
    --patience 20 \
    --lr 0.0003469 \
    --only_diff_at_end 1 \
    --hidden_size 2048 \
    --num_layers 3 \
    --masked_inquired_actions \
    --include_turns_in_state \
    --compute_metrics_flag

# Report job completion and wandb status
echo "Job completed"
echo "Wandb offline logs saved to: $WANDB_DIR"
echo "To sync these logs later, run: wandb sync $WANDB_DIR/offline-run-*"
