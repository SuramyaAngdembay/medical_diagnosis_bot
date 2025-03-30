#!/bin/bash
#SBATCH --job-name=medical_rl_training
#SBATCH --output=rl_training_%j.out
#SBATCH --error=rl_training_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:1           # Request 2 GPUs
#SBATCH --cpus-per-task=8      # Adjust based on your needs
#SBATCH --mem=30G              # Adjust based on your needs
#SBATCH --time=24:00:00        # Max time for the job

# Load necessary modules (uncomment and modify as needed)
# module load cuda/11.2
# module load cudnn/8.1

# Activate your Conda environment (modify paths as needed)
# source ~/miniconda3/etc/profile.d/conda.sh
 conda init
 conda activate pytorch_env_2

export WANDB_MODE=offline
# Set the base directory to the current directory where the script is run
BASE_DIR="$PWD"
DATA_DIR="$BASE_DIR/data"
RL_MODEL_DIR="$BASE_DIR/rl_model"
OUTPUT_DIR="$RL_MODEL_DIR/output"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

echo "Starting RL training job..."
echo "Base directory: $BASE_DIR"
echo "Data directory: $DATA_DIR"
echo "RL model directory: $RL_MODEL_DIR"
echo "Output directory: $OUTPUT_DIR"

# Move to the RL model directory
cd $RL_MODEL_DIR

# Run the training script
python main.py \
    --seed 42 \
    --train_data_path "$DATA_DIR/release_train_patients.zip" \
    --val_data_path "$DATA_DIR/release_validate_patients.zip" \
    --train \
    --trail 1 \
    --nu 2.826 \
    --mu 1.0 \
    --lr 0.000352 \
    --lamb 0.99 \
    --gamma 0.99 \
    --eval_batch_size 500 \
    --batch_size 500 \
    --EPOCHS 3 \
    --MAXSTEP 30 \
    --patience 20 \
    --eval_on_train_epoch_end \
    --evi_meta_path "$DATA_DIR/release_evidences.json" \
    --patho_meta_path "$DATA_DIR/release_conditions.json" \
    --save_dir "$OUTPUT_DIR"

# Report job completion
echo "Job completed"
