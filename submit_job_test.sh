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



echo "Starting RL training job..."
echo "Base directory: $BASE_DIR"
echo "Data directory: $DATA_DIR"
echo "RL model directory: $RL_MODEL_DIR"
echo "Output directory: $OUTPUT_DIR"

python test_rl_model.py

