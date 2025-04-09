#!/bin/bash
#SBATCH --job-name=medical_rl_testing
#SBATCH --output=rl_testing_%j.out
#SBATCH --error=rl_testing_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=24:00:00

# Activate your Conda environment
conda init
conda activate pytorch_env_2

# Set the base directory to the current directory where the script is run
BASE_DIR="$PWD"
DATA_DIR="$BASE_DIR/data"
RL_MODEL_DIR="$BASE_DIR/rl_model"
OUTPUT_DIR="$RL_MODEL_DIR/output"
OUTPUT_TEST_DIR="$RL_MODEL_DIR/output_test"

# Create output_test directory if it doesn't exist
mkdir -p $OUTPUT_TEST_DIR

echo "Starting RL testing job..."
echo "Base directory: $BASE_DIR"
echo "Data directory: $DATA_DIR"
echo "RL model directory: $RL_MODEL_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Test results directory: $OUTPUT_TEST_DIR"

# Move to the RL model directory
cd $RL_MODEL_DIR

# Run the testing script
python test.py \
    --seed 42 \
    --test_data_path "$DATA_DIR/release_test_patients.zip" \
    --evi_meta_path "$DATA_DIR/release_evidences.json" \
    --patho_meta_path "$DATA_DIR/release_conditions.json" \
    --checkpoint_dir "$OUTPUT_DIR" \
    --dataset casande \
    --threshold 1.0 \
    --mu 1.0 \
    --nu 2.826 \
    --trail 1 \
    --batch_size 500 \
    --MAXSTEP 30 \
    --save_dir "$OUTPUT_TEST_DIR"

# Report job completion
echo "Job completed" 