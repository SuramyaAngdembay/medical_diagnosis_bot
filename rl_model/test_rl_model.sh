#!/bin/bash

# Set the project root directory (adjust this if needed)
ROOT_DIR=/content/medical_diagnosis_bot
TEST_DATA=$ROOT_DIR/data/release_test_patients.zip
EVI_META=$ROOT_DIR/data/release_evidences.json
PATHO_META=$ROOT_DIR/data/release_conditions.json
OUTPUT_DIR=$ROOT_DIR/rl_model/output
RESULTS_DIR=$ROOT_DIR/rl_model/output_test

# Create results directory if it doesn't exist
mkdir -p $RESULTS_DIR

echo "Running RL model test..."
echo "Data paths:"
echo "  Test data: $TEST_DATA"
echo "  Evidence metadata: $EVI_META"
echo "  Pathology metadata: $PATHO_META"
echo "  Models directory: $OUTPUT_DIR"
echo "  Results directory: $RESULTS_DIR"

# Run the test script
python $ROOT_DIR/rl_model/test.py \
    --test_data_path $TEST_DATA \
    --evi_meta_path $EVI_META \
    --patho_meta_path $PATHO_META \
    --checkpoint_dir $OUTPUT_DIR \
    --dataset casande \
    --threshold 1.0 \
    --mu 1.0 \
    --nu 2.826 \
    --trail 1 \
    --batch_size 50 \
    --save_dir $RESULTS_DIR \
    --model_prefix "best_"

echo "Test completed. Results saved to $RESULTS_DIR" 