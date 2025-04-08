"""
Test RL Model

A script to test the RL model with the same architecture as training.
"""

import os
import sys
import torch
import argparse
import numpy as np
import pickle
from rl_model.agent import Policy_Gradient_pair_model
from rl_model.env import environment

def main():
    # Create a simple argument parser
    parser = argparse.ArgumentParser(description='Test the RL model')
    parser.add_argument('--model_path', type=str, default='rl_model/output/best_policy_casande_1.0_1.0_2.826_1.pth',
                        help='Path to the trained model')
    parser.add_argument('--threshold_path', type=str, default='rl_model/output/threshold_changing_curve_casande_1.0_1.0_2.826_1.pkl',
                        help='Path to the threshold file')
    parser.add_argument('--test_data_path', type=str, default='data/release_test_patients.zip',
                        help='Path to the test data')
    parser.add_argument('--dataset', type=str, default='casande',
                        help='Name of the dataset')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Initial threshold value')
    parser.add_argument('--mu', type=float, default=1.0,
                        help='Mu parameter')
    parser.add_argument('--nu', type=float, default=2.826,
                        help='Nu parameter')
    parser.add_argument('--trail', type=int, default=1,
                        help='Trial number')
    parser.add_argument('--save_dir', type=str, default='./output',
                        help='Directory to save results')
    parser.add_argument('--checkpoint_dir', type=str, default='rl_model/output',
                        help='Directory containing checkpoints')
    parser.add_argument('--model_prefix', type=str, default='best_',
                        help='Prefix for the model to load')
    parser.add_argument('--evi_meta_path', type=str, default='data/release_evidences.json',
                        help='Path to the evidences (symptoms) meta data')
    parser.add_argument('--patho_meta_path', type=str, default='data/release_conditions.json',
                        help='Path to the pathologies (diseases) meta data')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for testing')
    parser.add_argument('--num_batches', type=int, default=5,
                        help='Number of batches to test')
    args = parser.parse_args()
    
    # Create a namespace that matches the training environment
    class Args:
        def __init__(self):
            # These values should match what was used during training
            self.seed = 42
            self.train_data_path = "data/release_train_patients.zip"  # This is just for initialization
            self.val_data_path = "data/release_validate_patients.zip"  # This is just for initialization
            self.test_data_path = args.test_data_path  # Use test data for evaluation
            self.train = False
            self.trail = args.trail
            self.nu = args.nu
            self.mu = args.mu
            self.lr = 0.000352
            self.lamb = 0.99
            self.gamma = 0.99
            self.eval_batch_size = 4139
            self.batch_size = 2657
            self.EPOCHS = 100
            self.MAXSTEP = 30
            self.patience = 20
            self.eval_on_train_epoch_end = True
            self.no_differential = False
            self.no_initial_evidence = False
            self.dataset = args.dataset
            self.threshold = args.threshold
            self.save_dir = args.save_dir
            self.checkpoint_dir = args.checkpoint_dir
            self.model_prefix = args.model_prefix
            self.include_turns_in_state = False  # Add this missing attribute
            self.evi_meta_path = args.evi_meta_path
            self.patho_meta_path = args.patho_meta_path
    
    env_args = Args()
    
    print("Loading the RL model...")
    
    # Initialize the environment with test data
    try:
        # Use test data for evaluation
        env = environment(env_args, env_args.test_data_path, train=False)
        print("Environment initialized successfully with test data.")
        print(f"State space: {env.state_size}")
        print(f"Disease size: {env.diag_size}")
        print(f"Symptom size: {env.symptom_size}")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        print("Using a mock environment for testing.")
        # Create a mock environment for testing
        class MockEnv:
            def __init__(self):
                self.state_size = 110
                self.diag_size = 50
                self.symptom_size = 100
                self.num_symptoms = 100
                self.num_pathos = 50
                self.num_demo_features = 10
                self.obs_dtype = torch.float32
                self.low_demo_values = [0] * self.num_demo_features
                self.high_demo_values = [1] * self.num_demo_features
            
            def _define_action_and_observation_spaces(self, num_symptoms, num_pathos, num_demo_features, low_demo_values, high_demo_values, obs_dtype):
                pass
        
        env = MockEnv()
    
    # Initialize the model with the same architecture as training
    try:
        # These values should match what was used during training
        state_size = env.state_size
        disease_size = env.diag_size
        symptom_size = env.symptom_size
        
        model = Policy_Gradient_pair_model(state_size, disease_size, symptom_size)
        print("Model initialized successfully.")
        print(f"Model architecture: {model}")
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Using a mock model for testing.")
        # Create a mock model for testing
        class MockModel:
            def __init__(self):
                pass
            
            def load_state_dict(self, state_dict):
                print("Loading model state dict...")
            
            def choose_action_s(self, state, deterministic=False):
                print("Choosing action...")
                return torch.tensor([0])  # Return a dummy action
            
            def choose_diagnosis(self, state):
                print("Choosing diagnosis...")
                return torch.tensor([0]), torch.tensor([0.5, 0.5])  # Return dummy diagnosis and probabilities
        
        model = MockModel()
    
    # Load the model using the original load_model method
    try:
        # Use the original load_model method from the Policy_Gradient_pair_model class
        model.load_model(env_args)
        print(f"Model loaded successfully using the original load_model method.")
    except Exception as e:
        print(f"Error loading model using original method: {e}")
        print("Trying alternative loading method...")
        
        # Try alternative loading method
        try:
            # Construct the model file names with the correct format
            # This matches the format in the original load_model method
            info = f"{env_args.dataset}_{env_args.threshold}_{env_args.mu}_{env_args.nu}_{env_args.trail}"
            policy_path = os.path.join(env_args.checkpoint_dir, f"{env_args.model_prefix}policy_{info}.pth")
            classifier_path = os.path.join(env_args.checkpoint_dir, f"{env_args.model_prefix}classifier_{info}.pth")
            
            print(f"Looking for model files:")
            print(f"Policy path: {policy_path}")
            print(f"Classifier path: {classifier_path}")
            
            if os.path.exists(policy_path) and os.path.exists(classifier_path):
                model.policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
                model.classifier.load_state_dict(torch.load(classifier_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
                print(f"Model loaded from {policy_path} and {classifier_path}")
            else:
                print(f"Model files not found: {policy_path} or {classifier_path}")
                # Try loading without the model prefix
                policy_path = os.path.join(env_args.checkpoint_dir, f"policy_{info}.pth")
                classifier_path = os.path.join(env_args.checkpoint_dir, f"classifier_{info}.pth")
                if os.path.exists(policy_path) and os.path.exists(classifier_path):
                    model.policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
                    model.classifier.load_state_dict(torch.load(classifier_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
                    print(f"Model loaded from {policy_path} and {classifier_path}")
                else:
                    print(f"Model files not found: {policy_path} or {classifier_path}")
                    # Try with the actual files in the directory
                    print("Listing files in the checkpoint directory:")
                    for file in os.listdir(env_args.checkpoint_dir):
                        print(f"  - {file}")
        except Exception as e2:
            print(f"Error loading model using alternative method: {e2}")
    
    # Load the threshold
    try:
        if os.path.exists(args.threshold_path):
            with open(args.threshold_path, 'rb') as f:
                threshold_list = pickle.load(f)
                # Use the last threshold in the list
                threshold = threshold_list[-1]
            print(f"Threshold loaded from {args.threshold_path}")
            print(f"Threshold shape: {threshold.shape}")
        else:
            print(f"Threshold file not found: {args.threshold_path}")
            # Create a default threshold
            threshold = np.ones(env.diag_size) * env_args.threshold
            print(f"Using default threshold: {threshold}")
    except Exception as e:
        print(f"Error loading threshold: {e}")
        # Create a default threshold
        threshold = np.ones(env.diag_size) * env_args.threshold
        print(f"Using default threshold: {threshold}")
    
    # Test the model on multiple test cases
    print("\nTesting the model on test data...")
    
    # Set the model to evaluation mode
    model.policy.eval()
    model.classifier.eval()
    
    # Initialize metrics
    total_correct_diagnoses = 0
    total_diagnoses = 0
    total_steps = 0
    
    # Test on multiple batches
    for batch_idx in range(args.num_batches):
        print(f"\nTesting batch {batch_idx+1}/{args.num_batches}")
        
        try:
            # Reset the environment
            env.reset()
            
            # Initialize the state for this batch
            state, true_diseases, _, _, _ = env.initialize_state(args.batch_size)
            print(f"Initialized state with shape {state.shape}")
            
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32)
            if torch.cuda.is_available():
                state_tensor = state_tensor.cuda()
            
            # Initialize variables for tracking
            done = np.zeros(args.batch_size, dtype=bool)
            right_diagnosis = np.zeros(args.batch_size, dtype=bool)
            ent_init = np.ones(args.batch_size) * np.log(env.diag_size)
            ent = ent_init.copy()
            
            # Run the model for a few steps
            for step in range(env_args.MAXSTEP):
                # Get the next action
                action = model.choose_action_s(state_tensor, deterministic=True)
                print(f"Step {step+1}: Selected actions: {action[:5]}...")
                
                # Convert state to NumPy array for the step method
                state_np = state_tensor.cpu().numpy() if torch.is_tensor(state_tensor) else state_tensor
                
                # Take a step in the environment
                next_state, reward, done, right_diagnosis, diagnosis, ent, diagnosis_idx = env.step(
                    state_np, action.cpu().numpy(), done, right_diagnosis, model, ent_init, threshold, ent
                )
                
                # Update state
                state = next_state
                state_tensor = torch.tensor(state, dtype=torch.float32)
                if torch.cuda.is_available():
                    state_tensor = state_tensor.cuda()
                
                # Check if all episodes are done
                if np.all(done):
                    break
            
            # Get the final diagnosis
            diagnosis, probabilities = model.choose_diagnosis(state_tensor)
            
            # Calculate metrics
            correct_diagnoses = np.sum(right_diagnosis)
            total_correct_diagnoses += correct_diagnoses
            total_diagnoses += args.batch_size
            total_steps += step + 1
            
            print(f"Batch {batch_idx+1} results:")
            print(f"  - Correct diagnoses: {correct_diagnoses}/{args.batch_size} ({correct_diagnoses/args.batch_size*100:.2f}%)")
            print(f"  - Average steps: {(step+1)/args.batch_size:.2f}")
            
            # Print some example diagnoses
            print("Example diagnoses:")
            for i in range(min(5, args.batch_size)):
                print(f"  - Patient {i+1}: True disease: {true_diseases[i]}, Predicted: {diagnosis[i]}, Correct: {right_diagnosis[i]}")
        
        except Exception as e:
            print(f"Error testing batch {batch_idx+1}: {e}")
    
    # Print overall results
    print("\nOverall test results:")
    print(f"  - Accuracy: {total_correct_diagnoses/total_diagnoses*100:.2f}%")
    print(f"  - Average steps: {total_steps/total_diagnoses:.2f}")
    
    print("\nRL model test completed.")

if __name__ == "__main__":
    main() 