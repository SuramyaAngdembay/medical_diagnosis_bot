#!/usr/bin/env python
import argparse
import numpy as np
import torch
import os
import pickle
from env import environment
from agent import Policy_Gradient_pair_model
import logging
from tqdm import tqdm
from scipy.stats import entropy

def parse_args():
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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--MAXSTEP', type=int, default=30,
                        help='Maximum steps per episode')
    return parser.parse_args()

class Args:
    def __init__(self, args):
        # Training parameters (hardcoded to match training environment)
        self.seed = args.seed
        self.train_data_path = "data/release_train_patients.zip"  # Just for initialization
        self.val_data_path = "data/release_validate_patients.zip"  # Just for initialization
        self.test_data_path = args.test_data_path
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
        self.MAXSTEP = args.MAXSTEP
        self.patience = 20
        self.eval_on_train_epoch_end = True
        self.no_differential = False
        self.no_initial_evidence = False
        self.dataset = args.dataset
        self.threshold = args.threshold
        self.save_dir = args.save_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.model_prefix = args.model_prefix
        self.include_turns_in_state = False
        self.evi_meta_path = args.evi_meta_path
        self.patho_meta_path = args.patho_meta_path
        self.use_differential_diagnosis = True
        self.include_race_in_state = True
        self.include_ethnicity_in_state = True

def test(args):
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Initialize environment with explicit Args class
    env_args = Args(args)
    env = environment(env_args, args.test_data_path)
    
    # Initialize model with correct dimensions from environment
    model = Policy_Gradient_pair_model(
        state_size=env.state_size,
        disease_size=env.diag_size,
        symptom_size=env.symptom_size
    )
    
    # Load model weights
    try:
        model.load_model(env_args, prefix=env_args.model_prefix)
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # Load threshold file
    info = str(args.dataset) + '_' + str(args.threshold) + '_' + str(args.mu) + '_' + str(args.nu) + '_' + str(args.trail)
    threshold_path = os.path.join(args.checkpoint_dir, f"threshold_changing_curve_{info}.pkl")
    try:
        with open(threshold_path, 'rb') as f:
            threshold_list = pickle.load(f)
            # Use the last threshold from the list
            threshold = threshold_list[-1] if isinstance(threshold_list, list) else threshold_list
        print(f"Successfully loaded threshold file from {threshold_path}")
    except Exception as e:
        print(f"Error loading threshold file: {e}")
        return

    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    total_accuracy = 0
    total_steps = 0
    total_reward = 0
    num_episodes = 0
    
    # Testing loop
    with torch.no_grad():
        for episode in tqdm(range(args.batch_size), desc="Testing"):
            env.reset()  # Reset the environment
            
            # Use the actual batch size from arguments
            state, true_disease, true_diff_indices, true_diff_probas, _ = env.initialize_state(args.batch_size)
            
            episode_reward = 0
            done = np.zeros(args.batch_size, dtype=bool)
            steps = 0
            
            # Initial diagnosis
            a_d, p_d = model.choose_diagnosis(state)
            init_ent = entropy(p_d, axis=1)
            
            # Check if initial diagnosis is correct
            done = (init_ent < threshold[a_d])
            right_diag = (a_d == env.disease) & done
            
            ent = init_ent
            
            while not np.all(done) and steps < args.MAXSTEP:
                # Get action from model
                action = model.choose_action_s(state)
                
                # Take step in environment
                state_, reward_s, done, right_diag, final_idx, ent_, a_d_ = env.step(state, action, done, right_diag, model, init_ent, threshold, ent)
                
                # Calculate reward (average over batch)
                reward = np.mean(reward_s)
                
                # Update metrics
                episode_reward += reward
                steps += 1
                
                # Update state and entropy
                state = state_
                ent = ent_
                
                if np.all(done):
                    break
            
            # Calculate accuracy for this episode (average over batch)
            accuracy = np.mean(right_diag.astype(float))
            
            # Update total metrics
            total_accuracy += accuracy
            total_steps += steps
            total_reward += episode_reward
            num_episodes += 1
            
            # Print episode results
            if (episode + 1) % 10 == 0:
                print(f"\nEpisode {episode + 1}")
                print(f"Steps: {steps}")
                print(f"Reward: {episode_reward:.2f}")
                print(f"Accuracy: {accuracy:.2f}")
    
    # Calculate and print final metrics
    avg_accuracy = total_accuracy / num_episodes
    avg_steps = total_steps / num_episodes
    avg_reward = total_reward / num_episodes
    
    print("\nTest Results:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    
    # Create output_test directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save results
    results = {
        'accuracy': avg_accuracy,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward,
        'num_episodes': num_episodes,
        'test_parameters': {
            'dataset': args.dataset,
            'threshold': args.threshold,
            'mu': args.mu,
            'nu': args.nu,
            'trail': args.trail,
            'batch_size': args.batch_size,
            'MAXSTEP': args.MAXSTEP
        }
    }
    
    results_path = os.path.join(args.save_dir, f"test_results_{args.dataset}_{args.trail}.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    args = parse_args()
    test(args) 