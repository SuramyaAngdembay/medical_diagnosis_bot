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

def parse_args():
    parser = argparse.ArgumentParser(description='Test the trained RL model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_data_path', type=str, default='../data/release_test_patients.zip',
                      help='Path to test data')
    parser.add_argument('--evi_meta_path', type=str, default='../data/release_evidences.json',
                      help='Path to evidence metadata')
    parser.add_argument('--patho_meta_path', type=str, default='../data/release_conditions.json',
                      help='Path to pathology metadata')
    parser.add_argument('--checkpoint_dir', type=str, default='output',
                      help='Directory containing saved models')
    parser.add_argument('--dataset', type=str, default='casande',
                      help='Dataset name')
    parser.add_argument('--threshold', type=float, default=1.0,
                      help='Threshold for diagnosis')
    parser.add_argument('--mu', type=float, default=1.0,
                      help='Mu parameter')
    parser.add_argument('--nu', type=float, default=2.826,
                      help='Nu parameter')
    parser.add_argument('--trail', type=int, default=1,
                      help='Trial number')
    parser.add_argument('--batch_size', type=int, default=500,
                      help='Batch size for testing')
    parser.add_argument('--MAXSTEP', type=int, default=30,
                      help='Maximum steps per episode')
    parser.add_argument('--save_dir', type=str, default='output_test',
                      help='Directory to save test results')
    return parser.parse_args()

class Args:
    def __init__(self, args):
        self.no_initial_evidence = False
        self.MAXSTEP = args.MAXSTEP
        self.include_turns_in_state = True
        self.no_differential = False
        self.evi_meta_path = args.evi_meta_path
        self.patho_meta_path = args.patho_meta_path
        self.train = False  # Important: Set to False for testing
        self.use_differential_diagnosis = True
        self.include_race_in_state = True
        self.include_ethnicity_in_state = True

def test(args):
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Initialize environment
    env_args = Args(args)
    env = environment(env_args, args.test_data_path)
    
    # Initialize model
    model = Policy_Gradient_pair_model(
        state_size=env.observation_space.shape[0],
        disease_size=env.action_space.n,
        symptom_size=env.action_space.n
    )
    
    # Load model weights
    try:
        model.load_model(
            checkpoint_dir=args.checkpoint_dir,
            dataset=args.dataset,
            threshold=args.threshold,
            mu=args.mu,
            nu=args.nu,
            trail=args.trail
        )
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # Load threshold file
    threshold_path = os.path.join(
        args.checkpoint_dir,
        f"threshold_changing_curve_{args.dataset}_{args.trail}_{args.threshold}_{args.nu}_{args.trail}.pkl"
    )
    try:
        with open(threshold_path, 'rb') as f:
            threshold = pickle.load(f)
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
            state = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < args.MAXSTEP:
                # Get action from model
                action = model.choose_action(state)
                
                # Take step in environment
                next_state, reward, done, info = env.step(action)
                
                # Get diagnosis
                diagnosis = model.choose_diagnosis(state)
                
                # Update metrics
                episode_reward += reward
                steps += 1
                
                # Update state
                state = next_state
            
            # Calculate accuracy for this episode
            accuracy = 1.0 if info.get('correct_diagnosis', False) else 0.0
            
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