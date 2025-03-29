import modal
import os
import subprocess
import sys
import shutil

# Define the Modal app
app = modal.App("train-medical-models")

# Get the absolute paths to local directories and files
local_repo_dir = os.path.abspath(".")  # Current directory containing the repo
data_dir = os.path.abspath("data")

# List essential files with full paths
essential_files = {
    "train_zip": os.path.join(data_dir, "release_train_patients.zip"),
    "val_zip": os.path.join(data_dir, "release_validate_patients.zip"),
    "evi_json": os.path.join(data_dir, "release_evidences.json"),
    "cond_json": os.path.join(data_dir, "release_conditions.json")
}

# Print information about files to be mounted
for name, path in essential_files.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Will mount {name}: {path} ({size_mb:.2f} MB)")
    else:
        print(f"WARNING: {name} not found at {path}")

# Create an image with all required packages
image = modal.Image.debian_slim().apt_install(
    "git", 
    "wget",
    "bzip2",
    "ca-certificates"
).pip_install(
    # Install packages directly into the main Python environment
    "numpy==2.2.4",
    "pandas==2.2.3",
    "scipy==1.15.2", 
    "tqdm==4.67.1",
    "torch==2.6.0",
    "torchvision==0.21.0",
    "scikit-learn==1.6.1",
    "matplotlib==3.10.1",
    "wandb==0.19.8",
    "orion==0.2.7"  # The orion package we installed
)

# Create a directory in the container for data files
image = image.run_commands("mkdir -p /data")

# Add essential files individually to the image
image = image.add_local_file(local_path=essential_files["train_zip"], remote_path="/data/release_train_patients.zip")
image = image.add_local_file(local_path=essential_files["val_zip"], remote_path="/data/release_validate_patients.zip")
image = image.add_local_file(local_path=essential_files["evi_json"], remote_path="/data/release_evidences.json")
image = image.add_local_file(local_path=essential_files["cond_json"], remote_path="/data/release_conditions.json")

# Add local code directories to the image
image = image.add_local_dir(local_path=os.path.join(local_repo_dir, "rl_model"), remote_path="/root/rl_model")
image = image.add_local_dir(local_path=os.path.join(local_repo_dir, "basd"), remote_path="/root/basd")

@app.function(image=image, gpu="A100:2")
def check_data():
    """Check that data files are properly mounted"""
    # List files in the data directory to verify
    print("Data directory contents:")
    subprocess.run(["ls", "-la", "/data"], check=True)
    
    file_list = [
        "release_train_patients.zip",
        "release_validate_patients.zip",
        "release_evidences.json",
        "release_conditions.json"
    ]
    
    for filename in file_list:
        filepath = f"/data/{filename}"
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ {filename} found ({size_mb:.2f} MB)")
        else:
            print(f"✗ {filename} not found!")
            
    # Check that local code was mounted
    print("\nRL model directory contents:")
    subprocess.run(["ls", "-la", "/root/rl_model"], check=True)
    
    print("\nBASE model directory contents:")
    subprocess.run(["ls", "-la", "/root/basd"], check=True)
    
    return "Data and code check complete"

@app.function(image=image, gpu="A100:2")
def train_rl_model():
    """Train the RL model using A100 GPUs"""
    # Set up environment variables
    os.environ["WANDB_MODE"] = "offline"
    
    # Change to the RL model directory
    os.chdir("/root/rl_model")
    
    # Create output directory and remove old directories if they exist
    output_dir = "./output"
    if os.path.exists(output_dir):
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Print current working directory to debug
    print(f"Current working directory: {os.getcwd()}")
    print(f"Data files in: /data")
    subprocess.run(["ls", "-la", "/data"], check=True)
    
    # Run the training script directly with Python, using correct data paths
    # Further reduce batch size for initial test
    command = [
        "python", "main.py", 
        "--seed", "42", 
        "--train_data_path", "/data/release_train_patients.zip", 
        "--val_data_path", "/data/release_validate_patients.zip", 
        "--train", 
        "--trail", "1", 
        "--nu", "2.826", 
        "--mu", "1.0", 
        "--lr", "0.000352", 
        "--lamb", "0.99", 
        "--gamma", "0.99", 
        "--eval_batch_size", "500",  # Reduced for minimal testing
        "--batch_size", "500",       # Reduced for minimal testing
        "--EPOCHS", "3",             # Further reduced for faster testing
        "--MAXSTEP", "30", 
        "--patience", "20", 
        "--eval_on_train_epoch_end",
        "--evi_meta_path", "/data/release_evidences.json", 
        "--patho_meta_path", "/data/release_conditions.json",
        "--save_dir", output_dir
    ]
    
    print(f"Running command: {' '.join(command)}")
    
    subprocess.run(
        command,
        stdout=sys.stdout, 
        stderr=sys.stderr,
        check=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1")
    )

@app.function(image=image, gpu="A100:2")
def train_basd_model():
    """Train the BASD model using A100 GPUs"""
    # Set up environment variables
    os.environ["WANDB_MODE"] = "offline"
    
    # Change to the BASD model directory
    os.chdir("/root/basd")
    
    # Create output directories and remove old ones if they exist
    output_dirs = ["./model", "./result", "./logs"]
    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            print(f"Removing existing directory: {dir_path}")
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    
    # Print current working directory to debug
    print(f"Current working directory: {os.getcwd()}")
    print(f"Data files in: /data")
    subprocess.run(["ls", "-la", "/data"], check=True)
    
    # Run the training script directly with Python, using correct data paths
    # Further reduce batch size for initial test
    command = [
        "python", "main.py", 
        "--cuda_idx", "0", 
        "--seed", "2919", 
        "--n_workers", "4", 
        "--thres", "0.5", 
        "--min_rec_ratio", "1.0", 
        "--output", "./", 
        "--run_mode", "train", 
        "--evi_meta_path", "/data/release_evidences.json", 
        "--patho_meta_path", "/data/release_conditions.json", 
        "--eval_batch_size", "500",  # Reduced for minimal testing
        "--batch_size", "500",       # Reduced for minimal testing
        "--num_epochs", "3",         # Further reduced for faster testing 
        "--interaction_length", "30", 
        "--patience", "20", 
        "--data", "/data/release_train_patients.zip", 
        "--eval_data", "/data/release_validate_patients.zip", 
        "--lr", "0.0003469", 
        "--only_diff_at_end", "1", 
        "--hidden_size", "2048", 
        "--num_layers", "3", 
        "--masked_inquired_actions", 
        "--include_turns_in_state", 
        "--compute_metrics_flag"
    ]
    
    print(f"Running command: {' '.join(command)}")
    
    subprocess.run(
        command,
        stdout=sys.stdout, 
        stderr=sys.stderr,
        check=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1")
    )

@app.local_entrypoint()
def main():
    try:
        print("First, checking data and code...")
        check_data.remote()
        
        print("Starting RL model training...")
        train_rl_model.remote()
        
        print("Starting BASD model training...")
        train_basd_model.remote()
        
        print("Training jobs submitted. Check the Modal dashboard for progress.")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise

if __name__ == "__main__":
    main()