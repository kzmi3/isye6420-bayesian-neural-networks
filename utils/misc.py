import os
import json
import random
import torch
import numpy as np


def log_message(message, log_path='logs'):
    os.makedirs(log_path, exist_ok=True)  # Ensure the directory exists
    log_path = f'{log_path}/training_log.txt'
    with open(log_path, 'a') as f:
        f.write(message + '\n')  

def set_random_seed(seed=42):
    """
    Set the random seed for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_best_model_config(results_log_path):
    """
    Loads the best model based on validation loss from the results log.
    
    Args:
        results_log_path: Path to the results.json file.
    
    Returns:
        best_checkpoint_path: Path to best model checkpoint (best validation loss).
        best_config: Configuration of the experiment with the best validation loss.
    """
    # Load the results log
    with open(results_log_path, "r") as f:
        results = json.load(f)

    # Find the experiment with the lowest validation loss
    best_config = min(results, key=lambda x: x["best_val_loss"])
    print(f"Best checkpoint found at: {best_config["checkpoint_path"]}")
    print(f"Best validation loss: {best_config['best_val_loss']}")

    return best_config