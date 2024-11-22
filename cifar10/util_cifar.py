import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import os
import random
import json
from pathlib import Path

# Local imports
from baseline_module import *
from attack_module import *
from cifar_data_pre import *
from models import *

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_config(config_path):
    """Load configuration file"""
    with open(config_path) as f:
        return json.load(f)

def setup_paths():
    """Setup necessary paths and directories"""
    # Get the current file's directory (should be in cifar10 folder)
    current_path = Path(__file__).parent
    
    # Create necessary subdirectories
    save_model_path = current_path / 'saved_model'
    results_path = current_path / 'attack_result'
    
    # Create directories if they don't exist
    save_model_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    
    return current_path, save_model_path, results_path


