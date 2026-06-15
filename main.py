"""
	Main function to execute the complete experimental workflow
"""
import os
import random

import torch

output_dir = "results"

def set_seed(seed=42):
    """set seed to ensure reproducibility"""
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from src.preprocess.ubfc_rPPG_dataset_info import print_dataset_preview

def main():
    """
    Main function to execute the complete experimental workflow:
    """
    set_seed()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print_dataset_preview()

if __name__ == "__main__":
    main()
