"""Configuration file for experiments."""

import os
from pathlib import Path


class ExperimentConfig:
    """Configuration class for all experiments."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    RESULTS_DIR = PROJECT_ROOT / "results"
    RAW_DATA_DIR = RESULTS_DIR / "raw_data"
    PROCESSED_DIR = RESULTS_DIR / "processed"
    FIGURES_DIR = RESULTS_DIR / "figures"
    
    # Data parameters
    DATASET = 'CIFAR10'
    N_SAMPLES = 50000
    IMAGE_SIZE = 32
    N_CLASSES = 10
    DATA_AUGMENTATION = False
    
    # Training parameters
    BATCH_SIZE = 512
    LEARNING_RATE = 0.01
    EPOCHS = 50
    OPTIMIZER = 'SGD'
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    SCHEDULER = 'cosine'  # 'cosine', 'step', or None
    
    # Architecture parameters
    ARCHITECTURES = ['resnet18', 'vgg11', 'mlp']
    
    # Analysis parameters
    N_SMOOTHNESS_SAMPLES = 1000
    FOURIER_RESOLUTION = 64
    LIPSCHITZ_SAMPLES = 500
    GRADIENT_NORM_SAMPLES = 1000
    
    # Two-stage learning parameters
    STAGE1_SAMPLES = 50000
    STAGE2_SAMPLES = 50000
    NOISE_TYPE = 'uniform'  # 'uniform' or 'gaussian'
    
    # Progressive corruption parameters
    CORRUPTION_RATES = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Random seeds for reproducibility
    SEEDS = [42, 123, 456, 789, 2024]
    
    # Compute parameters
    DEVICE = 'cuda'  # Will fallback to 'cpu' if CUDA unavailable
    NUM_WORKERS = 2
    PIN_MEMORY = True
    USE_AMP = True  # Automatic Mixed Precision (faster training on modern GPUs)
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_CHECKPOINTS = True
    CHECKPOINT_INTERVAL = 50  # Save every N epochs
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories."""
        for dir_path in [cls.RESULTS_DIR, cls.RAW_DATA_DIR, 
                        cls.PROCESSED_DIR, cls.FIGURES_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_save_path(cls, experiment_name: str, model_name: str, 
                           seed: int, suffix: str = '') -> Path:
        """Generate standardized model save path."""
        filename = f"{experiment_name}_{model_name}_seed{seed}{suffix}.pt"
        return cls.RAW_DATA_DIR / filename
    
    @classmethod
    def get_results_save_path(cls, experiment_name: str, suffix: str = '') -> Path:
        """Generate standardized results save path."""
        filename = f"{experiment_name}_results{suffix}.pkl"
        return cls.PROCESSED_DIR / filename


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the appropriate device (CUDA or CPU)."""
    import torch
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu')

