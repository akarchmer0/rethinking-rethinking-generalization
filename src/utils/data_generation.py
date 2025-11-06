"""Data generation utilities for experiments."""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional, Literal


class RandomNoiseDataset(Dataset):
    """Dataset of random noise images."""
    
    def __init__(self, n_samples: int, image_size: int = 32, 
                 n_classes: int = 10, noise_type: Literal['uniform', 'gaussian'] = 'uniform',
                 random_labels: bool = True, seed: Optional[int] = None):
        """
        Args:
            n_samples: Number of samples to generate
            image_size: Size of square images
            n_classes: Number of classes
            noise_type: Type of noise ('uniform' or 'gaussian')
            random_labels: If True, use random labels; else use sequential labels
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.n_samples = n_samples
        self.image_size = image_size
        self.n_classes = n_classes
        
        # Generate random images
        if noise_type == 'uniform':
            self.images = torch.rand(n_samples, 3, image_size, image_size)
        elif noise_type == 'gaussian':
            self.images = torch.randn(n_samples, 3, image_size, image_size)
            # Clip to [0, 1] range
            self.images = torch.clamp(self.images * 0.5 + 0.5, 0, 1)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Generate labels
        if random_labels:
            self.labels = torch.randint(0, n_classes, (n_samples,))
        else:
            self.labels = torch.arange(n_samples) % n_classes
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.images[idx], self.labels[idx].item()


class ModelLabeledDataset(Dataset):
    """Dataset where labels come from a neural network."""
    
    def __init__(self, model: torch.nn.Module, n_samples: int, 
                 image_size: int = 32, noise_type: Literal['uniform', 'gaussian'] = 'uniform',
                 device: str = 'cuda', seed: Optional[int] = None):
        """
        Args:
            model: Trained model to use for labeling
            n_samples: Number of samples to generate
            image_size: Size of square images
            noise_type: Type of noise for images
            device: Device to run model on
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.n_samples = n_samples
        self.image_size = image_size
        
        # Generate random images
        if noise_type == 'uniform':
            self.images = torch.rand(n_samples, 3, image_size, image_size)
        elif noise_type == 'gaussian':
            self.images = torch.randn(n_samples, 3, image_size, image_size)
            self.images = torch.clamp(self.images * 0.5 + 0.5, 0, 1)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Generate labels using the model
        model.eval()
        model = model.to(device)
        self.labels = []
        
        with torch.no_grad():
            batch_size = 256
            for i in range(0, n_samples, batch_size):
                batch = self.images[i:i+batch_size].to(device)
                outputs = model(batch)
                preds = outputs.argmax(dim=1)
                self.labels.append(preds.cpu())
        
        self.labels = torch.cat(self.labels)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.images[idx], self.labels[idx].item()


def corrupt_labels(labels: torch.Tensor, corruption_rate: float, 
                   n_classes: int, seed: Optional[int] = None) -> torch.Tensor:
    """
    Randomly corrupt a fraction of labels.
    
    Args:
        labels: Original labels
        corruption_rate: Fraction of labels to corrupt (0.0 to 1.0)
        n_classes: Number of classes
        seed: Random seed
    
    Returns:
        Corrupted labels
    """
    if seed is not None:
        np.random.seed(seed)
    
    corrupted_labels = labels.clone()
    n_corrupt = int(len(labels) * corruption_rate)
    
    if n_corrupt > 0:
        # Select random indices to corrupt
        corrupt_indices = np.random.choice(len(labels), n_corrupt, replace=False)
        
        # Assign random labels to corrupted indices
        corrupted_labels[corrupt_indices] = torch.randint(0, n_classes, (n_corrupt,))
    
    return corrupted_labels


def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 4,
                       random_labels: bool = False, corruption_rate: float = 0.0,
                       seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 train and test data loaders.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        random_labels: If True, replace all labels with random labels
        corruption_rate: Fraction of labels to corrupt (0.0 to 1.0)
        seed: Random seed
    
    Returns:
        train_loader, test_loader
    """
    # Standard CIFAR-10 normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                     download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
    
    # Modify labels if requested
    if random_labels or corruption_rate > 0.0:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        train_labels = torch.tensor(train_dataset.targets)
        
        if random_labels:
            # Replace all labels with random labels
            train_labels = torch.randint(0, 10, (len(train_labels),))
        elif corruption_rate > 0.0:
            # Corrupt a fraction of labels
            train_labels = corrupt_labels(train_labels, corruption_rate, 10, seed)
        
        train_dataset.targets = train_labels.tolist()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers,
                           pin_memory=True)
    
    return train_loader, test_loader


def create_smooth_random_dataset(base_model: torch.nn.Module, n_samples: int,
                                 image_size: int = 32, 
                                 noise_type: Literal['uniform', 'gaussian'] = 'uniform',
                                 device: str = 'cuda',
                                 seed: Optional[int] = None) -> Dataset:
    """
    Create a dataset of random images labeled by a neural network.
    
    This creates a "smooth random function" by using a trained neural network
    to label random noise images.
    
    Args:
        base_model: Trained model to use for labeling
        n_samples: Number of samples to generate
        image_size: Size of square images
        noise_type: Type of noise for images
        device: Device to run model on
        seed: Random seed
    
    Returns:
        Dataset with random images labeled by the model
    """
    return ModelLabeledDataset(base_model, n_samples, image_size, 
                              noise_type, device, seed)


class RandomWalkDataset(Dataset):
    """Dataset generated via random walk in pixel space.
    
    At each step, exactly one pixel is changed to a random value,
    and the label is randomized. This creates data with much lower
    Lipschitz continuity than uniform random noise.
    """
    
    def __init__(self, n_samples: int, image_size: int = 32,
                 n_classes: int = 10, seed: Optional[int] = None,
                 continuation_from: Optional[Tuple[torch.Tensor, int]] = None):
        """
        Generate data via random walk in pixel space.
        
        Args:
            n_samples: Number of steps in random walk
            image_size: Size of square images (will be image_size x image_size x 3)
            n_classes: Number of classes
            seed: Random seed for reproducibility
            continuation_from: (image, label) tuple to continue walk from
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.n_samples = n_samples
        self.image_size = image_size
        self.n_classes = n_classes
        
        # Initialize storage
        self.images = torch.zeros(n_samples, 3, image_size, image_size)
        self.labels = torch.zeros(n_samples, dtype=torch.long)
        
        # Starting point
        if continuation_from is not None:
            current_image = continuation_from[0].clone()
            current_label = continuation_from[1]
        else:
            # Start with random image and label
            current_image = torch.rand(3, image_size, image_size)
            current_label = np.random.randint(0, n_classes)
        
        # Generate random walk
        for i in range(n_samples):
            # Store current state
            self.images[i] = current_image.clone()
            self.labels[i] = current_label
            
            # Take one step: change exactly 1 pixel
            channel = np.random.randint(0, 3)
            row = np.random.randint(0, image_size)
            col = np.random.randint(0, image_size)
            current_image[channel, row, col] = np.random.rand()
            
            # Randomize label
            current_label = np.random.randint(0, n_classes)
        
        # Store final state for potential continuation
        self.final_state = (current_image.clone(), current_label)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.images[idx], self.labels[idx].item()
    
    def get_final_state(self) -> Tuple[torch.Tensor, int]:
        """Get the final (image, label) state for continuation."""
        return self.final_state

