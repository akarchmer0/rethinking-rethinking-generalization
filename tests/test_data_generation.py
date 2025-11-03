"""Tests for data generation utilities."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pytest
import torch
import numpy as np

from src.utils.data_generation import (
    RandomNoiseDataset,
    corrupt_labels,
)


def test_random_noise_dataset_uniform():
    """Test random noise dataset with uniform noise."""
    dataset = RandomNoiseDataset(
        n_samples=100,
        image_size=32,
        n_classes=10,
        noise_type='uniform',
        random_labels=True,
        seed=42
    )
    
    assert len(dataset) == 100
    
    img, label = dataset[0]
    assert img.shape == (3, 32, 32)
    assert 0 <= label < 10
    assert torch.all((img >= 0) & (img <= 1))


def test_random_noise_dataset_gaussian():
    """Test random noise dataset with Gaussian noise."""
    dataset = RandomNoiseDataset(
        n_samples=100,
        image_size=32,
        n_classes=10,
        noise_type='gaussian',
        random_labels=True,
        seed=42
    )
    
    assert len(dataset) == 100
    
    img, label = dataset[0]
    assert img.shape == (3, 32, 32)
    assert 0 <= label < 10
    # Gaussian noise is clipped to [0, 1]
    assert torch.all((img >= 0) & (img <= 1))


def test_corrupt_labels():
    """Test label corruption."""
    labels = torch.arange(100)
    corruption_rate = 0.5
    
    corrupted = corrupt_labels(labels, corruption_rate, n_classes=10, seed=42)
    
    # Should have approximately corruption_rate * n labels different
    n_different = (corrupted != labels).sum().item()
    expected = int(100 * corruption_rate)
    
    # Allow some tolerance
    assert abs(n_different - expected) <= 10


def test_corrupt_labels_zero_rate():
    """Test that zero corruption rate doesn't change labels."""
    labels = torch.arange(100)
    
    corrupted = corrupt_labels(labels, 0.0, n_classes=10, seed=42)
    
    assert torch.all(corrupted == labels)


def test_corrupt_labels_full_rate():
    """Test full corruption rate."""
    labels = torch.arange(10)
    
    corrupted = corrupt_labels(labels, 1.0, n_classes=10, seed=42)
    
    # All labels should be corrupted
    assert (corrupted != labels).sum().item() == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

