"""Tests for analysis metrics."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from src.analysis.metrics import SmoothnessMetrics, GeneralizationMetrics
from src.models.architectures import SimpleMLP


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    model = SimpleMLP(input_size=100, hidden_sizes=[50, 20], num_classes=10)
    return model


@pytest.fixture
def test_dataloader():
    """Create a simple dataloader for testing."""
    X = torch.randn(100, 100)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10)


def test_gradient_norm(simple_model):
    """Test gradient norm computation."""
    x = torch.randn(5, 100)
    grad_norm = SmoothnessMetrics.gradient_norm(simple_model, x, device='cpu')
    
    assert isinstance(grad_norm, float)
    assert grad_norm >= 0


def test_spectral_norm(simple_model):
    """Test spectral norm computation."""
    spec_norm = SmoothnessMetrics.spectral_norm(simple_model)
    
    assert isinstance(spec_norm, float)
    assert spec_norm > 0


def test_path_norm(simple_model):
    """Test path norm computation."""
    p_norm = SmoothnessMetrics.path_norm(simple_model)
    
    assert isinstance(p_norm, float)
    assert p_norm > 0


def test_agreement_rate(simple_model, test_dataloader):
    """Test agreement rate between two models."""
    model2 = SimpleMLP(input_size=100, hidden_sizes=[50, 20], num_classes=10)
    
    agreement = GeneralizationMetrics.agreement_rate(
        simple_model, model2, test_dataloader, device='cpu'
    )
    
    assert isinstance(agreement, float)
    assert 0 <= agreement <= 100


def test_function_distance(simple_model, test_dataloader):
    """Test function distance computation."""
    model2 = SimpleMLP(input_size=100, hidden_sizes=[50, 20], num_classes=10)
    
    distance = GeneralizationMetrics.function_distance(
        simple_model, model2, test_dataloader, device='cpu'
    )
    
    assert isinstance(distance, float)
    assert distance >= 0


def test_test_accuracy(simple_model, test_dataloader):
    """Test accuracy computation."""
    accuracy = GeneralizationMetrics.test_accuracy(
        simple_model, test_dataloader, device='cpu'
    )
    
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

