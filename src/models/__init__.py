"""Neural network models and training utilities."""

from .architectures import get_model, ResNet18, VGG11, SimpleMLP
from .training import Trainer, train_model

__all__ = [
    'get_model',
    'ResNet18',
    'VGG11',
    'SimpleMLP',
    'Trainer',
    'train_model',
]

