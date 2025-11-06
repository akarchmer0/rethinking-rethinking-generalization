"""
Experiment 6: Random Walk Learning

Train networks on random walk data where each image differs from the previous
by exactly 1 pixel change. This creates data with much lower Lipschitz continuity
than uniform random noise, making memorization/interpolation significantly harder.

Includes:
1. Baseline training on random walk data
2. Smoothness analysis
3. Two-stage learning with two variants (walk and uniform)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import pickle
import numpy as np
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader

from utils.config import ExperimentConfig, set_random_seeds, get_device
from utils.data_generation import RandomWalkDataset, RandomNoiseDataset, ModelLabeledDataset
from models.architectures import get_model
from models.training import Trainer
from analysis.metrics import SmoothnessMetrics, GeneralizationMetrics


def train_on_random_walk(
    architecture: str = 'resnet18',
    n_samples: int = 50000,
    test_samples_walk: int = 10000,
    test_samples_uniform: int = 10000,
    epochs: int = 200,
    seed: int = 42,
    device: str = 'cuda',
    verbose: bool = True
) -> Tuple[nn.Module, Dict, Tuple[torch.Tensor, int]]:
    """
    Train a network on random walk data - Baseline experiment.
    
    Args:
        architecture: Model architecture to use
        n_samples: Number of training samples (walk steps)
        test_samples_walk: Number of test samples (walk continuation)
        test_samples_uniform: Number of uniform random test samples
        epochs: Number of training epochs
        seed: Random seed
        device: Device to train on
        verbose: Print progress
    
    Returns:
        (trained_model, results_dict, final_walk_state)
    """
    if verbose:
        print("="*70)
        print("BASELINE: Training on Random Walk Data")
        print("="*70)
    
    set_random_seeds(seed)
    device = torch.device(device)
    
    # Generate random walk training data
    if verbose:
        print(f"\nGenerating random walk with {n_samples} steps...")
    
    train_dataset = RandomWalkDataset(
        n_samples=n_samples,
        image_size=32,
        n_classes=10,
        seed=seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=ExperimentConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=ExperimentConfig.NUM_WORKERS,
        pin_memory=True
    )
    
    # Get final state for test set continuation
    final_state = train_dataset.get_final_state()
    
    # Generate test set: continuation of walk
    if verbose:
        print(f"Generating walk continuation test set ({test_samples_walk} steps)...")
    
    test_dataset_walk = RandomWalkDataset(
        n_samples=test_samples_walk,
        image_size=32,
        n_classes=10,
        seed=seed + 1000,
        continuation_from=final_state
    )
    
    test_loader_walk = DataLoader(
        test_dataset_walk,
        batch_size=ExperimentConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=ExperimentConfig.NUM_WORKERS,
        pin_memory=True
    )
    
    # Generate test set: uniform random
    if verbose:
        print(f"Generating uniform random test set ({test_samples_uniform} samples)...")
    
    test_dataset_uniform = RandomNoiseDataset(
        n_samples=test_samples_uniform,
        image_size=32,
        n_classes=10,
        noise_type='uniform',
        random_labels=True,
        seed=seed + 2000
    )
    
    test_loader_uniform = DataLoader(
        test_dataset_uniform,
        batch_size=ExperimentConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=ExperimentConfig.NUM_WORKERS,
        pin_memory=True
    )
    
    # Train model
    if verbose:
        print(f"\nTraining {architecture} on random walk data...")
    
    model = get_model(architecture, num_classes=10)
    trainer = Trainer(
        model,
        device,
        learning_rate=ExperimentConfig.LEARNING_RATE,
        momentum=ExperimentConfig.MOMENTUM,
        weight_decay=ExperimentConfig.WEIGHT_DECAY,
        scheduler_type=ExperimentConfig.SCHEDULER,
        use_amp=ExperimentConfig.USE_AMP
    )
    
    history = trainer.train(
        train_loader,
        test_loader=None,  # No test during training
        epochs=epochs,
        verbose=verbose
    )
    
    # Evaluate on both test sets
    if verbose:
        print("\nEvaluating on test sets...")
    
    test_acc_walk = GeneralizationMetrics.test_accuracy(
        model, test_loader_walk, device=device
    )
    
    test_acc_uniform = GeneralizationMetrics.test_accuracy(
        model, test_loader_uniform, device=device
    )
    
    if verbose:
        print(f"\nResults:")
        print(f"  Training accuracy: {history['train_acc'][-1]:.2f}%")
        print(f"  Test (walk continuation): {test_acc_walk:.2f}%")
        print(f"  Test (uniform random): {test_acc_uniform:.2f}%")
    
    results = {
        'architecture': architecture,
        'training_history': history,
        'final_train_acc': history['train_acc'][-1],
        'test_acc_walk': test_acc_walk,
        'test_acc_uniform': test_acc_uniform,
        'n_samples': n_samples,
        'epochs': epochs,
        'seed': seed
    }
    
    # Save model
    save_path = ExperimentConfig.get_model_save_path(
        'random_walk', f'{architecture}_baseline', seed
    )
    torch.save(model.state_dict(), save_path)
    
    return model, results, final_state


def analyze_random_walk_smoothness(
    model: nn.Module,
    n_samples: int = 1000,
    device: str = 'cuda',
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Analyze smoothness of model trained on random walk data.
    
    Args:
        model: Trained model
        n_samples: Number of samples for analysis
        device: Device to run on
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dictionary of smoothness metrics
    """
    if verbose:
        print("\n" + "="*70)
        print("SMOOTHNESS ANALYSIS: Random Walk Model")
        print("="*70)
    
    set_random_seeds(seed)
    device = torch.device(device)
    
    # Generate test data for analysis
    test_dataset = RandomWalkDataset(
        n_samples=n_samples,
        image_size=32,
        n_classes=10,
        seed=seed + 3000
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4
    )
    
    model = model.to(device)
    model.eval()
    
    if verbose:
        print("\nComputing smoothness metrics...")
    
    # Compute all smoothness metrics
    if verbose:
        print("  - Average gradient norm...")
    grad_norm = SmoothnessMetrics.average_gradient_norm(
        model, test_loader, n_samples=n_samples, device=device
    )
    
    if verbose:
        print("  - Spectral norm...")
    spectral_norm = SmoothnessMetrics.spectral_norm(model)
    
    if verbose:
        print("  - Path norm...")
    path_norm = SmoothnessMetrics.path_norm(model)
    
    if verbose:
        print("  - Local Lipschitz constant...")
    lipschitz_estimates = []
    sample_count = 0
    for inputs, _ in test_loader:
        if sample_count >= min(100, n_samples):
            break
        x = inputs[0:1].to(device)
        lipschitz = SmoothnessMetrics.local_lipschitz(
            model, x, radius=0.1, n_samples=50, device=device
        )
        lipschitz_estimates.append(lipschitz)
        sample_count += 1
    avg_lipschitz = np.mean(lipschitz_estimates)
    
    if verbose:
        print("  - Local variation...")
    variations = []
    sample_count = 0
    for inputs, _ in test_loader:
        if sample_count >= min(100, n_samples):
            break
        x = inputs[0:1].to(device)
        variation = SmoothnessMetrics.local_variation(
            model, x, epsilon=0.01, n_samples=50, device=device
        )
        variations.append(variation)
        sample_count += 1
    avg_variation = np.mean(variations)
    
    results = {
        'gradient_norm': grad_norm,
        'spectral_norm': spectral_norm,
        'path_norm': path_norm,
        'lipschitz_constant': avg_lipschitz,
        'local_variation': avg_variation
    }
    
    if verbose:
        print("\nSmootness metrics:")
        for key, value in results.items():
            print(f"  {key}: {value:.6f}")
    
    return results


def two_stage_random_walk(
    stage1_model: nn.Module,
    final_walk_state: Tuple[torch.Tensor, int],
    architecture: str = 'resnet18',
    n_samples: int = 50000,
    test_samples: int = 10000,
    epochs: int = 200,
    seed: int = 42,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict:
    """
    Two-stage learning with random walk.
    
    Stage 2a: Train Network_2 on random walk labeled by Network_1
    Stage 2b: Train Network_2 on uniform random labeled by Network_1
    
    Args:
        stage1_model: Network_1 trained on random walk
        final_walk_state: Final state from training walk
        architecture: Architecture for Network_2
        n_samples: Training samples for stage 2
        test_samples: Test samples
        epochs: Training epochs
        seed: Random seed
        device: Device to train on
        verbose: Print progress
    
    Returns:
        Dictionary with results from both variants
    """
    if verbose:
        print("\n" + "="*70)
        print("TWO-STAGE LEARNING: Random Walk")
        print("="*70)
    
    device = torch.device(device)
    stage1_model = stage1_model.to(device)
    stage1_model.eval()
    
    results = {}
    
    # ========================================
    # Variant A: Network_2 on random walk from Network_1
    # ========================================
    if verbose:
        print("\n" + "-"*70)
        print("VARIANT A: Training on Random Walk labeled by Network_1")
        print("-"*70)
    
    set_random_seeds(seed + 4000)
    
    # Generate new random walk
    if verbose:
        print(f"\nGenerating new random walk ({n_samples} steps)...")
    
    walk_dataset = RandomWalkDataset(
        n_samples=n_samples,
        image_size=32,
        n_classes=10,
        seed=seed + 4000
    )
    
    # Label with Network_1
    if verbose:
        print("Labeling with Network_1...")
    
    walk_images = walk_dataset.images.to(device)
    walk_labels = []
    
    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(walk_images), batch_size):
            batch = walk_images[i:i+batch_size]
            outputs = stage1_model(batch)
            preds = outputs.argmax(dim=1)
            walk_labels.append(preds.cpu())
    
    walk_labels = torch.cat(walk_labels)
    
    # Create dataset
    from torch.utils.data import TensorDataset
    stage2a_dataset = TensorDataset(walk_dataset.images, walk_labels)
    stage2a_loader = DataLoader(
        stage2a_dataset,
        batch_size=ExperimentConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=ExperimentConfig.NUM_WORKERS,
        pin_memory=True
    )
    
    # Train Network_2a
    if verbose:
        print(f"\nTraining Network_2a on walk data...")
    
    model2a = get_model(architecture, num_classes=10)
    trainer2a = Trainer(
        model2a,
        device,
        learning_rate=ExperimentConfig.LEARNING_RATE,
        momentum=ExperimentConfig.MOMENTUM,
        weight_decay=ExperimentConfig.WEIGHT_DECAY,
        scheduler_type=ExperimentConfig.SCHEDULER,
        use_amp=ExperimentConfig.USE_AMP
    )
    
    history2a = trainer2a.train(
        stage2a_loader,
        test_loader=None,
        epochs=epochs,
        verbose=verbose
    )
    
    # Generate test set (walk continuation)
    test_walk_final_state = walk_dataset.get_final_state()
    test_dataset_walk = RandomWalkDataset(
        n_samples=test_samples,
        image_size=32,
        n_classes=10,
        seed=seed + 5000,
        continuation_from=test_walk_final_state
    )
    
    # Label test set with Network_1
    test_images = test_dataset_walk.images.to(device)
    test_labels = []
    
    with torch.no_grad():
        for i in range(0, len(test_images), 256):
            batch = test_images[i:i+256]
            outputs = stage1_model(batch)
            preds = outputs.argmax(dim=1)
            test_labels.append(preds.cpu())
    
    test_labels = torch.cat(test_labels)
    test_dataset_walk_labeled = TensorDataset(test_images.cpu(), test_labels)
    test_loader_walk = DataLoader(test_dataset_walk_labeled, batch_size=256, shuffle=False)
    
    # Evaluate agreement
    agreement_walk = GeneralizationMetrics.agreement_rate(
        stage1_model, model2a, test_loader_walk, device=device
    )
    
    if verbose:
        print(f"\nVariant A Results:")
        print(f"  Final training accuracy: {history2a['train_acc'][-1]:.2f}%")
        print(f"  Agreement with Network_1: {agreement_walk:.2f}%")
    
    results['variant_a_walk'] = {
        'history': history2a,
        'final_train_acc': history2a['train_acc'][-1],
        'agreement': agreement_walk
    }
    
    # ========================================
    # Variant B: Network_2 on uniform random from Network_1
    # ========================================
    if verbose:
        print("\n" + "-"*70)
        print("VARIANT B: Training on Uniform Random labeled by Network_1")
        print("-"*70)
    
    set_random_seeds(seed + 6000)
    
    # Generate uniform random images
    if verbose:
        print(f"\nGenerating uniform random images ({n_samples} samples)...")
    
    uniform_dataset = RandomNoiseDataset(
        n_samples=n_samples,
        image_size=32,
        n_classes=10,
        noise_type='uniform',
        random_labels=False,  # Will label with Network_1
        seed=seed + 6000
    )
    
    # Label with Network_1
    if verbose:
        print("Labeling with Network_1...")
    
    uniform_images = uniform_dataset.images.to(device)
    uniform_labels = []
    
    with torch.no_grad():
        for i in range(0, len(uniform_images), 256):
            batch = uniform_images[i:i+256]
            outputs = stage1_model(batch)
            preds = outputs.argmax(dim=1)
            uniform_labels.append(preds.cpu())
    
    uniform_labels = torch.cat(uniform_labels)
    
    # Create dataset
    stage2b_dataset = TensorDataset(uniform_dataset.images, uniform_labels)
    stage2b_loader = DataLoader(
        stage2b_dataset,
        batch_size=ExperimentConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=ExperimentConfig.NUM_WORKERS,
        pin_memory=True
    )
    
    # Train Network_2b
    if verbose:
        print(f"\nTraining Network_2b on uniform random data...")
    
    model2b = get_model(architecture, num_classes=10)
    trainer2b = Trainer(
        model2b,
        device,
        learning_rate=ExperimentConfig.LEARNING_RATE,
        momentum=ExperimentConfig.MOMENTUM,
        weight_decay=ExperimentConfig.WEIGHT_DECAY,
        scheduler_type=ExperimentConfig.SCHEDULER,
        use_amp=ExperimentConfig.USE_AMP
    )
    
    history2b = trainer2b.train(
        stage2b_loader,
        test_loader=None,
        epochs=epochs,
        verbose=verbose
    )
    
    # Generate test set (uniform random)
    test_dataset_uniform = RandomNoiseDataset(
        n_samples=test_samples,
        image_size=32,
        n_classes=10,
        noise_type='uniform',
        random_labels=False,
        seed=seed + 7000
    )
    
    # Label test set with Network_1
    test_uniform_images = test_dataset_uniform.images.to(device)
    test_uniform_labels = []
    
    with torch.no_grad():
        for i in range(0, len(test_uniform_images), 256):
            batch = test_uniform_images[i:i+256]
            outputs = stage1_model(batch)
            preds = outputs.argmax(dim=1)
            test_uniform_labels.append(preds.cpu())
    
    test_uniform_labels = torch.cat(test_uniform_labels)
    test_dataset_uniform_labeled = TensorDataset(test_uniform_images.cpu(), test_uniform_labels)
    test_loader_uniform = DataLoader(test_dataset_uniform_labeled, batch_size=256, shuffle=False)
    
    # Evaluate agreement
    agreement_uniform = GeneralizationMetrics.agreement_rate(
        stage1_model, model2b, test_loader_uniform, device=device
    )
    
    if verbose:
        print(f"\nVariant B Results:")
        print(f"  Final training accuracy: {history2b['train_acc'][-1]:.2f}%")
        print(f"  Agreement with Network_1: {agreement_uniform:.2f}%")
    
    results['variant_b_uniform'] = {
        'history': history2b,
        'final_train_acc': history2b['train_acc'][-1],
        'agreement': agreement_uniform
    }
    
    return results


def run_random_walk_experiment(
    architecture: str = 'resnet18',
    n_samples: int = 50000,
    test_samples_walk: int = 10000,
    test_samples_uniform: int = 10000,
    epochs: int = 200,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Run complete random walk experiment.
    
    Includes:
    1. Baseline training on random walk
    2. Smoothness analysis
    3. Two-stage learning (both variants)
    
    Args:
        architecture: Model architecture
        n_samples: Training samples
        test_samples_walk: Test samples (walk continuation)
        test_samples_uniform: Test samples (uniform random)
        epochs: Training epochs
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Complete results dictionary
    """
    device = get_device()
    ExperimentConfig.create_directories()
    
    # Experiment 1: Baseline
    model, baseline_results, final_state = train_on_random_walk(
        architecture=architecture,
        n_samples=n_samples,
        test_samples_walk=test_samples_walk,
        test_samples_uniform=test_samples_uniform,
        epochs=epochs,
        seed=seed,
        device=device,
        verbose=verbose
    )
    
    # Experiment 2: Smoothness
    smoothness_results = analyze_random_walk_smoothness(
        model=model,
        n_samples=1000,
        device=device,
        seed=seed,
        verbose=verbose
    )
    
    # Experiment 3: Two-stage learning
    two_stage_results = two_stage_random_walk(
        stage1_model=model,
        final_walk_state=final_state,
        architecture=architecture,
        n_samples=n_samples,
        test_samples=test_samples_walk,
        epochs=epochs,
        seed=seed,
        device=device,
        verbose=verbose
    )
    
    # Compile all results
    complete_results = {
        'baseline': baseline_results,
        'smoothness': smoothness_results,
        'two_stage': two_stage_results,
        'config': {
            'architecture': architecture,
            'n_samples': n_samples,
            'test_samples_walk': test_samples_walk,
            'test_samples_uniform': test_samples_uniform,
            'epochs': epochs,
            'seed': seed
        }
    }
    
    # Save results
    results_path = ExperimentConfig.get_results_save_path('random_walk_experiment')
    with open(results_path, 'wb') as f:
        pickle.dump(complete_results, f)
    
    if verbose:
        print("\n" + "="*70)
        print("RANDOM WALK EXPERIMENT COMPLETE!")
        print("="*70)
        print(f"\nResults saved to {results_path}")
        print("\nSummary:")
        print(f"  Training accuracy: {baseline_results['final_train_acc']:.2f}%")
        print(f"  Test (walk): {baseline_results['test_acc_walk']:.2f}%")
        print(f"  Test (uniform): {baseline_results['test_acc_uniform']:.2f}%")
        print(f"\n  Two-stage agreement (walk): {two_stage_results['variant_a_walk']['agreement']:.2f}%")
        print(f"  Two-stage agreement (uniform): {two_stage_results['variant_b_uniform']['agreement']:.2f}%")
    
    return complete_results


if __name__ == '__main__':
    # Run random walk experiment
    results = run_random_walk_experiment(
        architecture='resnet18',
        n_samples=50000,
        test_samples_walk=10000,
        test_samples_uniform=10000,
        epochs=200,
        seed=42,
        verbose=True
    )


