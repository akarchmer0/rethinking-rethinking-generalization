"""
Experiment 2: Smoothness Analysis

Analyze and compare smoothness properties of networks trained on true vs random labels.
Measures:
- Average gradient norm
- Lipschitz constant estimation
- Spectral norm
- Path norm
- Local variation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pickle
import numpy as np
from typing import Dict, Optional
from torch.utils.data import DataLoader

from utils.config import ExperimentConfig, set_random_seeds, get_device
from utils.data_generation import get_cifar10_loaders, RandomNoiseDataset
from models.architectures import get_model
from analysis.metrics import SmoothnessMetrics


def analyze_model_smoothness(
    model: torch.nn.Module,
    dataloader: DataLoader,
    n_samples: int = 1000,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict:
    """
    Analyze smoothness of a single model.
    
    Args:
        model: Trained model
        dataloader: Data loader for analysis
        n_samples: Number of samples to use
        device: Device to run on
        verbose: Whether to print progress
    
    Returns:
        Dictionary of smoothness metrics
    """
    model = model.to(device)
    model.eval()
    
    if verbose:
        print("Computing smoothness metrics...")
    
    # Average gradient norm
    if verbose:
        print("  - Computing average gradient norm...")
    grad_norm = SmoothnessMetrics.average_gradient_norm(
        model, dataloader, n_samples=n_samples, device=device
    )
    
    # Spectral norm
    if verbose:
        print("  - Computing spectral norm...")
    spectral_norm = SmoothnessMetrics.spectral_norm(model)
    
    # Path norm
    if verbose:
        print("  - Computing path norm...")
    path_norm = SmoothnessMetrics.path_norm(model)
    
    # Local Lipschitz constant (sample a few points)
    if verbose:
        print("  - Estimating local Lipschitz constant...")
    lipschitz_estimates = []
    sample_count = 0
    
    for inputs, _ in dataloader:
        if sample_count >= min(100, n_samples):
            break
        
        # Take first image from batch
        x = inputs[0:1].to(device)
        lipschitz = SmoothnessMetrics.local_lipschitz(
            model, x, radius=0.1, n_samples=50, device=device
        )
        lipschitz_estimates.append(lipschitz)
        sample_count += 1
    
    avg_lipschitz = np.mean(lipschitz_estimates)
    
    # Local variation
    if verbose:
        print("  - Computing local variation...")
    variations = []
    sample_count = 0
    
    for inputs, _ in dataloader:
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


def run_smoothness_analysis(
    baseline_results_path: Optional[str] = None,
    architectures: list = ['resnet18', 'vgg11', 'mlp'],
    seeds: list = [42, 123, 456],
    n_samples: int = 1000,
    verbose: bool = True
) -> Dict:
    """
    Run smoothness analysis on trained models.
    
    Args:
        baseline_results_path: Path to baseline experiment results
        architectures: List of architectures to analyze
        seeds: Random seeds
        n_samples: Number of samples for analysis
        verbose: Whether to print progress
    
    Returns:
        Dictionary of smoothness analysis results
    """
    device = get_device()
    ExperimentConfig.create_directories()
    
    results = {
        'true_labels': {},
        'random_labels': {}
    }
    
    for arch in architectures:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Analyzing {arch.upper()}")
            print(f"{'='*60}")
        
        results['true_labels'][arch] = []
        results['random_labels'][arch] = []
        
        for seed in seeds:
            if verbose:
                print(f"\n--- Seed {seed} ---")
            
            set_random_seeds(seed)
            
            # Get data loader
            _, test_loader = get_cifar10_loaders(
                batch_size=128,
                num_workers=4,
                random_labels=False,
                seed=seed
            )
            
            # Analyze model trained with true labels
            if verbose:
                print("\nAnalyzing model trained with TRUE LABELS...")
            
            model_true = get_model(arch, num_classes=10)
            model_path = ExperimentConfig.get_model_save_path(
                'baseline', f'{arch}_true', seed
            )
            
            if model_path.exists():
                model_true.load_state_dict(torch.load(model_path))
                smoothness_true = analyze_model_smoothness(
                    model_true, test_loader, n_samples=n_samples,
                    device=device, verbose=verbose
                )
                results['true_labels'][arch].append({
                    'seed': seed,
                    'smoothness': smoothness_true
                })
            else:
                print(f"Warning: Model not found at {model_path}")
            
            # Analyze model trained with random labels
            if verbose:
                print("\nAnalyzing model trained with RANDOM LABELS...")
            
            model_random = get_model(arch, num_classes=10)
            model_path = ExperimentConfig.get_model_save_path(
                'baseline', f'{arch}_random', seed
            )
            
            if model_path.exists():
                model_random.load_state_dict(torch.load(model_path))
                smoothness_random = analyze_model_smoothness(
                    model_random, test_loader, n_samples=n_samples,
                    device=device, verbose=verbose
                )
                results['random_labels'][arch].append({
                    'seed': seed,
                    'smoothness': smoothness_random
                })
            else:
                print(f"Warning: Model not found at {model_path}")
    
    # Save results
    results_path = ExperimentConfig.get_results_save_path('smoothness_analysis')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Analysis completed!")
        print(f"Results saved to {results_path}")
        print(f"{'='*60}")
    
    return results


def compare_smoothness(results: Dict) -> None:
    """Compare smoothness between true and random label models."""
    import numpy as np
    
    print("\n" + "="*70)
    print("SMOOTHNESS COMPARISON")
    print("="*70)
    
    metrics = ['gradient_norm', 'spectral_norm', 'path_norm', 
              'lipschitz_constant', 'local_variation']
    
    for arch in results['true_labels'].keys():
        print(f"\n{arch.upper()}:")
        print("-" * 70)
        
        for metric in metrics:
            true_values = [r['smoothness'][metric] 
                          for r in results['true_labels'][arch]]
            random_values = [r['smoothness'][metric] 
                            for r in results['random_labels'][arch]]
            
            print(f"\n{metric}:")
            print(f"  True labels:   {np.mean(true_values):.6f} ± {np.std(true_values):.6f}")
            print(f"  Random labels: {np.mean(random_values):.6f} ± {np.std(random_values):.6f}")
            
            # Compute ratio
            ratio = np.mean(random_values) / (np.mean(true_values) + 1e-10)
            print(f"  Ratio (random/true): {ratio:.2f}x")


if __name__ == '__main__':
    # Run analysis (assumes baseline experiment has been run)
    results = run_smoothness_analysis(
        architectures=['resnet18'],
        seeds=[42],
        n_samples=1000,
        verbose=True
    )
    
    # Compare
    compare_smoothness(results)

