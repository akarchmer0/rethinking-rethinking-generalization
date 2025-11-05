"""
Experiment 5: Progressive Label Corruption

Train models with varying levels of label corruption to study the relationship
between data realizability and generalization.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pickle
import numpy as np
from typing import Dict, List, Optional

from utils.config import ExperimentConfig, set_random_seeds, get_device
from utils.data_generation import get_cifar10_loaders
from models.architectures import get_model
from models.training import Trainer
from analysis.metrics import SmoothnessMetrics


def train_with_corruption(
    architecture: str,
    corruption_rate: float,
    epochs: int = 200,
    seed: int = 42,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict:
    """
    Train a model with a specific label corruption rate.
    
    Args:
        architecture: Architecture to use
        corruption_rate: Fraction of labels to corrupt (0.0 to 1.0)
        epochs: Number of training epochs
        seed: Random seed
        device: Device to train on
        verbose: Whether to print progress
    
    Returns:
        Dictionary with training results
    """
    set_random_seeds(seed)
    
    # Get data loaders with corrupted labels
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=ExperimentConfig.BATCH_SIZE,
        num_workers=ExperimentConfig.NUM_WORKERS,
        random_labels=(corruption_rate >= 1.0),
        corruption_rate=corruption_rate if corruption_rate < 1.0 else 0.0,
        seed=seed
    )
    
    # Create and train model
    model = get_model(architecture, num_classes=10)
    trainer = Trainer(
        model,
        torch.device(device),
        learning_rate=ExperimentConfig.LEARNING_RATE,
        momentum=ExperimentConfig.MOMENTUM,
        weight_decay=ExperimentConfig.WEIGHT_DECAY,
        scheduler_type=ExperimentConfig.SCHEDULER,
        use_amp=ExperimentConfig.USE_AMP
    )
    
    if verbose:
        print(f"Training with {corruption_rate*100:.0f}% label corruption...")
    
    history = trainer.train(
        train_loader,
        test_loader,
        epochs=epochs,
        verbose=False  # Suppress per-epoch output for cleaner logs
    )
    
    results = {
        'history': history,
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1]
    }
    
    if verbose:
        print(f"  Train Acc: {results['final_train_acc']:.2f}%, "
              f"Test Acc: {results['final_test_acc']:.2f}%")
    
    # Save model
    save_path = ExperimentConfig.get_model_save_path(
        'progressive_corruption',
        f"{architecture}_corrupt{int(corruption_rate*100)}",
        seed
    )
    torch.save(model.state_dict(), save_path)
    
    return results, model


def analyze_corruption_smoothness(
    model: torch.nn.Module,
    device: str = 'cuda',
    n_samples: int = 500,
    verbose: bool = False
) -> Dict:
    """
    Analyze smoothness of a model trained with corrupted labels.
    
    Args:
        model: Trained model
        device: Device to run on
        n_samples: Number of samples for analysis
        verbose: Whether to print progress
    
    Returns:
        Dictionary of smoothness metrics
    """
    from torch.utils.data import DataLoader
    from utils.data_generation import get_cifar10_loaders
    
    _, test_loader = get_cifar10_loaders(batch_size=128, num_workers=4)
    
    # Compute key smoothness metrics
    grad_norm = SmoothnessMetrics.average_gradient_norm(
        model, test_loader, n_samples=n_samples, device=device
    )
    
    spectral_norm = SmoothnessMetrics.spectral_norm(model)
    path_norm = SmoothnessMetrics.path_norm(model)
    
    return {
        'gradient_norm': grad_norm,
        'spectral_norm': spectral_norm,
        'path_norm': path_norm
    }


def run_progressive_corruption_experiment(
    corruption_rates: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    architectures: List[str] = ['resnet18', 'vgg11', 'mlp'],
    seeds: List[int] = [42, 123, 456],
    epochs: int = 200,
    analyze_smoothness: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run progressive label corruption experiment.
    
    Args:
        corruption_rates: List of corruption rates to test
        architectures: List of architectures to train
        seeds: Random seeds
        epochs: Number of training epochs
        analyze_smoothness: Whether to analyze smoothness
        verbose: Whether to print progress
    
    Returns:
        Dictionary of results
    """
    device = get_device()
    ExperimentConfig.create_directories()
    
    results = {}
    
    for arch in architectures:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {arch.upper()}")
            print(f"{'='*60}")
        
        results[arch] = {}
        
        for corruption_rate in corruption_rates:
            if verbose:
                print(f"\n--- Corruption Rate: {corruption_rate*100:.0f}% ---")
            
            results[arch][corruption_rate] = []
            
            for seed in seeds:
                if verbose:
                    print(f"Seed {seed}:")
                
                set_random_seeds(seed)
                
                # Train model
                train_results, model = train_with_corruption(
                    architecture=arch,
                    corruption_rate=corruption_rate,
                    epochs=epochs,
                    seed=seed,
                    device=device,
                    verbose=verbose
                )
                
                # Analyze smoothness
                smoothness = None
                if analyze_smoothness:
                    if verbose:
                        print("  Analyzing smoothness...")
                    smoothness = analyze_corruption_smoothness(
                        model, device=device, n_samples=500, verbose=False
                    )
                    if verbose:
                        print(f"    Gradient norm: {smoothness['gradient_norm']:.6f}")
                
                results[arch][corruption_rate].append({
                    'seed': seed,
                    'training': train_results,
                    'smoothness': smoothness
                })
    
    # Save results
    results_path = ExperimentConfig.get_results_save_path('progressive_corruption')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Progressive corruption experiment completed!")
        print(f"Results saved to {results_path}")
        print(f"{'='*60}")
    
    return results


def summarize_corruption_results(results: Dict) -> None:
    """Summarize progressive corruption results."""
    import numpy as np
    
    print("\n" + "="*70)
    print("PROGRESSIVE CORRUPTION SUMMARY")
    print("="*70)
    
    for arch, arch_results in results.items():
        print(f"\n{arch.upper()}:")
        print("-" * 70)
        print(f"{'Corruption':<12} {'Train Acc':<15} {'Test Acc':<15} {'Gradient Norm':<15}")
        print("-" * 70)
        
        corruption_rates = sorted(arch_results.keys())
        
        for rate in corruption_rates:
            runs = arch_results[rate]
            
            train_accs = [r['training']['final_train_acc'] for r in runs]
            test_accs = [r['training']['final_test_acc'] for r in runs]
            
            grad_norms = [r['smoothness']['gradient_norm'] for r in runs 
                         if r['smoothness'] is not None]
            
            print(f"{rate*100:>6.0f}%      "
                  f"{np.mean(train_accs):>6.2f}±{np.std(train_accs):>4.2f}%    "
                  f"{np.mean(test_accs):>6.2f}±{np.std(test_accs):>4.2f}%    "
                  f"{np.mean(grad_norms) if grad_norms else 0:>8.6f}")
    
    # Compute correlation between corruption and generalization
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    for arch, arch_results in results.items():
        corruption_rates = sorted(arch_results.keys())
        
        all_corruptions = []
        all_test_accs = []
        all_grad_norms = []
        
        for rate in corruption_rates:
            runs = arch_results[rate]
            for run in runs:
                all_corruptions.append(rate)
                all_test_accs.append(run['training']['final_test_acc'])
                if run['smoothness']:
                    all_grad_norms.append(run['smoothness']['gradient_norm'])
        
        # Compute correlations
        if len(all_test_accs) > 0:
            corr_acc = np.corrcoef(all_corruptions, all_test_accs)[0, 1]
            print(f"\n{arch.upper()}:")
            print(f"  Correlation (corruption vs test accuracy): {corr_acc:.4f}")
            
            if len(all_grad_norms) > 0 and len(all_grad_norms) == len(all_corruptions):
                corr_smooth = np.corrcoef(all_corruptions, all_grad_norms)[0, 1]
                print(f"  Correlation (corruption vs gradient norm): {corr_smooth:.4f}")


if __name__ == '__main__':
    # Run progressive corruption experiment
    results = run_progressive_corruption_experiment(
        corruption_rates=[0.0, 0.25, 0.5, 0.75, 1.0],
        architectures=['resnet18'],
        seeds=[42],
        epochs=200,
        analyze_smoothness=True,
        verbose=True
    )
    
    # Summarize results
    summarize_corruption_results(results)

