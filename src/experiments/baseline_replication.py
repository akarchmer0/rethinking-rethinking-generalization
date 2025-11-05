"""
Experiment 1: Baseline Replication

Replicate Zhang et al.'s core findings:
- Train networks on CIFAR-10 with true labels vs random labels
- Achieve ~0 training error on both
- Measure test accuracy (should be ~90% for true, ~10% for random)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pickle
from typing import Dict, Optional
from tqdm import tqdm

from utils.config import ExperimentConfig, set_random_seeds, get_device
from utils.data_generation import get_cifar10_loaders
from models.architectures import get_model
from models.training import Trainer


def run_baseline_replication(
    architectures: list = ['resnet18', 'vgg11', 'mlp'],
    seeds: list = [42],
    epochs: int = 50,
    save_models: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run baseline replication experiment.
    
    Args:
        architectures: List of architectures to train
        seeds: Random seeds for reproducibility
        epochs: Number of training epochs
        save_models: Whether to save trained models
        verbose: Whether to print progress
    
    Returns:
        Dictionary of results
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
            print(f"Training {arch.upper()}")
            print(f"{'='*60}")
        
        results['true_labels'][arch] = []
        results['random_labels'][arch] = []
        
        for seed in seeds:
            if verbose:
                print(f"\n--- Seed {seed} ---")
            
            set_random_seeds(seed)
            
            # Train with true labels
            if verbose:
                print("\nTraining with TRUE LABELS...")
            
            train_loader, test_loader = get_cifar10_loaders(
                batch_size=ExperimentConfig.BATCH_SIZE,
                num_workers=ExperimentConfig.NUM_WORKERS,
                random_labels=False,
                seed=seed
            )
            
            model_true = get_model(arch, num_classes=10)
            trainer_true = Trainer(
                model_true,
                device,
                learning_rate=ExperimentConfig.LEARNING_RATE,
                momentum=ExperimentConfig.MOMENTUM,
                weight_decay=ExperimentConfig.WEIGHT_DECAY,
                scheduler_type=ExperimentConfig.SCHEDULER
            )
            
            history_true = trainer_true.train(
                train_loader,
                test_loader,
                epochs=epochs,
                verbose=verbose
            )
            
            # Save model
            if save_models:
                save_path = ExperimentConfig.get_model_save_path(
                    'baseline', f'{arch}_true', seed
                )
                torch.save(model_true.state_dict(), save_path)
            
            results['true_labels'][arch].append({
                'seed': seed,
                'history': history_true,
                'final_train_acc': history_true['train_acc'][-1],
                'final_test_acc': history_true['test_acc'][-1]
            })
            
            # Train with random labels
            if verbose:
                print("\nTraining with RANDOM LABELS...")
            
            train_loader, test_loader = get_cifar10_loaders(
                batch_size=ExperimentConfig.BATCH_SIZE,
                num_workers=ExperimentConfig.NUM_WORKERS,
                random_labels=True,
                seed=seed
            )
            
            model_random = get_model(arch, num_classes=10)
            trainer_random = Trainer(
                model_random,
                device,
                learning_rate=ExperimentConfig.LEARNING_RATE,
                momentum=ExperimentConfig.MOMENTUM,
                weight_decay=ExperimentConfig.WEIGHT_DECAY,
                scheduler_type=ExperimentConfig.SCHEDULER
            )
            
            history_random = trainer_random.train(
                train_loader,
                test_loader,
                epochs=epochs,
                verbose=verbose
            )
            
            # Save model
            if save_models:
                save_path = ExperimentConfig.get_model_save_path(
                    'baseline', f'{arch}_random', seed
                )
                torch.save(model_random.state_dict(), save_path)
            
            results['random_labels'][arch].append({
                'seed': seed,
                'history': history_random,
                'final_train_acc': history_random['train_acc'][-1],
                'final_test_acc': history_random['test_acc'][-1]
            })
            
            if verbose:
                print(f"\nResults for {arch} (seed {seed}):")
                print(f"  True labels  - Train: {history_true['train_acc'][-1]:.2f}%, "
                      f"Test: {history_true['test_acc'][-1]:.2f}%")
                print(f"  Random labels - Train: {history_random['train_acc'][-1]:.2f}%, "
                      f"Test: {history_random['test_acc'][-1]:.2f}%")
    
    # Save results
    results_path = ExperimentConfig.get_results_save_path('baseline_replication')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Experiment completed!")
        print(f"Results saved to {results_path}")
        print(f"{'='*60}")
    
    return results


def summarize_results(results: Dict) -> None:
    """Print summary of baseline replication results."""
    import numpy as np
    
    print("\n" + "="*70)
    print("BASELINE REPLICATION SUMMARY")
    print("="*70)
    
    for label_type in ['true_labels', 'random_labels']:
        print(f"\n{label_type.upper().replace('_', ' ')}:")
        print("-" * 70)
        
        for arch, arch_results in results[label_type].items():
            train_accs = [r['final_train_acc'] for r in arch_results]
            test_accs = [r['final_test_acc'] for r in arch_results]
            
            print(f"\n{arch.upper()}:")
            print(f"  Train Accuracy: {np.mean(train_accs):.2f}% ± {np.std(train_accs):.2f}%")
            print(f"  Test Accuracy:  {np.mean(test_accs):.2f}% ± {np.std(test_accs):.2f}%")


if __name__ == '__main__':
    # Run experiment
    results = run_baseline_replication(
        architectures=['resnet18'],  # Start with ResNet-18
        seeds=[42],  # Single seed for quick testing
        epochs=50,
        save_models=True,
        verbose=True
    )
    
    # Summarize
    summarize_results(results)

