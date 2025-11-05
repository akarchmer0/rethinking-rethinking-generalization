"""
Experiment 3: Two-Stage Learning (CRITICAL EXPERIMENT)

This is the key novel experiment:
Stage 1: Train Network_1 on random noise images with random labels
Stage 2: Use Network_1 as ground truth to train Network_2

The hypothesis is that Network_2 will generalize well to Network_1's function,
demonstrating that neural networks learn smooth functions even when trained on 
unrealizable data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pickle
import numpy as np
from typing import Dict, Optional, List
from torch.utils.data import DataLoader

from utils.config import ExperimentConfig, set_random_seeds, get_device
from utils.data_generation import RandomNoiseDataset, ModelLabeledDataset
from models.architectures import get_model
from models.training import Trainer
from analysis.metrics import GeneralizationMetrics


def train_stage1_model(
    architecture: str = 'resnet18',
    n_samples: int = 50000,
    noise_type: str = 'uniform',
    epochs: int = 200,
    seed: int = 42,
    device: str = 'cuda',
    verbose: bool = True
) -> torch.nn.Module:
    """
    Stage 1: Train a network on random noise with random labels.
    This creates a "smooth random function".
    
    Args:
        architecture: Architecture to use
        n_samples: Number of random noise images
        noise_type: Type of noise ('uniform' or 'gaussian')
        epochs: Number of training epochs
        seed: Random seed
        device: Device to train on
        verbose: Whether to print progress
    
    Returns:
        Trained model (Network_1)
    """
    if verbose:
        print("="*70)
        print("STAGE 1: Training on Random Noise with Random Labels")
        print("="*70)
    
    set_random_seeds(seed)
    
    # Create random noise dataset with random labels
    train_dataset = RandomNoiseDataset(
        n_samples=n_samples,
        image_size=32,
        n_classes=10,
        noise_type=noise_type,
        random_labels=True,
        seed=seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create and train model
    model = get_model(architecture, num_classes=10)
    trainer = Trainer(
        model,
        torch.device(device),
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        scheduler_type='cosine',
        use_amp=ExperimentConfig.USE_AMP
    )
    
    if verbose:
        print(f"\nTraining {architecture} on {n_samples} random noise images...")
    
    history = trainer.train(
        train_loader,
        test_loader=None,
        epochs=epochs,
        verbose=verbose
    )
    
    if verbose:
        print(f"\nStage 1 complete! Final training accuracy: {history['train_acc'][-1]:.2f}%")
    
    # Save model
    save_path = ExperimentConfig.get_model_save_path(
        'two_stage', f'{architecture}_stage1', seed
    )
    torch.save(model.state_dict(), save_path)
    
    return model


def train_stage2_model(
    stage1_model: torch.nn.Module,
    architecture: str = 'resnet18',
    n_samples: int = 50000,
    noise_type: str = 'uniform',
    epochs: int = 200,
    seed: int = 42,
    device: str = 'cuda',
    verbose: bool = True,
    track_agreement: bool = True,
    test_dataset: Optional[RandomNoiseDataset] = None
) -> tuple:
    """
    Stage 2: Train Network_2 to mimic Network_1's function.
    
    Args:
        stage1_model: Trained model from Stage 1
        architecture: Architecture for Network_2
        n_samples: Number of training samples (NEW random images)
        noise_type: Type of noise
        epochs: Number of training epochs
        seed: Random seed
        device: Device to train on
        verbose: Whether to print progress
        track_agreement: Whether to track agreement with stage1_model during training
        test_dataset: Test dataset for evaluation
    
    Returns:
        Tuple of (trained_model, history, agreement_history)
    """
    if verbose:
        print("\n" + "="*70)
        print("STAGE 2: Training Network_2 to Learn Network_1's Function")
        print("="*70)
    
    set_random_seeds(seed + 1000)  # Different seed for stage 2
    
    # Create dataset labeled by stage1_model
    train_dataset = ModelLabeledDataset(
        model=stage1_model,
        n_samples=n_samples,
        image_size=32,
        noise_type=noise_type,
        device=device,
        seed=seed + 1000
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create test loader if test dataset provided
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # Create and train Network_2
    model2 = get_model(architecture, num_classes=10)
    trainer = Trainer(
        model2,
        torch.device(device),
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        scheduler_type='cosine',
        use_amp=ExperimentConfig.USE_AMP
    )
    
    if verbose:
        print(f"\nTraining {architecture} on {n_samples} images labeled by Network_1...")
    
    # Track agreement if requested
    agreement_history = []
    
    if track_agreement and test_loader is not None:
        # Custom training loop to track agreement
        for epoch in range(1, epochs + 1):
            # Train one epoch
            train_loss, train_acc = trainer.train_epoch(train_loader)
            trainer.history['train_loss'].append(train_loss)
            trainer.history['train_acc'].append(train_acc)
            
            # Compute agreement with stage1_model
            agreement = GeneralizationMetrics.agreement_rate(
                model2, stage1_model, test_loader, device=device
            )
            agreement_history.append(agreement)
            
            # Update scheduler
            if trainer.scheduler is not None:
                trainer.scheduler.step()
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - "
                      f"Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_acc:.2f}%, "
                      f"Agreement: {agreement:.2f}%")
    else:
        # Standard training
        history = trainer.train(
            train_loader,
            test_loader=None,
            epochs=epochs,
            verbose=verbose
        )
    
    if verbose:
        print(f"\nStage 2 complete! Final training accuracy: {trainer.history['train_acc'][-1]:.2f}%")
        if agreement_history:
            print(f"Final agreement with Network_1: {agreement_history[-1]:.2f}%")
    
    # Save model
    save_path = ExperimentConfig.get_model_save_path(
        'two_stage', f'{architecture}_stage2', seed
    )
    torch.save(model2.state_dict(), save_path)
    
    return model2, trainer.history, agreement_history


def evaluate_generalization(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    test_dataset: RandomNoiseDataset,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict:
    """
    Evaluate how well Network_2 generalizes to Network_1's function.
    
    Args:
        model1: Network_1 (ground truth)
        model2: Network_2 (learner)
        test_dataset: Test dataset
        device: Device to run on
        verbose: Whether to print results
    
    Returns:
        Dictionary of evaluation metrics
    """
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Compute metrics
    agreement = GeneralizationMetrics.agreement_rate(
        model1, model2, test_loader, device=device
    )
    
    function_dist = GeneralizationMetrics.function_distance(
        model1, model2, test_loader, device=device
    )
    
    correlation = GeneralizationMetrics.output_correlation(
        model1, model2, test_loader, device=device
    )
    
    results = {
        'agreement_rate': agreement,
        'function_distance': function_dist,
        'output_correlation': correlation
    }
    
    if verbose:
        print("\n" + "="*70)
        print("GENERALIZATION EVALUATION")
        print("="*70)
        print(f"Agreement Rate: {agreement:.2f}%")
        print(f"Function Distance (MSE): {function_dist:.6f}")
        print(f"Output Correlation: {correlation:.4f}")
    
    return results


def run_sample_efficiency_experiment(
    stage1_model: torch.nn.Module,
    sample_sizes: List[int] = [1000, 5000, 10000, 25000, 50000],
    architecture: str = 'resnet18',
    noise_type: str = 'uniform',
    epochs: int = 200,
    seed: int = 42,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict:
    """
    Test sample efficiency: How many samples does Network_2 need?
    
    Args:
        stage1_model: Trained Network_1
        sample_sizes: List of training set sizes to test
        architecture: Architecture for Network_2
        noise_type: Type of noise
        epochs: Number of training epochs
        seed: Random seed
        device: Device to run on
        verbose: Whether to print progress
    
    Returns:
        Dictionary mapping sample sizes to performance
    """
    if verbose:
        print("\n" + "="*70)
        print("SAMPLE EFFICIENCY EXPERIMENT")
        print("="*70)
    
    # Create test dataset
    test_dataset = RandomNoiseDataset(
        n_samples=10000,
        image_size=32,
        n_classes=10,
        noise_type=noise_type,
        random_labels=False,
        seed=seed + 5000
    )
    
    # Label test dataset with stage1_model
    test_dataset_labeled = ModelLabeledDataset(
        model=stage1_model,
        n_samples=10000,
        image_size=32,
        noise_type=noise_type,
        device=device,
        seed=seed + 5000
    )
    
    results = {}
    
    for n_samples in sample_sizes:
        if verbose:
            print(f"\n--- Training with {n_samples} samples ---")
        
        model2, history, agreement_hist = train_stage2_model(
            stage1_model=stage1_model,
            architecture=architecture,
            n_samples=n_samples,
            noise_type=noise_type,
            epochs=epochs,
            seed=seed,
            device=device,
            verbose=False,
            track_agreement=True,
            test_dataset=test_dataset_labeled
        )
        
        # Evaluate
        eval_results = evaluate_generalization(
            stage1_model, model2, test_dataset_labeled,
            device=device, verbose=False
        )
        
        results[n_samples] = {
            'final_agreement': agreement_hist[-1] if agreement_hist else 0,
            'agreement_history': agreement_hist,
            'evaluation': eval_results
        }
        
        if verbose:
            print(f"Final agreement: {results[n_samples]['final_agreement']:.2f}%")
    
    return results


def run_two_stage_learning_experiment(
    architecture: str = 'resnet18',
    stage1_samples: int = 50000,
    stage2_samples: int = 50000,
    noise_type: str = 'uniform',
    epochs: int = 200,
    seed: int = 42,
    test_sample_efficiency: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run the complete two-stage learning experiment.
    
    Args:
        architecture: Architecture to use
        stage1_samples: Number of samples for stage 1
        stage2_samples: Number of samples for stage 2
        noise_type: Type of noise
        epochs: Number of training epochs
        seed: Random seed
        test_sample_efficiency: Whether to run sample efficiency tests
        verbose: Whether to print progress
    
    Returns:
        Complete results dictionary
    """
    device = get_device()
    ExperimentConfig.create_directories()
    set_random_seeds(seed)
    
    # Stage 1: Train on random noise
    model1 = train_stage1_model(
        architecture=architecture,
        n_samples=stage1_samples,
        noise_type=noise_type,
        epochs=epochs,
        seed=seed,
        device=device,
        verbose=verbose
    )
    
    # Create test dataset
    test_dataset = ModelLabeledDataset(
        model=model1,
        n_samples=10000,
        image_size=32,
        noise_type=noise_type,
        device=device,
        seed=seed + 10000
    )
    
    # Stage 2: Train to mimic Network_1
    model2, history2, agreement_hist = train_stage2_model(
        stage1_model=model1,
        architecture=architecture,
        n_samples=stage2_samples,
        noise_type=noise_type,
        epochs=epochs,
        seed=seed,
        device=device,
        verbose=verbose,
        track_agreement=True,
        test_dataset=test_dataset
    )
    
    # Evaluate generalization
    eval_results = evaluate_generalization(
        model1, model2, test_dataset,
        device=device, verbose=verbose
    )
    
    # Compile results
    results = {
        'architecture': architecture,
        'stage1_samples': stage1_samples,
        'stage2_samples': stage2_samples,
        'noise_type': noise_type,
        'epochs': epochs,
        'seed': seed,
        'stage2_history': history2,
        'agreement_over_epochs': agreement_hist,
        'final_agreement': agreement_hist[-1] if agreement_hist else 0,
        'evaluation': eval_results
    }
    
    # Sample efficiency experiment
    if test_sample_efficiency:
        sample_efficiency = run_sample_efficiency_experiment(
            stage1_model=model1,
            sample_sizes=[1000, 5000, 10000, 25000, 50000],
            architecture=architecture,
            noise_type=noise_type,
            epochs=epochs,
            seed=seed,
            device=device,
            verbose=verbose
        )
        
        results['sample_efficiency'] = {
            'sample_sizes': list(sample_efficiency.keys()),
            'accuracies': [sample_efficiency[n]['final_agreement'] 
                          for n in sample_efficiency.keys()],
            'details': sample_efficiency
        }
    
    # Save results
    results_path = ExperimentConfig.get_results_save_path('two_stage_learning')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    if verbose:
        print(f"\n{'='*70}")
        print("Two-stage learning experiment completed!")
        print(f"Results saved to {results_path}")
        print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    # Run the critical two-stage learning experiment
    results = run_two_stage_learning_experiment(
        architecture='resnet18',
        stage1_samples=50000,
        stage2_samples=50000,
        noise_type='uniform',
        epochs=200,
        seed=42,
        test_sample_efficiency=True,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    print(f"Network_2 agreement with Network_1: {results['final_agreement']:.2f}%")
    print(f"Output correlation: {results['evaluation']['output_correlation']:.4f}")
    
    if 'sample_efficiency' in results:
        print("\nSample Efficiency:")
        for n, acc in zip(results['sample_efficiency']['sample_sizes'],
                         results['sample_efficiency']['accuracies']):
            print(f"  {n:6d} samples: {acc:.2f}% agreement")

