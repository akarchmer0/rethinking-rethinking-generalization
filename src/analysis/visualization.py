"""Visualization utilities for analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def plot_learning_curves(histories: Dict[str, Dict], 
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 4)):
    """
    Plot training and test learning curves.
    
    Args:
        histories: Dictionary mapping experiment names to training histories
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    for name, history in histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], label=f'{name} (train)', alpha=0.7)
        if 'test_loss' in history and history['test_loss']:
            axes[0].plot(epochs, history['test_loss'], label=f'{name} (test)', 
                        linestyle='--', alpha=0.7)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    for name, history in histories.items():
        epochs = range(1, len(history['train_acc']) + 1)
        axes[1].plot(epochs, history['train_acc'], label=f'{name} (train)', alpha=0.7)
        if 'test_acc' in history and history['test_acc']:
            axes[1].plot(epochs, history['test_acc'], label=f'{name} (test)',
                        linestyle='--', alpha=0.7)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_smoothness_comparison(smoothness_results: Dict[str, Dict],
                               metrics: List[str] = ['gradient_norm', 'spectral_norm', 
                                                    'path_norm'],
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 6)):
    """
    Plot comparison of smoothness metrics.
    
    Args:
        smoothness_results: Dictionary mapping model names to smoothness metrics
        metrics: List of metrics to plot
        save_path: Path to save figure
        figsize: Figure size
    """
    # Prepare data
    data = []
    for model_name, results in smoothness_results.items():
        for metric in metrics:
            if metric in results:
                data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Value': results[metric]
                })
    
    df = pd.DataFrame(data)
    
    # Create subplot for each metric
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        metric_data = df[df['Metric'] == metric]
        sns.barplot(data=metric_data, x='Model', y='Value', ax=axes[i])
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_frequency_spectra(spectra: Dict[str, np.ndarray],
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 4)):
    """
    Plot 2D frequency spectra as heatmaps.
    
    Args:
        spectra: Dictionary mapping model names to 2D FFT arrays
        save_path: Path to save figure
        figsize: Figure size
    """
    n_models = len(spectra)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, spectrum) in zip(axes, spectra.items()):
        # Log scale for better visualization
        log_spectrum = np.log10(np.abs(spectrum) + 1e-10)
        
        im = ax.imshow(log_spectrum, cmap='viridis', aspect='auto')
        ax.set_title(name)
        ax.set_xlabel('Frequency X')
        ax.set_ylabel('Frequency Y')
        plt.colorbar(im, ax=ax, label='Log Magnitude')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_two_stage_results(results: Dict,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 6)):
    """
    Plot key results from two-stage learning experiment.
    
    Args:
        results: Dictionary containing two-stage learning results
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Agreement rate over epochs
    if 'agreement_over_epochs' in results:
        epochs = range(1, len(results['agreement_over_epochs']) + 1)
        axes[0].plot(epochs, results['agreement_over_epochs'])
        axes[0].axhline(y=results.get('final_agreement', 0), 
                       color='r', linestyle='--', 
                       label=f"Final: {results.get('final_agreement', 0):.2f}%")
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Agreement Rate (%)')
        axes[0].set_title('Network 2 Agreement with Network 1')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Sample efficiency
    if 'sample_efficiency' in results:
        sample_sizes = results['sample_efficiency']['sample_sizes']
        accuracies = results['sample_efficiency']['accuracies']
        axes[1].plot(sample_sizes, accuracies, marker='o')
        axes[1].set_xlabel('Training Samples')
        axes[1].set_ylabel('Agreement Rate (%)')
        axes[1].set_title('Sample Efficiency')
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_sample_efficiency(results: Dict[int, float],
                          baseline_samples: Optional[int] = None,
                          baseline_acc: Optional[float] = None,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (8, 6)):
    """
    Plot sample efficiency curves.
    
    Args:
        results: Dictionary mapping sample sizes to accuracies
        baseline_samples: Baseline number of samples
        baseline_acc: Baseline accuracy
        save_path: Path to save figure
        figsize: Figure size
    """
    sample_sizes = sorted(results.keys())
    accuracies = [results[s] for s in sample_sizes]
    
    plt.figure(figsize=figsize)
    plt.plot(sample_sizes, accuracies, marker='o', linewidth=2, markersize=8)
    
    if baseline_samples and baseline_acc:
        plt.axhline(y=baseline_acc, color='r', linestyle='--', 
                   label=f'Baseline ({baseline_samples} samples)')
        plt.axvline(x=baseline_samples, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Sample Efficiency')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_progressive_corruption(results: Dict[float, Dict],
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 4)):
    """
    Plot results from progressive label corruption experiment.
    
    Args:
        results: Dictionary mapping corruption rates to metrics
        save_path: Path to save figure
        figsize: Figure size
    """
    corruption_rates = sorted(results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot test accuracy vs corruption rate
    test_accs = [results[r]['test_accuracy'] for r in corruption_rates]
    axes[0].plot(corruption_rates, test_accs, marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Label Corruption Rate')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Generalization vs Label Corruption')
    axes[0].grid(True, alpha=0.3)
    
    # Plot smoothness vs corruption rate
    if 'smoothness' in results[corruption_rates[0]]:
        smoothness = [results[r]['smoothness'] for r in corruption_rates]
        axes[1].plot(corruption_rates, smoothness, marker='o', linewidth=2, markersize=8)
        axes[1].set_xlabel('Label Corruption Rate')
        axes[1].set_ylabel('Smoothness Metric')
        axes[1].set_title('Smoothness vs Label Corruption')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_comparison_grid(results: Dict[str, Dict[str, float]],
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 8)):
    """
    Create a grid comparison of multiple metrics across models.
    
    Args:
        results: Dictionary mapping model names to metric dictionaries
        save_path: Path to save figure
        figsize: Figure size
    """
    df = pd.DataFrame(results).T
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Value'})
    plt.title('Model Comparison Across Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Model')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_random_walk_comparison(results: Dict,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (15, 5)):
    """
    Compare random walk results to uniform random baseline.
    
    Creates a 3-subplot figure showing:
    1. Test accuracy: walk continuation vs uniform random
    2. Smoothness metrics comparison
    3. Two-stage agreement: walk vs uniform
    
    Args:
        results: Results dictionary from run_random_walk_experiment
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Subplot 1: Test Accuracy Comparison
    ax1 = axes[0]
    baseline = results['baseline']
    test_accs = [
        baseline['final_train_acc'],
        baseline['test_acc_walk'],
        baseline['test_acc_uniform']
    ]
    labels = ['Train', 'Test\n(Walk)', 'Test\n(Uniform)']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax1.bar(labels, test_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Baseline Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Subplot 2: Smoothness Metrics
    ax2 = axes[1]
    smoothness = results['smoothness']
    
    # Select key metrics
    metrics_to_plot = ['gradient_norm', 'lipschitz_constant', 'local_variation']
    metric_labels = ['Gradient\nNorm', 'Lipschitz\nConstant', 'Local\nVariation']
    metric_values = [smoothness[m] for m in metrics_to_plot]
    
    bars = ax2.bar(metric_labels, metric_values, color='#9b59b6', alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Metric Value', fontsize=12)
    ax2.set_title('Smoothness Metrics', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Subplot 3: Two-Stage Agreement
    ax3 = axes[2]
    two_stage = results['two_stage']
    
    agreements = [
        two_stage['variant_a_walk']['agreement'],
        two_stage['variant_b_uniform']['agreement']
    ]
    labels = ['Walk\nData', 'Uniform\nData']
    colors = ['#3498db', '#e74c3c']
    
    bars = ax3.bar(labels, agreements, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Agreement (%)', fontsize=12)
    ax3.set_title('Two-Stage Agreement', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Random Walk Experiment Results', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {save_path}")
    
    plt.show()

