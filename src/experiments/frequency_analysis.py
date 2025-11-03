"""
Experiment 4: Frequency Analysis

Analyze neural network functions in the frequency domain using Fourier transforms.
This reveals the spectral characteristics of learned functions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pickle
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import fft

from utils.config import ExperimentConfig, set_random_seeds, get_device
from models.architectures import get_model


def sample_function_on_grid(
    model: torch.nn.Module,
    resolution: int = 64,
    device: str = 'cuda',
    input_dim: int = 3072,
    n_classes: int = 10
) -> np.ndarray:
    """
    Sample neural network function on a 2D grid in input space.
    
    Args:
        model: Neural network model
        resolution: Grid resolution (resolution x resolution)
        device: Device to run on
        input_dim: Dimension of input space
        n_classes: Number of output classes
    
    Returns:
        2D array of function values
    """
    model = model.to(device)
    model.eval()
    
    # Create 2D grid in a random 2D subspace of input space
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xx, yy = np.meshgrid(x, y)
    
    # Random projection to 2D
    np.random.seed(42)
    proj_matrix = np.random.randn(input_dim, 2)
    proj_matrix = proj_matrix / np.linalg.norm(proj_matrix, axis=0)
    
    # Sample function on grid
    grid_values = np.zeros((resolution, resolution, n_classes))
    
    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                # Create point in 2D space
                point_2d = np.array([xx[i, j], yy[i, j]])
                
                # Project to input space
                point_input = proj_matrix @ point_2d
                
                # Add to mean point (use zeros as baseline)
                point_input = point_input.reshape(3, 32, 32)
                
                # Convert to tensor and get model output
                point_tensor = torch.from_numpy(point_input).float().unsqueeze(0).to(device)
                output = model(point_tensor)
                grid_values[i, j] = output.cpu().numpy()
    
    # Return first output dimension for visualization
    return grid_values[:, :, 0]


def fourier_analysis_2d(
    model: torch.nn.Module,
    resolution: int = 64,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Compute 2D Fourier transform of network function.
    
    Args:
        model: Neural network model
        resolution: Grid resolution
        device: Device to run on
    
    Returns:
        2D Fourier spectrum (magnitude)
    """
    # Sample function on grid
    function_grid = sample_function_on_grid(model, resolution, device)
    
    # Compute 2D FFT
    spectrum = fft.fft2(function_grid)
    spectrum = fft.fftshift(spectrum)
    
    # Return magnitude
    return np.abs(spectrum)


def directional_fourier(
    model: torch.nn.Module,
    n_directions: int = 100,
    n_points: int = 256,
    device: str = 'cuda',
    input_dim: int = 3072
) -> Dict[str, np.ndarray]:
    """
    Compute 1D Fourier transforms along random directions in input space.
    
    Args:
        model: Neural network model
        n_directions: Number of random directions to sample
        n_points: Number of points along each direction
        device: Device to run on
        input_dim: Dimension of input space
    
    Returns:
        Dictionary with frequencies and average spectrum
    """
    model = model.to(device)
    model.eval()
    
    spectra = []
    
    for _ in range(n_directions):
        # Random direction
        direction = np.random.randn(input_dim)
        direction = direction / np.linalg.norm(direction)
        direction = direction.reshape(3, 32, 32)
        
        # Sample along direction
        t = np.linspace(-2, 2, n_points)
        function_values = []
        
        with torch.no_grad():
            for ti in t:
                point = ti * direction
                point_tensor = torch.from_numpy(point).float().unsqueeze(0).to(device)
                output = model(point_tensor)
                function_values.append(output[0, 0].cpu().item())
        
        function_values = np.array(function_values)
        
        # Compute 1D FFT
        spectrum = fft.fft(function_values)
        spectrum_magnitude = np.abs(spectrum)
        spectra.append(spectrum_magnitude)
    
    # Average spectrum
    avg_spectrum = np.mean(spectra, axis=0)
    frequencies = fft.fftfreq(n_points)
    
    return {
        'frequencies': frequencies,
        'spectrum': avg_spectrum,
        'all_spectra': np.array(spectra)
    }


def analyze_frequency_content(
    model: torch.nn.Module,
    device: str = 'cuda',
    resolution: int = 64,
    n_directions: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Comprehensive frequency analysis of a model.
    
    Args:
        model: Neural network model
        device: Device to run on
        resolution: Resolution for 2D FFT
        n_directions: Number of directions for 1D FFT
        verbose: Whether to print progress
    
    Returns:
        Dictionary of frequency analysis results
    """
    if verbose:
        print("Performing frequency analysis...")
    
    # 2D Fourier analysis
    if verbose:
        print("  - Computing 2D Fourier transform...")
    spectrum_2d = fourier_analysis_2d(model, resolution, device)
    
    # Directional Fourier analysis
    if verbose:
        print("  - Computing directional Fourier transforms...")
    directional_results = directional_fourier(model, n_directions, device=device)
    
    # Compute frequency statistics
    # High frequency energy (outer half of spectrum)
    center = resolution // 2
    r = resolution // 4
    mask_low = np.zeros_like(spectrum_2d, dtype=bool)
    y, x = np.ogrid[:resolution, :resolution]
    mask_low[(x - center)**2 + (y - center)**2 <= r**2] = True
    
    low_freq_energy = np.sum(spectrum_2d[mask_low])
    high_freq_energy = np.sum(spectrum_2d[~mask_low])
    total_energy = low_freq_energy + high_freq_energy
    
    high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
    
    results = {
        'spectrum_2d': spectrum_2d,
        'directional_frequencies': directional_results['frequencies'],
        'directional_spectrum': directional_results['spectrum'],
        'low_freq_energy': low_freq_energy,
        'high_freq_energy': high_freq_energy,
        'high_freq_ratio': high_freq_ratio
    }
    
    if verbose:
        print(f"\nFrequency statistics:")
        print(f"  Low frequency energy: {low_freq_energy:.2e}")
        print(f"  High frequency energy: {high_freq_energy:.2e}")
        print(f"  High frequency ratio: {high_freq_ratio:.4f}")
    
    return results


def run_frequency_analysis(
    architectures: list = ['resnet18', 'vgg11', 'mlp'],
    seeds: list = [42, 123, 456],
    resolution: int = 64,
    n_directions: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Run frequency analysis on trained models.
    
    Args:
        architectures: List of architectures to analyze
        seeds: Random seeds
        resolution: Resolution for 2D FFT
        n_directions: Number of directions for 1D FFT
        verbose: Whether to print progress
    
    Returns:
        Dictionary of frequency analysis results
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
            
            # Analyze model trained with true labels
            if verbose:
                print("\nAnalyzing model trained with TRUE LABELS...")
            
            model_true = get_model(arch, num_classes=10)
            model_path = ExperimentConfig.get_model_save_path(
                'baseline', f'{arch}_true', seed
            )
            
            if model_path.exists():
                model_true.load_state_dict(torch.load(model_path))
                freq_analysis_true = analyze_frequency_content(
                    model_true, device=device, resolution=resolution,
                    n_directions=n_directions, verbose=verbose
                )
                results['true_labels'][arch].append({
                    'seed': seed,
                    'frequency_analysis': freq_analysis_true
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
                freq_analysis_random = analyze_frequency_content(
                    model_random, device=device, resolution=resolution,
                    n_directions=n_directions, verbose=verbose
                )
                results['random_labels'][arch].append({
                    'seed': seed,
                    'frequency_analysis': freq_analysis_random
                })
            else:
                print(f"Warning: Model not found at {model_path}")
    
    # Save results
    results_path = ExperimentConfig.get_results_save_path('frequency_analysis')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Frequency analysis completed!")
        print(f"Results saved to {results_path}")
        print(f"{'='*60}")
    
    return results


def compare_frequency_content(results: Dict) -> None:
    """Compare frequency content between true and random label models."""
    import numpy as np
    
    print("\n" + "="*70)
    print("FREQUENCY CONTENT COMPARISON")
    print("="*70)
    
    for arch in results['true_labels'].keys():
        print(f"\n{arch.upper()}:")
        print("-" * 70)
        
        true_hf_ratios = [r['frequency_analysis']['high_freq_ratio'] 
                         for r in results['true_labels'][arch]]
        random_hf_ratios = [r['frequency_analysis']['high_freq_ratio'] 
                           for r in results['random_labels'][arch]]
        
        print(f"\nHigh Frequency Ratio:")
        print(f"  True labels:   {np.mean(true_hf_ratios):.4f} ± {np.std(true_hf_ratios):.4f}")
        print(f"  Random labels: {np.mean(random_hf_ratios):.4f} ± {np.std(random_hf_ratios):.4f}")
        
        ratio = np.mean(random_hf_ratios) / (np.mean(true_hf_ratios) + 1e-10)
        print(f"  Ratio (random/true): {ratio:.2f}x")


if __name__ == '__main__':
    # Run frequency analysis (assumes baseline experiment has been run)
    results = run_frequency_analysis(
        architectures=['resnet18'],
        seeds=[42],
        resolution=64,
        n_directions=100,
        verbose=True
    )
    
    # Compare
    compare_frequency_content(results)

