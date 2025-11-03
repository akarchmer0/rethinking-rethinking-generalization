#!/usr/bin/env python
"""
Master script to run all experiments in sequence.

This script runs all five experiments and generates figures.
Warning: This may take several hours to complete!
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

import argparse
from src.utils.config import ExperimentConfig


def run_all_experiments(quick_test=False):
    """
    Run all experiments in sequence.
    
    Args:
        quick_test: If True, run with reduced parameters for quick testing
    """
    print("="*80)
    print("RUNNING ALL EXPERIMENTS")
    print("="*80)
    
    # Set parameters based on mode
    if quick_test:
        print("\n*** QUICK TEST MODE ***")
        print("Using reduced parameters for faster execution\n")
        architectures = ['resnet18']
        seeds = [42]
        epochs = 50
    else:
        print("\n*** FULL EXPERIMENT MODE ***")
        print("This will take several hours to complete\n")
        architectures = ['resnet18', 'vgg11', 'mlp']
        seeds = [42, 123, 456]
        epochs = 200
    
    # Import experiments
    from src.experiments import baseline_replication
    from src.experiments import smoothness_analysis
    from src.experiments import two_stage_learning
    from src.experiments import frequency_analysis
    from src.experiments import complexity_measures
    
    # Experiment 1: Baseline Replication
    print("\n" + "="*80)
    print("EXPERIMENT 1: BASELINE REPLICATION")
    print("="*80)
    baseline_results = baseline_replication.run_baseline_replication(
        architectures=architectures,
        seeds=seeds,
        epochs=epochs,
        save_models=True,
        verbose=True
    )
    baseline_replication.summarize_results(baseline_results)
    
    # Experiment 2: Smoothness Analysis
    print("\n" + "="*80)
    print("EXPERIMENT 2: SMOOTHNESS ANALYSIS")
    print("="*80)
    smoothness_results = smoothness_analysis.run_smoothness_analysis(
        architectures=architectures,
        seeds=seeds,
        n_samples=1000 if not quick_test else 500,
        verbose=True
    )
    smoothness_analysis.compare_smoothness(smoothness_results)
    
    # Experiment 3: Two-Stage Learning (KEY EXPERIMENT)
    print("\n" + "="*80)
    print("EXPERIMENT 3: TWO-STAGE LEARNING (CRITICAL)")
    print("="*80)
    two_stage_results = two_stage_learning.run_two_stage_learning_experiment(
        architecture='resnet18',
        stage1_samples=50000 if not quick_test else 10000,
        stage2_samples=50000 if not quick_test else 10000,
        noise_type='uniform',
        epochs=epochs,
        seed=seeds[0],
        test_sample_efficiency=not quick_test,  # Skip in quick mode
        verbose=True
    )
    
    print("\n" + "="*80)
    print("KEY FINDINGS FROM TWO-STAGE LEARNING:")
    print("="*80)
    print(f"Network_2 agreement with Network_1: {two_stage_results['final_agreement']:.2f}%")
    print(f"Output correlation: {two_stage_results['evaluation']['output_correlation']:.4f}")
    
    if 'sample_efficiency' in two_stage_results:
        print("\nSample Efficiency:")
        for n, acc in zip(two_stage_results['sample_efficiency']['sample_sizes'],
                         two_stage_results['sample_efficiency']['accuracies']):
            print(f"  {n:6d} samples: {acc:.2f}% agreement")
    
    # Experiment 4: Frequency Analysis
    print("\n" + "="*80)
    print("EXPERIMENT 4: FREQUENCY ANALYSIS")
    print("="*80)
    freq_results = frequency_analysis.run_frequency_analysis(
        architectures=architectures,
        seeds=seeds,
        resolution=64 if not quick_test else 32,
        n_directions=100 if not quick_test else 50,
        verbose=True
    )
    frequency_analysis.compare_frequency_content(freq_results)
    
    # Experiment 5: Progressive Corruption
    print("\n" + "="*80)
    print("EXPERIMENT 5: PROGRESSIVE CORRUPTION")
    print("="*80)
    corruption_results = complexity_measures.run_progressive_corruption_experiment(
        corruption_rates=[0.0, 0.25, 0.5, 0.75, 1.0],
        architectures=architectures,
        seeds=seeds,
        epochs=epochs,
        analyze_smoothness=True,
        verbose=True
    )
    complexity_measures.summarize_corruption_results(corruption_results)
    
    # Summary
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    print("\nResults saved to:")
    print(f"  {ExperimentConfig.PROCESSED_DIR}")
    print("\nTo generate figures, run:")
    print("  jupyter notebook notebooks/figure_generation.ipynb")
    print("\nOr explore results interactively:")
    print("  jupyter notebook notebooks/exploratory_analysis.ipynb")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run all generalization experiments'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with reduced parameters'
    )
    
    args = parser.parse_args()
    
    # Create directories
    ExperimentConfig.create_directories()
    
    # Run experiments
    try:
        run_all_experiments(quick_test=args.quick_test)
    except KeyboardInterrupt:
        print("\n\nExperiments interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

